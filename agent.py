"""Anthropic API integration: prompts, data context, and the agent loop.

Step 4 of the build order: the model can call the run_sql tool; the loop
executes queries with DuckDB, feeds results (or errors) back, and repeats
until the model returns a final text answer.
"""

import os
from pathlib import Path

import anthropic
import pandas as pd
from dotenv import load_dotenv

import json

from tools import make_chart, python_namespace, run_python, run_sql

MODEL = "claude-sonnet-5"
MAX_TOKENS = 16000
MAX_TOOL_CALLS = 8       # hard cap per question, prevents runaway loops
MAX_RETRIES_PER_QUERY = 2  # consecutive failed queries before giving up

SYSTEM_PROMPT = """\
You are an experienced data analyst helping a user understand a CSV file \
they uploaded. You are given the dataset's schema, an automatic data \
quality profile, and the log of cleaning actions that were applied to the \
working copy.

You have two tools:
- run_sql: DuckDB SQL against the dataset, available as a table named \
`data`. SELECT queries only. Prefer this for filtering, aggregation, \
grouping, sorting, and lookups.
- run_python: Python with `df` (the working dataframe), pandas as `pd`, \
and numpy as `np`. scipy, statsmodels, and sklearn are NOT available. \
Variables persist between run_python calls within a question, like \
notebook cells. Prefer this for statistics (correlations, distributions, \
quantiles), reshaping/pivoting, and anything SQL makes awkward. For \
significance testing, implement it with numpy: a permutation test for \
group differences, a bootstrap (np.random.choice, replace=True) for \
confidence intervals. Print results or end with an expression; keep \
outputs small (head(), round(), describe()).
- make_chart: render a Plotly chart from a simple spec. Chart from \
"last_result" (the dataframe returned by your most recent successful \
run_sql/run_python call — aggregate first, then chart) or from \
"working_data" (the full dataset, e.g. for histograms and scatters).

Chart rules:
- Only chart when a visual adds something beyond the numbers. A single \
number or a two-row comparison does not need a chart.
- Bar charts for comparisons, line for time series, scatter for \
relationships, histogram/box for distributions and outlier discussions. \
Pie only for shares across at most 5 categories.
- If a line chart's zoomed y-axis exaggerates a small change, say so in \
your answer.

Follow the analyst workflow, in order:
1. Understand the ask. Restate ambiguous questions as a concrete, \
answerable version, and state your assumptions explicitly in the answer.
2. Check the data can answer it using the schema and profile. If it \
cannot, say so plainly instead of running queries. Never fabricate.
3. Plan the minimal set of steps before running anything.
4. Execute with the right tool. Never guess a number a tool can compute.
5. Verify before answering: sanity-check that row counts are plausible, \
group totals reconcile with overall totals, and percentages sum to ~100. \
Cross-check an important number with a second cheap query when warranted.
6. Communicate finding first, supporting numbers second, caveats third \
(nulls, small samples, outliers, incomplete periods, applied cleaning).

Rules:
- Every specific number or named ranking in your answer must come from \
a tool call made while answering this question, or verbatim from the \
data profile above. Never state a figure from memory or estimation. If \
you want to add a side observation you have not computed, either run \
the query for it or phrase it as a suggestion to explore ("I can check \
whether…"), never as a stated fact. This includes rankings: never claim \
what would lead an alternative ranking unless you ran that ranking's \
query — values fetched for one ranking do not establish another.
- Only reference columns that exist in the schema. Never invent column \
names or values.
- If a query or code errors, read the error, fix it, and retry.
- Never imply causation from correlation; say "is associated with".
- Round sensibly in prose (12.3%, $1.2M) but never round inside \
intermediate calculations.
- Flag small sample sizes and incomplete periods instead of reporting \
them as confident findings.
- Mention cleaning actions or data quality issues when they affect the \
interpretation of an answer.
- Keep answers concise and readable for a non-technical user.
"""

RUN_SQL_TOOL = {
    "name": "run_sql",
    "description": (
        "Run a read-only DuckDB SQL query against the uploaded dataset. "
        "The dataset is a single table named `data`. Use this for "
        "filtering, aggregation, grouping, sorting, and lookups. Only "
        "reference columns that exist in the provided schema. SELECT "
        "statements only."
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The SQL query to run against the table `data`.",
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}

RUN_PYTHON_TOOL = {
    "name": "run_python",
    "description": (
        "Run Python code against the uploaded dataset. The working "
        "dataframe is available as `df`, with pandas as `pd` and numpy as "
        "`np`. scipy, statsmodels, and sklearn do not exist here — write "
        "statistical tests in plain numpy (permutation test for "
        "significance, bootstrap for confidence intervals). Variables "
        "persist between calls within a question, like notebook cells. "
        "Use this for statistics, reshaping, and anything SQL makes "
        "awkward. The printed output and the value of the last expression "
        "are returned; output is truncated, so keep it small (head(), "
        "describe(), round())."
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to run. `df`, `pd`, and `np` are available.",
            }
        },
        "required": ["code"],
        "additionalProperties": False,
    },
}

MAKE_CHART_TOOL = {
    "name": "make_chart",
    "description": (
        "Render a Plotly chart from a simple spec. Only use it when a "
        "visual genuinely adds something beyond the numbers in your "
        "answer. source='last_result' charts the dataframe returned by "
        "your most recent successful run_sql/run_python call (aggregate "
        "first, then chart); source='working_data' charts the full "
        "dataset (for histograms, box plots, scatters). Bars are "
        "automatically sorted by value with a zero-baseline y-axis; pie "
        "charts with more than 5 categories become sorted bar charts."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "chart_type": {
                "type": "string",
                "enum": ["bar", "line", "scatter", "histogram", "box", "pie"],
            },
            "source": {"type": "string", "enum": ["working_data", "last_result"]},
            "x": {
                "type": "string",
                "description": "Column for the x-axis (category, date, or "
                "numeric value for histograms; group labels for box plots).",
            },
            "y": {
                "type": "string",
                "description": "Column for the y-axis / values. Required for "
                "every type except histogram (and optional for box).",
            },
            "color": {
                "type": "string",
                "description": "Optional column to color/group series by.",
            },
            "title": {"type": "string", "description": "Chart title (required)."},
        },
        "required": ["chart_type", "source", "title"],
        "additionalProperties": False,
    },
}

TOOLS = [RUN_SQL_TOOL, RUN_PYTHON_TOOL, MAKE_CHART_TOOL]


def get_client() -> anthropic.Anthropic | None:
    """Return an Anthropic client, or None when no API key is configured.

    Checks Streamlit secrets first (how Streamlit Community Cloud provides
    the key), then falls back to the local .env file.
    """
    key = None
    try:
        import streamlit as st
        key = st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        key = None  # no secrets file, or running outside Streamlit
    if not key:
        load_dotenv(Path(__file__).parent / ".env")
        key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    return anthropic.Anthropic(api_key=key)


def build_data_context(view: str, profile: dict, default_log: list[dict],
                       user_log: list[dict]) -> str:
    """Render the schema, profile, and cleaning log as text for the model."""
    overview = profile["overview"]
    lines = [
        f"# Dataset context (the user is viewing the {view})",
        "",
        f"Rows: {overview['n_rows']:,} | Columns: {overview['n_cols']} | "
        f"Duplicate rows: {overview['duplicate_rows']:,}",
        "",
        "## Schema and per-column profile",
    ]
    for col in profile["columns"]:
        parts = [
            f"- {col['name']} ({col['kind']}, dtype={col['dtype']}): "
            f"{col['null_pct']}% null, {col['n_unique']:,} unique"
        ]
        if col["flags"]:
            parts.append(f", flags: {', '.join(col['flags'])}")
        stats = col["stats"]
        if stats and "mean" in stats:
            parts.append(
                f", min={stats['min']:,.4g}, max={stats['max']:,.4g}, "
                f"mean={stats['mean']:,.4g}, median={stats['median']:,.4g}, "
                f"std={stats['std']:,.4g}, IQR outliers={stats['iqr_outliers']}"
            )
        elif stats:
            parts.append(f", range {stats['min']} to {stats['max']}")
        if col["top_values"]:
            top = ", ".join(
                f"{v['value']!r} ({v['count']:,} rows, {v['pct']}%)"
                for v in col["top_values"]
            )
            parts.append(f", top values: {top}")
        lines.append("".join(parts))

    lines += ["", "## Data quality findings"]
    if profile["issues"]:
        lines += [f"- [{i['column']}] {i['issue']}" for i in profile["issues"]]
    else:
        lines.append("- none detected")

    lines += ["", "## Cleaning log (already applied to the working copy)"]
    if default_log or user_log:
        lines += [
            f"- [automatic] {e['column']}: {e['action']} — {e['affected']:,} "
            f"affected ({e['why']})"
            for e in default_log
        ]
        lines += [
            f"- [user-approved] {e['column']}: {e['action']} — "
            f"{e['affected']:,} affected ({e['why']})"
            for e in user_log
        ]
    else:
        lines.append("- no cleaning was needed")

    return "\n".join(lines)


SUGGESTION_LABELS = ["Aggregation (SQL)", "Statistics (Python)", "Chart", "Data limits"]

# Dataset-agnostic fallbacks, used when the generation call fails.
GENERIC_QUESTIONS = [
    "Which group or category has the highest total in this data?",
    "Are any two numeric columns meaningfully correlated?",
    "Show me the distribution of the main numeric column as a chart.",
    "How will these numbers trend next month?",
]


def suggest_questions(client: anthropic.Anthropic, data_context: str) -> list[dict]:
    """One cheap model call producing 4 example questions for a fresh
    upload — one per category in SUGGESTION_LABELS. Falls back to
    generic phrasings on any failure."""
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=500,
            system="You write example questions for a data analysis chat app.",
            messages=[{
                "role": "user",
                "content": (
                    f"{data_context}\n\n"
                    "Write exactly 4 short, natural questions a user could ask "
                    "about this dataset, in this order: (1) an aggregation "
                    "question answerable with SQL; (2) a statistical question "
                    "needing a correlation or significance check; (3) a "
                    "question that deserves a chart; (4) a question this "
                    "dataset cannot fully answer, so the analyst must say so "
                    "honestly. Under 18 words each. Reference real column "
                    "names or values from the context."
                ),
            }],
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "questions": {
                                "type": "array",
                                "items": {"type": "string"},
                            }
                        },
                        "required": ["questions"],
                        "additionalProperties": False,
                    },
                }
            },
        )
        text = next(b.text for b in response.content if b.type == "text")
        questions = json.loads(text)["questions"]
    except Exception:
        questions = GENERIC_QUESTIONS
    questions = (list(questions) + GENERIC_QUESTIONS)[:4]
    return [
        {"label": label, "question": q}
        for label, q in zip(SUGGESTION_LABELS, questions)
    ]


def answer_question(client: anthropic.Anthropic, chat_history: list[dict],
                    data_context: str, df: pd.DataFrame) -> tuple[str, list[dict], int]:
    """The agent loop: ask the model, execute any tool calls, feed the
    results back, and repeat until it returns a final text answer.

    Returns (answer_text, tool_events, n_turns) where each tool event
    records the executed input, whether it succeeded, its text output,
    and any result dataframe/figure for display in the UI; n_turns is
    the number of model calls the loop made.
    """
    # Strip UI-only fields (like stored tool events) from past turns.
    messages = [{"role": m["role"], "content": m["content"]} for m in chat_history]
    system = [
        {"type": "text", "text": SYSTEM_PROMPT},
        {"type": "text", "text": data_context, "cache_control": {"type": "ephemeral"}},
    ]

    tool_events: list[dict] = []
    tool_calls = 0
    consecutive_errors = 0
    # One shared namespace per question so run_python calls behave like
    # notebook cells: variables defined in one call persist to the next.
    py_namespace = python_namespace(df)
    # The dataframe from the most recent successful tool call, so
    # make_chart can plot what the agent just computed.
    last_result = None
    n_turns = 0

    while True:
        n_turns += 1
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            thinking={"type": "adaptive"},
            system=system,
            tools=TOOLS,
            messages=messages,
        )
        tool_uses = [b for b in response.content if b.type == "tool_use"]

        if not tool_uses:
            text = "".join(b.text for b in response.content if b.type == "text")
            if response.stop_reason == "refusal":
                text = "The model declined to answer this question."
            return text or "(no answer returned)", tool_events, n_turns

        # Echo the assistant turn back unchanged (including thinking blocks),
        # then answer every tool call in a single user message.
        messages.append({"role": "assistant", "content": response.content})
        results = []
        for tool_use in tool_uses:
            if tool_use.name == "make_chart":
                source = json.dumps(tool_use.input, indent=2)
            else:
                source = tool_use.input.get("query") or tool_use.input.get("code", "")
            if tool_calls >= MAX_TOOL_CALLS:
                outcome = {
                    "ok": False,
                    "result": None,
                    "output": f"Tool call limit ({MAX_TOOL_CALLS}) reached. Do not "
                    "run more queries or code; give your best final answer from "
                    "the results you already have, and state what is missing.",
                }
            else:
                if tool_use.name == "run_sql":
                    outcome = run_sql(source, df)
                elif tool_use.name == "run_python":
                    outcome = run_python(source, namespace=py_namespace)
                else:
                    outcome = make_chart(tool_use.input, df, last_result)
                tool_calls += 1
                if outcome["ok"] and isinstance(outcome["result"], pd.DataFrame):
                    last_result = outcome["result"]
                tool_events.append({"tool": tool_use.name, "input": source, **outcome})

            if outcome["ok"]:
                consecutive_errors = 0
                content = outcome["output"]
            else:
                consecutive_errors += 1
                content = outcome["output"]
                if consecutive_errors > MAX_RETRIES_PER_QUERY:
                    content += (
                        "\nRetry limit reached. Do not retry this query; explain "
                        "in your final answer what you could not compute and why."
                    )
                else:
                    remaining = MAX_RETRIES_PER_QUERY - consecutive_errors + 1
                    content += f"\nYou may fix and retry ({remaining} retry(ies) left)."
            results.append({
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": content,
                "is_error": not outcome["ok"],
            })
        messages.append({"role": "user", "content": results})
