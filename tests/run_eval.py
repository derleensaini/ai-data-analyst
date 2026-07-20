"""Run the eval questions through the agent exactly the way the app does.

Usage:
    .venv/bin/python tests/run_eval.py [path/to/csv] [--model MODEL] [--out FILE]

Defaults: the agent's configured model, the Sephora sample CSV, and
tests/eval_results.md. The model is the ONLY thing that changes between
runs — same agent code, same system prompt, same questions. Writes a
results table with an empty pass/fail column for hand-grading (no
auto-grading), headed by wall-clock time and token totals.
"""

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

import agent
from agent import answer_question, build_data_context, get_client
from profiling import clean_dataframe, profile_dataframe

DEFAULT_DATA = PROJECT_ROOT / "data" / "product_info.csv"
DEFAULT_OUT = PROJECT_ROOT / "tests" / "eval_results.md"

# $/million tokens for cost estimates; cache reads bill at 0.1x the input
# rate, cache writes at 1.25x. Prices need updating when they change.
PRICES = {
    "claude-opus-4-8": {"input": 5.00, "output": 25.00},
    # Claude Sonnet 5 introductory pricing (through 2026-08-31);
    # standard is $3.00 / $15.00.
    "claude-sonnet-5": {"input": 2.00, "output": 10.00},
}

QUESTIONS = [
    "How many products are in the dataset?",
    "How many unique brands are there?",
    "What is the most expensive product and how much does it cost?",
    "Which primary category has the most products, and how many?",
    "Which brand has the highest total loves count?",
    "What is the average price of Fragrance products versus the overall average?",
    "Do Sephora-exclusive products have a higher average rating than non-exclusive ones?",
    "What percentage of products are out of stock?",
    "Which product sold the most units last month?",
    "What is the average customer age?",
    "What is the most popular product?",
]


class UsageMeter:
    """Wraps client.messages.create to total token usage across the run."""

    def __init__(self, client):
        self.input = self.output = self.cache_read = self.cache_write = 0
        self._create = client.messages.create
        client.messages.create = self._counted

    def _counted(self, **kwargs):
        response = self._create(**kwargs)
        usage = response.usage
        self.input += usage.input_tokens
        self.output += usage.output_tokens
        self.cache_read += usage.cache_read_input_tokens or 0
        self.cache_write += usage.cache_creation_input_tokens or 0
        return response

    def cost(self, model: str) -> float | None:
        prices = PRICES.get(model)
        if prices is None:
            return None
        return (
            self.input * prices["input"]
            + self.cache_write * prices["input"] * 1.25
            + self.cache_read * prices["input"] * 0.10
            + self.output * prices["output"]
        ) / 1_000_000


def md_cell(text: str) -> str:
    """Make arbitrary text safe inside a markdown table cell."""
    return text.replace("|", "\\|").replace("\n", "<br>").strip()


def tools_summary(tool_events: list[dict]) -> str:
    counts = Counter(e["tool"] for e in tool_events)
    failed = sum(1 for e in tool_events if not e["ok"])
    parts = [f"{tool}×{n}" for tool, n in counts.items()]
    if failed:
        parts.append(f"({failed} failed)")
    return ", ".join(parts) if parts else "none"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", nargs="?", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", default=agent.MODEL,
                        help="Model to run with (default: the agent's configured model)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    # The only variable between runs: same agent code, prompt, and questions.
    agent.MODEL = args.model

    client = get_client()
    if client is None:
        sys.exit("No ANTHROPIC_API_KEY found — add it to .env first.")
    meter = UsageMeter(client)

    print(f"Model: {args.model}")
    print(f"Loading {args.data} …")
    raw = pd.read_csv(args.data)
    print(f"  {len(raw):,} rows, {raw.shape[1]} columns")
    print("Cleaning and profiling (same as the app) …")
    work, default_log = clean_dataframe(raw)
    context = build_data_context(
        "cleaned working copy", profile_dataframe(work), default_log, []
    )
    print(f"  working copy: {len(work):,} rows | context: {len(context):,} chars\n")

    rows = []
    run_start = time.time()
    for i, question in enumerate(QUESTIONS, 1):
        print(f"[{i}/{len(QUESTIONS)}] {question}")
        start = time.time()
        try:
            answer, events, turns = answer_question(
                client, [{"role": "user", "content": question}], context, work
            )
        except Exception as exc:  # keep going; record the failure
            answer, events, turns = f"ERROR: {type(exc).__name__}: {exc}", [], 0
        elapsed = time.time() - start
        print(f"    done in {elapsed:.0f}s | turns: {turns} | tools: {tools_summary(events)}\n")
        rows.append({
            "n": i,
            "question": question,
            "answer": answer,
            "tools": tools_summary(events),
            "turns": turns,
        })
    total_time = time.time() - run_start

    cost = meter.cost(args.model)
    cost_text = f"${cost:.2f}" if cost is not None else "unknown pricing"
    summary = [
        f"Model: `{args.model}` | Dataset: `{args.data}`",
        "",
        f"- Total wall-clock time: **{total_time:.0f}s**",
        f"- Input tokens: **{meter.input:,}** (+ {meter.cache_write:,} cache "
        f"writes, {meter.cache_read:,} cache reads)",
        f"- Output tokens: **{meter.output:,}**",
        f"- Estimated cost: **{cost_text}**"
        + (" (introductory Sonnet pricing, $2/$10 per MTok)"
           if args.model == "claude-sonnet-5" else ""),
    ]

    lines = [
        "# Eval results — hand-grade the pass/fail column",
        "",
        *summary,
        "",
        "Model answers below are unedited.",
        "",
        "| # | Question | Agent's final answer | Tools called | Turns | Pass/Fail |",
        "|---|----------|----------------------|--------------|-------|-----------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['n']} | {md_cell(r['question'])} | {md_cell(r['answer'])} "
            f"| {r['tools']} | {r['turns']} |  |"
        )
    table = "\n".join(lines) + "\n"
    args.out.write_text(table, encoding="utf-8")
    print(f"Saved {args.out}\n")
    print(table)


if __name__ == "__main__":
    main()
