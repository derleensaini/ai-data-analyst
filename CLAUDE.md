# AI Data Analyst Agent

## What this is
A Streamlit app where users upload a CSV, ask questions about it in
plain English, and get answers backed by real SQL queries, Python
analysis, and charts. An LLM agent decides which tools to use and
loops until it has a defensible answer. The agent must behave like an
experienced data analyst, not a query engine: inspect first, clean
deliberately, state assumptions, verify results, communicate clearly.

## Stack
- Python 3.11+
- Streamlit for the UI
- pandas for data handling
- DuckDB for running SQL against uploaded dataframes
- Plotly for charts
- Anthropic API for the agent (tool calling)

## Architecture
User question -> agent orchestrator -> tools -> interpret -> answer

The agent has three tools:
1. run_sql(query) - runs SQL via DuckDB against the working dataframe
2. run_python(code) - executes Python with the dataframe available as df
3. make_chart(spec) - renders a Plotly chart from a simple spec

The agent loop: send question + data schema + data profile to the
model, execute any tool it calls, feed results back, repeat until it
returns a final text answer. If tool code errors, feed the error back
and let it retry (max 2 retries). Cap the loop at 8 tool calls per
question to prevent runaway loops.

## Build order (do not skip ahead)
1. CSV upload + data preview + automatic data profile
2. Data cleaning step with a visible cleaning log
3. Chat box wired to the Anthropic API with schema + profile in the prompt
4. Tools, added one at a time: run_sql, then run_python, then make_chart
5. Agent loop with retry on errors
6. Show executed SQL/Python in an expandable section
7. Sample datasets, eval question set, polish
8. Deploy to Streamlit Community Cloud

---

## The analyst workflow (the agent must follow this order)

A real analyst never jumps straight to the answer. Every question is
handled in this sequence:

1. **Understand the ask.** Restate ambiguous questions as a concrete,
   answerable version. If a critical ambiguity would change the answer,
   ask the user one clarifying question instead of guessing.
2. **Check the data can answer it.** Confirm the needed columns exist,
   have usable types, and have enough non-null rows. If the data cannot
   answer the question, say so plainly. Never fabricate.
3. **Plan the analysis.** Decide the minimal set of steps (filter,
   join, aggregate, compare) before running anything.
4. **Execute.** Prefer SQL for filtering/aggregation, Python for
   statistics, reshaping, and anything SQL makes awkward.
5. **Verify.** Sanity-check the result (rules below) before presenting.
6. **Communicate.** Finding first, numbers second, caveats third.

## Automatic data profiling (runs on every upload)

Before any question is answered, profile the data and show the user a
data quality summary. The profile is also injected into the agent's
context. Profile every column for:

- Row count, column count, memory footprint
- Null count and null percentage per column
- Duplicate rows (full-row duplicates) and suspected duplicate keys
- Data type per column, and columns with mixed or wrong types
  (numbers stored as strings, "N/A"/"null"/"-" placeholder strings,
  currency symbols or commas inside numeric columns)
- Date columns that fail to parse, mixed date formats, and impossible
  dates (in the future when they shouldn't be, or before 1900)
- Numeric columns: min, max, mean, median, std, and outlier count
- Categorical columns: cardinality, top 5 values with frequencies,
  and near-duplicate categories ("NY" vs "New York" vs "new york",
  leading/trailing whitespace, case inconsistencies)
- Constant columns (one unique value) and fully-unique columns
  (likely IDs), flagged so they are excluded from aggregations

## Data cleaning rules

Cleaning is deliberate, visible, and reversible. Never silently
modify data.

- Keep the raw dataframe untouched. All cleaning happens on a working
  copy. The user can always revert.
- Every cleaning action is appended to a visible **cleaning log**:
  what was changed, how many rows/values were affected, and why.
- Default cleaning applied automatically (and logged):
  - Strip leading/trailing whitespace from string columns
  - Standardize obvious null placeholders ("", "N/A", "null", "-",
    "n/a", "NULL") to true NaN
  - Parse date-like columns to datetime where >=95% of values parse
  - Convert numeric-looking string columns (strip $ , % symbols)
    where >=95% of values convert
  - Drop exact full-row duplicates, and report how many were dropped
- Cleaning that requires user confirmation (never automatic):
  - Dropping rows with nulls in key columns
  - Imputing missing values (and if imputing, prefer median for
    skewed numerics, mode for categoricals, and always disclose it)
  - Merging near-duplicate category labels
  - Removing or capping outliers (see below)
- Nulls are information. Before dropping or imputing, check whether
  missingness is concentrated (one time period, one category) because
  that changes the interpretation of results.

## Outlier handling

Outliers are findings first, problems second. The agent must never
delete outliers silently.

- Detection: use the IQR rule (outside Q1 - 1.5*IQR to Q3 + 1.5*IQR)
  as the default. For roughly normal columns, also report |z| > 3.
  Report the count and percentage of outliers per numeric column in
  the profile.
- Diagnose before treating. Classify each outlier situation as one of:
  1. Data entry error (age of 250, negative quantity) -> fix or null it
  2. Unit/format mistake (cents vs dollars, ms vs s) -> convert
  3. Legitimate extreme value (one huge order, a viral day) -> keep it
- Legitimate extremes are usually the most interesting rows in the
  dataset. Mention them in answers when they materially move an
  average or total.
- When outliers distort an aggregate, prefer reporting the median (or
  both mean and median) over deleting rows. If removal or capping
  (winsorizing) is genuinely warranted, ask the user, do it on the
  working copy, log it, and state in every affected answer that
  outliers were excluded.
- Always report sensitivity when it matters: "average order value is
  $87, or $64 excluding the two orders above $10k."

## Analyst behavior in answers

- When a question is ambiguous ("best", "top", "recently"), the agent
  states its assumption explicitly in the answer, or asks one
  clarifying question if the ambiguity really matters.
- The agent sanity-checks results before answering: group totals
  reconcile to overall totals, percentages sum to ~100, row counts
  are plausible, joins did not duplicate rows, date ranges match the
  data.
- Cross-check important numbers with a second method when cheap
  (e.g. verify a SQL aggregate with a quick pandas calculation).
- Answers follow this shape: finding first in plain English, then the
  supporting numbers, then caveats (missing data, small samples,
  incomplete periods, excluded outliers).
- Flag small sample sizes. A "top category" based on 4 rows gets a
  caveat, not a confident claim.
- Watch for incomplete periods. If the last month of data is partial,
  say so instead of reporting a fake decline.
- The agent never invents column names or values. Before running SQL
  or Python, it validates that referenced columns exist in the schema.
- Never imply causation from correlation. Use "is associated with,"
  not "causes."
- Round sensibly in prose (12.3%, $1.2M) but never round inside
  intermediate calculations.

## Chart rules

- Bar charts sorted by value for comparisons, line charts for time
  series, scatter for relationships, histogram/box plot for
  distributions and outlier discussions. Avoid pie charts beyond 4-5
  categories.
- Every chart has a title, labeled axes, and readable formatting
  (thousands separators, sensible date ticks, no scientific notation
  for money).
- Bar charts for quantities start the y-axis at zero. Line charts may
  zoom, but note it when the zoom exaggerates a change.
- Only chart when it adds something. A single number does not need a
  chart.

## Code quality rules

- Keep code simple and readable. No LangChain or agent frameworks.
- Small number of files: app.py (UI), agent.py (loop + prompts),
  tools.py (tool implementations), profiling.py (profile + cleaning).
- run_python executes in a restricted namespace: df, pandas, numpy
  only. No file system access, no network, output size capped.
- Handle failure gracefully: bad CSV encodings (try utf-8, then
  latin-1), empty files, files with no header, huge files (sample or
  warn above ~200MB).
- One feature at a time. Explain what you wrote after each change.
- Never hardcode API keys. Use a .env file and .gitignore it.

## Evaluation

- Maintain tests/eval_questions.md: 10-15 questions per sample
  dataset with hand-verified answers.
- Include easy lookups, aggregations, grouped comparisons, time-based
  questions, at least two trick questions the data cannot answer
  (correct behavior is saying so), and at least two ambiguous
  questions (correct behavior is stating the assumption).
- Track the score across changes. If a change drops the score,
  investigate before moving on.
