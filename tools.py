"""Tool implementations for the agent: run_sql and run_python."""

import ast
import contextlib
import io

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px

MAX_RESULT_ROWS = 100
MAX_OUTPUT_CHARS = 4000

import builtins as _builtins

# Modules run_python code (and numpy/pandas internals, which lazily import
# at runtime) may import. Everything else — os, sys, subprocess, socket,
# importlib, ctypes, pathlib, io, ... — is rejected.
ALLOWED_IMPORTS = {
    "numpy", "pandas", "math", "statistics", "random", "datetime", "re",
    "itertools", "collections", "functools", "warnings",
}


def _limited_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split(".")[0] in ALLOWED_IMPORTS:
        return _builtins.__import__(name, globals, locals, fromlist, level)
    raise ImportError(
        f"import of {name!r} is not allowed in this sandbox. "
        f"Allowed imports: {', '.join(sorted(ALLOWED_IMPORTS))}."
    )


# The only builtins exposed to run_python code. Notably absent: open
# (blocks the file system) and exec/eval/compile. __import__ is the
# limited version above — numpy internals need it at runtime.
SAFE_BUILTINS = {
    name: getattr(_builtins, name)
    for name in (
        "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter",
        "float", "format", "frozenset", "int", "isinstance", "issubclass",
        "len", "list", "map", "max", "min", "next", "print", "range",
        "repr", "reversed", "round", "set", "slice", "sorted", "str", "sum",
        "tuple", "type", "zip", "Exception", "ValueError", "TypeError",
        "KeyError", "IndexError", "ZeroDivisionError", "StopIteration",
    )
}
SAFE_BUILTINS["__import__"] = _limited_import


def run_sql(query: str, df: pd.DataFrame) -> dict:
    """Execute a SQL query against the working dataframe with DuckDB.

    Returns {"ok": bool, "output": str, "result": DataFrame | None} where
    "output" is the text fed back to the model and "result" is kept for
    display in the UI.
    """
    con = duckdb.connect()
    try:
        con.register("data", df)
        # EXPLAIN binds the query without running it, so references to
        # columns or tables that don't exist in the schema fail here with
        # a clear binder error before anything executes.
        con.execute(f"EXPLAIN {query}")
        result = con.execute(query).fetchdf()
    except duckdb.Error as exc:
        columns = ", ".join(df.columns)
        return {
            "ok": False,
            "result": None,
            "output": f"SQL error: {exc}\nAvailable columns in table `data`: {columns}",
        }
    finally:
        con.close()

    shown = result.head(MAX_RESULT_ROWS)
    text = shown.to_string(index=False, max_colwidth=80)
    notes = [f"{len(result):,} row(s) returned."]
    if len(result) > MAX_RESULT_ROWS:
        notes.append(f"Showing the first {MAX_RESULT_ROWS} rows only.")
    if len(text) > MAX_OUTPUT_CHARS:
        text = text[:MAX_OUTPUT_CHARS]
        notes.append("(output truncated)")
    return {"ok": True, "result": result, "output": " ".join(notes) + "\n" + text}


def _format_value(value) -> str:
    """Render the last expression's value compactly for the model."""
    if isinstance(value, pd.DataFrame):
        header = f"[DataFrame: {len(value):,} rows x {value.shape[1]} columns"
        if len(value) > MAX_RESULT_ROWS:
            header += f", showing first {MAX_RESULT_ROWS}"
        return header + "]\n" + value.head(MAX_RESULT_ROWS).to_string(max_colwidth=80)
    if isinstance(value, pd.Series):
        header = f"[Series: {len(value):,} values"
        if len(value) > MAX_RESULT_ROWS:
            header += f", showing first {MAX_RESULT_ROWS}"
        return header + "]\n" + value.head(MAX_RESULT_ROWS).to_string()
    return repr(value)


def python_namespace(df: pd.DataFrame) -> dict:
    """Build a fresh restricted namespace for one question's Python cells."""
    return {
        "__builtins__": SAFE_BUILTINS,
        "df": df.copy(),  # protect the working copy from silent mutation
        "pd": pd,
        "pandas": pd,
        "np": np,
        "numpy": np,
    }


def _imported_roots(tree: ast.AST) -> set[str]:
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            roots.add((node.module or "").split(".")[0] if node.level == 0 else "")
    return roots


def run_python(code: str, df: pd.DataFrame | None = None,
               namespace: dict | None = None) -> dict:
    """Execute Python code in a restricted namespace.

    Available names: df (a copy of the working dataframe), pd/pandas,
    np/numpy, a whitelist of harmless builtins, and imports from
    ALLOWED_IMPORTS only. No file access, no exec/eval. Pass the same
    `namespace` across calls so variables persist like notebook cells.
    Returns printed output plus the value of the last expression, capped
    at MAX_OUTPUT_CHARS.
    """
    if namespace is None:
        namespace = python_namespace(df)
    stdout = io.StringIO()
    try:
        tree = ast.parse(code)
        # Reject disallowed imports up front with actionable guidance — a
        # bare ImportError mid-run sends the model down dead ends.
        bad_imports = _imported_roots(tree) - ALLOWED_IMPORTS
        if bad_imports:
            return {
                "ok": False,
                "result": None,
                "output": (
                    f"Import of {', '.join(sorted(bad_imports))} is not allowed "
                    "in this sandbox — scipy, statsmodels, and sklearn are NOT "
                    "installed. Only `df`, pandas (`pd`), and numpy (`np`) are "
                    "available. Implement statistics with numpy instead: a "
                    "permutation test (shuffle labels with np.random.shuffle, "
                    "recompute the group difference many times, compare to the "
                    "observed difference) for significance, or a bootstrap "
                    "(np.random.choice with replace=True) for confidence "
                    "intervals."
                ),
            }
        # If the code ends in a bare expression, evaluate it separately so
        # its value can be returned (like a notebook cell).
        last_expr = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last_expr = ast.Expression(tree.body[-1].value)
            tree.body = tree.body[:-1]
        with contextlib.redirect_stdout(stdout):
            exec(compile(tree, "<analysis>", "exec"), namespace)
            value = (
                eval(compile(last_expr, "<analysis>", "eval"), namespace)
                if last_expr is not None
                else None
            )
    except Exception as exc:
        printed = stdout.getvalue()
        error = f"Python error: {type(exc).__name__}: {exc}"
        if printed:
            error = f"Output before the error:\n{printed[:1000]}\n{error}"
        return {"ok": False, "result": None, "output": error}

    parts = []
    printed = stdout.getvalue().strip()
    if printed:
        parts.append(printed)
    if value is not None:
        parts.append(_format_value(value))
    output = "\n".join(parts) if parts else "(code ran successfully but produced no output — print something or end with an expression)"
    if len(output) > MAX_OUTPUT_CHARS:
        output = output[:MAX_OUTPUT_CHARS] + "\n…(output truncated)"
    result_df = value if isinstance(value, pd.DataFrame) else None
    return {"ok": True, "result": result_df, "output": output}


PIE_MAX_CATEGORIES = 5
CHART_TYPES = ("bar", "line", "scatter", "histogram", "box", "pie")


def make_chart(spec: dict, working_df: pd.DataFrame,
               last_result: pd.DataFrame | None) -> dict:
    """Build a Plotly figure from a simple chart spec.

    The spec chooses its data: "working_data" (the working dataframe) or
    "last_result" (the dataframe from the agent's most recent successful
    run_sql / run_python call). The chart rules from CLAUDE.md are
    enforced here, not left to the model: bars sorted by value with a
    zero-baseline y-axis, a required title, labeled axes, thousands
    separators, and pie charts with more than PIE_MAX_CATEGORIES
    categories fall back to a sorted bar chart.
    """
    def fail(msg: str) -> dict:
        return {"ok": False, "result": None, "figure": None, "output": f"Chart error: {msg}"}

    chart_type = spec.get("chart_type")
    if chart_type not in CHART_TYPES:
        return fail(f"chart_type must be one of {', '.join(CHART_TYPES)}.")
    title = (spec.get("title") or "").strip()
    if not title:
        return fail("a non-empty title is required.")

    if spec.get("source") == "last_result":
        if last_result is None:
            return fail(
                "there is no previous result to chart — run a run_sql or "
                "run_python step that returns a dataframe first, or use "
                'source "working_data".'
            )
        data = last_result
    else:
        data = working_df

    x, y, color = spec.get("x"), spec.get("y"), spec.get("color")
    needs_y = chart_type in ("bar", "line", "scatter", "box", "pie")
    if not x and chart_type != "box":
        return fail("x is required for this chart type.")
    if needs_y and not y:
        return fail(f"y is required for a {chart_type} chart.")
    missing = [c for c in (x, y, color) if c and c not in data.columns]
    if missing:
        return fail(
            f"column(s) {', '.join(missing)} not found in the {spec.get('source')} "
            f"data. Available columns: {', '.join(data.columns)}."
        )

    notes = []
    if chart_type == "pie" and data[x].nunique() > PIE_MAX_CATEGORIES:
        chart_type = "bar"
        notes.append(
            f"Pie chart rejected ({data[x].nunique()} categories > "
            f"{PIE_MAX_CATEGORIES}); rendered a sorted bar chart instead."
        )

    if chart_type == "bar":
        d = data.sort_values(y, ascending=False)
        fig = px.bar(d, x=x, y=y, color=color, title=title)
        fig.update_yaxes(rangemode="tozero")
    elif chart_type == "line":
        fig = px.line(data.sort_values(x), x=x, y=y, color=color, title=title)
    elif chart_type == "scatter":
        fig = px.scatter(data, x=x, y=y, color=color, title=title)
    elif chart_type == "histogram":
        plot_data = data
        values = pd.to_numeric(data[x], errors="coerce").dropna()
        # Long-tail guard: when a few extreme values would stretch the
        # x-axis until the bulk of the data is unreadable, clip the view
        # at the 99th percentile — and say so on the chart itself.
        if len(values) >= 50:
            p99 = values.quantile(0.99)
            if p99 > 0 and values.max() > 2.5 * p99:
                mask = pd.to_numeric(data[x], errors="coerce") <= p99
                plot_data = data[mask]
                n_hidden = int(len(values) - mask.sum())
                pct = 100 * n_hidden / len(values)
                notes.append(
                    f"Long right tail: the x-axis is clipped at the 99th "
                    f"percentile ({p99:,.4g}); {n_hidden} value(s) ({pct:.1f}%) "
                    f"between {p99:,.4g} and {values.max():,.4g} are not shown. "
                    "Mention this tail in your answer — extremes are findings."
                )
        fig = px.histogram(plot_data, x=x, color=color, title=title)
        if plot_data is not data:
            fig.add_annotation(
                text=f"{n_hidden} value(s) above {p99:,.4g} not shown (top 1%)",
                xref="paper", yref="paper", x=0.99, y=0.98,
                showarrow=False, font=dict(size=11, color="gray"),
            )
    elif chart_type == "box":
        fig = px.box(data, x=x, y=y, color=color, title=title)
    else:  # pie, already within the category limit
        fig = px.pie(data, names=x, values=y, title=title)

    if chart_type != "pie":
        # Thousands separators, no scientific notation on value axes.
        fig.update_yaxes(tickformat=",")
        if chart_type == "histogram" and pd.api.types.is_numeric_dtype(data[x]):
            fig.update_xaxes(tickformat=",")

    notes.insert(0, f"Chart created: {title!r} ({chart_type}, {len(data):,} data rows).")
    return {"ok": True, "result": None, "figure": fig, "output": " ".join(notes)}
