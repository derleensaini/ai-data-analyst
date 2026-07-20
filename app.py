"""Streamlit UI: CSV upload, data preview, and the data quality summary."""

import csv
import html
import io
from pathlib import Path

import anthropic
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from agent import answer_question, build_data_context, get_client, suggest_questions
from profiling import (
    apply_actions,
    clean_dataframe,
    fmt_num,
    human_bytes,
    impossible_date_mask,
    iqr_fences,
    near_duplicate_categories,
    profile_dataframe,
)

MAX_VISIBLE_FINDINGS = 8

ACCENT = "#1E40AF"

# Charts inherit the app theme: white panel, slate grid, blue-led colorway
# with one warm contrast color. Registered once; every make_chart figure
# picks it up as the plotly default.
pio.templates["analyst"] = go.layout.Template(
    layout=go.Layout(
        colorway=[ACCENT, "#D97706", "#0E7490", "#7C3AED", "#059669",
                  "#DC2626", "#475569"],
        font=dict(family="Fira Sans, sans-serif", color="#0F172A"),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        xaxis=dict(gridcolor="#E8EDF5", zerolinecolor="#CBD5E1"),
        yaxis=dict(gridcolor="#E8EDF5", zerolinecolor="#CBD5E1"),
        margin=dict(t=56, r=24, b=48, l=56),
    )
)
pio.templates.default = "plotly_white+analyst"

APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fira+Sans:wght@400;500;600;700&family=Fira+Code:wght@400;500&display=swap');

/* Comfortable reading width for the chat-first main column */
.block-container { max-width: 880px; padding-top: 3.6rem; }

/* Compact header */
.app-header { display: flex; align-items: baseline; gap: .75rem;
  padding: .1rem 0 .7rem; border-bottom: 1px solid #E2E8F0;
  margin-bottom: .6rem; }
.app-header svg { align-self: center; flex: none; }
.app-name { font-size: 1.25rem; font-weight: 700; color: #0F172A; }
.app-tag { color: #64748B; font-size: .92rem; }

/* Suggestion chips */
div[class*="st-key-suggestion_"] button {
  border-radius: 999px; border: 1px solid #1E40AF44;
  background: #1E40AF0D; color: #1E40AF;
  font-size: .88rem; padding: .35rem 1rem; min-height: 44px;
  transition: background .15s ease, color .15s ease; }
div[class*="st-key-suggestion_"] button:hover {
  background: #1E40AF; color: #FFFFFF; border-color: #1E40AF; }

/* Cleaning-log timeline */
.tl-item { position: relative; margin-left: .35rem; padding: 0 0 .7rem 1rem;
  border-left: 2px solid #DBEAFE; font-size: .85rem; line-height: 1.45; }
.tl-item::before { content: ""; position: absolute; left: -5px; top: .3rem;
  width: 8px; height: 8px; border-radius: 50%; background: #1E40AF; }
.tl-auto::before { background: #94A3B8; }
.tl-why { color: #64748B; font-size: .78rem; }

/* Code panels inside the tool expander */
.stExpander pre { border: 1px solid #DBE3EF; border-radius: 8px; }

/* Empty state */
.empty-state { text-align: center; padding: 4.5rem 1rem 5rem; color: #475569; }
.empty-state h3 { color: #0F172A; margin: .9rem 0 .3rem; font-weight: 600; }
.empty-state p { margin: 0 auto; max-width: 26rem; }
</style>
"""

HEADER_HTML = """
<div class="app-header">
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#1E40AF"
       stroke-width="1.8" stroke-linecap="round" aria-hidden="true">
    <path d="M3 3v18h18"/><rect x="7" y="12" width="3" height="6" rx=".6"/>
    <rect x="12" y="8" width="3" height="10" rx=".6"/>
    <rect x="17" y="5" width="3" height="13" rx=".6"/>
  </svg>
  <span class="app-name">AI Data Analyst</span>
  <span class="app-tag">Ask questions about your CSV; every answer is backed
  by real queries.</span>
</div>
"""

EMPTY_STATE_HTML = """
<div class="empty-state">
  <svg width="46" height="46" viewBox="0 0 24 24" fill="none" stroke="#1E40AF"
       stroke-width="1.5" stroke-linecap="round" aria-hidden="true">
    <path d="M3 3v18h18"/><rect x="7" y="12" width="3" height="6" rx=".6"/>
    <rect x="12" y="8" width="3" height="10" rx=".6"/>
    <rect x="17" y="5" width="3" height="13" rx=".6"/>
  </svg>
  <h3>Load a dataset to begin</h3>
  <p>Pick a data source in the sidebar: upload your own CSV, or try a
  built-in sample with suggested questions.</p>
</div>
"""

MAX_FILE_MB = 200
SAMPLE_ROWS = 200_000
DATA_DIR = Path(__file__).parent / "data"

# Built-in sample datasets. Adding one = drop a CSV in data/ and add an
# entry here (path, one-line description, and 4 hardcoded suggestions —
# one per category: SQL aggregation, Python statistics, chart, data limits).
SAMPLE_DATASETS = {
    "Sephora products": {
        "path": DATA_DIR / "product_info.csv",
        "description": "8,494 beauty products from Sephora: brands, prices, "
        "ratings, review counts, categories, and popularity ('loves').",
        "suggestions": [
            {"label": "Aggregation (SQL)",
             "question": "Which brand has the highest total loves count?"},
            {"label": "Statistics (Python)",
             "question": "Do Sephora-exclusive products have higher ratings "
             "than non-exclusive ones, and is the difference significant?"},
            {"label": "Chart",
             "question": "Show me the number of products by primary category "
             "as a chart."},
            {"label": "Data limits",
             "question": "Which product sold the most units last month?"},
        ],
    },
    "Unicorn companies": {
        "path": DATA_DIR / "unicorn_companies.csv",
        "description": "1,062 private companies valued at $1B+: industry, "
        "location, founding year, investors, valuation, and funding raised.",
        "suggestions": [
            {"label": "Aggregation (SQL)",
             "question": "Which country has the most unicorn companies?"},
            {"label": "Statistics (Python)",
             "question": "Is a company's valuation correlated with how much "
             "funding it raised, and how strong is the relationship?"},
            {"label": "Chart",
             "question": "Show me total unicorn valuation by industry as a chart."},
            {"label": "Data limits",
             "question": "Which of these companies was most profitable last year?"},
        ],
    },
}


def load_csv(raw: bytes) -> tuple[pd.DataFrame, list[str]]:
    """Read CSV bytes robustly. Returns (df, notes about how it was read)."""
    notes = []
    if not raw.strip():
        raise ValueError("The file is empty.")

    size_mb = len(raw) / 1024**2
    nrows = None
    if size_mb > MAX_FILE_MB:
        nrows = SAMPLE_ROWS
        notes.append(
            f"File is {size_mb:,.0f} MB — only the first {SAMPLE_ROWS:,} rows were "
            "loaded to keep the app responsive."
        )

    df = None
    for encoding in ("utf-8", "latin-1"):
        try:
            sample = raw[: 64 * 1024].decode(encoding, errors="replace")
            try:
                has_header = csv.Sniffer().has_header(sample)
            except csv.Error:
                has_header = True
            df = pd.read_csv(
                io.BytesIO(raw),
                encoding=encoding,
                header=0 if has_header else None,
                nrows=nrows,
            )
            if encoding != "utf-8":
                notes.append(f"File was not valid UTF-8; read with {encoding} encoding.")
            if not has_header:
                df.columns = [f"column_{i + 1}" for i in range(df.shape[1])]
                notes.append(
                    "No header row detected; columns were named column_1, column_2, …"
                )
            break
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError:
            raise ValueError("The file has no parsable rows.")
    if df is None:
        raise ValueError("Could not decode the file as UTF-8 or latin-1.")
    if df.empty and df.columns.empty:
        raise ValueError("The file has no parsable rows.")
    return df, notes


def rebuild_working_state() -> None:
    """Recompute the working copy: default cleaning, then replay every
    user-confirmed action in order. Called on upload and whenever the
    action list changes."""
    base, default_log = clean_dataframe(st.session_state.raw_df)
    df, user_log = apply_actions(base, st.session_state.user_actions)
    st.session_state.clean_df = df
    st.session_state.default_log = default_log
    st.session_state.user_log = user_log
    st.session_state.clean_profile = profile_dataframe(df)


def add_action(action_type: str, column: str, params: dict | None = None) -> None:
    st.session_state.action_counter += 1
    st.session_state.user_actions.append({
        "id": st.session_state.action_counter,
        "type": action_type,
        "column": column,
        "params": params or {},
    })
    rebuild_working_state()
    st.rerun()


def undo_action(action_id: int) -> None:
    st.session_state.user_actions = [
        a for a in st.session_state.user_actions if a["id"] != action_id
    ]
    rebuild_working_state()
    st.rerun()


def dismiss_issue(issue_id: str) -> None:
    st.session_state.dismissed.add(issue_id)
    st.rerun()


# Which applied action types settle which card, so a handled issue stops
# being offered even if the profiler still detects a remnant of it.
HANDLED_BY = {
    "nulls": {"drop_null_rows", "impute"},
    "nullgroup": {"drop_null_rows"},
    "categories": {"merge_categories"},
    "outliers": {"cap_outliers", "remove_outliers", "note"},
    "dates": {"null_impossible_dates", "remove_impossible_dates"},
}


def issue_resolved(issue_id: str) -> bool:
    if issue_id in st.session_state.dismissed:
        return True
    group, column = issue_id.split(":", 1)
    return any(
        a["type"] in HANDLED_BY[group] and a["column"] == column
        for a in st.session_state.user_actions
    )


def render_null_group_card(df: pd.DataFrame, group: dict) -> bool:
    """One card for columns that are null on exactly the same rows."""
    names = group["columns"]
    group_id = "nullgroup:" + "+".join(names)
    n = int(df[names[0]].isna().sum())
    if n == 0 or issue_resolved(group_id):
        return False
    with st.expander(f"🕳️ {' & '.join(names)} — {n:,} rows missing together"):
        st.write(
            f"These {len(names)} columns are null on exactly the same rows — "
            "one pattern, not separate problems. Sample of affected rows:"
        )
        st.dataframe(df[df[names[0]].isna()].head(10), use_container_width=True)
        c1, c2 = st.columns(2)
        if c1.button(f"Drop {n:,} rows", key=f"group_drop_{names[0]}"):
            add_action("drop_null_rows", names[0])
        if c2.button("Leave as is", key=f"group_leave_{names[0]}"):
            dismiss_issue(group_id)
    return True


def render_null_card(df: pd.DataFrame, col: dict) -> bool:
    name = col["name"]
    n = int(df[name].isna().sum())
    if n == 0 or "mostly empty" in col["flags"] or issue_resolved(f"nulls:{name}"):
        return False
    with st.expander(f"🕳️ {name} — {n:,} nulls ({100 * n / len(df):.1f}%)"):
        st.write("Sample of rows where this column is missing:")
        st.dataframe(df[df[name].isna()].head(10), use_container_width=True)
        s = df[name]
        if pd.api.types.is_numeric_dtype(s):
            value, strategy = s.median(), "median"
            impute_label = None if pd.isna(value) else f"Impute median ({fmt_num(value)})"
        else:
            mode = s.mode()
            strategy = "mode"
            impute_label = None if mode.empty else f"Impute mode ({mode.iloc[0]!r})"
        c1, c2, c3 = st.columns(3)
        if c1.button(f"Drop {n:,} rows", key=f"null_drop_{name}"):
            add_action("drop_null_rows", name)
        if impute_label and c2.button(impute_label, key=f"null_impute_{name}"):
            add_action("impute", name, {"strategy": strategy})
        if c3.button("Leave as is", key=f"null_leave_{name}"):
            dismiss_issue(f"nulls:{name}")
    return True


def render_category_card(df: pd.DataFrame, col: dict) -> bool:
    name = col["name"]
    if col["kind"] != "categorical" or issue_resolved(f"categories:{name}"):
        return False
    values = df[name].dropna().astype(str)
    groups = near_duplicate_categories(values)
    if not groups:
        return False
    counts = values.value_counts()
    mapping, rows = {}, []
    for group in groups:
        canonical = max(group, key=lambda v: counts.get(v, 0))
        for variant in group:
            rows.append({
                "label": variant,
                "rows": int(counts.get(variant, 0)),
                "becomes": canonical,
            })
            if variant != canonical:
                mapping[variant] = canonical
    n_affected = sum(int(counts.get(v, 0)) for v in mapping)
    with st.expander(f"🔀 {name} — {len(groups)} near-duplicate label group(s)"):
        st.write("Labels that differ only by case or spacing; merging keeps the most frequent spelling:")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        c1, c2 = st.columns(2)
        if c1.button(f"Merge ({n_affected:,} values change)", key=f"cat_merge_{name}"):
            add_action("merge_categories", name, {"mapping": mapping})
        if c2.button("Leave as is", key=f"cat_leave_{name}"):
            dismiss_issue(f"categories:{name}")
    return True


def render_outlier_card(df: pd.DataFrame, col: dict) -> bool:
    name = col["name"]
    stats = col["stats"]
    # Only genuine numeric columns: IDs, binary flags, zero-inflated counts,
    # and mostly-empty columns are excluded from outlier treatment.
    if (col["kind"] != "numeric" or not stats or not stats.get("iqr_outliers")
            or stats.get("zero_inflated") or issue_resolved(f"outliers:{name}")):
        return False
    values = df[name].dropna()
    if values.empty:
        return False
    low, high = iqr_fences(values)
    mask = (df[name] < low) | (df[name] > high)
    n = int(mask.sum())
    if n == 0:
        return False
    with st.expander(f"📈 {name} — {n:,} outlier(s) outside [{fmt_num(low)}, {fmt_num(high)}]"):
        st.write(
            "Outliers are findings first, problems second: they may be data entry "
            "errors, unit mistakes, or legitimate extremes worth keeping. "
            "Most extreme rows:"
        )
        by_distance = (df.loc[mask, name] - values.median()).abs().sort_values(ascending=False)
        st.dataframe(df.loc[by_distance.index].head(10), use_container_width=True)
        c1, c2, c3 = st.columns(3)
        if c1.button("Keep them", key=f"out_keep_{name}"):
            dismiss_issue(f"outliers:{name}")
        if c2.button(f"Cap at [{fmt_num(low)}, {fmt_num(high)}]", key=f"out_cap_{name}"):
            add_action("cap_outliers", name)
        if c3.button(f"Remove {n:,} rows", key=f"out_remove_{name}"):
            add_action("remove_outliers", name)
        note = st.text_input(
            "Or keep them and record a note in the cleaning log:",
            key=f"out_note_text_{name}",
            placeholder="e.g. two legitimate $10k+ orders pull the mean up",
        )
        if st.button("Leave with note", key=f"out_note_{name}", disabled=not note.strip()):
            add_action("note", name, {"text": note.strip()})
    return True


def render_impossible_dates_card(df: pd.DataFrame, col: dict) -> bool:
    name = col["name"]
    if col["kind"] != "date" or issue_resolved(f"dates:{name}"):
        return False
    mask = impossible_date_mask(df[name])
    n = int(mask.sum())
    if n == 0:
        return False
    n_future = int((df[name] > pd.Timestamp.now(tz=getattr(df[name].dt, "tz", None))).sum())
    with st.expander(f"📅 {name} — {n:,} impossible date(s)"):
        st.write(
            f"{n_future:,} in the future, {n - n_future:,} before 1900. "
            "These are usually data entry errors. Affected rows:"
        )
        st.dataframe(df[mask].head(10), use_container_width=True)
        c1, c2, c3 = st.columns(3)
        if c1.button(f"Set {n:,} to null", key=f"date_null_{name}"):
            add_action("null_impossible_dates", name)
        if c2.button(f"Remove {n:,} rows", key=f"date_remove_{name}"):
            add_action("remove_impossible_dates", name)
        if c3.button("Keep them", key=f"date_keep_{name}"):
            dismiss_issue(f"dates:{name}")
    return True


def render_issue_cards(df: pd.DataFrame, profile: dict) -> None:
    st.markdown("#### Review flagged issues")
    st.caption(
        "Nothing is applied until you click an action. Applied actions modify "
        "only the working copy, are logged below, and can be undone."
    )
    # Importance order: shared-null patterns, then single-column nulls,
    # then near-duplicate labels, then dates, then genuine outliers.
    n_cards = 0
    grouped_cols: set = set()
    for group in profile.get("null_groups", []):
        grouped_cols |= set(group["columns"])
        n_cards += render_null_group_card(df, group)
    for col in profile["columns"]:
        if col["name"] not in grouped_cols:
            n_cards += render_null_card(df, col)
    for col in profile["columns"]:
        n_cards += render_category_card(df, col)
    for col in profile["columns"]:
        n_cards += render_impossible_dates_card(df, col)
    for col in profile["columns"]:
        n_cards += render_outlier_card(df, col)
    if n_cards == 0:
        st.success("No flagged issues left to review.")


def _timeline_item(entry: dict, auto: bool) -> None:
    css_class = "tl-item tl-auto" if auto else "tl-item"
    tag = "automatic" if auto else "your call"
    st.markdown(
        f'<div class="{css_class}"><b>{html.escape(str(entry["column"]))}</b> '
        f"· {html.escape(entry['action'])} · {entry['affected']:,} affected "
        f'<span class="tl-why">({tag})</span><br>'
        f'<span class="tl-why">{html.escape(entry["why"])}</span></div>',
        unsafe_allow_html=True,
    )


def render_cleaning_log() -> None:
    st.markdown("#### Cleaning log")
    default_log = st.session_state.default_log
    user_log = st.session_state.user_log
    if not default_log and not user_log:
        st.caption("No cleaning has been applied.")
        return
    for entry in default_log:
        _timeline_item(entry, auto=True)
    for entry in user_log:
        col_text, col_undo = st.columns([4, 1], vertical_alignment="center")
        with col_text:
            _timeline_item(entry, auto=False)
        if col_undo.button("Undo", key=f"undo_{entry['action_id']}"):
            undo_action(entry["action_id"])


def render_quality_summary(profile: dict) -> None:
    overview = profile["overview"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{overview['n_rows']:,}")
    c2.metric("Columns", f"{overview['n_cols']:,}")
    c3.metric("Memory", human_bytes(overview["memory_bytes"]))
    c4.metric("Duplicate rows", f"{overview['duplicate_rows']:,}")

    issues = profile["issues"]
    if issues:
        st.warning(f"**{len(issues)} data quality finding(s)**")
        st.markdown(
            "\n".join(
                f"- **{i['column']}**: {i['issue']}"
                for i in issues[:MAX_VISIBLE_FINDINGS]
            )
        )
        if len(issues) > MAX_VISIBLE_FINDINGS:
            with st.expander(f"Show all {len(issues)} findings"):
                st.markdown(
                    "\n".join(
                        f"- **{i['column']}**: {i['issue']}"
                        for i in issues[MAX_VISIBLE_FINDINGS:]
                    )
                )
    else:
        st.success("No data quality issues detected.")

    st.markdown("#### Column overview")
    st.dataframe(
        pd.DataFrame(
            {
                "column": col["name"],
                "type": col["kind"],
                "dtype": col["dtype"],
                "nulls": f"{col['null_count']:,} ({col['null_pct']}%)",
                "unique": f"{col['n_unique']:,}",
                "flags": ", ".join(col["flags"]),
                "issues": len(col["issues"]),
            }
            for col in profile["columns"]
        ),
        use_container_width=True,
        hide_index=True,
    )

    numeric_cols = [c for c in profile["columns"] if c["stats"] and "mean" in c["stats"]]
    if numeric_cols:
        with st.expander("Numeric column details"):
            st.dataframe(
                pd.DataFrame(
                    {
                        "column": col["name"],
                        "min": col["stats"]["min"],
                        "max": col["stats"]["max"],
                        "mean": col["stats"]["mean"],
                        "median": col["stats"]["median"],
                        "std": col["stats"]["std"],
                        "IQR outliers": f"{col['stats']['iqr_outliers']:,} "
                        f"({col['stats']['iqr_outlier_pct']}%)",
                    }
                    for col in numeric_cols
                ),
                use_container_width=True,
                hide_index=True,
            )

    date_cols = [
        c for c in profile["columns"]
        if c["stats"] and c["kind"] in ("date", "date-as-string")
    ]
    if date_cols:
        with st.expander("Date column details"):
            st.dataframe(
                pd.DataFrame(
                    {
                        "column": col["name"],
                        "type": col["kind"],
                        "earliest": col["stats"]["min"],
                        "latest": col["stats"]["max"],
                    }
                    for col in date_cols
                ),
                use_container_width=True,
                hide_index=True,
            )

    cat_cols = [c for c in profile["columns"] if c["top_values"]]
    if cat_cols:
        with st.expander("Categorical column details (top 5 values)"):
            st.dataframe(
                pd.DataFrame(
                    {
                        "column": col["name"],
                        "cardinality": col["n_unique"],
                        "top values": ", ".join(
                            f"{v['value']} ({v['count']:,} · {v['pct']}%)"
                            for v in col["top_values"]
                        ),
                    }
                    for col in cat_cols
                ),
                use_container_width=True,
                hide_index=True,
            )


TOOL_LABELS = {"run_sql": ("SQL", "sql"), "run_python": ("Python", "python"),
               "make_chart": ("Chart spec", "json")}


def render_figures(tool_events: list[dict], key_prefix: str) -> None:
    """Render charts inline in the chat message, under the answer text."""
    for i, event in enumerate(tool_events):
        if event["ok"] and event.get("figure") is not None:
            st.plotly_chart(
                event["figure"], use_container_width=True, key=f"{key_prefix}_fig{i}"
            )


def render_tool_events(tool_events: list[dict]) -> None:
    """Show each executed SQL query / Python snippet / chart spec."""
    if not tool_events:
        return
    with st.expander(f"Queries, code, and charts ({len(tool_events)})"):
        for i, event in enumerate(tool_events, 1):
            label, language = TOOL_LABELS[event["tool"]]
            st.markdown(f"**Step {i} — {label}**" + ("" if event["ok"] else " — failed"))
            st.code(event["input"], language=language)
            if not event["ok"]:
                st.error(f"```text\n{event['output']}\n```")
            elif event["result"] is not None:
                st.dataframe(event["result"], use_container_width=True)
            else:
                st.text(event["output"])


def render_suggestions(client, use_clean: bool) -> str | None:
    """Show 4 clickable example-question chips; returns the clicked
    question, if any. Only called while the conversation is empty."""
    suggestions = st.session_state.get("suggestions")
    if suggestions is None:
        # Uploaded CSV: generate once from the schema + profile, then cache
        # in session_state so reruns don't repeat the call.
        with st.spinner("Thinking of example questions…"):
            view = "cleaned working copy" if use_clean else "raw data"
            context = build_data_context(
                view,
                st.session_state["clean_profile" if use_clean else "raw_profile"],
                st.session_state["default_log"] if use_clean else [],
                st.session_state["user_log"] if use_clean else [],
            )
            suggestions = suggest_questions(client, context)
        st.session_state["suggestions"] = suggestions

    clicked = None
    st.caption("Try one of these:")
    columns = st.columns(2)
    for i, suggestion in enumerate(suggestions):
        with columns[i % 2]:
            st.caption(suggestion["label"])
            if st.button(
                suggestion["question"], key=f"suggestion_{i}",
                use_container_width=True,
            ):
                clicked = suggestion["question"]
    return clicked


def render_chat(use_clean: bool) -> None:
    st.subheader("Ask questions")
    client = get_client()
    if client is None:
        st.warning(
            "**No Anthropic API key found.** To enable the chat, create a "
            "file named `.env` in the project folder containing:\n\n"
            "```\nANTHROPIC_API_KEY=sk-ant-your-key-here\n```\n\n"
            "then restart the app. Get a key at console.anthropic.com."
        )
        return

    for idx, msg in enumerate(st.session_state["chat_history"]):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            render_figures(msg.get("tool_events", []), key_prefix=f"msg{idx}")
            render_tool_events(msg.get("tool_events", []))

    # Suggestion chips appear only while the conversation is empty. A
    # clicked chip becomes the question, exactly as if it had been typed.
    clicked = None
    if not st.session_state["chat_history"]:
        clicked = render_suggestions(client, use_clean)

    question = st.chat_input("Ask a question about your data") or clicked
    if not question:
        return

    st.session_state["chat_history"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    view = "cleaned working copy" if use_clean else "raw data"
    data_context = build_data_context(
        view,
        st.session_state["clean_profile" if use_clean else "raw_profile"],
        st.session_state["default_log"] if use_clean else [],
        st.session_state["user_log"] if use_clean else [],
    )
    df = st.session_state["clean_df" if use_clean else "raw_df"]
    with st.chat_message("assistant"):
        try:
            with st.spinner("Analyzing…"):
                answer, tool_events, _ = answer_question(
                    client, st.session_state["chat_history"], data_context, df
                )
            st.markdown(answer)
            # Same key the message gets when re-rendered from history on
            # the next rerun (it is appended right after this block).
            render_figures(
                tool_events,
                key_prefix=f"msg{len(st.session_state['chat_history'])}",
            )
            render_tool_events(tool_events)
        except anthropic.AuthenticationError:
            st.error(
                "The Anthropic API rejected your key. Check ANTHROPIC_API_KEY "
                "in your .env file."
            )
            return
        except anthropic.RateLimitError:
            st.error("Rate limited by the Anthropic API — wait a moment and try again.")
            return
        except anthropic.APIConnectionError:
            st.error("Could not reach the Anthropic API. Check your internet connection.")
            return
        except anthropic.APIStatusError as exc:
            st.error(f"Anthropic API error ({exc.status_code}): {exc.message}")
            return
    st.session_state["chat_history"].append(
        {"role": "assistant", "content": answer, "tool_events": tool_events}
    )
    # Re-render cleanly from history (also removes the suggestion chips
    # that were drawn earlier in this same run).
    st.rerun()


def main() -> None:
    st.set_page_config(
        page_title="AI Data Analyst", page_icon="📊",
        layout="centered", initial_sidebar_state="expanded",
    )
    st.markdown(APP_CSS, unsafe_allow_html=True)
    st.markdown(HEADER_HTML, unsafe_allow_html=True)

    # ---- Sidebar: data setup ------------------------------------------
    ready = True
    with st.sidebar:
        st.markdown("#### Data source")
        source = st.radio(
            "Data source", ("Upload a CSV", "Sample dataset"),
            horizontal=True, label_visibility="collapsed",
        )
        if source == "Upload a CSV":
            uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
            if uploaded is None:
                ready = False
            else:
                file_key = ("upload", uploaded.name, uploaded.size)
                read_bytes = uploaded.getvalue
                suggestions = None  # generated from the profile on first chat view
        else:
            sample_name = st.selectbox("Sample dataset", list(SAMPLE_DATASETS))
            config = SAMPLE_DATASETS[sample_name]
            st.caption(config["description"])
            file_key = ("sample", sample_name)
            read_bytes = config["path"].read_bytes
            suggestions = config["suggestions"]

    if not ready:
        st.markdown(EMPTY_STATE_HTML, unsafe_allow_html=True)
        return

    if st.session_state.get("file_key") != file_key:
        try:
            with st.spinner("Reading, profiling, and cleaning the file…"):
                df, notes = load_csv(read_bytes())
                st.session_state.update(
                    file_key=file_key,
                    load_notes=notes,
                    raw_df=df,
                    raw_profile=profile_dataframe(df),
                    user_actions=[],
                    dismissed=set(),
                    action_counter=0,
                    chat_history=[],
                    suggestions=suggestions,
                )
                rebuild_working_state()
        except (ValueError, OSError) as exc:
            st.session_state.pop("file_key", None)
            st.error(str(exc))
            return

    # ---- Sidebar: cleaning --------------------------------------------
    with st.sidebar:
        for note in st.session_state["load_notes"]:
            st.info(note)
        st.divider()
        st.markdown("#### Cleaning")
        use_clean = st.toggle(
            "Use cleaned working copy",
            value=True,
            help="Turn off to view the raw data. The raw file is always "
            "kept untouched, so you can revert anytime.",
        )
        if use_clean:
            render_issue_cards(
                st.session_state["clean_df"], st.session_state["clean_profile"]
            )
            st.divider()
            render_cleaning_log()
        else:
            st.caption("Viewing raw data — the cleaning panel is hidden "
                       "until you switch back.")

    # ---- Main area: data at a glance, then the chat -------------------
    df = st.session_state["clean_df" if use_clean else "raw_df"]
    view = "cleaned working copy" if use_clean else "raw data"
    profile = st.session_state["clean_profile" if use_clean else "raw_profile"]

    with st.expander(f"Data preview — {view}, {len(df):,} rows"):
        st.caption(f"First {min(len(df), 100)} of {len(df):,} rows.")
        st.dataframe(df.head(100), use_container_width=True)
    with st.expander(f"Data quality — {len(profile['issues'])} finding(s)"):
        st.caption(f"Profile of the {view}.")
        render_quality_summary(profile)

    render_chat(use_clean)


if __name__ == "__main__":
    main()
