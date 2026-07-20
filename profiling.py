"""Automatic data profiling and default cleaning for uploaded CSVs.

Builds the data-quality profile described in CLAUDE.md: overview stats,
duplicate detection, and per-column null/type/outlier/category checks.
The same profile dict will later be injected into the agent's context.

Also applies the default cleaning rules to a working copy, producing a
cleaning log of what changed, how much, and why. The raw dataframe is
never modified.
"""

import math
import re
import warnings

import pandas as pd

# Placeholder strings that usually mean "missing". Compared case-insensitively
# after stripping whitespace.
NULL_PLACEHOLDERS = {"", "n/a", "null", "-"}

ID_NAME_RE = re.compile(r"(^id$|_id$|^id_|\bid\b|uuid|guid|_key$)", re.IGNORECASE)
DATE_NAME_RE = re.compile(r"(date|time|_at$|^dob$|day|month|year)", re.IGNORECASE)

# Coarse format buckets used to detect mixed date formats in one column.
DATE_FORMAT_PATTERNS = [
    ("YYYY-MM-DD", re.compile(r"^\d{4}-\d{2}-\d{2}")),
    ("D/M/Y or M/D/Y", re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4}")),
    ("D-M-Y or M-D-Y", re.compile(r"^\d{1,2}-\d{1,2}-\d{2,4}")),
    ("month name", re.compile(r"[A-Za-z]{3,9}")),
]


def _parse_dates(s: pd.Series) -> pd.Series:
    """Parse strings to datetime, one format inference per value.

    format="mixed" parses each value independently; without it pandas infers
    one format from the first value and rejects everything else.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(s, errors="coerce", format="mixed")
        if not pd.api.types.is_datetime64_any_dtype(parsed):
            # Mixed timezones parse to object dtype; force UTC to get one dtype.
            parsed = pd.to_datetime(s, errors="coerce", format="mixed", utc=True)
    return parsed


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Apply the default cleaning rules from CLAUDE.md to a copy of df.

    Returns (working copy, cleaning log). Each log entry records what was
    changed, how many rows/values were affected, and why. The input
    dataframe is never modified; reverting means using the raw dataframe.
    """
    out = df.copy()
    log = []

    def record(column, action, affected, why):
        log.append({"column": column, "action": action, "affected": affected, "why": why})

    for col in out.columns:
        s = out[col]
        if not (pd.api.types.is_object_dtype(s) or isinstance(s.dtype, pd.StringDtype)):
            continue

        stripped = s.str.strip()
        n_ws = int(((s != stripped) & s.notna()).sum())
        if n_ws:
            out[col] = s = stripped
            record(
                col, "stripped whitespace", n_ws,
                "leading/trailing whitespace makes identical values look distinct",
            )

        is_placeholder = s.str.lower().isin(NULL_PLACEHOLDERS) & s.notna()
        n_ph = int(is_placeholder.sum())
        if n_ph:
            out[col] = s = s.mask(is_placeholder)
            record(
                col, "replaced placeholder text with null", n_ph,
                'strings like "N/A", "null", "-" mean missing but hide from null counts',
            )

        non_null = s.dropna()
        if non_null.empty:
            continue

        # Numbers stored as strings (checked before dates so that year-like
        # columns such as "2021" become numbers, not dates).
        as_numeric = pd.to_numeric(non_null.str.replace(r"[$,%\s]", "", regex=True), errors="coerce")
        if as_numeric.notna().mean() >= 0.95:
            n_failed = int(as_numeric.isna().sum())
            out[col] = pd.to_numeric(s.str.replace(r"[$,%\s]", "", regex=True), errors="coerce")
            why = "numbers stored as text ($ , % symbols removed) cannot be aggregated"
            if n_failed:
                why += f"; {n_failed} unconvertible value(s) became null"
            record(col, "converted text to numbers", len(non_null) - n_failed, why)
            continue

        # Dates stored as strings.
        parsed = _parse_dates(non_null)
        if parsed.notna().mean() >= 0.95:
            n_failed = int(parsed.isna().sum())
            out[col] = _parse_dates(s)
            why = "dates stored as text cannot be used for time analysis"
            if n_failed:
                why += f"; {n_failed} unparseable value(s) became null"
            record(col, "parsed text to dates", len(non_null) - n_failed, why)

    n_dupes = int(out.duplicated().sum())
    if n_dupes:
        out = out.drop_duplicates().reset_index(drop=True)
        record(
            "(whole rows)", "dropped exact duplicate rows", n_dupes,
            "identical rows double-count every aggregate",
        )

    return out, log


def fmt_num(x) -> str:
    """User-facing number format: thousands separators, no scientific
    notation, at most 2 decimal places."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return str(x)
    text = f"{x:,.2f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def iqr_fences(values: pd.Series) -> tuple[float, float]:
    q1, q3 = values.quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)
    # A column that cannot be negative never gets a negative fence.
    if len(values) and float(values.min()) >= 0:
        low = max(low, 0.0)
    return low, high


def impossible_date_mask(s: pd.Series) -> pd.Series:
    """True for dates in the future or before 1900 (NaT stays False)."""
    tz = getattr(s.dt, "tz", None)
    return (s > pd.Timestamp.now(tz=tz)) | (s < pd.Timestamp("1900-01-01", tz=tz))


def apply_actions(df: pd.DataFrame, actions: list[dict]) -> tuple[pd.DataFrame, list[dict]]:
    """Replay user-confirmed cleaning actions on the working copy.

    Returns (df, log entries). Counts and values (fences, medians) are
    recomputed at replay time, so the log always reflects what actually
    happened — including after an earlier action is undone.
    """
    df = df.copy()
    log = []
    for action in actions:
        df, entry = _apply_action(df, action)
        entry["action_id"] = action["id"]
        log.append(entry)
    return df, log


def _apply_action(df: pd.DataFrame, action: dict) -> tuple[pd.DataFrame, dict]:
    col = action["column"]
    kind = action["type"]
    params = action["params"]

    if kind == "merge_categories":
        mapping = {k: v for k, v in params["mapping"].items() if k != v}
        n = int(df[col].isin(mapping).sum())
        df[col] = df[col].replace(mapping)
        return df, _log_entry(
            col, "merged near-duplicate labels", n,
            "the same label spelled differently splits one group into several",
        )

    if kind == "drop_null_rows":
        n = int(df[col].isna().sum())
        df = df[df[col].notna()].reset_index(drop=True)
        return df, _log_entry(
            col, "dropped rows with nulls", n,
            "user chose to exclude rows missing this column",
        )

    if kind == "impute":
        s = df[col]
        n = int(s.isna().sum())
        if params["strategy"] == "median":
            value = s.median()
            what = f"imputed median ({value:,.4g})"
        else:
            value = s.mode().iloc[0]
            what = f"imputed mode ({value!r})"
        df[col] = s.fillna(value)
        return df, _log_entry(
            col, what, n,
            "user chose to fill missing values; imputed values are estimates, not observations",
        )

    if kind == "cap_outliers":
        low, high = iqr_fences(df[col].dropna())
        n = int(((df[col] < low) | (df[col] > high)).sum())
        df[col] = df[col].clip(low, high)
        return df, _log_entry(
            col, f"capped outliers to [{fmt_num(low)}, {fmt_num(high)}]", n,
            "user chose winsorizing; aggregates on this column understate the extremes",
        )

    if kind == "remove_outliers":
        low, high = iqr_fences(df[col].dropna())
        mask = (df[col] < low) | (df[col] > high)
        n = int(mask.sum())
        df = df[~mask].reset_index(drop=True)
        return df, _log_entry(
            col, f"removed outlier rows outside [{fmt_num(low)}, {fmt_num(high)}]", n,
            "user chose removal; every result now excludes these extremes",
        )

    if kind == "null_impossible_dates":
        mask = impossible_date_mask(df[col])
        n = int(mask.sum())
        df[col] = df[col].mask(mask)
        return df, _log_entry(
            col, "set impossible dates to null", n,
            "dates in the future or before 1900 treated as data entry errors",
        )

    if kind == "remove_impossible_dates":
        mask = impossible_date_mask(df[col])
        n = int(mask.sum())
        df = df[~mask].reset_index(drop=True)
        return df, _log_entry(
            col, "removed rows with impossible dates", n,
            "dates in the future or before 1900 treated as data entry errors",
        )

    if kind == "note":
        return df, _log_entry(col, "analyst note (no data change)", 0, params["text"])

    raise ValueError(f"unknown cleaning action: {kind}")


def _log_entry(column: str, action: str, affected: int, why: str) -> dict:
    return {"column": column, "action": action, "affected": affected, "why": why}


def _shared_null_groups(df: pd.DataFrame, columns: list[dict]) -> tuple[list[dict], set]:
    """Columns whose null masks are byte-identical (missing on exactly the
    same rows) are one pattern, not N separate findings."""
    masks: dict[bytes, list[str]] = {}
    for col in columns:
        if col["null_count"] > 0 and "mostly empty" not in col["flags"]:
            key = df[col["name"]].isna().values.tobytes()
            masks.setdefault(key, []).append(col["name"])
    groups, grouped = [], set()
    for names in masks.values():
        if len(names) >= 2:
            count = next(c["null_count"] for c in columns if c["name"] == names[0])
            groups.append({"columns": names, "count": count})
            grouped |= set(names)
    return groups, grouped


def _issue_priority(text: str) -> int:
    """Sort order for findings: mostly-empty and shared-null patterns first,
    then plain nulls, near-duplicate labels, everything else, and genuine
    numeric outliers last."""
    if "mostly empty" in text or "entirely null" in text:
        return 0
    if " nulls (" in text:
        return 2
    if "near-duplicate" in text:
        return 3
    if "outlier" in text or "|z|" in text:
        return 5
    return 4


def profile_dataframe(df: pd.DataFrame) -> dict:
    """Profile every column and return the full data-quality summary."""
    n_rows = len(df)
    columns = [_profile_column(df[col], n_rows) for col in df.columns]
    null_groups, grouped_cols = _shared_null_groups(df, columns)

    duplicate_rows = int(df.duplicated().sum())
    issues = []
    if duplicate_rows:
        pct = 100 * duplicate_rows / n_rows if n_rows else 0
        issues.append({
            "column": "(whole rows)",
            "issue": f"{duplicate_rows:,} exact duplicate rows ({pct:.1f}%)",
            "priority": 2,
        })
    for group in null_groups:
        names = group["columns"]
        issues.append({
            "column": " + ".join(names),
            "issue": f"{group['count']:,} rows are missing "
            f"{' and '.join(names)} together (same rows)",
            "priority": 1,
        })
    for key in _suspected_duplicate_keys(df, columns):
        issues.append({
            "column": key["column"],
            "issue": f"looks like a key but has {key['duplicate_values']:,} duplicated values",
            "priority": 4,
        })
    for col in columns:
        for issue in col["issues"]:
            # Members of a shared-null group are covered by the merged entry.
            if col["name"] in grouped_cols and " nulls (" in issue:
                continue
            issues.append({
                "column": col["name"],
                "issue": issue,
                "priority": _issue_priority(issue),
            })
    issues.sort(key=lambda i: i["priority"])

    return {
        "overview": {
            "n_rows": n_rows,
            "n_cols": df.shape[1],
            "memory_bytes": int(df.memory_usage(deep=True).sum()),
            "duplicate_rows": duplicate_rows,
        },
        "columns": columns,
        "null_groups": null_groups,
        "issues": issues,
    }


def _suspected_duplicate_keys(df: pd.DataFrame, columns: list[dict]) -> list[dict]:
    """Columns that look like unique keys (by name or near-uniqueness) but
    contain duplicated values."""
    suspects = []
    for col in columns:
        non_null = df[col["name"]].dropna()
        if non_null.empty:
            continue
        n_dupes = len(non_null) - non_null.nunique()
        looks_like_key = (
            ID_NAME_RE.search(col["name"]) is not None
            or col["unique_ratio"] >= 0.99
        )
        # Only near-unique columns count: a key column that is mostly
        # duplicated (like a foreign key) is not a broken key, and flagging
        # its duplicates is noise.
        if (looks_like_key and col["unique_ratio"] >= 0.9
                and n_dupes > 0 and "constant" not in col["flags"]):
            suspects.append({"column": col["name"], "duplicate_values": int(n_dupes)})
    return suspects


def _profile_column(s: pd.Series, n_rows: int) -> dict:
    null_count = int(s.isna().sum())
    non_null = s.dropna()
    n_unique = int(non_null.nunique())
    info = {
        "name": str(s.name),
        "dtype": str(s.dtype),
        "kind": "empty",
        "null_count": null_count,
        "null_pct": round(100 * null_count / n_rows, 1) if n_rows else 0.0,
        "n_unique": n_unique,
        "unique_ratio": n_unique / len(non_null) if len(non_null) else 0.0,
        "flags": [],
        "issues": [],
        "stats": None,
        "top_values": None,
    }
    if non_null.empty:
        info["flags"].append("mostly empty")
        info["issues"].append("column is entirely null")
        return info
    # Mostly-empty columns get one flag and are excluded from every other
    # check — profiling the 5% of values that exist is noise.
    if info["null_pct"] > 90:
        info["kind"] = "mostly empty"
        info["flags"].append("mostly empty")
        info["issues"].append(
            f"mostly empty ({info['null_pct']}% null) — excluded from other checks"
        )
        return info
    if null_count:
        info["issues"].append(f"{null_count:,} nulls ({info['null_pct']}%)")

    if n_unique == 1:
        info["flags"].append("constant")
        info["issues"].append("constant column (single value) — exclude from aggregations")
    elif n_unique == len(non_null):
        info["flags"].append("unique")

    if pd.api.types.is_bool_dtype(s):
        info["kind"] = "boolean"
        info["top_values"] = _top_values(non_null)
    elif pd.api.types.is_numeric_dtype(s):
        # Binary flag columns (0/1 and the like): outlier detection is
        # meaningless — a fence of [0, 0] would flag every 1.
        if n_unique <= 2:
            info["kind"] = "boolean"
            info["flags"].append("binary")
            info["top_values"] = _top_values(non_null)
        # ID-like columns (named like identifiers, or near-unique integers)
        # get no distribution stats: outlier counts and means on an ID are
        # noise, and the flag tells the agent to exclude them from
        # aggregations.
        elif ID_NAME_RE.search(info["name"]) is not None or (
            pd.api.types.is_integer_dtype(s) and info["unique_ratio"] >= 0.95
        ):
            info["kind"] = "id"
            info["flags"].append("id")
            info["stats"] = {"min": float(non_null.min()), "max": float(non_null.max())}
        else:
            info["kind"] = "numeric"
            _add_numeric_stats(non_null, info)
    elif pd.api.types.is_datetime64_any_dtype(s):
        info["kind"] = "date"
        _add_date_checks(non_null, info)
    else:
        _profile_object_column(non_null, info)
    return info


def _add_numeric_stats(non_null: pd.Series, info: dict) -> None:
    values = non_null.astype(float)
    info["stats"] = {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "std": float(values.std()) if len(values) > 1 else 0.0,
    }

    # Zero-inflated counts: when most values are 0, whole-column IQR fences
    # collapse and flag every nonzero value. Compute on the nonzero subset
    # and say so.
    outlier_values = values
    zero_note = ""
    zero_share = float((values == 0).mean())
    if zero_share > 0.5:
        info["stats"]["zero_inflated"] = True
        nonzero = values[values != 0]
        if len(nonzero) < 10:
            info["stats"]["iqr_outliers"] = 0
            info["stats"]["iqr_outlier_pct"] = 0.0
            return
        outlier_values = nonzero
        zero_note = (
            f" (column is {zero_share:.0%} zeros; fences computed on the "
            f"{len(nonzero):,} nonzero values)"
        )

    low, high = iqr_fences(outlier_values)
    n_out = int(((outlier_values < low) | (outlier_values > high)).sum())
    pct = 100 * n_out / len(outlier_values)
    info["stats"]["iqr_outliers"] = n_out
    info["stats"]["iqr_outlier_pct"] = round(pct, 1)
    if n_out:
        info["issues"].append(
            f"{n_out:,} outliers ({pct:.1f}%) outside IQR fences "
            f"[{fmt_num(low)}, {fmt_num(high)}]{zero_note}"
        )

    # For roughly normal columns, also report the |z| > 3 count.
    std = info["stats"]["std"]
    if len(values) >= 30 and std > 0 and abs(float(values.skew())) < 1:
        n_z = int((((values - values.mean()) / std).abs() > 3).sum())
        info["stats"]["z_outliers"] = n_z
        if n_z:
            info["issues"].append(f"{n_z:,} values with |z| > 3 (column is roughly normal)")


def _add_date_checks(non_null: pd.Series, info: dict) -> None:
    dates = _parse_dates(non_null)
    if dates.dt.tz is not None:
        dates = dates.dt.tz_localize(None)
    info["stats"] = {
        "min": str(dates.min().date()),
        "max": str(dates.max().date()),
    }
    n_future = int((dates > pd.Timestamp.now()).sum())
    n_ancient = int((dates < pd.Timestamp("1900-01-01")).sum())
    if n_future:
        info["issues"].append(f"{n_future} dates in the future")
    if n_ancient:
        info["issues"].append(f"{n_ancient} dates before 1900")


def _profile_object_column(non_null: pd.Series, info: dict) -> None:
    as_str = non_null.astype(str)
    stripped = as_str.str.strip()

    n_whitespace = int((as_str != stripped).sum())
    if n_whitespace:
        info["issues"].append(f"{n_whitespace} values with leading/trailing whitespace")

    is_placeholder = stripped.str.lower().isin(NULL_PLACEHOLDERS)
    n_placeholder = int(is_placeholder.sum())
    if n_placeholder:
        examples = sorted(set(stripped[is_placeholder].head(20)))[:3]
        info["issues"].append(
            f"{n_placeholder} placeholder nulls ({', '.join(repr(e) for e in examples)}) "
            "not counted as real nulls"
        )
    real = stripped[~is_placeholder]
    if real.empty:
        info["kind"] = "text"
        info["issues"].append("all values are null placeholders")
        return

    # Numbers stored as strings, possibly with $ , % symbols.
    cleaned = real.str.replace(r"[$,%\s]", "", regex=True)
    as_numeric = pd.to_numeric(cleaned, errors="coerce")
    if as_numeric.notna().mean() >= 0.95:
        info["kind"] = "numeric-as-string"
        symbols = " with $/,/% symbols" if (cleaned != real).any() else ""
        info["issues"].append(
            f"numeric values stored as strings{symbols} "
            f"({as_numeric.notna().mean():.0%} convert cleanly)"
        )
        _add_numeric_stats(as_numeric.dropna(), info)
        return

    # Dates stored as strings.
    parsed = _parse_dates(real)
    parsed_pct = parsed.notna().mean()
    name_hints_date = DATE_NAME_RE.search(info["name"]) is not None
    if parsed_pct >= 0.95 or (name_hints_date and parsed_pct >= 0.5):
        info["kind"] = "date-as-string"
        n_failed = int(parsed.isna().sum())
        if n_failed:
            info["issues"].append(f"{n_failed} date values fail to parse")
        formats = {
            label
            for label, pattern in DATE_FORMAT_PATTERNS
            if real[parsed.notna()].str.contains(pattern).any()
        }
        if len(formats) > 1:
            info["issues"].append(f"mixed date formats ({', '.join(sorted(formats))})")
        _add_date_checks(real[parsed.notna()], info)
        info["issues"].append("dates stored as strings — parse to datetime before use")
        return

    # Plain categorical / free text.
    info["kind"] = "categorical" if info["unique_ratio"] < 0.5 or info["n_unique"] <= 20 else "text"
    info["top_values"] = _top_values(non_null)
    if info["kind"] == "categorical" and info["n_unique"] <= 1000:
        near_dupes = near_duplicate_categories(as_str)
        if near_dupes:
            shown = "; ".join(" / ".join(repr(v) for v in group) for group in near_dupes[:3])
            more = f" (+{len(near_dupes) - 3} more groups)" if len(near_dupes) > 3 else ""
            info["issues"].append(f"near-duplicate categories: {shown}{more}")


def _top_values(non_null: pd.Series) -> list[dict]:
    counts = non_null.astype(str).value_counts().head(5)
    return [
        {"value": value, "count": int(count), "pct": round(100 * count / len(non_null), 1)}
        for value, count in counts.items()
    ]


def near_duplicate_categories(values: pd.Series) -> list[list[str]]:
    """Groups of raw labels that collapse to the same normalized form,
    e.g. 'NY ' / 'ny' / 'NY'."""
    normalized = values.str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
    groups = pd.Series(values.values).groupby(normalized.values).unique()
    return [sorted(g) for g in groups if len(g) > 1]


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:,.1f} {unit}" if unit != "B" else f"{n:,.0f} B"
        n /= 1024
    return f"{n:,.1f} TB"
