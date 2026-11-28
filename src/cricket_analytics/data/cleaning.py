"""Clean and validate the per-match ball-by-ball sheets."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd

from cricket_analytics.data import columns as C
from cricket_analytics.data.parameters import SymbolLegend

logger = logging.getLogger(__name__)

_FALSY_FLAGS = {"", "0", "0.0", "no", "n", "false"}


def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map case/whitespace variants (``"Ball Type"``) onto canonical names."""
    canonical = {name.lower(): name for name in (C.BALL_TYPE, C.SHOT, C.CONTROL, C.DISMISSAL)}
    renames = {}
    for col in df.columns:
        key = " ".join(str(col).split()).lower()
        if key in canonical and col != canonical[key]:
            renames[col] = canonical[key]
    return df.rename(columns=renames)


def _to_flag(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return (series.fillna(0) != 0).astype(float)
    present = series.notna() & ~series.astype(str).str.strip().str.lower().isin(_FALSY_FLAGS)
    return present.astype(float)


def _drop_invalid(df: pd.DataFrame, column: str, valid: Iterable[Any], sheet: str) -> pd.DataFrame:
    """Drop rows whose ``column`` value is not in ``valid``, logging what was removed."""
    mask = df[column].isin(list(valid))
    if not mask.all():
        bad = sorted(map(str, df.loc[~mask, column].unique()))
        logger.warning(
            "%s: dropping %d row(s) with invalid %r values %s",
            sheet,
            int((~mask).sum()),
            column,
            bad,
        )
    return df[mask]


def _normalize_symbols(series: pd.Series) -> pd.Series:
    return series.map(lambda v: str(v).strip().upper() if pd.notna(v) else v)


def _normalize_dismissal(
    series: pd.Series, legend: SymbolLegend, aliases: Mapping[str, str]
) -> pd.Series:
    upper_aliases = {str(k).strip().upper(): str(v).strip().upper() for k, v in aliases.items()}
    by_name = legend.dismissal_name_to_symbol()

    def normalize(value: Any) -> Any:
        if pd.isna(value) or (isinstance(value, (int, float)) and value == 0):
            return C.NO_DISMISSAL
        text = str(value).strip().upper()
        if not text or text == "0":
            return C.NO_DISMISSAL
        text = upper_aliases.get(text, text)
        return by_name.get(text, text)

    return series.map(normalize)


def clean_match_df(
    df: pd.DataFrame, legend: SymbolLegend, cleaning: Mapping[str, Any], sheet: str = "match"
) -> pd.DataFrame:
    """Clean one ``<team>_vs_<team>`` sheet.

    Mirrors the notebook's ``clean_df``: drop ``Sno`` and empty rows, rename
    player columns, coerce flag columns, fill gaps, and drop rows whose
    categorical codes are not in the ``parameters`` legend.
    """
    df = df.drop(columns=["Sno"], errors="ignore").dropna(how="all").reset_index(drop=True)
    df = df.rename(columns=dict(cleaning.get("column_renames", {})))
    df = _canonicalize_columns(df)

    for required in (C.BATSMAN, C.BALL_TYPE, C.SHOT, C.TARGET):
        if required not in df.columns:
            raise KeyError(f"{sheet}: required column {required!r} missing; got {list(df.columns)}")

    if C.DISMISSAL in df.columns:
        df[C.DISMISSAL] = _normalize_dismissal(
            df[C.DISMISSAL], legend, cleaning.get("dismissal_aliases", {})
        )
        if C.WICKET in df.columns:
            dismissed = df[C.DISMISSAL] != C.NO_DISMISSAL
            df.loc[dismissed & df[C.WICKET].isna(), C.WICKET] = 1.0

    for col in cleaning.get("unary_columns", []):
        if col in df.columns and (df[col].isna().any() or df[col].nunique() <= 1):
            df = df.drop(columns=col)
    for col in cleaning.get("binary_columns", []):
        if col in df.columns:
            df[col] = _to_flag(df[col])

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    df[numeric_cols] = df[numeric_cols].fillna(0)

    df[C.SHOT] = _normalize_symbols(df[C.SHOT])
    df[C.BALL_TYPE] = _normalize_symbols(df[C.BALL_TYPE])
    df = _drop_invalid(df, C.SHOT, legend.shot_symbols, sheet)
    df = _drop_invalid(df, C.BALL_TYPE, legend.ball_symbols, sheet)

    if C.CONTROL in df.columns:
        df[C.CONTROL] = pd.to_numeric(df[C.CONTROL], errors="coerce").replace(-1.0, 0.0)
        df = _drop_invalid(df, C.CONTROL, C.CONTROL_VALUES, sheet)
        df[C.CONTROL] = df[C.CONTROL].astype(float)

    df[C.TARGET] = pd.to_numeric(df[C.TARGET], errors="coerce").round(6)
    df = _drop_invalid(df, C.TARGET, C.EFFECTIVENESS_VALUES, sheet)

    if C.DISMISSAL in df.columns:
        df = _drop_invalid(df, C.DISMISSAL, legend.dismissal_symbols | {C.NO_DISMISSAL}, sheet)

    remaining = [c for c in df.columns if df[c].isna().any()]
    if remaining:
        df[remaining] = df[remaining].fillna(0)
    return df.reset_index(drop=True)


def build_matches_df(
    match_sheets: Mapping[str, pd.DataFrame], legend: SymbolLegend, cleaning: Mapping[str, Any]
) -> pd.DataFrame:
    """Clean every match sheet and stack them into one table."""
    frames = []
    for name, sheet in match_sheets.items():
        cleaned = clean_match_df(sheet, legend, cleaning, sheet=name)
        logger.info("%s: kept %d/%d rows", name, len(cleaned), len(sheet.dropna(how="all")))
        frames.append(cleaned)
    matches = pd.concat(frames, ignore_index=True)

    corrections = dict(cleaning.get("batsman_name_corrections", {}))
    if corrections:
        matches[C.BATSMAN] = matches[C.BATSMAN].replace(corrections)

    object_cols = matches.select_dtypes(include="object").columns
    matches[object_cols] = matches[object_cols].astype(str)
    return matches


def unique_values(df: pd.DataFrame) -> dict[str, list[Any]]:
    """Unique values per column (excluding ``Sno``), for quick data audits."""
    return {col: list(df[col].unique()) for col in df.columns if col != "Sno"}


__all__ = ["build_matches_df", "clean_match_df", "unique_values"]
