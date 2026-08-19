"""Minimal workbook → ball-by-ball frame reader.

This is a stopgap so the modelling code can be exercised end to end today. The
authoritative cleaning implementation is the notebook extraction in PR #11
(``cricket_analytics.cleaning.build_matches_df``); when that lands, this module
should delegate to it instead of re-implementing the rules.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

COLUMN_RENAMES = {
    "Player Name": "Batsman Name",
    "Player ID": "Batsman Player ID",
    "Bowler": "Bowler Name",
    "Player ID.1": "Bowler Player ID",
    "X": "Effectiveness",
}
FLAG_COLUMNS = ("Wide", "No ball", "Wicket", "Hit the Bat")
UNARY_COLUMNS = ("Pitch Type", "Match ID", "Sno")

VALID_CONTROL = (0.0, 1.0)
VALID_EFFECTIVENESS = (-1.0, *(round(step / 10.0, 1) for step in range(11)))


def _legend(parameters: pd.DataFrame, column: str) -> set[str]:
    return set(parameters[column].dropna().astype(str))


def clean_match_frame(
    frame: pd.DataFrame, ball_symbols: set[str], shot_symbols: set[str], sheet: str
) -> pd.DataFrame:
    frame = frame.dropna(how="all").rename(columns=COLUMN_RENAMES)
    frame = frame.drop(columns=[col for col in UNARY_COLUMNS if col in frame.columns])

    for column in FLAG_COLUMNS:
        if column in frame.columns:
            frame[column] = frame[column].notna().astype(float)

    frame["Control"] = pd.to_numeric(frame["Control"], errors="coerce")
    frame["Effectiveness"] = pd.to_numeric(frame["Effectiveness"], errors="coerce")

    valid = (
        frame["Ball type"].astype(str).isin(ball_symbols)
        & frame["Type of Shot"].astype(str).isin(shot_symbols)
        & frame["Control"].isin(VALID_CONTROL)
        & frame["Effectiveness"].round(1).isin(VALID_EFFECTIVENESS)
    )
    if not valid.all():
        warnings.warn(
            f"{sheet}: dropping {int((~valid).sum())} row(s) failing symbol or "
            "value-domain validation",
            stacklevel=2,
        )
    return frame[valid].reset_index(drop=True)


def matches_frame_from_workbook(path: str | Path) -> pd.DataFrame:
    """Stack every ``<team>_vs_<team>`` sheet into one validated frame."""
    excel = pd.ExcelFile(path)
    if "parameters" not in excel.sheet_names:
        raise KeyError(
            f"{path} has no 'parameters' sheet; sheets are {excel.sheet_names}"
        )
    parameters = excel.parse("parameters")
    ball_symbols = _legend(parameters, "ball symbol")
    shot_symbols = _legend(parameters, "shot symbol")

    match_sheets = [name for name in excel.sheet_names if "vs" in name.lower()]
    if not match_sheets:
        raise KeyError(
            f"{path} has no '<team>_vs_<team>' match sheets; "
            f"sheets are {excel.sheet_names}"
        )

    frames = [
        clean_match_frame(excel.parse(sheet), ball_symbols, shot_symbols, sheet)
        for sheet in match_sheets
    ]
    return pd.concat(frames, ignore_index=True)
