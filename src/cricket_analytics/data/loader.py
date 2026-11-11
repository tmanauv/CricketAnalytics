"""Read the multi-sheet Excel workbook into per-sheet DataFrames."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

CONTROL_SHEET = "control_sheet"
PARAMETERS_SHEET = "parameters"
ANALYSIS_PARAMETERS_SHEET = "analysis_parameters"


def normalize_sheet_name(sheet_name: str) -> str:
    """``"IND vs AFG"`` -> ``"ind_vs_afg"``."""
    return "_".join(sheet_name.lower().split())


def is_match_sheet(name: str) -> bool:
    return "_vs_" in normalize_sheet_name(name)


def load_workbook(path: str | Path) -> dict[str, pd.DataFrame]:
    """Return every sheet of the workbook keyed by its normalized name."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Workbook not found at {path}. Place 'Cricket Data Set.xlsx' there "
            "(see data/README.md) or generate a synthetic one with "
            "`python tools/make_synthetic_workbook.py`."
        )
    sheets = pd.read_excel(path, sheet_name=None)
    return {normalize_sheet_name(name): df for name, df in sheets.items()}


def require_sheets(sheets: dict[str, pd.DataFrame], *names: str) -> None:
    missing = [name for name in names if name not in sheets]
    if missing:
        raise KeyError(f"Workbook is missing required sheet(s): {missing}")


def match_sheets(sheets: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    matches = {name: df for name, df in sheets.items() if is_match_sheet(name)}
    if not matches:
        raise ValueError("Workbook has no '<team>_vs_<team>' match sheets.")
    return matches
