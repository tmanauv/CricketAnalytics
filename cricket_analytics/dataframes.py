"""Workbook loading and DataFrame inspection helpers."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import pandas as pd

MATCH_NAME_MARKER = "vs"


def dataframe_names(sheet_names: Iterable[str]) -> list[str]:
    """Map workbook sheet names to the ``<snake_case>_df`` keys used as keys."""
    return ["_".join(name.lower().split()) + "_df" for name in sheet_names]


def load_workbook(path: str) -> dict[str, pd.DataFrame]:
    """Read every sheet of ``path`` into a dict keyed by ``dataframe_names``."""
    xls = pd.ExcelFile(path)
    names = dataframe_names(xls.sheet_names)
    return {name: pd.read_excel(xls, index) for index, name in enumerate(names)}


def filter_names(
    names: Iterable[str],
    contains: str | None = None,
    excludes: str | None = None,
) -> Iterator[str]:
    """Yield names containing ``contains`` and not containing ``excludes``.

    Replaces the repeated ``if 'vs' not in df_name: continue`` guards.
    """
    for name in names:
        if contains is not None and contains not in name:
            continue
        if excludes is not None and excludes in name:
            continue
        yield name


def unique_values(df: pd.DataFrame) -> list[tuple[str, list]]:
    """Per-column unique values, skipping the source workbook's ``Sno`` index."""
    return [(col, list(df[col].unique())) for col in df.columns if col != "Sno"]


def print_unique_values(frames: dict[str, pd.DataFrame], **kwargs) -> None:
    """Print ``unique_values`` for every frame selected by ``filter_names``."""
    for name in filter_names(frames.keys(), **kwargs):
        print(name)
        for col, values in unique_values(frames[name]):
            print(f"{col} : {values}")
        print()


def symbol_list(parameters_df: pd.DataFrame, column: str) -> list[str]:
    """Upper-cased legend symbols of ``column`` in the parameters sheet."""
    return [str(value).upper() for value in parameters_df[column].dropna()]
