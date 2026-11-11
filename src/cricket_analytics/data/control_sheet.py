"""Reshape the wide ``control_sheet`` into one frame per ball type."""

from __future__ import annotations

import pandas as pd

from cricket_analytics.data.parameters import SymbolLegend

BLOCK_COLUMNS = [
    "Runs Scored",
    "Correct execution of shot according to the ball",
    "Effectiveness of the shot execution",
]


def clean_control_sheet(control_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Drop empty rows/helper columns and forward-fill the merged shot-type cells.

    Returns the per-ball-type column blocks and the aligned ``Shot Type`` series.
    """
    df = control_df.rename(columns={control_df.columns[0]: "Shot Type"})
    if len(df.columns) > 1 and str(df.columns[1]).startswith("Unnamed"):
        df = df.drop(columns=df.columns[1])
    df = df.dropna(how="all")
    shot_type = df.pop("Shot Type")
    shot_type = shot_type.where(shot_type.map(lambda v: isinstance(v, str))).ffill()
    return df, shot_type


def split_by_ball_type(control_df: pd.DataFrame, legend: SymbolLegend) -> dict[str, pd.DataFrame]:
    """Split the cross-tab into ``{ball type name: shots frame}``.

    Each ball type occupies a block of three columns whose first header is the
    ball symbol (matched case-insensitively against the ``parameters`` sheet).
    """
    blocks, shot_type = clean_control_sheet(control_df)
    if blocks.shape[1] % len(BLOCK_COLUMNS):
        raise ValueError(
            f"Control sheet has {blocks.shape[1]} value columns; expected a multiple of "
            f"{len(BLOCK_COLUMNS)}."
        )

    frames: dict[str, pd.DataFrame] = {}
    for start in range(0, blocks.shape[1], len(BLOCK_COLUMNS)):
        block = blocks.iloc[:, start : start + len(BLOCK_COLUMNS)].copy()
        ball_symbol = str(block.columns[0]).strip().upper()
        if ball_symbol not in legend.ball_types:
            raise KeyError(f"Control sheet ball symbol {ball_symbol!r} not in parameters sheet.")
        block.columns = BLOCK_COLUMNS
        block.insert(0, "Ball Type", ball_symbol)
        block.insert(1, "Shot Type", shot_type)
        frames[legend.ball_types[ball_symbol]] = block.reset_index(drop=True)
    return frames
