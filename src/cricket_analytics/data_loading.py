"""Load the multi-sheet Excel workbook and reshape the control sheet."""

from __future__ import annotations

import pandas as pd

CONTROL_SHEET_COLUMNS = [
    "Runs Scored",
    "Correct execution of shot according to the ball",
    "Effectiveness of the shot execution",
]


def sheet_to_df_name(sheet_name: str) -> str:
    """Convert an Excel sheet name into a DataFrame key, e.g. ``IND vs AFG`` -> ``ind_vs_afg_df``."""
    return "_".join(sheet_name.lower().split()) + "_df"


def load_workbook(path: str) -> dict[str, pd.DataFrame]:
    """Load every sheet of the workbook into a dict keyed by DataFrame name."""
    xls = pd.ExcelFile(path)
    names = [sheet_to_df_name(sheet) for sheet in xls.sheet_names]
    return {
        names[index]: pd.read_excel(xls, index) for index in range(len(names))
    }


def prepare_control_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise the raw control sheet: name columns, drop fully-empty rows."""
    df = df.rename(columns={"Unnamed: 0": "Shot Type", "Sl": "SL"})
    if "Unnamed: 1" in df.columns:
        df = df.drop("Unnamed: 1", axis=1)
    return df.dropna(how="all")


def forward_fill_shot_type(shot_type: pd.Series) -> pd.Series:
    """Fill non-string entries with the closest preceding string value."""
    filled = shot_type.copy()
    for index, value in filled.items():
        if not isinstance(value, str):
            filled[index] = filled[index - 1]
    return filled


def build_ball_type_dict(parameters_df: pd.DataFrame) -> dict[str, str]:
    """Map ball symbols (e.g. ``FT``) to ball-type names from the parameters sheet."""
    ball_type_df = parameters_df.loc[:, ["Ball Type", "ball symbol"]].dropna()
    return dict(zip(ball_type_df["ball symbol"], ball_type_df["Ball Type"]))


def generate_ballwise_dfs(
    control_df: pd.DataFrame,
    shot_type: pd.Series,
    ball_type_dict: dict[str, str],
) -> dict[str, pd.DataFrame]:
    """Split the control sheet's 3-column blocks into one frame per ball type."""
    ballwise: dict[str, pd.DataFrame] = {}
    for col_index in range(0, control_df.shape[1], 3):
        block = control_df.iloc[:, col_index : col_index + 3].copy()
        ball_symbol = block.columns[0]
        ball_type = ball_type_dict[ball_symbol]

        block.columns = CONTROL_SHEET_COLUMNS
        block.insert(0, "Ball Type", ball_symbol)
        block.insert(1, "Shot Type", shot_type)
        ballwise[f"{ball_type}_shots_df"] = block
    return ballwise
