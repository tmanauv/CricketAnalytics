"""Clean per-match ball-by-ball frames and stack them into one dataset."""

from __future__ import annotations

import pandas as pd

UNARY_COLS = ["Pitch Type", "Match ID"]
BINARY_COLS = ["Wide", "No ball", "Wicket", "Hit the Bat"]
COLUMN_RENAMES = {
    "Player Name": "Batsman Name",
    "Player ID": "Batsman Player ID",
    "Bowler": "Bowler Name",
    "Player ID.1": "Bowler Player ID",
    "X": "Effectiveness",
}
BATSMAN_NAME_FIXES = {
    "Kedar Jadhav": "Kedhar Jadhav",
    "Shikar Dhawan": "Shikhar Dhawan",
}
CONTROL_VALUES = [0.0, 1.0]
EFFECTIVENESS_VALUES = [x / 10.0 for x in range(11)] + [-1.0]


def unique_values(df: pd.DataFrame) -> list[tuple[str, list]]:
    """Return ``(column, unique values)`` pairs for every column except ``Sno``."""
    return [(col, list(df[col].unique())) for col in df.columns if col != "Sno"]


def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Drop constant columns, binarise presence columns, and zero-fill the rest."""
    cols_nan = [col for col, nan_count in df.isna().sum().items() if nan_count != 0]

    for col in cols_nan:
        if col in UNARY_COLS:
            df.drop(col, axis=1, inplace=True)
        elif col in BINARY_COLS:
            df[col] = df[col].notnull().astype("float")
        else:
            df[col] = df[col].fillna(value=0)

    return df


def drop_incorrect_data(
    df: pd.DataFrame,
    column_name: str,
    child_values: list,
    parent_values: list,
) -> None:
    """Drop rows whose ``column_name`` value is not in ``parent_values`` (in place)."""
    incorrect_values = [val for val in child_values if val not in parent_values]
    if not incorrect_values:
        return
    df.drop(df[df[column_name].isin(incorrect_values)].index, inplace=True)


def validate_column_values(
    df: pd.DataFrame, parameters_df: pd.DataFrame
) -> pd.DataFrame:
    """Normalise symbols against the parameters legend and drop invalid rows."""
    ball_symbols = [
        s.upper() for s in parameters_df["ball symbol"].dropna().tolist()
    ]
    shot_symbols = [
        s.upper() for s in parameters_df["shot symbol"].dropna().tolist()
    ]
    dismissal_symbols = [
        s.upper() for s in parameters_df["dismisal symbol"].dropna().tolist()
    ]
    dismissal_dict = dict(
        zip(
            parameters_df["Dismissal Kind"].dropna(),
            parameters_df["dismisal symbol"].dropna(),
        )
    )

    for col_name, col_values in unique_values(df):
        lowered = col_name.lower()
        if "shot" in lowered:
            col_values = [str(v).upper() for v in col_values]
            drop_incorrect_data(df, col_name, col_values, shot_symbols)
        elif lowered == "ball type":
            col_values = [str(v).upper() for v in col_values]
            drop_incorrect_data(df, col_name, col_values, ball_symbols)
        elif "control" in lowered:
            if -1.0 in col_values:
                df[col_name] = df[col_name].replace(-1.0, 0.0)
                col_values.remove(-1.0)
            drop_incorrect_data(df, col_name, col_values, CONTROL_VALUES)
        elif "effectiveness" in lowered:
            drop_incorrect_data(df, col_name, col_values, EFFECTIVENESS_VALUES)
        elif "dismissal" in lowered:
            col_values = [v for v in col_values if v != 0]
            if "C" in col_values:
                df[col_name] = df[col_name].replace("C", "Ct")
                col_values = ["Ct" if v == "C" else v for v in col_values]
            for col_value in col_values:
                if len(col_value) > 3:
                    df[col_name] = df[col_name].replace(
                        col_value, dismissal_dict[col_value]
                    )
            col_values = [
                dismissal_dict[v] if len(v) > 3 else v for v in col_values
            ]
            col_values = [v.upper() for v in col_values]
            drop_incorrect_data(df, col_name, col_values, dismissal_symbols)

    return df


def clean_match_df(df: pd.DataFrame, parameters_df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline for a single ``<team>_vs_<team>`` match frame."""
    df = df.drop("Sno", axis=1)
    df = df.dropna(how="all")
    df = df.reset_index(drop=True)
    df = df.rename(columns=COLUMN_RENAMES)

    non_nan_df = clean_missing_values(df)
    return validate_column_values(non_nan_df, parameters_df)


def normalize_batsman_names(df: pd.DataFrame) -> pd.DataFrame:
    """Fix known misspellings of batsman names."""
    df["Batsman Name"] = df["Batsman Name"].replace(BATSMAN_NAME_FIXES)
    return df


def build_matches_df(
    df_dict: dict[str, pd.DataFrame], parameters_df: pd.DataFrame
) -> pd.DataFrame:
    """Clean and stack every ``<team>_vs_<team>`` frame into one dataset."""
    matches_df = pd.DataFrame()
    for df_name, df in df_dict.items():
        if "vs" not in df_name:
            continue
        cleaned = clean_match_df(df, parameters_df)
        matches_df = pd.concat([matches_df, cleaned], ignore_index=True)
    return normalize_batsman_names(matches_df)
