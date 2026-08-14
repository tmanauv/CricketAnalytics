"""Feature selection, encoding, scaling, and train/test splitting."""

from __future__ import annotations

import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def split_variable_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split columns into (categorical, continuous) by dtype."""
    categorical: list[str] = []
    continuous: list[str] = []
    for column in df.columns:
        if df[column].dtype != "object":
            continuous.append(column)
        else:
            categorical.append(column)
    return categorical, continuous


def chi_square_filter(
    df: pd.DataFrame,
    categorical_cols: list[str],
    target: str = "Effectiveness",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Drop categorical columns not significantly related to the target."""
    for col in categorical_cols:
        contingency_table = pd.crosstab(df[col], df[target])
        if contingency_table.empty:
            continue
        _, p_value, _, _ = chi2_contingency(contingency_table)
        if p_value > alpha:
            df = df.drop(col, axis=1)
    return df


def label_encode(df: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    """Label-encode the given categorical columns (as strings)."""
    le = LabelEncoder()
    for col in df.columns:
        if col in categorical_cols:
            df[col] = le.fit_transform(df[col].astype(str))
    return df


def scale_and_split(
    df: pd.DataFrame,
    target: str = "Effectiveness",
    test_size: float = 0.20,
    random_state: int = 42,
):
    """Standard-scale features and split into train/test sets."""
    X = df.drop(target, axis=1)
    y = df[target]

    scaler = StandardScaler()
    scaled_X = scaler.fit_transform(X)

    return train_test_split(
        scaled_X, y, test_size=test_size, random_state=random_state
    )
