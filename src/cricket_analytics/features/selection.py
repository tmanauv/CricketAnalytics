"""Categorical feature selection by chi-square against the target."""

from __future__ import annotations

import logging

import pandas as pd
from scipy.stats import chi2_contingency

logger = logging.getLogger(__name__)


def split_feature_types(df: pd.DataFrame, target: str) -> tuple[list[str], list[str]]:
    """Return ``(categorical, continuous)`` column names, excluding ``target``."""
    categorical, continuous = [], []
    for col in df.columns:
        if col == target:
            continue
        (continuous if pd.api.types.is_numeric_dtype(df[col]) else categorical).append(col)
    return categorical, continuous


def chi2_select(
    df: pd.DataFrame, categorical: list[str], target: str, alpha: float = 0.05
) -> tuple[list[str], dict[str, float]]:
    """Keep categorical columns significantly related to ``target`` (p <= alpha)."""
    kept, p_values = [], {}
    for col in categorical:
        table = pd.crosstab(df[col], df[target])
        if table.shape[0] < 2 or table.shape[1] < 2:
            logger.info("chi2: dropping %r (constant column or target)", col)
            p_values[col] = 1.0
            continue
        _, p_value, _, _ = chi2_contingency(table)
        p_values[col] = float(p_value)
        if p_value <= alpha:
            kept.append(col)
        else:
            logger.info("chi2: dropping %r (p=%.4f > %.2f)", col, p_value, alpha)
    return kept, p_values
