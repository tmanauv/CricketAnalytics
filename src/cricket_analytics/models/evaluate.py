"""Metrics and cross-validated baseline comparison."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score

from cricket_analytics.models.registry import make_model


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
    }


def cross_validate_models(
    X: pd.DataFrame,
    y: pd.Series,
    models: Iterable[str],
    cv_folds: int = 5,
    scoring: str = "neg_mean_absolute_error",
    random_state: int | None = None,
) -> pd.DataFrame:
    """Cross-validate each named model on the training data.

    Scores from ``neg_*`` scorers are negated so lower is better for errors.
    """
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    sign = -1.0 if scoring.startswith("neg_") else 1.0
    metric = scoring.removeprefix("neg_")
    rows = []
    for name in models:
        scores = sign * cross_val_score(
            make_model(name, random_state), X, y, cv=cv, scoring=scoring
        )
        rows.append({"model": name, f"{metric}_mean": scores.mean(), f"{metric}_std": scores.std()})
    return pd.DataFrame(rows).sort_values(f"{metric}_mean", ascending=sign < 0, ignore_index=True)
