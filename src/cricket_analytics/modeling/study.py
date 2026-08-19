"""Run the effectiveness study end to end on a prepared ball-by-ball frame."""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor

from .aggregates import BATSMAN_COLUMN
from .pipeline import (
    baseline_estimators,
    build_pipeline,
    cross_validate_model,
    evaluate,
    group_cv,
    grouped_train_test_split,
    split_features_target,
)

RANDOM_SEED = 12345


def default_estimators(random_state: int = RANDOM_SEED) -> dict[str, BaseEstimator]:
    return {
        # Ridge rather than the notebook's plain ``LinearRegression``: one-hot
        # encoding bowlers and shot types makes the design matrix collinear, on
        # which an unregularised fit produces absurd coefficients and R² values
        # many orders of magnitude below zero.
        "ridge": Ridge(alpha=1.0, random_state=random_state),
        "decision_tree": DecisionTreeRegressor(random_state=random_state),
        "random_forest": RandomForestRegressor(
            n_estimators=200, random_state=random_state
        ),
        "gradient_boosting": GradientBoostingRegressor(random_state=random_state),
    }


def run_study(
    frame: pd.DataFrame,
    estimators: dict[str, BaseEstimator] | None = None,
    test_size: float = 0.2,
    n_splits: int = 5,
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Cross-validate on held-in batsmen, then score once on held-out batsmen.

    Returns one row per model with the grouped-CV R² (mean ± std over folds) and
    the single-shot test metrics. Baselines are included so the numbers can be
    read as "did the model beat predicting the mean".
    """
    features, target = split_features_target(frame)
    X_train, X_test, y_train, y_test = grouped_train_test_split(
        features, target, test_size=test_size, random_state=random_state
    )

    rows = []
    for name, estimator in baseline_estimators().items():
        # Baselines consume the raw frame (no encoding or scaling to fit), so
        # they bypass the preprocessing pipeline.
        scores = cross_val_score(
            clone(estimator),
            X_train,
            y_train,
            groups=X_train[BATSMAN_COLUMN],
            cv=group_cv(X_train, n_splits),
            scoring="r2",
        )
        fitted = clone(estimator).fit(X_train, y_train)
        rows.append(
            {
                "model": name,
                "kind": "baseline",
                "cv_r2_mean": scores.mean(),
                "cv_r2_std": scores.std(),
                **evaluate(fitted, X_test, y_test).as_dict(),
            }
        )

    for name, estimator in (estimators or default_estimators(random_state)).items():
        scores = cross_validate_model(
            frame, X_train, y_train, estimator, n_splits=n_splits
        )
        fitted = build_pipeline(frame, clone(estimator)).fit(X_train, y_train)
        rows.append(
            {
                "model": name,
                "kind": "model",
                "cv_r2_mean": scores.mean(),
                "cv_r2_std": scores.std(),
                **evaluate(fitted, X_test, y_test).as_dict(),
            }
        )

    return pd.DataFrame(rows).sort_values(
        "cv_r2_mean", ascending=False, ignore_index=True
    )
