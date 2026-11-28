"""Grid-search tuning on the training split, evaluated once on the test split.

The notebook picked hyper-parameters by their test-set R²; here the search is
cross-validated on the training data only.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline

from cricket_analytics.models.evaluate import regression_metrics
from cricket_analytics.models.registry import MODEL_STEP, make_model, prefixed_grid


def tune_model(
    name: str,
    grid: Mapping[str, list],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = 5,
    scoring: str = "r2",
    random_state: int | None = None,
    n_jobs: int | None = None,
) -> GridSearchCV:
    search = GridSearchCV(
        make_model(name, random_state),
        prefixed_grid(dict(grid)),
        cv=KFold(n_splits=cv_folds, shuffle=True, random_state=random_state),
        scoring=scoring,
        n_jobs=n_jobs,
        refit=True,
    )
    return search.fit(X_train, y_train)


def tune_models(
    grids: Mapping[str, Mapping[str, list]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv_folds: int = 5,
    scoring: str = "r2",
    random_state: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    """Tune every model in ``grids``; return a results table and the refit estimators."""
    rows, best = [], {}
    for name, grid in grids.items():
        search = tune_model(name, grid, X_train, y_train, cv_folds, scoring, random_state)
        estimator = search.best_estimator_
        best[name] = estimator
        params = {k.removeprefix(f"{MODEL_STEP}__"): v for k, v in search.best_params_.items()}
        rows.append(
            {
                "model": name,
                f"cv_{scoring}": search.best_score_,
                **{
                    f"test_{k}": v
                    for k, v in regression_metrics(y_test, estimator.predict(X_test)).items()
                },
                "best_params": params,
            }
        )
    results = pd.DataFrame(rows).sort_values(f"cv_{scoring}", ascending=False, ignore_index=True)
    return results, best


def feature_importances(estimator: Pipeline, feature_names: list[str]) -> pd.Series | None:
    """Feature importances of the pipeline's model step, if it exposes them."""
    model = estimator.named_steps[MODEL_STEP]
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return None
    return pd.Series(importances, index=feature_names).sort_values(ascending=False)
