"""Model evaluation helpers shared by the notebook's modeling cells."""

from __future__ import annotations

import itertools
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score

METRIC_COLUMNS = ["reg", "R2", "MAE", "MSE", "RMSE", "param.keys", "param.values"]


def cross_validate_regressors(
    regressors: Mapping[str, Any],
    X,
    y,
    cv: int = 5,
    scoring: str = "neg_mean_absolute_error",
) -> dict[str, float]:
    """Cross-validate each regressor and return its mean absolute error.

    Replaces the block that instantiated ``reg1``…``reg8``, called
    ``cross_val_score`` eight times and printed eight negated means.
    """
    errors: dict[str, float] = {}
    for name, regressor in regressors.items():
        scores = cross_val_score(regressor, X, y, cv=cv, scoring=scoring)
        errors[name] = float(-scores.mean())
        print(f"{name} mean absolute error: ", errors[name])
    return errors


def fit_and_evaluate(
    reg, param: Mapping[str, Any], X_train, y_train, X_test, y_test
) -> list:
    """Fit ``reg`` with ``param`` and return one :data:`METRIC_COLUMNS` row."""
    reg.set_params(**param)
    reg.fit(X_train, y_train)

    y_predict = reg.predict(X_test)
    mse = mean_squared_error(y_test, y_predict)
    return [
        reg.__class__.__name__,
        r2_score(y_test, y_predict),
        mean_absolute_error(y_test, y_predict),
        mse,
        mse**0.5,
        param.keys(),
        param.values(),
    ]


def evaluate_param_grid(
    reg,
    grid: Mapping[str, Sequence[Any]],
    X_train,
    y_train,
    X_test,
    y_test,
    progress_param: str | None = None,
) -> list[list]:
    """Fit ``reg`` once per point of the cartesian product of ``grid``.

    Replaces the hand-written nested ``for`` loops, whose depth grew with the
    number of tuned hyper-parameters. ``progress_param`` names the grid entry
    whose value is echoed as a progress indicator.
    """
    rows = []
    keys = list(grid)
    for values in itertools.product(*(grid[key] for key in keys)):
        param = dict(zip(keys, values))
        if progress_param is not None:
            total = len(grid[progress_param])
            sys.stdout.write(f"\r{progress_param}: {param[progress_param]}/{total}...")
        rows.append(fit_and_evaluate(reg, param, X_train, y_train, X_test, y_test))
    return rows


def results_frame(rows: Iterable[list]) -> pd.DataFrame:
    """Build the ``R2``-sorted results table from :func:`fit_and_evaluate` rows."""
    frame = pd.DataFrame(data=list(rows), columns=METRIC_COLUMNS, index=None)
    return frame.sort_values(by=["R2"], ascending=False)


def report_param_scores(
    estimator_factory: Callable[..., Any],
    results: pd.DataFrame,
    X_train,
    y_train,
    X_test,
    y_test,
) -> list[tuple[dict[str, Any], float]]:
    """Refit ``estimator_factory`` on each row of ``results`` and print its R2.

    ``results`` is a slice of :func:`results_frame` (typically the top rows for
    one estimator); its ``param.keys`` / ``param.values`` pairs are the
    hyper-parameters to refit.
    """
    scored = []
    for keys, values in zip(results["param.keys"], results["param.values"]):
        param = dict(zip(list(keys), list(values)))
        reg = estimator_factory(**param)
        reg.fit(X_train, y_train)
        score = r2_score(y_test, reg.predict(X_test))
        scored.append((param, score))
        print(param)
        print(score)
        print()
    return scored


def plot_feature_importances(reg, feature_names, title: str = "Top Features"):
    """Horizontal bar plot of ``reg.feature_importances_``, ascending."""
    import matplotlib.pyplot as plt

    importances = pd.Series(reg.feature_importances_, index=feature_names)
    axes = importances.sort_values().plot(kind="barh", figsize=(15, 10), title=title)
    plt.show()
    return axes
