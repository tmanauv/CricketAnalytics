import numpy as np
import pandas as pd
import pytest

from cricket_analytics.models.evaluate import cross_validate_models, regression_metrics
from cricket_analytics.models.registry import AVAILABLE_MODELS, make_model, prefixed_grid
from cricket_analytics.models.tuning import feature_importances, tune_models


@pytest.fixture
def regression_data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(120, 3)), columns=["a", "b", "c"])
    y = pd.Series(2 * X["a"] - X["b"] + rng.normal(scale=0.1, size=120))
    return X, y


def test_regression_metrics():
    metrics = regression_metrics([0.0, 1.0], [0.0, 0.5])
    assert metrics["mae"] == pytest.approx(0.25)
    assert metrics["rmse"] == pytest.approx(np.sqrt(0.125))


@pytest.mark.parametrize("name", AVAILABLE_MODELS)
def test_make_model_fits(name, regression_data):
    X, y = regression_data
    model = make_model(name, random_state=0).fit(X, y)
    assert model.predict(X).shape == (len(X),)


def test_make_model_unknown():
    with pytest.raises(KeyError):
        make_model("nope")


def test_prefixed_grid():
    assert prefixed_grid({"max_depth": (3, 5)}) == {"model__max_depth": [3, 5]}


def test_cross_validate_models_sorted(regression_data):
    X, y = regression_data
    results = cross_validate_models(X, y, ["linear", "decision_tree"], cv_folds=3, random_state=0)
    assert results["model"].tolist()[0] == "linear"
    assert results["mean_absolute_error_mean"].is_monotonic_increasing


def test_tune_models(regression_data):
    X, y = regression_data
    results, best = tune_models(
        {"decision_tree": {"max_depth": [2, 4]}},
        X[:90],
        y[:90],
        X[90:],
        y[90:],
        cv_folds=3,
        random_state=0,
    )
    assert results.loc[0, "best_params"]["max_depth"] in (2, 4)
    assert {"test_r2", "test_mae", "test_rmse"} <= set(results.columns)
    importances = feature_importances(best["decision_tree"], list(X.columns))
    assert importances.index[0] == "a"
    assert feature_importances(make_model("linear").fit(X, y), list(X.columns)) is None
