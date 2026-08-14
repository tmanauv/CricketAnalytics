import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from cricket_analytics.modeling import (
    METRIC_COLUMNS,
    cross_validate_regressors,
    evaluate_param_grid,
    fit_and_evaluate,
    report_param_scores,
    results_frame,
)


@pytest.fixture
def dataset():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))
    y = X[:, 0] * 2 - X[:, 1] + rng.normal(scale=0.1, size=60)
    return X[:40], X[40:], y[:40], y[40:]


def test_cross_validate_regressors_reports_positive_mae(dataset):
    X_train, _, y_train, _ = dataset

    errors = cross_validate_regressors(
        {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(random_state=0),
        },
        X_train,
        y_train,
    )

    assert set(errors) == {"Linear Regression", "Decision Tree"}
    assert all(error > 0 for error in errors.values())


def test_fit_and_evaluate_row_shape_and_rmse(dataset):
    X_train, X_test, y_train, y_test = dataset

    row = fit_and_evaluate(
        DecisionTreeRegressor(random_state=0),
        {"max_depth": 3},
        X_train,
        y_train,
        X_test,
        y_test,
    )

    assert len(row) == len(METRIC_COLUMNS)
    name, _r2, mae, mse, rmse, keys, values = row
    assert name == "DecisionTreeRegressor"
    assert rmse == pytest.approx(mse**0.5)
    assert (list(keys), list(values)) == (["max_depth"], [3])
    assert 0 < mae


def test_evaluate_param_grid_covers_cartesian_product(dataset):
    X_train, X_test, y_train, y_test = dataset
    grid = {"max_depth": [2, 3], "min_samples_leaf": [1, 2, 4]}

    rows = evaluate_param_grid(
        DecisionTreeRegressor(random_state=0),
        grid,
        X_train,
        y_train,
        X_test,
        y_test,
        progress_param="max_depth",
    )

    assert len(rows) == 6
    params = {tuple(row[6]) for row in rows}
    assert params == {(2, 1), (2, 2), (2, 4), (3, 1), (3, 2), (3, 4)}


def test_results_frame_sorted_by_r2_descending(dataset):
    X_train, X_test, y_train, y_test = dataset
    rows = evaluate_param_grid(
        DecisionTreeRegressor(random_state=0),
        {"max_depth": [1, 2, 3]},
        X_train,
        y_train,
        X_test,
        y_test,
    )

    frame = results_frame(rows)

    assert list(frame.columns) == METRIC_COLUMNS
    assert frame["R2"].is_monotonic_decreasing


def test_report_param_scores_refits_each_row(dataset, capsys):
    X_train, X_test, y_train, y_test = dataset
    frame = results_frame(
        evaluate_param_grid(
            DecisionTreeRegressor(random_state=0),
            {"max_depth": [2, 4]},
            X_train,
            y_train,
            X_test,
            y_test,
        )
    )

    scored = report_param_scores(
        lambda **param: DecisionTreeRegressor(random_state=0, **param),
        frame,
        X_train,
        y_train,
        X_test,
        y_test,
    )

    assert [param for param, _ in scored] == [
        {"max_depth": depth}
        for depth in frame["param.values"].apply(lambda v: next(iter(v)))
    ]
    assert [score for _, score in scored] == pytest.approx(list(frame["R2"]))
    assert capsys.readouterr().out
