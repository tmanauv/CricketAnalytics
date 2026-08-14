import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from cricket_analytics.evaluation import fit_and_evaluate


class TestFitAndEvaluate:
    @pytest.fixture
    def linear_data(self):
        rng = np.random.default_rng(0)
        X = rng.random((50, 2))
        y = 2 * X[:, 0] + 3 * X[:, 1]
        return X[:40], y[:40], X[40:], y[40:]

    def test_perfect_fit_metrics(self, linear_data):
        X_train, y_train, X_test, y_test = linear_data

        name, r2, mae, mse, rmse, _, _ = fit_and_evaluate(
            LinearRegression(), {}, X_train, y_train, X_test, y_test
        )

        assert name == "LinearRegression"
        assert r2 == pytest.approx(1.0)
        assert mae == pytest.approx(0.0, abs=1e-9)
        assert mse == pytest.approx(0.0, abs=1e-9)
        assert rmse == pytest.approx(0.0, abs=1e-9)

    def test_rmse_is_sqrt_of_mse(self, linear_data):
        X_train, y_train, X_test, y_test = linear_data
        y_test = y_test + 1.0  # introduce error

        _, _, _, mse, rmse, _, _ = fit_and_evaluate(
            LinearRegression(), {}, X_train, y_train, X_test, y_test
        )

        assert rmse == pytest.approx(np.sqrt(mse))

    def test_params_are_applied(self, linear_data):
        X_train, y_train, X_test, y_test = linear_data
        param = {"fit_intercept": False}

        _, _, _, _, _, keys, values = fit_and_evaluate(
            LinearRegression(), param, X_train, y_train, X_test, y_test
        )

        assert list(keys) == ["fit_intercept"]
        assert list(values) == [False]
