"""Fit regressors and report standard regression metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def fit_and_evaluate(reg, param: dict, X_train, y_train, X_test, y_test) -> list:
    """Fit ``reg`` with ``param`` and return name + R2/MAE/MSE/RMSE + params."""
    reg.set_params(**param)
    reg.fit(X_train, y_train)

    y_predict = reg.predict(X_test)
    r2 = r2_score(y_test, y_predict)
    mae = mean_absolute_error(y_test, y_predict)
    mse = mean_squared_error(y_test, y_predict)
    rmse = float(np.sqrt(mse))

    return [reg.__class__.__name__, r2, mae, mse, rmse, param.keys(), param.values()]
