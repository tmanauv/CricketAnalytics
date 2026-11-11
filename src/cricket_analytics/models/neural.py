"""Keras MLP regressor (requires the ``dl`` extra: ``pip install -e '.[dl]'``)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from cricket_analytics.models.evaluate import regression_metrics


def _keras():
    try:
        from tensorflow import keras
    except ImportError as exc:  # pragma: no cover - exercised only without TF
        raise ImportError(
            "The MLP stage needs TensorFlow. Install it with `pip install -e '.[dl]'`."
        ) from exc
    return keras


def build_mlp(
    input_dim: int, hidden_units: Sequence[int] = (128, 32, 8), learning_rate: float = 1e-3
):
    keras = _keras()
    model = keras.Sequential([keras.Input(shape=(input_dim,))])
    for units in hidden_units:
        model.add(keras.layers.Dense(units, activation="relu"))
    model.add(keras.layers.Dense(1, activation="linear"))
    model.compile(
        loss="mean_squared_error", optimizer=keras.optimizers.Adam(learning_rate=learning_rate)
    )
    return model


def train_mlp(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    mlp_cfg: dict[str, Any],
):
    """Fit the MLP with early stopping on a validation slice of the training data.

    Returns ``(model, scaler, test_metrics, history)``.
    """
    keras = _keras()
    keras.utils.set_random_seed(int(mlp_cfg.get("random_state", 42)))

    scaler = StandardScaler().fit(X_train)
    model = build_mlp(
        X_train.shape[1],
        hidden_units=mlp_cfg.get("hidden_units", (128, 32, 8)),
        learning_rate=float(mlp_cfg.get("learning_rate", 1e-3)),
    )
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=int(mlp_cfg.get("patience", 50)),
        restore_best_weights=True,
    )
    history = model.fit(
        scaler.transform(X_train),
        np.asarray(y_train, dtype=float),
        validation_split=float(mlp_cfg.get("validation_split", 0.2)),
        epochs=int(mlp_cfg.get("epochs", 500)),
        batch_size=int(mlp_cfg.get("batch_size", 100)),
        callbacks=[early_stopping],
        verbose=0,
    )
    y_pred = model.predict(scaler.transform(X_test), verbose=0).ravel()
    return model, scaler, regression_metrics(y_test, y_pred), history.history
