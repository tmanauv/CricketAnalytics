"""Leak-free modelling pipeline for shot effectiveness.

Everything the notebook does outside a fold — encoding, scaling, per-batsman
aggregates — happens inside an sklearn :class:`~sklearn.pipeline.Pipeline` here,
and every split is grouped by batsman, because rows of the same batsman share
aggregate features and are not independent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    GroupShuffleSplit,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .aggregates import BATSMAN_COLUMN, BatsmanScoreFeatures

TARGET_COLUMN = "Effectiveness"

#: Columns that determine the target arithmetically and must never be features.
#: ``Effectiveness`` is ``-1.0`` exactly when the ball is a wicket (see
#: ``data/README.md``), so ``Wicket``/``Dismissal kind`` give a model the answer
#: for every negative target. The notebook feeds both to its regressors.
DETERMINISTIC_LEAK_COLUMNS: tuple[str, ...] = ("Wicket", "Dismissal kind")

#: Identity columns: used to group the split and to key the aggregates, never
#: encoded as features. A label-encoded batsman id is meaningless for a batsman
#: the model has never seen, which is precisely the grouped-split case.
IDENTITY_COLUMNS: tuple[str, ...] = (BATSMAN_COLUMN, "Batsman Player ID")

CATEGORICAL_FEATURES: tuple[str, ...] = ("Ball type", "Type of Shot", "Bowler Name")
NUMERIC_FEATURES: tuple[str, ...] = ("Wide", "No ball", "Hit the Bat", "Control")


def feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split the frame's columns into the categorical / numeric features used."""
    categorical = [col for col in CATEGORICAL_FEATURES if col in frame.columns]
    numeric = [col for col in NUMERIC_FEATURES if col in frame.columns]
    return categorical, numeric


def drop_leaky_columns(frame: pd.DataFrame) -> pd.DataFrame:
    present = [col for col in DETERMINISTIC_LEAK_COLUMNS if col in frame.columns]
    return frame.drop(columns=present)


def build_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    categorical, numeric = feature_columns(frame)
    aggregate_inputs = [
        column
        for column in (BATSMAN_COLUMN, "Type of Shot", "Ball type", "Control")
        if column in frame.columns
    ]
    return ColumnTransformer(
        transformers=[
            (
                "batsman_scores",
                Pipeline(
                    [
                        ("scores", BatsmanScoreFeatures()),
                        ("scale", StandardScaler()),
                    ]
                ),
                aggregate_inputs,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
            ("numeric", StandardScaler(), numeric),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(frame: pd.DataFrame, estimator: BaseEstimator) -> Pipeline:
    """Preprocessing + estimator, all fit inside whatever fold wraps it."""
    return Pipeline(
        [("preprocess", build_preprocessor(frame)), ("estimator", estimator)]
    )


class BallTypeMeanRegressor(RegressorMixin, BaseEstimator):
    """Predict the training mean effectiveness of the ball type.

    A model that cannot beat this has learned nothing about shot selection.
    """

    def __init__(self, ball_column: str = "Ball type") -> None:
        self.ball_column = ball_column

    def fit(self, X: pd.DataFrame, y) -> BallTypeMeanRegressor:
        target = pd.Series(np.asarray(y, dtype=float), index=X.index)
        self.global_mean_ = float(target.mean())
        self.means_ = target.groupby(X[self.ball_column], sort=False).mean()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (
            self.means_.reindex(X[self.ball_column])
            .fillna(self.global_mean_)
            .to_numpy(dtype=float)
        )


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the target and remove the columns that trivially reveal it."""
    if TARGET_COLUMN not in frame.columns:
        raise KeyError(
            f"{TARGET_COLUMN!r} column is required; got {list(frame.columns)}"
        )
    target = frame[TARGET_COLUMN].astype(float)
    features = drop_leaky_columns(frame.drop(columns=[TARGET_COLUMN]))
    return features, target


def grouped_train_test_split(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Hold out whole batsmen, so no batsman appears on both sides."""
    groups = features[BATSMAN_COLUMN]
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_index, test_index = next(splitter.split(features, target, groups=groups))
    return (
        features.iloc[train_index],
        features.iloc[test_index],
        target.iloc[train_index],
        target.iloc[test_index],
    )


def group_cv(features: pd.DataFrame, n_splits: int = 5) -> GroupKFold:
    """A ``GroupKFold`` with at most one fold per available batsman."""
    n_groups = features[BATSMAN_COLUMN].nunique()
    return GroupKFold(n_splits=max(2, min(n_splits, n_groups)))


def cross_validate_model(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    target: pd.Series,
    estimator: BaseEstimator,
    n_splits: int = 5,
    scoring: str = "r2",
) -> np.ndarray:
    """Group-aware CV of the full pipeline, refit from scratch in every fold."""
    pipeline = build_pipeline(frame, clone(estimator))
    return cross_val_score(
        pipeline,
        features,
        target,
        groups=features[BATSMAN_COLUMN],
        cv=group_cv(features, n_splits),
        scoring=scoring,
    )


def tune(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    target: pd.Series,
    estimator: BaseEstimator,
    param_grid: dict[str, list],
    n_splits: int = 4,
    scoring: str = "r2",
) -> GridSearchCV:
    """Select hyper-parameters with an inner grouped CV on the training data.

    The test split is never passed to this function; the notebook instead scored
    every grid point with ``r2_score(y_test, ...)`` and then reported the best of
    those same scores.
    """
    search = GridSearchCV(
        build_pipeline(frame, clone(estimator)),
        param_grid={f"estimator__{key}": value for key, value in param_grid.items()},
        cv=group_cv(features, n_splits),
        scoring=scoring,
        n_jobs=1,
    )
    search.fit(features, target, groups=features[BATSMAN_COLUMN])
    return search


@dataclass(frozen=True)
class Metrics:
    r2: float
    mae: float
    mse: float
    rmse: float

    def as_dict(self) -> dict[str, float]:
        return {"r2": self.r2, "mae": self.mae, "mse": self.mse, "rmse": self.rmse}


def evaluate(model, features: pd.DataFrame, target: pd.Series) -> Metrics:
    predictions = model.predict(features)
    mse = float(mean_squared_error(target, predictions))
    return Metrics(
        r2=float(r2_score(target, predictions)),
        mae=float(mean_absolute_error(target, predictions)),
        mse=mse,
        # np.sqrt rather than `squared=False`, which scikit-learn 1.6 removed.
        rmse=float(np.sqrt(mse)),
    )


def baseline_estimators() -> dict[str, BaseEstimator]:
    return {
        "mean": DummyRegressor(strategy="mean"),
        "ball_type_mean": BallTypeMeanRegressor(),
    }
