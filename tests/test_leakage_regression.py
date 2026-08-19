"""Demonstrates, and guards against, the two leaks in the notebook pipeline.

Both tests build data whose target is pure noise — there is nothing to learn, so
any pipeline reporting a good score is reporting leakage, not skill.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from cricket_analytics.modeling.aggregates import SCORE_COLUMNS, batsman_scores
from cricket_analytics.modeling.pipeline import (
    build_pipeline,
    cross_validate_model,
    grouped_train_test_split,
)

BALL_TYPES = ["OC", "LC", "Y", "B", "GY", "LB"]
SHOT_TYPES = ["D", "Dr", "DR", "ODR", "C", "P", "Sw", "RS", "F", "R", "LG", "SC", "L"]


# The size of the leak scales with how much of a batsman's aggregate comes from
# a single ball: with ~3 balls per batsman each row's own target dominates its
# own features, which makes the effect visible at test-suite scale. The leak
# exists at any size, it is just diluted when batsmen face hundreds of balls.
def noise_frame(rows: int = 3000, batsmen: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Ball-by-ball rows whose Effectiveness is independent of every feature."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "Batsman Name": [f"B{index}" for index in rng.integers(0, batsmen, rows)],
            "Bowler Name": [f"P{index}" for index in rng.integers(0, 5, rows)],
            "Ball type": rng.choice(BALL_TYPES, rows),
            "Type of Shot": rng.choice(SHOT_TYPES, rows),
            "Wide": 0.0,
            "No ball": 0.0,
            "Hit the Bat": rng.integers(0, 2, rows).astype(float),
            "Control": rng.integers(0, 2, rows).astype(float),
            "Effectiveness": rng.choice([step / 10 for step in range(11)], rows),
        }
    )


def notebook_style_r2(frame: pd.DataFrame, seed: int = 0) -> float:
    """The notebook's own recipe: aggregate over everything, then split randomly.

    Mirrors cells 51-98 — scores merged onto every row from the full dataset,
    ``LabelEncoder``/``StandardScaler`` fit before the split, and a random split
    that puts the same batsman on both sides.
    """
    scores = batsman_scores(frame, frame["Effectiveness"])
    merged = frame.join(scores, on="Batsman Name").fillna(0.0)

    encoded = merged.copy()
    for column in encoded.columns:
        if encoded[column].dtype == object:
            encoded[column] = LabelEncoder().fit_transform(encoded[column].astype(str))

    target = encoded.pop("Effectiveness")
    scaled = StandardScaler().fit_transform(encoded)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled, target, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=seed).fit(
        X_train, y_train
    )
    return float(r2_score(y_test, model.predict(X_test)))


def test_notebook_recipe_scores_well_on_pure_noise() -> None:
    frame = noise_frame()
    # Nothing is learnable here, yet the notebook's recipe reports skill: the
    # per-batsman scores carry each row's own target, and the random split lets
    # the model look its training batsmen up again at test time.
    assert notebook_style_r2(frame) > 0.15


def test_leak_free_pipeline_scores_no_better_than_chance_on_pure_noise() -> None:
    frame = noise_frame()
    scores = cross_validate_model(
        frame,
        *_features_and_target(frame),
        RandomForestRegressor(n_estimators=50, random_state=0),
        n_splits=4,
    )
    # R² of an honest model on noise sits at or below zero.
    assert scores.mean() < 0.05
    assert notebook_style_r2(frame) > scores.mean()


def test_holdout_batsman_features_are_independent_of_their_own_target() -> None:
    frame = noise_frame(rows=900, batsmen=12, seed=1)
    features, target = _features_and_target(frame)
    X_train, X_test, y_train, _ = grouped_train_test_split(features, target)

    pipeline = build_pipeline(
        frame, RandomForestRegressor(n_estimators=10, random_state=0)
    )
    pipeline.fit(X_train, y_train)
    scores = (
        pipeline.named_steps["preprocess"]
        .named_transformers_["batsman_scores"]
        .named_steps["scores"]
    )

    # No held-out batsman contributed to the learned score table, so their
    # aggregate features are the training fallback and cannot vary with their own
    # effectiveness — whereas the notebook's merge gives each held-out row a
    # score computed partly from that very row.
    assert not set(scores.scores_.index) & set(X_test["Batsman Name"])
    transformed = scores.transform(X_test)
    assert set(transformed.columns) == set(SCORE_COLUMNS)
    assert (transformed.nunique() == 1).all()

    leaky = batsman_scores(features, target).reindex(X_test["Batsman Name"])
    assert leaky.nunique().max() > 1


def _features_and_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return frame.drop(columns=["Effectiveness"]), frame["Effectiveness"].astype(float)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_leak_free_cv_is_stable_across_seeds(seed: int) -> None:
    frame = noise_frame(rows=900, batsmen=12, seed=seed)
    scores = cross_validate_model(
        frame,
        *_features_and_target(frame),
        RandomForestRegressor(n_estimators=50, random_state=seed),
        n_splits=3,
    )
    assert scores.mean() < 0.05
