from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cricket_analytics.modeling.aggregates import (
    SCORE_COLUMNS,
    BatsmanScoreFeatures,
    ScoreSpec,
    batsman_scores,
)


@pytest.fixture
def balls() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Batsman Name": ["A", "A", "A", "A", "B", "B"],
            "Type of Shot": ["DR", "C", "D", "D", "DR", "D"],
            "Ball type": ["OC", "LB", "Y", "GY", "OC", "OC"],
            "Control": [1.0, 0.0, 1.0, 0.0, 1.0, 1.0],
        }
    )


@pytest.fixture
def effectiveness() -> pd.Series:
    return pd.Series([1.0, 0.5, 0.0, 0.2, 0.4, 0.8])


def test_score_matches_the_notebook_formula(
    balls: pd.DataFrame, effectiveness: pd.Series
) -> None:
    scores = batsman_scores(balls, effectiveness)
    # A played 2 of 4 balls off side ('DR', 'C') with mean effectiveness 0.75.
    assert scores.loc["A", "off_side_score"] == pytest.approx(0.5 * 0.75)
    # B played 1 of 2 off side with effectiveness 0.4.
    assert scores.loc["B", "off_side_score"] == pytest.approx(0.5 * 0.4)
    # Spin requires the shot list *and* a spin ball type: A's 'C'/'LB' and
    # 'D'/'GY' qualify, so 2 of 4 balls with mean effectiveness 0.35.
    assert scores.loc["A", "spin_ball_score"] == pytest.approx(0.5 * 0.35)


def test_missing_subset_yields_nan_not_zero(
    balls: pd.DataFrame, effectiveness: pd.Series
) -> None:
    scores = batsman_scores(balls, effectiveness)
    # Neither batsman played a leg-side shot.
    assert scores["leg_side_score"].isna().all()
    # B faced no spin ball at all.
    assert np.isnan(scores.loc["B", "spin_ball_score"])


def test_control_score_uses_the_control_flag_not_a_shot_list(
    balls: pd.DataFrame, effectiveness: pd.Series
) -> None:
    scores = batsman_scores(balls, effectiveness)
    # A's controlled balls are rows 0 and 2 (effectiveness 1.0 and 0.0).
    assert scores.loc["A", "control_score"] == pytest.approx(0.5 * 0.5)


def test_transformer_scores_are_learned_only_from_fit_rows(
    balls: pd.DataFrame, effectiveness: pd.Series
) -> None:
    train = balls[balls["Batsman Name"] == "A"]
    test = balls[balls["Batsman Name"] == "B"]

    transformer = BatsmanScoreFeatures().fit(train, effectiveness.loc[train.index])
    transformed = transformer.transform(test)

    assert list(transformed.columns) == list(SCORE_COLUMNS)
    # B is unseen, so every row gets the training-set mean of each score, i.e.
    # no information about B's own effectiveness reaches the features.
    expected = transformer.scores_.mean(axis=0).fillna(0.0)
    for column in SCORE_COLUMNS:
        assert transformed[column].nunique() == 1
        assert transformed[column].iloc[0] == pytest.approx(expected[column])


def test_transform_preserves_row_order_and_index(
    balls: pd.DataFrame, effectiveness: pd.Series
) -> None:
    transformer = BatsmanScoreFeatures().fit(balls, effectiveness)
    shuffled = balls.iloc[[5, 0, 3]]
    transformed = transformer.transform(shuffled)

    assert transformed.index.tolist() == [5, 0, 3]
    assert transformed.loc[0, "off_side_score"] == pytest.approx(
        transformer.scores_.loc["A", "off_side_score"]
    )


def test_fit_without_target_is_rejected(balls: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="needs y at fit time"):
        BatsmanScoreFeatures().fit(balls)


def test_custom_spec_can_combine_filters(
    balls: pd.DataFrame, effectiveness: pd.Series
) -> None:
    spec = ScoreSpec(
        "controlled_off_cutter",
        shot_types=("DR",),
        ball_types=("OC",),
        require_control=True,
    )
    scores = batsman_scores(balls, effectiveness, specs=(spec,))
    assert scores.loc["A", "controlled_off_cutter"] == pytest.approx(0.25 * 1.0)
    assert scores.loc["B", "controlled_off_cutter"] == pytest.approx(0.5 * 0.4)
