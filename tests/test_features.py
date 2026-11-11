import pandas as pd
import pytest

from cricket_analytics.data import columns as C
from cricket_analytics.features.batsman_scores import compute_batsman_scores, shot_group_score
from cricket_analytics.features.builder import FeatureBuilder
from cricket_analytics.features.selection import chi2_select, split_feature_types


@pytest.fixture
def deliveries():
    return pd.DataFrame(
        {
            C.BATSMAN: ["A", "A", "A", "A", "B", "B"],
            C.SHOT: ["DR", "DR", "D", "P", "D", "D"],
            C.BALL_TYPE: ["LB", "OC", "LB", "OC", "GY", "GY"],
            C.CONTROL: [1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            C.TARGET: [0.8, 0.4, 0.2, 0.6, 0.1, 0.3],
        }
    )


def test_shot_group_score(deliveries):
    scores = shot_group_score(deliveries, "off_side", shots=["DR"]).set_index(C.BATSMAN)
    assert scores.loc["A", "pct_off_side_shots"] == pytest.approx(50.0)
    assert scores.loc["A", "avg_effectiveness_off_side_shots"] == pytest.approx(0.6)
    assert scores.loc["A", "off_side_score"] == pytest.approx(0.3)
    assert scores.loc["B", "off_side_score"] == 0.0


def test_shot_group_score_filters(deliveries):
    spin = shot_group_score(deliveries, "spin", shots=["D"], ball_types=["GY", "LB"]).set_index(
        C.BATSMAN
    )
    assert spin.loc["B", "spin_score"] == pytest.approx(0.2)
    control = shot_group_score(deliveries, "control", controlled_only=True).set_index(C.BATSMAN)
    assert control.loc["A", "control_score"] == pytest.approx(0.75 * (1.6 / 3))


def test_compute_batsman_scores_keeps_every_batsman(deliveries):
    scores = compute_batsman_scores(deliveries, {"leg_side": {"shots": ["P"]}})
    assert scores[C.BATSMAN].tolist() == ["A", "B"]
    assert scores["leg_side_score"].tolist() == pytest.approx([0.15, 0.0])


def test_split_feature_types(deliveries):
    categorical, continuous = split_feature_types(deliveries, C.TARGET)
    assert categorical == [C.BATSMAN, C.SHOT, C.BALL_TYPE]
    assert continuous == [C.CONTROL]


def test_chi2_select_drops_constant_and_unrelated():
    df = pd.DataFrame(
        {
            "related": ["a"] * 50 + ["b"] * 50,
            "constant": ["x"] * 100,
            "noise": ["p", "q"] * 50,
            C.TARGET: [0.0] * 50 + [1.0] * 50,
        }
    )
    kept, p_values = chi2_select(df, ["related", "constant", "noise"], C.TARGET)
    assert kept == ["related"]
    assert p_values["constant"] == 1.0


def test_feature_builder_handles_unseen_rows(cfg, deliveries):
    cfg["features"]["chi2_alpha"] = 1.0
    builder = FeatureBuilder(cfg["features"]).fit(deliveries)
    unseen = deliveries.head(1).assign(**{C.BATSMAN: "Z", C.SHOT: "SW"})
    X, y = builder.transform(unseen)
    assert list(X.columns) == builder.feature_names_
    assert X.loc[0, C.SHOT] == -1
    assert X.filter(like="_score").iloc[0].eq(0.0).all()
    assert y.tolist() == [0.8]


def test_feature_builder_drop_columns(cfg, deliveries):
    cfg["features"]["drop_columns"] = [C.CONTROL]
    builder = FeatureBuilder(cfg["features"]).fit(deliveries)
    assert C.CONTROL not in builder.feature_names_
