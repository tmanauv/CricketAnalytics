import pandas as pd
import pytest

from cricket_analytics.features import (
    SPECS_BY_SCORE,
    batsman_group_score,
    merge_scores,
    score_frames,
)


@pytest.fixture
def matches_df() -> pd.DataFrame:
    rows = [
        ("Rohit Sharma", "DR", "GY", 1.0, 0.8),
        ("Rohit Sharma", "C", "LB", 0.0, 0.4),
        ("Rohit Sharma", "D", "Y", 1.0, 0.6),
        ("Rohit Sharma", "P", "GY", 1.0, 1.0),
        ("Virat Kohli", "RS", "OC", 0.0, 0.2),
        ("Virat Kohli", "L", "Y", 1.0, 0.5),
        ("Virat Kohli", "S", "LB", 1.0, 0.9),
        ("KL Rahul", "D", "Y", 0.0, -1.0),
        ("KL Rahul", "L", "OC", 1.0, 0.3),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "Batsman Name",
            "Type of Shot",
            "Ball type",
            "Control",
            "Effectiveness",
        ],
    )


def legacy_off_side_score(df: pd.DataFrame) -> pd.DataFrame:
    """The notebook's original ``off_side_score``, kept as a parity oracle."""
    off_side_shots = ["DR", "C", "RS"]

    off_side_shots_count = (
        df[df["Type of Shot"].isin(off_side_shots)]
        .groupby("Batsman Name")
        .agg(
            off_side_shots_count=("Type of Shot", "count"),
            sum_effectiveness=("Effectiveness", "sum"),
        )
    )
    grouped_df = df.groupby("Batsman Name").agg(
        total_no_of_balls_faced=("Type of Shot", "count")
    )

    off_side_score_df = off_side_shots_count.merge(grouped_df, on="Batsman Name")
    off_side_score_df = off_side_score_df.reset_index()

    off_side_score_df["Highest percentage of off side shots"] = (
        off_side_score_df["off_side_shots_count"]
        / off_side_score_df["total_no_of_balls_faced"]
    ) * 100
    off_side_score_df["Highest effectiveness average of off side shots"] = (
        off_side_score_df["sum_effectiveness"]
        / off_side_score_df["off_side_shots_count"]
    )
    off_side_score_df["off_side_score"] = (
        off_side_score_df["Highest percentage of off side shots"] / 100
    ) * off_side_score_df["Highest effectiveness average of off side shots"]

    off_side_score_df = off_side_score_df.sort_values(
        by=["off_side_score"], ascending=False
    )

    return off_side_score_df.loc[
        :,
        [
            "Batsman Name",
            "Highest percentage of off side shots",
            "Highest effectiveness average of off side shots",
            "off_side_score",
        ],
    ]


def legacy_spin_ball_score(df: pd.DataFrame) -> pd.DataFrame:
    """The notebook's original ``spin_ball_score``, kept as a parity oracle."""
    spin_ball_shots = ["D", "Dr", "C", "P", "Sw", "RS", "ODR", "F", "R", "LG", "SC"]
    spin_ball = ["GY", "LB"]

    spin_ball_shots_count = (
        df[df["Type of Shot"].isin(spin_ball_shots) & df["Ball type"].isin(spin_ball)]
        .groupby("Batsman Name")
        .agg(
            spin_ball_shots_count=("Type of Shot", "count"),
            sum_effectiveness=("Effectiveness", "sum"),
        )
    )
    grouped_df = df.groupby("Batsman Name").agg(
        total_no_of_balls_faced=("Type of Shot", "count")
    )

    spin_ball_score_df = spin_ball_shots_count.merge(grouped_df, on="Batsman Name")
    spin_ball_score_df = spin_ball_score_df.reset_index()

    spin_ball_score_df["Highest percentage of spin ball shots"] = (
        spin_ball_score_df["spin_ball_shots_count"]
        / spin_ball_score_df["total_no_of_balls_faced"]
    ) * 100
    spin_ball_score_df["Highest effectiveness average of spin ball shots"] = (
        spin_ball_score_df["sum_effectiveness"]
        / spin_ball_score_df["spin_ball_shots_count"]
    )
    spin_ball_score_df["spin_ball_score"] = (
        spin_ball_score_df["Highest percentage of spin ball shots"] / 100
    ) * spin_ball_score_df["Highest effectiveness average of spin ball shots"]

    spin_ball_score_df = spin_ball_score_df.sort_values(
        by=["spin_ball_score"], ascending=False
    )

    return spin_ball_score_df.loc[
        :,
        [
            "Batsman Name",
            "Highest percentage of spin ball shots",
            "Highest effectiveness average of spin ball shots",
            "spin_ball_score",
        ],
    ]


def legacy_control_score(df: pd.DataFrame) -> pd.DataFrame:
    """The notebook's original ``control_score``, kept as a parity oracle."""
    shot_count_df = (
        df[df["Control"] == 1.0]
        .groupby("Batsman Name")
        .agg(
            controlled_shots_count=("Type of Shot", "count"),
            sum_effectiveness=("Effectiveness", "sum"),
        )
    )
    grouped_df = df.groupby("Batsman Name").agg(
        total_no_of_balls_faced=("Type of Shot", "count")
    )

    control_score_df = shot_count_df.merge(grouped_df, on="Batsman Name")
    control_score_df = control_score_df.reset_index()

    control_score_df["Highest percentage of controlled shots"] = (
        control_score_df["controlled_shots_count"]
        / control_score_df["total_no_of_balls_faced"]
    ) * 100
    control_score_df["Highest effectiveness average of controlled shots"] = (
        control_score_df["sum_effectiveness"]
        / control_score_df["controlled_shots_count"]
    )
    control_score_df["control_score"] = (
        control_score_df["Highest percentage of controlled shots"] / 100
    ) * control_score_df["Highest effectiveness average of controlled shots"]

    control_score_df = control_score_df.sort_values(
        by=["control_score"], ascending=False
    )

    return control_score_df.loc[
        :,
        [
            "Batsman Name",
            "Highest percentage of controlled shots",
            "Highest effectiveness average of controlled shots",
            "control_score",
        ],
    ]


@pytest.mark.parametrize(
    ("score_column", "legacy"),
    [
        ("off_side_score", legacy_off_side_score),
        ("spin_ball_score", legacy_spin_ball_score),
        ("control_score", legacy_control_score),
    ],
)
def test_matches_legacy_implementation(matches_df, score_column, legacy):
    expected = legacy(matches_df)
    actual = batsman_group_score(matches_df, SPECS_BY_SCORE[score_column])

    pd.testing.assert_frame_equal(actual, expected)


def test_score_is_share_times_mean_effectiveness(matches_df):
    result = batsman_group_score(matches_df, SPECS_BY_SCORE["defense_score"]).set_index(
        "Batsman Name"
    )

    # KL Rahul faced 2 defensive shots ("D", "L") out of 2 balls, mean -0.35.
    assert result.loc["KL Rahul", "Highest percentage of defense shots"] == 100.0
    assert result.loc["KL Rahul", "Highest effectiveness average of defense shots"] == (
        pytest.approx(-0.35)
    )
    assert result.loc["KL Rahul", "defense_score"] == pytest.approx(-0.35)


def test_batsman_without_group_shot_is_absent(matches_df):
    result = batsman_group_score(matches_df, SPECS_BY_SCORE["leg_side_score"])

    # Only Rohit ("P") and Virat ("S") play leg side shots in the fixture.
    assert list(result["Batsman Name"]) == ["Virat Kohli", "Rohit Sharma"]


def test_score_frames_and_merge(matches_df):
    frames = score_frames(matches_df)
    assert set(frames) == {f"{spec}_df" for spec in SPECS_BY_SCORE}

    merged = merge_scores(matches_df, frames)
    for score_column in SPECS_BY_SCORE:
        assert score_column in merged.columns
    # Every score requires the batsman to appear in all seven frames.
    assert set(merged["Batsman Name"]) == {"Rohit Sharma"}
