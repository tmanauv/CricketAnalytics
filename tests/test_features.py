import pandas as pd
import pytest

from cricket_analytics.features import (
    aggressive_shots_score,
    bouncy_track_score,
    compute_all_scores,
    control_score,
    defense_score,
    leg_side_score,
    merge_scores,
    off_side_score,
    spin_ball_score,
)


class TestOffSideScore:
    def test_score_computation(self, matches_df):
        result = off_side_score(matches_df)
        rohit = result[result["Batsman Name"] == "Rohit"].iloc[0]

        # Rohit: off-side shots DR, C -> 2 of 4 balls, effectiveness 0.8 + 0.6
        assert rohit["Highest percentage of off side shots"] == pytest.approx(50.0)
        assert rohit[
            "Highest effectiveness average of off side shots"
        ] == pytest.approx(0.7)
        assert rohit["off_side_score"] == pytest.approx(0.35)

    def test_sorted_descending(self, matches_df):
        result = off_side_score(matches_df)

        scores = result["off_side_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_excludes_batsmen_without_group_shots(self):
        df = pd.DataFrame(
            {
                "Batsman Name": ["A", "B"],
                "Type of Shot": ["DR", "L"],
                "Ball type": ["FT", "FT"],
                "Control": [1.0, 1.0],
                "Effectiveness": [0.5, 0.5],
            }
        )

        result = off_side_score(df)

        assert result["Batsman Name"].tolist() == ["A"]


class TestLegSideScore:
    def test_score_computation(self, matches_df):
        result = leg_side_score(matches_df)
        rohit = result[result["Batsman Name"] == "Rohit"].iloc[0]

        # Rohit: leg-side shot P -> 1 of 4 balls, effectiveness 0.4
        assert rohit["leg_side_score"] == pytest.approx(0.25 * 0.4)


class TestBouncyTrackScore:
    def test_score_computation(self, matches_df):
        result = bouncy_track_score(matches_df)
        rohit = result[result["Batsman Name"] == "Rohit"].iloc[0]

        # Rohit: bouncy-track shots C(0.6), P(0.4), D(0.2) -> 3 of 4 balls
        assert rohit["bouncy_track_score"] == pytest.approx(0.75 * 0.4)


class TestAggressiveShotsScore:
    def test_score_computation(self, matches_df):
        result = aggressive_shots_score(matches_df)
        virat = result[result["Batsman Name"] == "Virat"].iloc[0]

        # Virat: aggressive shots RS(1.0), F(0.5), LG(0.7) -> 3 of 4 balls
        assert virat["aggressive_score"] == pytest.approx(
            0.75 * (2.2 / 3)
        )


class TestDefenseScore:
    def test_score_computation(self, matches_df):
        result = defense_score(matches_df)
        rohit = result[result["Batsman Name"] == "Rohit"].iloc[0]

        # Rohit: defense shot D -> 1 of 4 balls, effectiveness 0.2
        assert rohit["defense_score"] == pytest.approx(0.25 * 0.2)


class TestSpinBallScore:
    def test_requires_spin_ball_and_shot(self, matches_df):
        result = spin_ball_score(matches_df)
        rohit = result[result["Batsman Name"] == "Rohit"].iloc[0]

        # Rohit: spin balls GY(shot C, 0.6) and LB(shot P, 0.4) -> 2 of 4
        assert rohit["spin_ball_score"] == pytest.approx(0.5 * 0.5)

    def test_non_spin_balls_excluded(self):
        df = pd.DataFrame(
            {
                "Batsman Name": ["A", "A"],
                "Type of Shot": ["C", "C"],
                "Ball type": ["FT", "GY"],
                "Control": [1.0, 1.0],
                "Effectiveness": [0.4, 0.8],
            }
        )

        result = spin_ball_score(df)
        a = result.iloc[0]

        assert a["spin_ball_score"] == pytest.approx(0.5 * 0.8)


class TestControlScore:
    def test_only_controlled_shots_counted(self, matches_df):
        result = control_score(matches_df)
        virat = result[result["Batsman Name"] == "Virat"].iloc[0]

        # Virat: controlled shots RS(1.0), LG(0.7) -> 2 of 4 balls
        assert virat["control_score"] == pytest.approx(0.5 * 0.85)


class TestComputeAllScores:
    def test_returns_all_seven_scores(self, matches_df):
        result = compute_all_scores(matches_df)

        assert set(result.keys()) == {
            "off_side_score",
            "leg_side_score",
            "bouncy_track_score",
            "aggressive_score",
            "defense_score",
            "spin_ball_score",
            "control_score",
        }
        for score_col, score_df in result.items():
            assert score_col in score_df.columns
            assert "Batsman Name" in score_df.columns


class TestMergeScores:
    def test_adds_score_columns(self, matches_df):
        score_dfs = compute_all_scores(matches_df)

        result = merge_scores(matches_df, score_dfs)

        for score_col in score_dfs:
            assert score_col in result.columns

    def test_scores_aligned_by_batsman(self, matches_df):
        score_dfs = {"off_side_score": off_side_score(matches_df)}

        result = merge_scores(matches_df, score_dfs)
        rohit_scores = result[result["Batsman Name"] == "Rohit"][
            "off_side_score"
        ].unique()

        assert len(rohit_scores) == 1
        assert rohit_scores[0] == pytest.approx(0.35)
