"""Per-batsman aggregate scores derived from shot selection and effectiveness."""

from __future__ import annotations

import pandas as pd

OFF_SIDE_SHOTS = ["DR", "C", "RS"]
LEG_SIDE_SHOTS = ["P", "ODR", "LG", "F", "S"]
BOUNCY_TRACK_SHOTS = ["D", "P", "C", "L"]
AGGRESSIVE_SHOTS = ["Dr", "C", "P", "Sw", "RS", "ODR", "F", "R", "LG", "SC"]
DEFENSE_SHOTS = ["D", "L"]
SPIN_BALL_SHOTS = ["D", "Dr", "C", "P", "Sw", "RS", "ODR", "F", "R", "LG", "SC"]
SPIN_BALLS = ["GY", "LB"]

SCORE_NAMES = [
    "off_side_score",
    "leg_side_score",
    "bouncy_track_score",
    "aggressive_score",
    "defense_score",
    "spin_ball_score",
    "control_score",
]


def _shot_group_score(
    df: pd.DataFrame, mask: pd.Series, label: str, score_col: str
) -> pd.DataFrame:
    """Score batsmen on how often and how effectively they play a shot group.

    ``score`` = (share of balls answered with the group's shots) x (mean
    effectiveness of those shots).
    """
    percentage_col = f"Highest percentage of {label}"
    average_col = f"Highest effectiveness average of {label}"

    counts = (
        df[mask]
        .groupby("Batsman Name")
        .agg(
            shots_count=("Type of Shot", "count"),
            sum_effectiveness=("Effectiveness", "sum"),
        )
    )
    totals = df.groupby("Batsman Name").agg(
        total_no_of_balls_faced=("Type of Shot", "count")
    )

    score_df = counts.merge(totals, on="Batsman Name").reset_index()
    score_df[percentage_col] = (
        score_df["shots_count"] / score_df["total_no_of_balls_faced"]
    ) * 100
    score_df[average_col] = (
        score_df["sum_effectiveness"] / score_df["shots_count"]
    )
    score_df[score_col] = (score_df[percentage_col] / 100) * score_df[average_col]

    score_df = score_df.sort_values(by=[score_col], ascending=False)
    return score_df.loc[:, ["Batsman Name", percentage_col, average_col, score_col]]


def off_side_score(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["Type of Shot"].isin(OFF_SIDE_SHOTS)
    return _shot_group_score(df, mask, "off side shots", "off_side_score")


def leg_side_score(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["Type of Shot"].isin(LEG_SIDE_SHOTS)
    return _shot_group_score(df, mask, "leg side shots", "leg_side_score")


def bouncy_track_score(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["Type of Shot"].isin(BOUNCY_TRACK_SHOTS)
    return _shot_group_score(df, mask, "bouncy track shots", "bouncy_track_score")


def aggressive_shots_score(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["Type of Shot"].isin(AGGRESSIVE_SHOTS)
    return _shot_group_score(df, mask, "aggressive shots", "aggressive_score")


def defense_score(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["Type of Shot"].isin(DEFENSE_SHOTS)
    return _shot_group_score(df, mask, "defense shots", "defense_score")


def spin_ball_score(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["Type of Shot"].isin(SPIN_BALL_SHOTS) & df["Ball type"].isin(
        SPIN_BALLS
    )
    return _shot_group_score(df, mask, "spin ball shots", "spin_ball_score")


def control_score(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["Control"] == 1.0
    return _shot_group_score(df, mask, "controlled shots", "control_score")


def compute_all_scores(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Compute every per-batsman aggregate score frame."""
    return {
        "off_side_score": off_side_score(df),
        "leg_side_score": leg_side_score(df),
        "bouncy_track_score": bouncy_track_score(df),
        "aggressive_score": aggressive_shots_score(df),
        "defense_score": defense_score(df),
        "spin_ball_score": spin_ball_score(df),
        "control_score": control_score(df),
    }


def merge_scores(
    matches_df: pd.DataFrame, score_dfs: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Merge each score column into the ball-by-ball frame on batsman name."""
    for score_col, score_df in score_dfs.items():
        matches_df = pd.merge(
            matches_df,
            score_df.loc[:, ["Batsman Name", score_col]],
            on="Batsman Name",
        )
    return matches_df
