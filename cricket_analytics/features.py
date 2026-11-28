"""Per-batsman aggregate scores.

The notebook defined seven functions (``off_side_score``, ``leg_side_score``,
``bouncy_track_score``, ``aggressive_shots_score``, ``defense_score``,
``spin_ball_score``, ``control_score``) that were textually identical apart
from the subset of rows they counted and the names of the resulting columns.
:func:`batsman_group_score` is that shared body; :data:`SCORE_SPECS` holds the
per-score differences.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

BATSMAN_COLUMN = "Batsman Name"
SHOT_COLUMN = "Type of Shot"
BALL_COLUMN = "Ball type"
CONTROL_COLUMN = "Control"
EFFECTIVENESS_COLUMN = "Effectiveness"


@dataclass(frozen=True)
class ScoreSpec:
    """How one per-batsman score selects its rows and names its columns.

    ``score_column`` is the final score column (e.g. ``off_side_score``);
    ``label`` is the human-readable group used in the percentage and
    effectiveness column names (e.g. ``off side shots``).
    """

    score_column: str
    label: str
    shot_types: tuple[str, ...] = ()
    ball_types: tuple[str, ...] = ()
    require_control: bool = False

    def mask(self, df: pd.DataFrame) -> pd.Series:
        selected = pd.Series(True, index=df.index)
        if self.shot_types:
            selected &= df[SHOT_COLUMN].isin(self.shot_types)
        if self.ball_types:
            selected &= df[BALL_COLUMN].isin(self.ball_types)
        if self.require_control:
            selected &= df[CONTROL_COLUMN] == 1.0
        return selected

    @property
    def percentage_column(self) -> str:
        return f"Highest percentage of {self.label}"

    @property
    def effectiveness_column(self) -> str:
        return f"Highest effectiveness average of {self.label}"

    @property
    def frame_name(self) -> str:
        return f"{self.score_column}_df"


SCORE_SPECS: tuple[ScoreSpec, ...] = (
    ScoreSpec(
        score_column="off_side_score",
        label="off side shots",
        shot_types=("DR", "C", "RS"),
    ),
    ScoreSpec(
        score_column="leg_side_score",
        label="leg side shots",
        shot_types=("P", "ODR", "LG", "F", "S"),
    ),
    ScoreSpec(
        score_column="bouncy_track_score",
        label="bouncy track shots",
        shot_types=("D", "P", "C", "L"),
    ),
    ScoreSpec(
        score_column="aggressive_score",
        label="aggressive shots",
        shot_types=("Dr", "C", "P", "Sw", "RS", "ODR", "F", "R", "LG", "SC"),
    ),
    ScoreSpec(
        score_column="defense_score",
        label="defense shots",
        shot_types=("D", "L"),
    ),
    ScoreSpec(
        score_column="spin_ball_score",
        label="spin ball shots",
        shot_types=("D", "Dr", "C", "P", "Sw", "RS", "ODR", "F", "R", "LG", "SC"),
        ball_types=("GY", "LB"),
    ),
    ScoreSpec(
        score_column="control_score",
        label="controlled shots",
        require_control=True,
    ),
)

SPECS_BY_SCORE: dict[str, ScoreSpec] = {spec.score_column: spec for spec in SCORE_SPECS}


def batsman_group_score(df: pd.DataFrame, spec: ScoreSpec) -> pd.DataFrame:
    """Score every batsman on the row group described by ``spec``.

    The score is the share of a batsman's balls that fall in the group times
    the mean effectiveness of those balls, and rows are sorted by it
    descending. Batsmen with no ball in the group are absent from the result.
    """
    group_counts = (
        df[spec.mask(df)]
        .groupby(BATSMAN_COLUMN)
        .agg(
            group_shots_count=(SHOT_COLUMN, "count"),
            sum_effectiveness=(EFFECTIVENESS_COLUMN, "sum"),
        )
    )
    totals = df.groupby(BATSMAN_COLUMN).agg(
        total_no_of_balls_faced=(SHOT_COLUMN, "count")
    )

    score_df = group_counts.merge(totals, on=BATSMAN_COLUMN).reset_index()
    score_df[spec.percentage_column] = (
        score_df["group_shots_count"] / score_df["total_no_of_balls_faced"]
    ) * 100
    score_df[spec.effectiveness_column] = (
        score_df["sum_effectiveness"] / score_df["group_shots_count"]
    )
    score_df[spec.score_column] = (score_df[spec.percentage_column] / 100) * score_df[
        spec.effectiveness_column
    ]

    score_df = score_df.sort_values(by=[spec.score_column], ascending=False)
    return score_df.loc[
        :,
        [
            BATSMAN_COLUMN,
            spec.percentage_column,
            spec.effectiveness_column,
            spec.score_column,
        ],
    ]


def score_frames(
    df: pd.DataFrame, specs: tuple[ScoreSpec, ...] = SCORE_SPECS
) -> dict[str, pd.DataFrame]:
    """Compute every score, keyed by ``<score_column>_df``."""
    return {spec.frame_name: batsman_group_score(df, spec) for spec in specs}


def merge_scores(df: pd.DataFrame, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Attach the score column of each frame in ``frames`` onto ``df``."""
    merged = df
    for name, score_df in frames.items():
        score_column = name.removesuffix("_df")
        merged = pd.merge(
            merged, score_df.loc[:, [BATSMAN_COLUMN, score_column]], on=BATSMAN_COLUMN
        )
    return merged
