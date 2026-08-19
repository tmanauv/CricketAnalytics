"""Per-batsman aggregate features, computed without target leakage.

The notebook builds seven per-batsman scores of the shape

    score = (matching balls / balls faced) * mean(Effectiveness over matching balls)

and merges them back onto every ball of that batsman (``SDP.ipynb`` cells
51-77). Because the mean is taken over *all* of the batsman's balls, each row's
own target contributes to its own features, and the features are computed once
over the whole dataset — before any train/test split. Both are target leakage.

:class:`BatsmanScoreFeatures` computes the same quantities as an sklearn
transformer: the scores are learned in ``fit`` from training rows only and
applied by lookup in ``transform``, so inside a cross-validation loop a
validation row never contributes to the features it is scored with.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

BATSMAN_COLUMN = "Batsman Name"
SHOT_COLUMN = "Type of Shot"
BALL_COLUMN = "Ball type"
CONTROL_COLUMN = "Control"


@dataclass(frozen=True)
class ScoreSpec:
    """One per-batsman score: which balls count towards it."""

    name: str
    shot_types: tuple[str, ...] = ()
    ball_types: tuple[str, ...] = ()
    require_control: bool = False

    def mask(self, frame: pd.DataFrame) -> pd.Series:
        selected = pd.Series(True, index=frame.index)
        if self.shot_types:
            selected &= frame[SHOT_COLUMN].isin(self.shot_types)
        if self.ball_types:
            selected &= frame[BALL_COLUMN].isin(self.ball_types)
        if self.require_control:
            selected &= frame[CONTROL_COLUMN].astype(float) == 1.0
        return selected


# Shot lists are copied verbatim from the notebook, including its mixed casing
# ('DR' for the off-side list, 'Dr' for the aggressive list) — the symbols are
# case-sensitive in the source workbook and changing them here would silently
# change which balls are counted.
SCORE_SPECS: tuple[ScoreSpec, ...] = (
    ScoreSpec("off_side_score", shot_types=("DR", "C", "RS")),
    ScoreSpec("leg_side_score", shot_types=("P", "ODR", "LG", "F", "S")),
    ScoreSpec("bouncy_track_score", shot_types=("D", "P", "C", "L")),
    ScoreSpec(
        "aggressive_score",
        shot_types=("Dr", "C", "P", "Sw", "RS", "ODR", "F", "R", "LG", "SC"),
    ),
    ScoreSpec("defense_score", shot_types=("D", "L")),
    ScoreSpec(
        "spin_ball_score",
        shot_types=("D", "Dr", "C", "P", "Sw", "RS", "ODR", "F", "R", "LG", "SC"),
        ball_types=("GY", "LB"),
    ),
    ScoreSpec("control_score", require_control=True),
)

SCORE_COLUMNS: tuple[str, ...] = tuple(spec.name for spec in SCORE_SPECS)


def batsman_scores(
    frame: pd.DataFrame,
    target: pd.Series,
    specs: tuple[ScoreSpec, ...] = SCORE_SPECS,
) -> pd.DataFrame:
    """Compute every score per batsman from ``frame`` / ``target``.

    Returns a frame indexed by batsman with one column per spec. A batsman with
    no matching balls for a spec gets ``NaN`` there (the notebook produced the
    same gap by way of an inner merge that dropped the batsman entirely).
    """
    effectiveness = pd.Series(np.asarray(target, dtype=float), index=frame.index)
    balls_faced = frame.groupby(BATSMAN_COLUMN, sort=False)[SHOT_COLUMN].count()

    scores = pd.DataFrame(index=balls_faced.index)
    for spec in specs:
        subset = frame[spec.mask(frame)]
        if subset.empty:
            scores[spec.name] = np.nan
            continue
        grouped = effectiveness.loc[subset.index].groupby(
            subset[BATSMAN_COLUMN], sort=False
        )
        matching = grouped.count()
        share = matching / balls_faced
        mean_effectiveness = grouped.sum() / matching
        scores[spec.name] = share * mean_effectiveness

    scores.index.name = BATSMAN_COLUMN
    return scores


class BatsmanScoreFeatures(TransformerMixin, BaseEstimator):
    """Learn the per-batsman scores on fit, apply them by lookup on transform.

    Batsmen unseen at fit time (the normal case under a grouped split, where
    whole batsmen are held out) get the training-set mean of each score, so the
    feature carries no information about the held-out batsman rather than
    leaking one.
    """

    def __init__(self, specs: tuple[ScoreSpec, ...] = SCORE_SPECS) -> None:
        self.specs = specs

    def fit(self, X: pd.DataFrame, y=None) -> BatsmanScoreFeatures:
        if y is None:
            raise ValueError(
                "BatsmanScoreFeatures needs y at fit time: the scores are "
                "target aggregates and must be computed from training rows only."
            )
        self.score_columns_ = [spec.name for spec in self.specs]
        self.scores_ = batsman_scores(X, y, self.specs)
        # Fallback for unseen batsmen and for batsmen with no matching balls.
        self.fallback_ = self.scores_.mean(axis=0).fillna(0.0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        lookup = self.scores_.reindex(X[BATSMAN_COLUMN])
        lookup.index = X.index
        return lookup.fillna(self.fallback_)[self.score_columns_]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(self.score_columns_, dtype=object)
