"""Per-batsman aggregate scores (off side, leg side, bouncy track, ...).

For a shot group, a batsman's score is::

    (share of balls faced played with a group shot) * (mean effectiveness of those shots)

which is the formula used by the notebook's ``off_side_score`` ... ``control_score``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd

from cricket_analytics.data import columns as C


def shot_group_score(
    df: pd.DataFrame,
    name: str,
    shots: Iterable[str] | None = None,
    ball_types: Iterable[str] | None = None,
    controlled_only: bool = False,
) -> pd.DataFrame:
    """Return ``[Batsman Name, pct_<name>, avg_effectiveness_<name>, <name>_score]``."""
    mask = pd.Series(True, index=df.index)
    if shots is not None:
        mask &= df[C.SHOT].isin([s.upper() for s in shots])
    if ball_types is not None:
        mask &= df[C.BALL_TYPE].isin([b.upper() for b in ball_types])
    if controlled_only:
        mask &= df[C.CONTROL] == 1.0

    group = (
        df[mask]
        .groupby(C.BATSMAN)
        .agg(shots_count=(C.SHOT, "count"), sum_effectiveness=(C.TARGET, "sum"))
    )
    faced = df.groupby(C.BATSMAN).agg(balls_faced=(C.SHOT, "count"))
    scores = faced.join(group, how="left").fillna(0.0)

    pct = scores["shots_count"] / scores["balls_faced"] * 100
    avg = (scores["sum_effectiveness"] / scores["shots_count"]).where(
        scores["shots_count"] > 0, 0.0
    )
    out = pd.DataFrame(
        {
            f"pct_{name}_shots": pct,
            f"avg_effectiveness_{name}_shots": avg,
            f"{name}_score": pct / 100 * avg,
        }
    )
    return out.reset_index().sort_values(f"{name}_score", ascending=False, ignore_index=True)


def compute_batsman_scores(
    df: pd.DataFrame, score_groups: Mapping[str, Mapping[str, Any]]
) -> pd.DataFrame:
    """One row per batsman with a ``<group>_score`` column per configured group."""
    result = pd.DataFrame({C.BATSMAN: sorted(df[C.BATSMAN].unique())})
    for name, spec in score_groups.items():
        scores = shot_group_score(
            df,
            name,
            shots=spec.get("shots"),
            ball_types=spec.get("ball_types"),
            controlled_only=bool(spec.get("controlled_only", False)),
        )
        result = result.merge(scores[[C.BATSMAN, f"{name}_score"]], on=C.BATSMAN, how="left")
    return result.fillna(0.0)
