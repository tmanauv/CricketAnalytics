"""Figures from the notebook, written to disk instead of displayed inline."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from cricket_analytics.data import columns as C  # noqa: E402


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_shot_spread(
    matches: pd.DataFrame, ball_type_names: Mapping[str, str], out_dir: Path
) -> list[Path]:
    """One count plot of shot types per ball type."""
    sns.set_style("whitegrid")
    paths = []
    for ball_type in sorted(matches[C.BALL_TYPE].unique()):
        subset = matches[matches[C.BALL_TYPE] == ball_type]
        fig, ax = plt.subplots(figsize=(15, 5))
        sns.countplot(x=C.SHOT, data=subset, order=sorted(subset[C.SHOT].unique()), ax=ax)
        ax.set_title(f"Shot Type Spread vs {ball_type_names.get(ball_type, ball_type)} ball")
        paths.append(_save(fig, out_dir / f"shot_spread_{ball_type.lower()}.png"))
    return paths


def plot_correlation_heatmap(df: pd.DataFrame, path: Path) -> Path:
    corr = df.select_dtypes("number").corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, cmap="Blues", annot=True, fmt=".2f", annot_kws={"size": 7}, ax=ax)
    return _save(fig, path)


def plot_feature_importances(importances: pd.Series, title: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, max(4, 0.35 * len(importances))))
    importances.sort_values().plot(kind="barh", ax=ax, title=title)
    return _save(fig, path)
