"""Leak-free modelling of shot effectiveness (see ``ROADMAP.md``, Phase 2)."""

from .aggregates import (
    SCORE_COLUMNS,
    SCORE_SPECS,
    BatsmanScoreFeatures,
    ScoreSpec,
    batsman_scores,
)
from .ingest import matches_frame_from_workbook
from .pipeline import (
    DETERMINISTIC_LEAK_COLUMNS,
    BallTypeMeanRegressor,
    Metrics,
    baseline_estimators,
    build_pipeline,
    build_preprocessor,
    cross_validate_model,
    evaluate,
    grouped_train_test_split,
    split_features_target,
    tune,
)
from .study import default_estimators, run_study

__all__ = [
    "DETERMINISTIC_LEAK_COLUMNS",
    "SCORE_COLUMNS",
    "SCORE_SPECS",
    "BallTypeMeanRegressor",
    "BatsmanScoreFeatures",
    "Metrics",
    "ScoreSpec",
    "baseline_estimators",
    "batsman_scores",
    "build_pipeline",
    "build_preprocessor",
    "cross_validate_model",
    "default_estimators",
    "evaluate",
    "grouped_train_test_split",
    "matches_frame_from_workbook",
    "run_study",
    "split_features_target",
    "tune",
]
