"""Shared utilities extracted from ``SDP.ipynb``.

The notebook repeated the same handful of patterns many times over (one
per-batsman scoring function per shot group, one cross-validation block per
regressor, one feature-importance plot per model). Those patterns live here
so the notebook only carries the parts that differ.
"""

from cricket_analytics.dataframes import (
    dataframe_names,
    filter_names,
    load_workbook,
    print_unique_values,
    symbol_list,
    unique_values,
)
from cricket_analytics.features import (
    SCORE_SPECS,
    SPECS_BY_SCORE,
    ScoreSpec,
    batsman_group_score,
    merge_scores,
    score_frames,
)
from cricket_analytics.modeling import (
    METRIC_COLUMNS,
    cross_validate_regressors,
    evaluate_param_grid,
    fit_and_evaluate,
    plot_feature_importances,
    report_param_scores,
    results_frame,
)

__all__ = [
    "METRIC_COLUMNS",
    "SCORE_SPECS",
    "SPECS_BY_SCORE",
    "ScoreSpec",
    "batsman_group_score",
    "cross_validate_regressors",
    "dataframe_names",
    "evaluate_param_grid",
    "filter_names",
    "fit_and_evaluate",
    "load_workbook",
    "merge_scores",
    "plot_feature_importances",
    "print_unique_values",
    "report_param_scores",
    "results_frame",
    "score_frames",
    "symbol_list",
    "unique_values",
]
