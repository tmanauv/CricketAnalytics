"""Pipeline stages. Each stage reads the previous stage's artifacts from disk.

prepare  -> data/interim/{matches.csv, legend.json, control/*.csv}
features -> data/processed/{train,test}.csv, models/feature_builder.joblib
baselines-> reports/baseline_cv.csv
tune     -> reports/tuning_results.csv, models/<model>.joblib, reports/figures/
mlp      -> reports/mlp_metrics.json, models/mlp.keras
plots    -> reports/figures/
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from cricket_analytics.config import resolve_path
from cricket_analytics.data.cleaning import build_matches_df
from cricket_analytics.data.control_sheet import split_by_ball_type
from cricket_analytics.data.loader import (
    CONTROL_SHEET,
    PARAMETERS_SHEET,
    load_workbook,
    match_sheets,
    require_sheets,
)
from cricket_analytics.data.parameters import parse_parameters
from cricket_analytics.features.builder import FeatureBuilder

logger = logging.getLogger(__name__)

Config = dict[str, Any]


def _path(cfg: Config, key: str) -> Path:
    return resolve_path(cfg["paths"][key])


def _require(path: Path, stage: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run the `{stage}` stage first.")
    return path


def _load_split(cfg: Config) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    processed = _path(cfg, "processed_dir")
    target = cfg["features"]["target"]
    train = pd.read_csv(_require(processed / "train.csv", "features"))
    test = pd.read_csv(_require(processed / "test.csv", "features"))
    return train.drop(columns=target), train[target], test.drop(columns=target), test[target]


def prepare_data(cfg: Config) -> pd.DataFrame:
    """Load the workbook, clean every match sheet and write the interim tables."""
    sheets = load_workbook(_path(cfg, "workbook"))
    require_sheets(sheets, PARAMETERS_SHEET)
    legend = parse_parameters(sheets[PARAMETERS_SHEET])
    interim = _path(cfg, "interim_dir")
    interim.mkdir(parents=True, exist_ok=True)

    if CONTROL_SHEET in sheets:
        control_dir = interim / "control"
        control_dir.mkdir(exist_ok=True)
        for ball_type, frame in split_by_ball_type(sheets[CONTROL_SHEET], legend).items():
            frame.to_csv(
                control_dir / f"{'_'.join(ball_type.lower().split())}_shots.csv", index=False
            )

    matches = build_matches_df(match_sheets(sheets), legend, cfg["cleaning"])
    matches.to_csv(interim / "matches.csv", index=False)
    (interim / "legend.json").write_text(
        json.dumps(
            {
                "ball_types": legend.ball_types,
                "shot_types": legend.shot_types,
                "dismissals": legend.dismissals,
            },
            indent=2,
        )
    )
    logger.info("prepare: wrote %d cleaned deliveries to %s", len(matches), interim)
    return matches


def build_features(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split deliveries into train/test and build model inputs fit on train only."""
    matches = pd.read_csv(_require(_path(cfg, "interim_dir") / "matches.csv", "prepare"))
    train_rows, test_rows = train_test_split(
        matches, test_size=cfg["split"]["test_size"], random_state=cfg["split"]["random_state"]
    )
    builder = FeatureBuilder(cfg["features"]).fit(train_rows)
    target = builder.target

    frames = []
    for rows in (train_rows, test_rows):
        X, y = builder.transform(rows)
        frames.append(X.assign(**{target: y.to_numpy()}))
    train, test = frames

    processed = _path(cfg, "processed_dir")
    processed.mkdir(parents=True, exist_ok=True)
    train.to_csv(processed / "train.csv", index=False)
    test.to_csv(processed / "test.csv", index=False)
    builder.batsman_scores_.to_csv(processed / "batsman_scores.csv", index=False)

    models_dir = _path(cfg, "models_dir")
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(builder, models_dir / "feature_builder.joblib")
    logger.info(
        "features: %d train / %d test rows, features=%s",
        len(train),
        len(test),
        builder.feature_names_,
    )
    return train, test


def train_baselines(cfg: Config) -> pd.DataFrame:
    """Cross-validate the baseline regressors on the training split."""
    from cricket_analytics.models.evaluate import cross_validate_models

    X_train, y_train, _, _ = _load_split(cfg)
    base = cfg["baselines"]
    results = cross_validate_models(
        X_train,
        y_train,
        base["models"],
        base["cv_folds"],
        base["scoring"],
        random_state=cfg["split"]["random_state"],
    )
    reports = _path(cfg, "reports_dir")
    reports.mkdir(parents=True, exist_ok=True)
    results.to_csv(reports / "baseline_cv.csv", index=False)
    logger.info("baselines:\n%s", results.to_string(index=False))
    return results


def tune_models(cfg: Config) -> pd.DataFrame:
    """Grid-search the configured models, save them and their feature importances."""
    from cricket_analytics.models.tuning import feature_importances
    from cricket_analytics.models.tuning import tune_models as _tune
    from cricket_analytics.visualization.plots import plot_feature_importances

    X_train, y_train, X_test, y_test = _load_split(cfg)
    tuning = cfg["tuning"]
    results, estimators = _tune(
        tuning["grids"],
        X_train,
        y_train,
        X_test,
        y_test,
        tuning["cv_folds"],
        tuning["scoring"],
        tuning["random_state"],
    )

    reports, models_dir = _path(cfg, "reports_dir"), _path(cfg, "models_dir")
    reports.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    results.assign(best_params=results["best_params"].map(json.dumps)).to_csv(
        reports / "tuning_results.csv", index=False
    )
    for name, estimator in estimators.items():
        joblib.dump(estimator, models_dir / f"{name}.joblib")
        importances = feature_importances(estimator, list(X_train.columns))
        if importances is not None:
            plot_feature_importances(
                importances,
                f"Top features: {name}",
                reports / "figures" / f"feature_importances_{name}.png",
            )
    logger.info("tune:\n%s", results.drop(columns="best_params").to_string(index=False))
    return results


def train_mlp(cfg: Config) -> dict[str, float]:
    """Train the Keras MLP (needs the ``dl`` extra)."""
    from cricket_analytics.models.neural import train_mlp as _train_mlp

    X_train, y_train, X_test, y_test = _load_split(cfg)
    model, scaler, metrics, history = _train_mlp(X_train, y_train, X_test, y_test, cfg["mlp"])

    reports, models_dir = _path(cfg, "reports_dir"), _path(cfg, "models_dir")
    reports.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    model.save(models_dir / "mlp.keras")
    joblib.dump(scaler, models_dir / "mlp_scaler.joblib")
    (reports / "mlp_metrics.json").write_text(
        json.dumps({"test": metrics, "epochs_trained": len(history["loss"])}, indent=2)
    )
    logger.info("mlp: %s", metrics)
    return metrics


def make_plots(cfg: Config) -> list[Path]:
    """Shot-spread plots per ball type and the feature correlation heatmap."""
    from cricket_analytics.visualization.plots import plot_correlation_heatmap, plot_shot_spread

    interim = _path(cfg, "interim_dir")
    matches = pd.read_csv(_require(interim / "matches.csv", "prepare"))
    legend = json.loads((interim / "legend.json").read_text())
    figures = _path(cfg, "reports_dir") / "figures"

    paths = plot_shot_spread(matches, legend["ball_types"], figures)
    train_csv = _path(cfg, "processed_dir") / "train.csv"
    if train_csv.exists():
        paths.append(plot_correlation_heatmap(pd.read_csv(train_csv), figures / "correlation.png"))
    logger.info("plots: wrote %d figures to %s", len(paths), figures)
    return paths


STAGES = {
    "prepare": prepare_data,
    "features": build_features,
    "baselines": train_baselines,
    "tune": tune_models,
    "mlp": train_mlp,
    "plots": make_plots,
}
DEFAULT_RUN = ("prepare", "features", "baselines", "tune", "plots")


def run_pipeline(cfg: Config, stages: tuple[str, ...] = DEFAULT_RUN) -> None:
    for stage in stages:
        logger.info("=== stage: %s ===", stage)
        STAGES[stage](cfg)


__all__ = ["STAGES", "run_pipeline"]
