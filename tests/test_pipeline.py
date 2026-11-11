import json

import pandas as pd
import pytest

from cricket_analytics.cli import main
from cricket_analytics.config import load_config
from cricket_analytics.pipeline import (
    build_features,
    make_plots,
    prepare_data,
    train_baselines,
    tune_models,
)


def test_load_config_overlay(tmp_path):
    override = tmp_path / "override.yaml"
    override.write_text("split:\n  test_size: 0.3\n")
    cfg = load_config(override)
    assert cfg["split"]["test_size"] == 0.3
    assert cfg["split"]["random_state"] == 42


def test_stage_requires_previous_stage(cfg):
    with pytest.raises(FileNotFoundError, match="prepare"):
        build_features(cfg)


def test_end_to_end(cfg, tmp_path):
    matches = prepare_data(cfg)
    assert (tmp_path / "interim" / "matches.csv").exists()
    assert len(list((tmp_path / "interim" / "control").glob("*.csv"))) == 8
    assert (
        json.loads((tmp_path / "interim" / "legend.json").read_text())["ball_types"]["GY"]
        == "Googly"
    )

    train, test = build_features(cfg)
    assert len(train) + len(test) == len(matches)
    assert list(train.columns) == list(test.columns)
    assert (tmp_path / "models" / "feature_builder.joblib").exists()

    baselines = train_baselines(cfg)
    assert set(baselines["model"]) == {"linear", "decision_tree"}

    results = tune_models(cfg)
    assert set(results["model"]) == {"decision_tree", "random_forest"}
    assert (tmp_path / "models" / "random_forest.joblib").exists()
    saved = pd.read_csv(tmp_path / "reports" / "tuning_results.csv")
    assert json.loads(saved.loc[0, "best_params"])

    figures = make_plots(cfg)
    assert any(p.name == "correlation.png" for p in figures)


def test_mlp_stage(cfg, tmp_path):
    pytest.importorskip("tensorflow")
    from cricket_analytics.pipeline import train_mlp

    prepare_data(cfg)
    build_features(cfg)
    metrics = train_mlp(cfg)
    assert set(metrics) == {"r2", "mae", "mse", "rmse"}
    assert (tmp_path / "models" / "mlp.keras").exists()


def test_cli_rejects_unknown_stage():
    with pytest.raises(SystemExit):
        main(["bogus"])
