from pathlib import Path

import pytest

from cricket_analytics.config import load_config
from cricket_analytics.data.loader import load_workbook
from cricket_analytics.data.parameters import parse_parameters
from cricket_analytics.data.synthetic import make_synthetic_workbook


@pytest.fixture(scope="session")
def workbook_path(tmp_path_factory) -> Path:
    return make_synthetic_workbook(tmp_path_factory.mktemp("data") / "Cricket Data Set.xlsx")


@pytest.fixture(scope="session")
def sheets(workbook_path):
    return load_workbook(workbook_path)


@pytest.fixture(scope="session")
def legend(sheets):
    return parse_parameters(sheets["parameters"])


@pytest.fixture
def cfg(tmp_path, workbook_path):
    config = load_config()
    config["paths"] = {
        "workbook": str(workbook_path),
        "interim_dir": str(tmp_path / "interim"),
        "processed_dir": str(tmp_path / "processed"),
        "models_dir": str(tmp_path / "models"),
        "reports_dir": str(tmp_path / "reports"),
    }
    config["baselines"]["models"] = ["linear", "decision_tree"]
    config["baselines"]["cv_folds"] = 3
    config["tuning"]["cv_folds"] = 3
    config["tuning"]["grids"] = {
        "decision_tree": {"max_depth": [3, 5]},
        "random_forest": {"n_estimators": [5], "max_depth": [3]},
    }
    config["mlp"].update({"epochs": 3, "patience": 2, "hidden_units": [8]})
    return config
