import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import extract_results as extractor
from extract_results import extract_results, main


def _code(source, outputs):
    return {"cell_type": "code", "source": source, "outputs": outputs}


def _stream(text):
    return {"output_type": "stream", "name": "stdout", "text": text}


def _result(value):
    return {"output_type": "execute_result", "data": {"text/plain": value}}


def test_parses_cv_mae_and_dedupes_repeated_models():
    nb = {
        "cells": [
            _code(
                "print(...)",
                [
                    _stream("0:\tlearn: 0.36\n"),
                    _stream("SVM mean absolute error:  0.2\n"),
                    _stream("Linear Regression mean absolute error:  9.0\n"),
                    _stream("Decision Tree Regressor mean absolute error:  0.03\n"),
                    _stream("Linear Regression mean absolute error:  8.0\n"),
                ],
            )
        ]
    }
    models = extract_results(nb)["cv"]["models"]
    assert [m["estimator"] for m in models] == ["DecisionTreeRegressor", "SVR", "LinearRegression"]
    assert models[-1]["cv_mae"] == 9.0


def test_parses_tuned_r2_and_mlp_mse():
    nb = {
        "cells": [
            _code(
                "reg = cb.CatBoostRegressor(random_state=1)\nreg.fit(X_train, y_train)\nr2_score(y_test, y_pred)",
                [_result("0.91")],
            ),
            _code(
                "reg = RandomForestRegressor(max_depth=8)\nr2_score(y_test, y_pred)\n",
                [_result("0.95")],
            ),
            _code("print(...)", [_stream("Standardized: -0.02 (0.01) MSE\n")]),
        ]
    }
    results = extract_results(nb)
    assert [m["estimator"] for m in results["tuned"]["models"]] == [
        "RandomForestRegressor",
        "CatBoostRegressor",
    ]
    assert results["mlp"]["variants"] == [{"variant": "Standardized", "mse": 0.02, "std": 0.01, "cell": 2}]


def test_committed_results_match_notebook(tmp_path):
    out = tmp_path / "results.json"
    main(["--output", str(out)])
    committed = json.loads((REPO_ROOT / "results" / "model_results.json").read_text())
    assert json.loads(out.read_text()) == committed


@pytest.mark.parametrize("layout", ["root", "notebooks", "both", "missing", "explicit", "explicit_missing"])
def test_notebook_path_selection(tmp_path, monkeypatch, layout):
    monkeypatch.setattr(extractor, "REPO_ROOT", tmp_path)
    preferred = tmp_path / "notebooks" / "SDP.ipynb"
    monkeypatch.setattr(extractor, "DEFAULT_NOTEBOOK", preferred)
    preferred.parent.mkdir()
    root = tmp_path / "SDP.ipynb"
    if layout != "missing":
        root.write_text(json.dumps({"cells": []}))
    if layout in {"notebooks", "both", "explicit", "explicit_missing"}:
        preferred.write_text(json.dumps({"cells": [_code("", [_stream("SVM mean absolute error: 0.2")])]}))
        if layout == "notebooks":
            root.unlink()
    out = tmp_path / "results.json"
    args = ["--output", str(out)]
    if layout.startswith("explicit"):
        args += ["--notebook", str(root if layout == "explicit" else tmp_path / "missing.ipynb")]
    if layout in {"missing", "explicit_missing"}:
        with pytest.raises(FileNotFoundError):
            main(args)
        assert not out.exists()
    else:
        main(args)
        models = json.loads(out.read_text())["cv"]["models"]
        assert len(models) == (1 if layout in {"notebooks", "both"} else 0)


def test_source_records_extracted_notebook(tmp_path):
    notebook = tmp_path / "out.ipynb"
    notebook.write_text(
        json.dumps({"cells": [_code("print(score)", [_stream("SVM mean absolute error: 0.2")])]})
    )
    out = tmp_path / "results.json"
    main(["--notebook", str(notebook), "--output", str(out)])
    results = json.loads(out.read_text())
    assert results["source"] == str(notebook.resolve())
    assert [m["estimator"] for m in results["cv"]["models"]] == ["SVR"]
