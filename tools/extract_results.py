"""Extract the final model results recorded in ``SDP.ipynb`` into JSON.

The dashboard reads ``results/model_results.json`` rather than re-running the
notebook, because the source workbook is not committed. Re-run the notebook,
then re-run this script to refresh the dashboard:

    python tools/extract_results.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NOTEBOOK = REPO_ROOT / "SDP.ipynb"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "model_results.json"

CV_MAE_LINE = re.compile(r"^(?P<label>.+?) mean absolute error:\s*(?P<value>\S+)\s*$")
MLP_MSE_LINE = re.compile(
    r"^(?P<label>Baseline|Standardized):\s*(?P<mean>-?[\d.]+)\s*\((?P<std>[\d.]+)\)\s*MSE\s*$"
)
TUNED_MODEL = re.compile(r"^reg\s*=\s*(?:\w+\.)?(?P<estimator>\w+)\(", re.MULTILINE)

ESTIMATOR_BY_LABEL = {
    "SVM": "SVR",
    "Linear Regression": "LinearRegression",
    "Random Forest": "RandomForestRegressor",
    "XGBoost": "XGBRegressor",
    "GradientBoosting Regressor": "GradientBoostingRegressor",
    "Decision Tree Regressor": "DecisionTreeRegressor",
    "Cat Boost Regressor": "CatBoostRegressor",
}


def _source(cell: dict) -> str:
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else src


def _output_text(cell: dict) -> list[str]:
    texts = []
    for output in cell.get("outputs", []):
        if "text" in output:
            text = output["text"]
        else:
            text = output.get("data", {}).get("text/plain", "")
        texts.append("".join(text) if isinstance(text, list) else text)
    return texts


def extract_results(notebook: dict) -> dict:
    cv_mae: dict[str, dict] = {}
    tuned_r2: dict[str, dict] = {}
    mlp_cv_mse: dict[str, dict] = {}

    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = _source(cell)
        outputs = _output_text(cell)

        for line in "\n".join(outputs).splitlines():
            line = line.strip()
            if match := CV_MAE_LINE.match(line):
                label = match["label"].strip()
                cv_mae.setdefault(
                    label,
                    {
                        "model": label,
                        "estimator": ESTIMATOR_BY_LABEL.get(label, label),
                        "cv_mae": float(match["value"]),
                        "cell": index,
                    },
                )
            elif match := MLP_MSE_LINE.match(line):
                mlp_cv_mse[match["label"]] = {
                    "variant": match["label"],
                    "mse": abs(float(match["mean"])),
                    "std": float(match["std"]),
                    "cell": index,
                }

        estimator = TUNED_MODEL.search(source)
        if estimator and source.rstrip().endswith("r2_score(y_test, y_pred)"):
            for output in cell.get("outputs", []):
                if output.get("output_type") != "execute_result":
                    continue
                text = "".join(output.get("data", {}).get("text/plain", ""))
                try:
                    r2 = float(text.strip())
                except ValueError:
                    continue
                tuned_r2[estimator["estimator"]] = {
                    "estimator": estimator["estimator"],
                    "r2": r2,
                    "cell": index,
                }

    return {
        "source": "SDP.ipynb",
        "cv": {
            "metric": "MAE",
            "folds": 5,
            "description": "5-fold cross-validated mean absolute error on the training split.",
            "models": sorted(cv_mae.values(), key=lambda m: m["cv_mae"]),
        },
        "tuned": {
            "metric": "R2",
            "description": "Held-out R² of the hyper-parameter-tuned models.",
            "models": sorted(tuned_r2.values(), key=lambda m: -m["r2"]),
        },
        "mlp": {
            "metric": "MSE",
            "description": "10-fold cross-validated MSE of the Keras MLP.",
            "variants": list(mlp_cv_mse.values()),
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    notebook = json.loads(args.notebook.read_text(encoding="utf-8"))
    results = extract_results(notebook)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
