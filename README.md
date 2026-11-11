# CricketAnalytics

Predicting the **effectiveness of a cricket shot** from the bowler, ball
type, batsman, and other contextual conditions, using ball-by-ball data
scraped from ESPN Cricinfo and a stack of classical and deep-learning
regressors.

## Project

In this project I have implemented Machine Learning techniques to predict
the effectiveness of a cricket shot played based on bowler, ball type, and
other such conditions. Ball-by-ball data was scraped from ESPN Cricinfo and
a novel approach was developed for assigning an *effectiveness scale* to
each shot based on ball type, outcome, and other contextual conditions.
`scikit-learn`, `XGBoost`, `CatBoost`, and a Keras MLP are used to fit and
compare regressors over the resulting dataset.

The pipeline from the original [`notebooks/SDP.ipynb`](notebooks/SDP.ipynb)
is packaged as `cricket_analytics` (under `src/`) and driven by one script
per stage under `scripts/`:

| Stage       | Script                                    | What it does                                                                 | Outputs |
| ----------- | ----------------------------------------- | ---------------------------------------------------------------------------- | ------- |
| `prepare`   | `scripts/data/prepare_data.py`            | Load the workbook, reshape the control sheet, clean and stack match sheets.  | `data/interim/` |
| `features`  | `scripts/features/build_features.py`      | Train/test split, per-batsman scores, chi-square selection, encoding.        | `data/processed/`, `models/feature_builder.joblib` |
| `baselines` | `scripts/models/train_baselines.py`       | 5-fold CV MAE for SVR, Linear, RF, XGBoost, GBM, Decision Tree, CatBoost.    | `reports/baseline_cv.csv` |
| `tune`      | `scripts/models/tune_models.py`           | Grid-search RF / XGBoost / Decision Tree / CatBoost, evaluate on test split. | `reports/tuning_results.csv`, `models/*.joblib`, feature-importance plots |
| `mlp`       | `scripts/models/train_mlp.py`             | Keras MLP with early stopping (needs the `dl` extra).                        | `reports/mlp_metrics.json`, `models/mlp.keras` |
| `plots`     | `scripts/visualization/make_plots.py`     | Shot-spread per ball type and feature correlation heatmap.                   | `reports/figures/` |

`scripts/run_pipeline.py` runs `prepare features baselines tune plots` in
order (or any stages you pass it). Every script accepts `--config`,
`--workbook` and `--log-level`. All tunables (paths, cleaning aliases, shot
groups, split, model grids, MLP settings) live in
[`configs/default.yaml`](configs/default.yaml); a YAML passed with
`--config` is merged on top of it.

### Differences from the notebook

The package keeps the notebook's cleaning rules and features, with these
corrections:

- Batsman scores, chi-square selection, encoding and scaling are **fit on
  the training split only**, and scaling happens inside each CV fold.
- Hyper-parameters are chosen by cross-validation on the training split, not
  by test-set R²; the test split is scored once per tuned model.
- Batsmen with no shots in a group get a score of 0 instead of being dropped
  by an inner merge.
- Shot / ball codes are validated case-insensitively and normalized to
  upper case, so `Dr` and `DR` are the same shot. Dismissal aliases and
  batsman-name fixes apply to every match sheet, not only `ind_vs_afg`.
- Rows dropped during validation are logged instead of removed silently.

Known remaining caveat: `Wicket` / `Dismissal kind` deterministically imply
`Effectiveness = -1`. Add them to `features.drop_columns` to exclude them.

## Dataset

The pipeline expects a workbook named **`Cricket Data Set.xlsx`** at
`data/raw/` with sheets:

- one *control sheet* (shot-by-ball-type cross-tab)
- one *parameters* sheet (ball-symbol / shot-symbol / dismissal-symbol
  legends)
- one *analysis_parameters* sheet
- one sheet per match, named `<team1>_vs_<team2>` (e.g. `IND_vs_AFG`)
  containing ball-by-ball rows.

The source dataset is **not committed** to this repo. See
[`data/README.md`](data/README.md) for the expected schema and provenance.
`python tools/make_synthetic_workbook.py` writes a synthetic workbook with
the same layout for local runs, tests and CI.

## Setup

Tested on Python 3.10 and 3.11.

```bash
git clone https://github.com/tmanauv/CricketAnalytics.git
cd CricketAnalytics
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip

pip install -e ".[dev]"            # package + tests/lint
pip install -e ".[dev,dl,notebook]"  # + TensorFlow MLP and Jupyter
```

## Run

```bash
# Real data: copy the workbook to data/raw/Cricket Data Set.xlsx
# No data: generate a synthetic workbook instead
python tools/make_synthetic_workbook.py

python scripts/run_pipeline.py                    # prepare -> features -> baselines -> tune -> plots
python scripts/models/train_mlp.py                # optional Keras MLP
python scripts/run_pipeline.py prepare features   # any subset of stages
cricket-analytics --config my_overrides.yaml      # same CLI, installed entry point
```

Equivalent `make` targets: `make pipeline`, `make prepare`, `make tune`,
`make mlp`, `make test`, `make lint`, ...

## Development

```bash
pytest              # unit + end-to-end tests on the synthetic workbook
ruff check . && ruff format --check .
```

CI (`.github/workflows/ci.yml`) runs lint, tests and a full pipeline smoke
run on the synthetic workbook for Python 3.10 and 3.11.

## Repository layout

```
.
├── configs/default.yaml           # all pipeline settings
├── data/                          # raw / interim / processed (gitignored), see data/README.md
├── notebooks/SDP.ipynb            # original notebook (reference)
├── scripts/
│   ├── data/prepare_data.py
│   ├── features/build_features.py
│   ├── models/{train_baselines,tune_models,train_mlp}.py
│   ├── visualization/make_plots.py
│   └── run_pipeline.py
├── src/cricket_analytics/
│   ├── config.py, cli.py, pipeline.py
│   ├── data/            # loader, parameters legend, control sheet, cleaning, synthetic workbook
│   ├── features/        # batsman scores, chi-square selection, FeatureBuilder
│   ├── models/          # model registry, CV evaluation, tuning, Keras MLP
│   └── visualization/   # plots
├── tests/                         # pytest suite
├── tools/make_synthetic_workbook.py
├── Makefile
├── pyproject.toml
└── requirements.txt               # pinned notebook environment
```

## Results

The notebook's only saved outputs are the 5-fold CV MAE scores; its tuned R²
and MLP cells were never executed. Run the pipeline on the real workbook to
populate `reports/` (`baseline_cv.csv`, `tuning_results.csv`,
`mlp_metrics.json`).

## License

This project is released under the [MIT License](LICENSE).
