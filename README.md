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

The end-to-end pipeline today lives in [`SDP.ipynb`](SDP.ipynb):

1. Load multi-sheet Excel data into per-sheet DataFrames.
2. Reshape the *control sheet* (one column block per ball type) into
   per-ball-type shot frames.
3. Stack all `<team>_vs_<team>` match sheets and clean missing values, bad
   categorical codes, and inconsistent column names.
4. Engineer per-batsman aggregate features (off-side, leg-side, bouncy
   track, aggressive shots, defensive shots, spin, control).
5. Filter categorical features by chi-square against the target.
6. Encode + scale and split into train/test.
7. Cross-validate eight regressors (SVR, Linear, RF, XGB, GBM, DT,
   CatBoost), then hyper-parameter-tune the best ones.
8. Train and evaluate a Keras MLP for comparison.

> **Note:** the modeling pipeline has known issues (target leakage in the
> aggregate features, hyper-parameter selection on the test set, pre-split
> `fit_transform`) that are tracked in the repository's improvement plan
> and will be addressed in subsequent PRs. The reported numbers in the
> notebook today should be treated as preliminary.

## Dataset

The notebook expects a workbook named **`Cricket Data Set.xlsx`** in the
repository root, with sheets:

- one *control sheet* (shot-by-ball-type cross-tab)
- one *parameters* sheet (ball-symbol / shot-symbol / dismissal-symbol
  legends)
- one *analysis_parameters* sheet
- one sheet per match, named `<team1>_vs_<team2>` (e.g. `IND_vs_AFG`)
  containing ball-by-ball rows.

The source dataset is **not committed** to this repo. See
[`data/README.md`](data/README.md) for the expected schema and provenance.

## Setup

Tested on Python 3.10 and 3.11.

```bash
# 1. Clone
git clone https://github.com/tmanauv/CricketAnalytics.git
cd CricketAnalytics

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Register the Jupyter kernel used by the notebook
python -m ipykernel install --user --name sdp --display-name "Python (sdp)"
```

Place your `Cricket Data Set.xlsx` in the repo root (it is `.gitignore`d so
it will not be committed accidentally).

## Run

```bash
# Open the notebook in JupyterLab and run all cells
jupyter lab SDP.ipynb
```

Alternatively, execute the notebook headlessly:

```bash
pip install papermill
papermill SDP.ipynb out.ipynb
```

## Repository layout

```
.
├── README.md              # this file
├── LICENSE                # MIT
├── requirements.txt       # pinned Python dependencies
├── .gitignore
├── SDP.ipynb              # end-to-end notebook
└── data/
    └── README.md          # dataset schema & provenance
```

## Results

The notebook currently reports cross-validated MAE for eight baseline
regressors and tuned R² scores for Random Forest, XGBoost, Decision Tree,
CatBoost, and the Keras MLP. Numbers will be re-reported here after the
pipeline-correctness PRs land — see the improvement plan for details.

## License

This project is released under the [MIT License](LICENSE).
