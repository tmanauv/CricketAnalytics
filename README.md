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
`scikit-learn`, `XGBoost`, `CatBoost`, `LightGBM`, and a Keras MLP are
used to fit and compare regressors over the resulting dataset.

### Pipeline

1. **Scrape** — ball-by-ball data from ESPN Cricinfo (`cricket-analytics scrape`).
2. **Prepare** — clean, validate, and engineer leak-free features (`cricket-analytics prepare`).
3. **Train** — nested cross-validation across classical + deep-learning models (`cricket-analytics train`).
4. **Evaluate** — compare models on held-out test set (`cricket-analytics evaluate`).

## Dataset

The scraper targets ESPN Cricinfo ball-by-ball commentary to build the
dataset automatically. The legacy `Cricket Data Set.xlsx` format is also
supported via the backward-compatible loader. See
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

# 3. Install the package in editable mode
pip install --upgrade pip
pip install -e ".[dev,notebook]"
```

## Run

```bash
# CLI commands
cricket-analytics scrape --config configs/scraper.yaml
cricket-analytics prepare
cricket-analytics train
cricket-analytics evaluate <model_path>

# Or explore interactively
jupyter lab notebooks/SDP.ipynb
```

## Repository Layout

```
.
├── pyproject.toml                 # project metadata & dependencies
├── README.md
├── LICENSE                        # MIT
├── .gitignore
├── configs/
│   ├── scraper.yaml               # scraper targets & settings
│   └── train.yaml                 # training hyperparameters & config
├── src/
│   └── cricket_analytics/
│       ├── __init__.py
│       ├── cli.py                 # Typer CLI entry point
│       ├── scraper/               # ESPN Cricinfo scraper
│       ├── data/                  # loading, cleaning, feature engineering
│       ├── models/                # classical ML + deep learning pipelines
│       └── visualisation/         # plots & charts
├── notebooks/
│   └── SDP.ipynb                  # legacy exploration notebook
├── tests/                         # pytest suite
├── data/
│   ├── README.md                  # dataset schema & provenance
│   ├── raw/                       # scraped data (gitignored)
│   ├── interim/                   # cleaned intermediate data (gitignored)
│   └── processed/                 # model-ready features (gitignored)
└── .github/
    └── workflows/
        └── ci.yml                 # lint + test on push/PR
```

## Results

Model comparison results will be reported here once the full pipeline is
operational. The legacy notebook (`notebooks/SDP.ipynb`) contains
preliminary results with known methodological issues (see the overhaul
plan for details).

## License

This project is released under the [MIT License](LICENSE).
