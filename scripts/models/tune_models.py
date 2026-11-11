"""Grid-search RF / XGBoost / Decision Tree / CatBoost and evaluate on the test split."""

from cricket_analytics.cli import run_stage
from cricket_analytics.pipeline import tune_models

if __name__ == "__main__":
    run_stage(tune_models, __doc__)
