"""Cross-validate the baseline regressors on the training split."""

from cricket_analytics.cli import run_stage
from cricket_analytics.pipeline import train_baselines

if __name__ == "__main__":
    run_stage(train_baselines, __doc__)
