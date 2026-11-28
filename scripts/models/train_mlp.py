"""Train the Keras MLP regressor (requires the dl extra)."""

from cricket_analytics.cli import run_stage
from cricket_analytics.pipeline import train_mlp

if __name__ == "__main__":
    run_stage(train_mlp, __doc__)
