"""Split deliveries into train/test and build leak-free model features."""

from cricket_analytics.cli import run_stage
from cricket_analytics.pipeline import build_features

if __name__ == "__main__":
    run_stage(build_features, __doc__)
