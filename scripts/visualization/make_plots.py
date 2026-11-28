"""Write shot-spread and correlation figures to reports/figures/."""

from cricket_analytics.cli import run_stage
from cricket_analytics.pipeline import make_plots

if __name__ == "__main__":
    run_stage(make_plots, __doc__)
