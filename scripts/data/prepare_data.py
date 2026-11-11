"""Load the Excel workbook, clean every match sheet and write data/interim/."""

from cricket_analytics.cli import run_stage
from cricket_analytics.pipeline import prepare_data

if __name__ == "__main__":
    run_stage(prepare_data, __doc__)
