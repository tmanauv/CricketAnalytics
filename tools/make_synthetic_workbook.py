"""Write a synthetic ``Cricket Data Set.xlsx`` for local runs, tests and CI."""

import argparse

from cricket_analytics.config import load_config, resolve_path
from cricket_analytics.data.synthetic import make_synthetic_workbook

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=None, help="Defaults to paths.workbook in the config")
    parser.add_argument("--n-balls", type=int, default=300, help="Deliveries per match sheet")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    output = args.output or resolve_path(load_config()["paths"]["workbook"])
    print(make_synthetic_workbook(output, n_balls=args.n_balls, seed=args.seed))
