"""Command-line entry points shared by ``scripts/`` and the ``cricket-analytics`` command."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable, Sequence
from typing import Any

from cricket_analytics.config import load_config


def _base_parser(description: str | None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=None, help="YAML overriding configs/default.yaml")
    parser.add_argument("--workbook", default=None, help="Override paths.workbook")
    parser.add_argument("--log-level", default="INFO")
    return parser


def _setup(args: argparse.Namespace) -> dict[str, Any]:
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    if args.workbook:
        cfg["paths"]["workbook"] = args.workbook
    return cfg


def run_stage(
    stage: Callable[[dict[str, Any]], Any],
    description: str | None = None,
    argv: Sequence[str] | None = None,
) -> None:
    """Run a single pipeline stage with the common ``--config`` options."""
    stage(_setup(_base_parser(description).parse_args(argv)))


def main(argv: Sequence[str] | None = None) -> None:
    from cricket_analytics.pipeline import DEFAULT_RUN, STAGES, run_pipeline

    parser = _base_parser("Run CricketAnalytics pipeline stages.")
    parser.add_argument(
        "stages",
        nargs="*",
        default=list(DEFAULT_RUN),
        metavar="STAGE",
        help=f"Stages to run in order, from {sorted(STAGES)} (default: {' '.join(DEFAULT_RUN)})",
    )
    args = parser.parse_args(argv)
    unknown = [s for s in args.stages if s not in STAGES]
    if unknown:
        parser.error(f"unknown stage(s) {unknown}; choose from {sorted(STAGES)}")
    run_pipeline(_setup(args), tuple(args.stages))


if __name__ == "__main__":
    main()
