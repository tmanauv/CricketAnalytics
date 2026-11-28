"""Run the full pipeline: prepare -> features -> baselines -> tune -> plots.

Pass stage names to run a subset, e.g. ``python scripts/run_pipeline.py prepare features mlp``.
"""

from cricket_analytics.cli import main

if __name__ == "__main__":
    main()
