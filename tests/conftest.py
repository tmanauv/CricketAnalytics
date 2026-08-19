import sys
from pathlib import Path

# `tools/` is a scripts directory, not an installed package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
