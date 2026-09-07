"""Explicitly run initial PP only; never execute a repair or timed comparison."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lns2_selector.evaluation.path_quality_preflight import contained
from lns2_selector.evaluation.pressure_reset import run_resets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/path_quality_pressure_reset_v2.json")
    parser.add_argument("--run-resets", action="store_true", required=True)
    args = parser.parse_args()
    run_resets(ROOT, contained(ROOT, args.config))
