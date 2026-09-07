"""Diagnose frozen task-generation failures without invoking a solver."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lns2_selector.evaluation.path_quality_preflight import contained
from lns2_selector.evaluation.pressure_capacity import diagnose


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/path_quality_pressure_pilot_v1.json")
    parser.add_argument("--output", default="build/path-quality-pressure-capacity-v1")
    args = parser.parse_args()
    diagnose(ROOT, contained(ROOT, args.config), contained(ROOT, args.output))
