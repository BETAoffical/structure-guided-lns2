from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.stride_robuststep import (  # noqa: E402
    run_robuststep_design,
    run_robuststep_seed_depth,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the consumed STRIDE robust-step label design analysis."
    )
    parser.add_argument(
        "--analysis", choices=("design", "seed-depth"), default="design"
    )
    parser.add_argument(
        "--config", default="configs/stride_robuststep_design.json"
    )
    parser.add_argument(
        "--output", default="build/stride-robuststep-design-v1"
    )
    arguments = parser.parse_args()
    runner = (
        run_robuststep_seed_depth
        if arguments.analysis == "seed-depth"
        else run_robuststep_design
    )
    report = runner(arguments.config, arguments.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
