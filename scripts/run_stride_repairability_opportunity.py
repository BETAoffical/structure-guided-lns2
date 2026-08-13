from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.stride_repairability_opportunity import (  # noqa: E402
    analyze_repairability_opportunity,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit quality-constrained PP-repairability opportunity"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    report = analyze_repairability_opportunity(arguments.config, arguments.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
