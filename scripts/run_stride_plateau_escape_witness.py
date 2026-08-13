from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.stride_plateau_escape_witness import (  # noqa: E402
    analyze_plateau_escape_witnesses,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze frozen first-repeat plateaus and their escape witnesses"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=16)
    arguments = parser.parse_args()
    report = analyze_plateau_escape_witnesses(
        arguments.config,
        arguments.output,
        workers=arguments.workers,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
