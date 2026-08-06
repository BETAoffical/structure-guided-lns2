from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.stride_robustaction_state_selection import (  # noqa: E402
    build_robustaction_combined_state_selection,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the preregistered result-blind RobustAction combined "
            "source-state selection."
        )
    )
    parser.add_argument(
        "--config",
        default=(
            "configs/"
            "stride_robustaction_structpool_combined_state_selection_design.json"
        ),
    )
    parser.add_argument(
        "--output",
        default="build/stride-robustaction-combined-state-selection-v1",
    )
    arguments = parser.parse_args()
    report = build_robustaction_combined_state_selection(
        config_path=arguments.config,
        output=arguments.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if bool(report["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
