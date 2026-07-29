from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.stall_preaction_cohort import (  # noqa: E402
    prepare_stall_preaction_cohort,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare an outcome-blind plus enriched pre-action v2 stall cohort; "
            "this command does not execute PP trials or train a controller."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    report = prepare_stall_preaction_cohort(
        arguments.config,
        arguments.output,
        resume=arguments.resume,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
