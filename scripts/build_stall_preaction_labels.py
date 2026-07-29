from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.stall_preaction_labels import (  # noqa: E402
    build_stall_preaction_labels,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Join a registered pre-action cohort with paired full-pool Oracle labels."
    )
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--oracle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    report = build_stall_preaction_labels(
        arguments.cohort,
        arguments.oracle,
        arguments.output,
        resume=arguments.resume,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
