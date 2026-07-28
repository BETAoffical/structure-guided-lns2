from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.stall_oracle import audit_stall_probe  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Classify a full-pool same-state v2 stall probe."
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-trials", type=int, default=4)
    parser.add_argument("--stable-fraction", type=float, default=0.75)
    arguments = parser.parse_args()
    report = audit_stall_probe(
        arguments.source,
        arguments.output,
        minimum_trials=arguments.minimum_trials,
        stable_fraction=arguments.stable_fraction,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
