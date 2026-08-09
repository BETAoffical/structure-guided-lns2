from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.stride_safeslot_label_readiness import (  # noqa: E402
    analyze_safeslot_label_readiness,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit SafeSlot label readiness")
    parser.add_argument(
        "--config", default="configs/stride_safeslot_label_readiness_v1.json"
    )
    parser.add_argument(
        "--output", default="build/stride-safeslot-label-readiness-v1"
    )
    args = parser.parse_args()
    report = analyze_safeslot_label_readiness(args.config, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["integrity_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
