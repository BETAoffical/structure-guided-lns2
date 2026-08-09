from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stride_safeslot_gate import train_safeslot_gate


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train and nested-map evaluate the offline SafeSlot gate"
    )
    parser.add_argument(
        "--config", default="configs/stride_safeslot_gate_v1.json"
    )
    parser.add_argument(
        "--output", default="build/stride-safeslot-gate-training-v1"
    )
    args = parser.parse_args()
    report = train_safeslot_gate(config_path=args.config, output=args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["offline_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
