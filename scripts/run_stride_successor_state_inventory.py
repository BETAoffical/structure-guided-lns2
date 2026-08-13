from __future__ import annotations

import argparse
import json

from experiments.stride_successor_state_inventory import (
    analyze_successor_state_inventory,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inventory exact states after frozen PreTail forced actions."
    )
    parser.add_argument(
        "--config",
        default="configs/stride_successor_state_inventory_v1_registration.json",
    )
    parser.add_argument(
        "--output", default="build/stride-successor-state-inventory-v1"
    )
    arguments = parser.parse_args()
    report = analyze_successor_state_inventory(arguments.config, arguments.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["integrity_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
