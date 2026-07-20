from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from research.studies.sequential.repair_order_probe import (  # noqa: E402
    CollectionLockError,
    run_repair_order_probe,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe PP repair-order effects at policy-visited InitLNS states."
    )
    parser.add_argument("--config", default="research/configs/sequential/repair_order_probe.json")
    parser.add_argument("--output", default="build/initlns-repair-order-probe-v1")
    parser.add_argument(
        "--phase",
        choices=("diagnose", "dry-run", "collect", "analyze", "all"),
        default="all",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-states", type=int)
    arguments = parser.parse_args()
    try:
        report = run_repair_order_probe(
            arguments.config,
            arguments.output,
            phase=arguments.phase,
            resume=arguments.resume,
            smoke_states=arguments.smoke_states,
        )
    except CollectionLockError as error:
        print(json.dumps({"status": "locked", "error": str(error)}), file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
