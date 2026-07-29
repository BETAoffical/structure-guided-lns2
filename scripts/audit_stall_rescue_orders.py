from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.stall_rescue_order_audit import audit_stall_rescue_orders  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit deterministic v2 rank-order rescue rules on exact Oracles."
    )
    parser.add_argument("--oracle-batch", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    report = audit_stall_rescue_orders(
        arguments.oracle_batch,
        arguments.output,
        resume=arguments.resume,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
