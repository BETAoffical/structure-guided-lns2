#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stall_trigger_policy_audit import audit_stall_trigger_policies


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit same-neighborhood stall triggers without changing v2 actions."
    )
    parser.add_argument("--extension", required=True)
    parser.add_argument("--oracle-batch", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = audit_stall_trigger_policies(
        args.extension,
        args.oracle_batch,
        args.output,
        resume=args.resume,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
