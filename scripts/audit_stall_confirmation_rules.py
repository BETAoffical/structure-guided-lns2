#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stall_confirmation_rule_audit import audit_stall_confirmation_rules


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit conservative observed-stall rules on frozen v2 traces."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = audit_stall_confirmation_rules(
        args.config, args.output, resume=args.resume
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
