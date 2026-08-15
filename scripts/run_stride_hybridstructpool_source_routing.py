#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
NATIVE = ROOT / "build" / "linux" / "project"
if NATIVE.is_dir() and str(NATIVE) not in sys.path:
    sys.path.insert(0, str(NATIVE))

from experiments.stride_hybridstructpool_source_routing import (  # noqa: E402
    analyze,
    run,
    zero_solver_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("audit", "run", "analyze"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.command == "audit":
        result = zero_solver_audit(args.config, args.output)
    elif args.command == "analyze":
        result = analyze(args.config, args.output)
    else:
        result = run(
            args.config,
            args.output,
            resume=args.resume,
            dry_run=args.dry_run,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if args.dry_run or result.get("integrity_passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
