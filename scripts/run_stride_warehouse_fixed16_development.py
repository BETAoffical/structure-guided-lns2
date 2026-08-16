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

from experiments.stride_warehouse_fixed16_development import (  # noqa: E402
    analyze,
    collect,
    plan,
    qualify,
    run,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the preregistered Warehouse fixed16 development screen. "
            "The plan command is read-only and never generates or resets the "
            "reserved final namespace."
        )
    )
    parser.add_argument(
        "command", choices=("plan", "qualify", "collect", "analyze", "run")
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.command == "plan":
        result = plan(Path(args.config))
    else:
        if not args.output:
            parser.error(f"--output is required for {args.command}")
        output = Path(args.output)
        if args.command == "qualify":
            result = qualify(
                Path(args.config), output, resume=args.resume, dry_run=args.dry_run
            )
        elif args.command == "collect":
            result = collect(
                Path(args.config), output, resume=args.resume, dry_run=args.dry_run
            )
        elif args.command == "analyze":
            if args.dry_run:
                parser.error("--dry-run is not valid for analyze")
            result = analyze(Path(args.config), output)
        else:
            result = run(
                Path(args.config), output, resume=args.resume, dry_run=args.dry_run
            )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
