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

from experiments.stride_warehouse_fixed16_development_v2 import (  # noqa: E402
    analyze,
    collect,
    plan,
    prepare_development_dataset,
    qualify,
    run,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the qualification-conditioned single-map Warehouse fixed16 "
            "development ablation. No command reads or imports the old r2 reset "
            "or episode artifacts."
        )
    )
    parser.add_argument(
        "command", choices=("plan", "prepare-q0", "qualify", "collect", "analyze", "run")
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = Path(args.config)
    if args.command == "plan":
        result = plan(config)
    else:
        if not args.output:
            parser.error(f"--output is required for {args.command}")
        output = Path(args.output)
        if args.command == "prepare-q0":
            if args.resume:
                parser.error("--resume is not valid for prepare-q0")
            result = plan(config) if args.dry_run else prepare_development_dataset(config, output)
        elif args.command == "qualify":
            result = qualify(config, output, resume=args.resume, dry_run=args.dry_run)
        elif args.command == "collect":
            result = collect(config, output, resume=args.resume, dry_run=args.dry_run)
        elif args.command == "analyze":
            if args.dry_run or args.resume:
                parser.error("--dry-run/--resume are not valid for analyze")
            result = analyze(config, output)
        else:
            result = run(config, output, resume=args.resume, dry_run=args.dry_run)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
