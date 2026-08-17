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

from experiments.stride_warehouse_compactcut import (  # noqa: E402
    plan,
    prepare_q0,
    qualify,
    run,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare and reset-qualify the independent compact-cut Warehouse "
            "benchmark. This runner cannot execute a controller, a policy "
            "episode, or formal TTF collection."
        )
    )
    parser.add_argument("command", choices=("plan", "prepare-q0", "qualify", "run"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = Path(args.config)
    if args.command == "plan":
        if args.resume or args.dry_run:
            parser.error("--resume/--dry-run are not valid for plan")
        result = plan(config)
    else:
        if not args.output:
            parser.error(f"--output is required for {args.command}")
        output = Path(args.output)
        if args.command == "prepare-q0":
            if args.resume or args.dry_run:
                parser.error("--resume/--dry-run are not valid for prepare-q0")
            result = prepare_q0(config, output)
        elif args.command == "qualify":
            result = qualify(config, output, resume=args.resume, dry_run=args.dry_run)
        else:
            result = run(config, output, resume=args.resume, dry_run=args.dry_run)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
