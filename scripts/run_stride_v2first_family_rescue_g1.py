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

from experiments.stride_v2first_family_rescue_g1 import (  # noqa: E402
    analyze_batch_a,
    analyze_final,
    plan,
    run,
    run_batch_a,
    run_batch_b,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the fresh-seed V2-first, family-isolated single-rescue G1 screen."
        )
    )
    parser.add_argument(
        "command",
        choices=(
            "plan",
            "run",
            "run-batch-a",
            "analyze-batch-a",
            "run-batch-b",
            "analyze-final",
        ),
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = Path(args.config)
    if args.command == "plan":
        if args.output or args.resume or args.dry_run:
            parser.error("plan accepts only --config")
        result = plan(config)
    else:
        if not args.output:
            parser.error(f"--output is required for {args.command}")
        output = Path(args.output)
        if args.command in {"analyze-batch-a", "analyze-final"}:
            if args.resume or args.dry_run:
                parser.error("analysis commands do not accept --resume/--dry-run")
            result = (
                analyze_batch_a(config, output)
                if args.command == "analyze-batch-a"
                else analyze_final(config, output)
            )
        elif args.command == "run-batch-a":
            result = run_batch_a(
                config, output, resume=args.resume, dry_run=args.dry_run
            )
        elif args.command == "run-batch-b":
            result = run_batch_b(
                config, output, resume=args.resume, dry_run=args.dry_run
            )
        else:
            result = run(
                config, output, resume=args.resume, dry_run=args.dry_run
            )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
