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

from experiments.stride_v2first_consensus16_warehouse_n800_supply import (  # noqa: E402
    analyze_final,
    analyze_screen,
    plan,
    run,
    run_formal,
    run_screen,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the preregistered N800 warehouse state-supply screen and "
            "prefix-screened consensus16 TTF test."
        )
    )
    parser.add_argument(
        "command",
        choices=(
            "plan",
            "run",
            "run-screen",
            "analyze-screen",
            "run-formal",
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
        if args.command in {"analyze-screen", "analyze-final"}:
            if args.resume or args.dry_run:
                parser.error("analysis commands do not accept --resume/--dry-run")
            result = (
                analyze_screen(config, output)
                if args.command == "analyze-screen"
                else analyze_final(config, output)
            )
        elif args.command == "run-screen":
            result = run_screen(
                config, output, resume=args.resume, dry_run=args.dry_run
            )
        elif args.command == "run-formal":
            result = run_formal(
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
