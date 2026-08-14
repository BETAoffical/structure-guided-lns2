from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
NATIVE_BUILD = PROJECT_ROOT / "build" / "linux" / "project"
if NATIVE_BUILD.is_dir() and str(NATIVE_BUILD) not in sys.path:
    sys.path.insert(0, str(NATIVE_BUILD))

from experiments.stride_failure_informed_rescue_continuation import (  # noqa: E402
    analyze_collection,
    run_collection,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "collect-initial",
            "analyze-initial",
            "collect-extension",
            "analyze-extended",
            "dry-run",
        ),
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit-cases", type=int)
    args = parser.parse_args()
    if args.command == "dry-run":
        result = run_collection(
            args.config,
            args.output,
            phase="initial",
            dry_run=True,
            limit_cases=args.limit_cases,
        )
    elif args.command.startswith("collect-"):
        result = run_collection(
            args.config,
            args.output,
            phase=args.command.removeprefix("collect-"),
            resume=args.resume,
            limit_cases=args.limit_cases,
        )
    else:
        result = analyze_collection(
            args.config,
            args.output,
            phase="initial" if args.command == "analyze-initial" else "extended",
            limit_cases=args.limit_cases,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
