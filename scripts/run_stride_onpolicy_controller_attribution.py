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

from experiments.stride_onpolicy_controller_attribution import (  # noqa: E402
    analyze_collection,
    run_collection,
)


def main(argv: list[str] | None = None) -> int:
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
    parser.add_argument("--limit-tasks", type=int)
    args = parser.parse_args(argv)
    if args.command.startswith("collect-"):
        result = run_collection(
            args.config,
            args.output,
            phase=args.command.removeprefix("collect-"),
            resume=args.resume,
            limit_tasks=args.limit_tasks,
        )
    elif args.command.startswith("analyze-"):
        result = analyze_collection(
            args.config,
            args.output,
            phase=args.command.removeprefix("analyze-"),
            limit_tasks=args.limit_tasks,
        )
    else:
        result = run_collection(
            args.config,
            args.output,
            phase="initial",
            dry_run=True,
            limit_tasks=args.limit_tasks,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
