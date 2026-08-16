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

from experiments.stride_structshell_rollback_aware_ttf import (  # noqa: E402
    analyze_final,
    analyze_screen,
    run,
    run_extension,
    run_screen,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the preregistered two-stage rollback-aware bounded TTF confirmation."
    )
    parser.add_argument(
        "command",
        choices=(
            "run",
            "run-screen",
            "analyze-screen",
            "run-extension",
            "analyze-final",
        ),
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config, output = Path(args.config), Path(args.output)
    if args.command == "run":
        result = run(config, output, resume=args.resume, dry_run=args.dry_run)
    elif args.command == "run-screen":
        result = run_screen(
            config, output, resume=args.resume, dry_run=args.dry_run
        )
    elif args.command == "run-extension":
        result = run_extension(
            config, output, resume=args.resume, dry_run=args.dry_run
        )
    elif args.command == "analyze-screen":
        result = analyze_screen(config, output)
    else:
        result = analyze_final(config, output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
