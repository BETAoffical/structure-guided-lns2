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

from experiments.stride_tailswitch import (  # noqa: E402
    analyze_tailswitch,
    run_tailswitch,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run STRIDE TailSwitch v1")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--limit-states", type=int)
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--config", required=True)
    analyze.add_argument("--output", required=True)
    analyze.add_argument("--expected-states", type=int)
    arguments = parser.parse_args()
    if arguments.command == "run":
        result = run_tailswitch(
            arguments.config,
            arguments.output,
            resume=arguments.resume,
            dry_run=arguments.dry_run,
            limit_states=arguments.limit_states,
        )
    else:
        result = analyze_tailswitch(
            arguments.config,
            arguments.output,
            expected_states=arguments.expected_states,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
