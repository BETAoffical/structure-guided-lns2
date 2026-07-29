from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NATIVE_BUILD = PROJECT_ROOT / "build" / "linux" / "project"
sys.path.insert(0, str(PROJECT_ROOT))
if NATIVE_BUILD.is_dir():
    sys.path.insert(0, str(NATIVE_BUILD))

from experiments.stall_training_extension import (  # noqa: E402
    run_stall_training_extension,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Extend unfinished policy_train v2 episodes and discover exact "
            "stall-specific label candidates without training or action changes."
        )
    )
    parser.add_argument("--source", required=True)
    parser.add_argument(
        "--config",
        help=(
            "Explicit source protocol for a directly written collection. "
            "Every protocol field must match the registered source run."
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-decisions", type=int, default=60)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--task-id", action="append", dest="task_ids")
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    report = run_stall_training_extension(
        arguments.source,
        arguments.output,
        source_config=arguments.config,
        target_decisions=arguments.target_decisions,
        workers=arguments.workers,
        task_ids=arguments.task_ids,
        resume=arguments.resume,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
