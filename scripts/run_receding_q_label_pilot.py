#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
NATIVE_ROOT = PROJECT_ROOT / "build" / "linux" / "project"
if NATIVE_ROOT.is_dir() and str(NATIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(NATIVE_ROOT))

from experiments._common import resolve_cli_path  # noqa: E402
from experiments.receding_q_pilot import (  # noqa: E402
    run_receding_q_label_pilot,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Collect fixed-H3 long-horizon labels for every actual 4/8/16 "
            "candidate. The root action is explicit and subsequent repairs "
            "are independently replanned by official Adaptive."
        )
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--state-count", type=int, default=12)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument(
        "--continuation-teacher",
        choices=("official_adaptive",),
        default="official_adaptive",
    )
    parser.add_argument("--split", default="policy_train")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    report = run_receding_q_label_pilot(
        source=resolve_cli_path(PROJECT_ROOT, arguments.source),
        output=resolve_cli_path(PROJECT_ROOT, arguments.output),
        state_count=arguments.state_count,
        trials=arguments.trials,
        horizon=arguments.horizon,
        continuation_teacher=arguments.continuation_teacher,
        split=arguments.split,
        workers=arguments.workers,
        smoke_only=arguments.smoke_only,
        resume=arguments.resume,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
