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

from experiments.stalled_state_probe import run_stalled_state_probe  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay one stalled v2 state and compare one-repair alternatives."
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--solver-seed", type=int, required=True)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--auto-terminal-stall", action="store_true")
    target.add_argument("--decision-index", type=int)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    report = run_stalled_state_probe(
        arguments.source,
        arguments.output,
        task_id=arguments.task_id,
        solver_seed=arguments.solver_seed,
        trials=arguments.trials,
        auto_terminal_stall=arguments.auto_terminal_stall,
        decision_index=arguments.decision_index,
        resume=arguments.resume,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
