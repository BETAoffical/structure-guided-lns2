from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.repair_collection import (  # noqa: E402
    CollectionLockError,
    run_collection,
)


def _integers(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("expected a comma-separated integer list")
    return result


def _strings(value: str) -> list[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("expected a comma-separated string list")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect official LNS2 repair trajectories and counterfactual rollouts."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--config", default="configs/repair_collection_pilot.json"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--phase",
        choices=("qualify", "baseline", "counterfactual", "all"),
        default="all",
    )
    parser.add_argument("--splits", help="comma-separated dataset splits")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--max-states", type=int)
    parser.add_argument("--max-seed-agents", type=int)
    parser.add_argument("--neighborhood-sizes", type=_integers)
    parser.add_argument("--trials", type=int)
    parser.add_argument("--horizons", type=_integers)
    parser.add_argument("--task-ids", type=_strings)
    parser.add_argument("--episode-time-limit", type=float)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    try:
        summary = run_collection(
            dataset=arguments.dataset,
            config_path=arguments.config,
            output=arguments.output,
            phase=arguments.phase,
            splits=(
                [
                    value.strip()
                    for value in arguments.splits.split(",")
                    if value.strip()
                ]
                if arguments.splits
                else None
            ),
            workers=arguments.workers,
            resume=arguments.resume,
            max_episodes=arguments.max_episodes,
            max_states=arguments.max_states,
            max_seed_agents=arguments.max_seed_agents,
            neighborhood_sizes=arguments.neighborhood_sizes,
            trials=arguments.trials,
            horizons=arguments.horizons,
            task_ids=arguments.task_ids,
            episode_time_limit=arguments.episode_time_limit,
            dry_run=arguments.dry_run,
        )
    except CollectionLockError as error:
        print(json.dumps({"status": "locked", "error": str(error)}), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print(json.dumps({"status": "interrupted"}), file=sys.stderr)
        return 130
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if arguments.dry_run:
        return 0
    errors = sum(
        int(summary.get(section, {}).get("error_count", 0))
        for section in ("qualification", "baseline", "counterfactual")
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
