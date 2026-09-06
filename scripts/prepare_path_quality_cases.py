"""Prepare static inputs. No collect, reset, training, or timing command exists."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments._common import read_json
from lns2_selector.evaluation.path_quality_preflight import contained, prepare, publish


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/path_quality_preflight_v1.json")
    parser.add_argument("--verify", action="store_true", help="Check frozen preparation without writing")
    args = parser.parse_args()
    config = contained(ROOT, args.config)
    report = prepare(ROOT, config)
    publish(ROOT, report, read_json(config)["output"], verify=args.verify)
    print(json.dumps({key: report[key] for key in (
        "status", "map_count", "task_count", "eligible_tasks", "quarantined_tasks",
        "eligible_instance_seeds", "proposed_first_feasible_episode_count", "proposed_total_episode_count", "solver_calls",
        "timed_episodes_started", "fingerprint",
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
