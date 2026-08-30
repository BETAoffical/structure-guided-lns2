from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.stride_hierarchical_ch_compact_flow_balanced_source_v1 import (  # noqa: E402
    collect_source_episodes,
    materialize_source_dataset,
    plan_source_collection,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the independent compact-flow balanced-coverage source cohort."
    )
    parser.add_argument(
        "--config",
        default=(
            "configs/"
            "stride_hierarchical_ch_compact_flow_balanced_source_v1.json"
        ),
    )
    parser.add_argument(
        "--output",
        default="build/stride-hierarchical-ch-compact-flow-balanced-source-v1",
    )
    parser.add_argument(
        "--phase", choices=("materialize", "plan", "collect"), required=True
    )
    parser.add_argument("--workers", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.dry_run and args.resume:
        parser.error("--dry-run and --resume are mutually exclusive")
    if args.phase == "materialize" and args.workers is not None:
        parser.error("--workers is only valid for plan/collect")
    common = {
        "config_path": args.config,
        "output": args.output,
        "dry_run": args.dry_run,
        "resume": args.resume,
    }
    if args.phase == "materialize":
        report = materialize_source_dataset(**common)
    elif args.phase == "plan":
        report = plan_source_collection(workers=args.workers, **common)
    else:
        report = collect_source_episodes(workers=args.workers, **common)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
