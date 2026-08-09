from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stride_maze_tail_action_replay import (  # noqa: E402
    analyze_action_replay,
    collect_action_replay,
    prepare_action_replay,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare, collect, and analyze paired Maze first-divergence actions."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument(
        "--config", default="configs/stride_maze_tail_action_replay_preflight_v1.json"
    )
    prepare.add_argument("--output", required=True)

    collect = subparsers.add_parser("collect")
    collect.add_argument("--config", required=True)
    collect.add_argument("--output", required=True)
    collect.add_argument("--resume", action="store_true")

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--config", required=True)
    analyze.add_argument("--collection", required=True)
    analyze.add_argument("--output", required=True)

    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_action_replay(args.config, args.output)
    elif args.command == "collect":
        result = collect_action_replay(args.config, args.output, resume=args.resume)
    else:
        result = analyze_action_replay(args.config, args.collection, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
