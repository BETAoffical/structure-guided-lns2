from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stride_marginalpool_action_replay import (  # noqa: E402
    analyze_marginalpool_action_replay,
    collect_marginalpool_action_replay,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the preregistered MarginalPool all-candidate action replay."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect = subparsers.add_parser("collect")
    collect.add_argument("--config", required=True)
    collect.add_argument("--output", required=True)
    collect.add_argument("--mode", choices=("preflight", "full"), required=True)
    collect.add_argument("--workers", type=int)
    collect.add_argument("--resume", action="store_true")
    collect.add_argument("--preflight-output")
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--config", required=True)
    analyze.add_argument("--collection", required=True)
    analyze.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "collect":
        result = collect_marginalpool_action_replay(
            config_path=args.config,
            output=args.output,
            mode=args.mode,
            workers=args.workers,
            resume=args.resume,
            preflight_output=args.preflight_output,
        )
    else:
        result = analyze_marginalpool_action_replay(
            config_path=args.config,
            collection=args.collection,
            output=args.output,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
