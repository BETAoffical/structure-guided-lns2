from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stride_residualhazard import (  # noqa: E402
    analyze_residualhazard,
    collect_residualhazard,
    smoke_residualhazard,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect and analyze preregistered one-step residual hazards."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect")
    collect.add_argument(
        "--config", default="configs/stride_residualhazard_v1_registration.json"
    )
    collect.add_argument("--output", required=True)
    collect.add_argument("--resume", action="store_true")

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument(
        "--config", default="configs/stride_residualhazard_v1_registration.json"
    )
    analyze.add_argument("--collection", required=True)
    analyze.add_argument("--output", required=True)

    smoke = subparsers.add_parser("smoke")
    smoke.add_argument(
        "--config", default="configs/stride_residualhazard_v1_registration.json"
    )
    smoke.add_argument("--output", required=True)

    args = parser.parse_args()
    if args.command == "collect":
        result = collect_residualhazard(args.config, args.output, resume=args.resume)
    elif args.command == "analyze":
        result = analyze_residualhazard(args.config, args.collection, args.output)
    else:
        result = smoke_residualhazard(args.config, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
