from __future__ import annotations

import argparse
import json

from experiments.stride_compact_nativeorder_repair import analyze, collect_phase


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run STRIDE Compact Native-Order Repair v1"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("collect-initial", "collect-extension"):
        current = subparsers.add_parser(command)
        current.add_argument("--config", required=True)
        current.add_argument("--output", required=True)
        current.add_argument("--resume", action="store_true")
    initial = subparsers.add_parser("analyze-initial")
    initial.add_argument("--config", required=True)
    initial.add_argument("--initial", required=True)
    extended = subparsers.add_parser("analyze-extended")
    extended.add_argument("--config", required=True)
    extended.add_argument("--initial", required=True)
    extended.add_argument("--extension", required=True)
    args = parser.parse_args()
    if args.command == "collect-initial":
        result = collect_phase(
            config_path=args.config,
            output=args.output,
            phase="initial",
            resume=args.resume,
        )
    elif args.command == "collect-extension":
        result = collect_phase(
            config_path=args.config,
            output=args.output,
            phase="extension",
            resume=args.resume,
        )
    elif args.command == "analyze-initial":
        result = analyze(args.config, args.initial)
    else:
        result = analyze(args.config, args.initial, args.extension)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") == "complete" or result.get("integrity_passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
