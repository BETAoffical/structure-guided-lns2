from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
NATIVE_BUILD = ROOT / "build" / "linux" / "project"
if NATIVE_BUILD.is_dir() and str(NATIVE_BUILD) not in sys.path:
    sys.path.insert(0, str(NATIVE_BUILD))

from experiments.stride_successor_repairability import (  # noqa: E402
    analyze_successor_repairability,
    collect_successor_repairability,
    prepare_successor_repairability,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the bounded STRIDE successor-repairability mechanism probe"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    names = (
        "prepare",
        "collect-preflight",
        "analyze-preflight",
        "collect-initial",
        "analyze-initial",
        "collect-extension",
        "analyze-extended",
        "run-preflight",
    )
    for name in names:
        command = commands.add_parser(name)
        command.add_argument(
            "--config",
            default=str(
                ROOT
                / "configs"
                / "stride_successor_repairability_v1_registration.json"
            ),
        )
        command.add_argument(
            "--output",
            default=str(ROOT / "build" / "stride-successor-repairability-v1"),
        )
        if name in {
            "prepare",
            "collect-preflight",
            "collect-initial",
            "collect-extension",
            "run-preflight",
        }:
            command.add_argument("--workers", type=int, default=16)
        if name in {
            "prepare",
            "collect-preflight",
            "collect-initial",
            "collect-extension",
            "run-preflight",
        }:
            command.add_argument("--no-resume", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = Path(args.config)
    output = Path(args.output)
    if args.command == "prepare":
        result = prepare_successor_repairability(
            config_path=config,
            output=output,
            workers=args.workers,
            resume=not args.no_resume,
        )
    elif args.command.startswith("collect-"):
        mode = args.command.removeprefix("collect-")
        result = collect_successor_repairability(
            config_path=config,
            output=output,
            mode=mode,
            workers=args.workers,
            resume=not args.no_resume,
        )
    elif args.command.startswith("analyze-"):
        mode = args.command.removeprefix("analyze-")
        result = analyze_successor_repairability(
            config_path=config, output=output, mode=mode
        )
    else:
        preparation = prepare_successor_repairability(
            config_path=config,
            output=output,
            workers=args.workers,
            resume=not args.no_resume,
        )
        if preparation.get("passed") is not True:
            result = preparation
        else:
            result = collect_successor_repairability(
                config_path=config,
                output=output,
                mode="preflight",
                workers=args.workers,
                resume=not args.no_resume,
            )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
