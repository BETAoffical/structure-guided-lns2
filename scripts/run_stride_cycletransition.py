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

from experiments.stride_cycletransition import (  # noqa: E402
    analyze_cycletransition,
    collect_cycletransition,
    prepare_cycletransition,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run STRIDE CycleTransition pilot")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "collect", "analyze", "run"):
        command = subparsers.add_parser(name)
        command.add_argument(
            "--config",
            default=str(
                ROOT / "configs" / "stride_cycletransition_pilot_v1_registration.json"
            ),
        )
        command.add_argument(
            "--output",
            default=str(ROOT / "build" / "stride-cycletransition-pilot-v1"),
        )
        if name in {"prepare", "collect", "run"}:
            command.add_argument("--workers", type=int, default=16)
        if name in {"collect", "run"}:
            command.add_argument("--no-resume", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = Path(args.config)
    output = Path(args.output)
    if args.command == "prepare":
        result = prepare_cycletransition(
            config_path=config, output=output, workers=args.workers
        )
    elif args.command == "collect":
        result = collect_cycletransition(
            config_path=config,
            output=output,
            workers=args.workers,
            resume=not args.no_resume,
        )
    elif args.command == "analyze":
        result = analyze_cycletransition(
            config_path=config, collection=output, output=output
        )
    else:
        preparation = prepare_cycletransition(
            config_path=config, output=output, workers=args.workers
        )
        if preparation.get("passed") is not True:
            result = preparation
        else:
            collection = collect_cycletransition(
                config_path=config,
                output=output,
                workers=args.workers,
                resume=not args.no_resume,
            )
            result = (
                analyze_cycletransition(
                    config_path=config, collection=output, output=output
                )
                if collection.get("status") == "complete"
                else collection
            )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
