from __future__ import annotations

import argparse
import json

from experiments.stride_nativeorder_transactionalrepair import (
    analyze,
    analyze_qualification,
    collect_phase,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run STRIDE Native-Order TransactionalRepair v1"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("qualify", "collect-initial", "collect-extension"):
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

    arguments = parser.parse_args()
    if arguments.command in {"qualify", "collect-initial", "collect-extension"}:
        phase = {
            "qualify": "qualification",
            "collect-initial": "initial",
            "collect-extension": "extension",
        }[arguments.command]
        result = collect_phase(
            config_path=arguments.config,
            output=arguments.output,
            phase=phase,
            resume=arguments.resume,
        )
        if arguments.command == "qualify" and result.get("status") == "complete":
            result = analyze_qualification(arguments.config, arguments.output)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if result.get("passed") is True else 1
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("status") == "complete" else 1
    if arguments.command == "analyze-initial":
        result = analyze(arguments.config, arguments.initial)
    else:
        result = analyze(arguments.config, arguments.initial, arguments.extension)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("integrity_passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
