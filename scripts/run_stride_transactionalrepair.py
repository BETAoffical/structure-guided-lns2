from __future__ import annotations

import argparse
import json

from experiments.stride_transactionalrepair import (
    analyze_transactional_audit,
    collect_transactional_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run STRIDE TransactionalRepair v1")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect")
    collect.add_argument("--config", required=True)
    collect.add_argument("--output", required=True)
    collect.add_argument("--resume", action="store_true")

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--config", required=True)
    analyze.add_argument("--output", required=True)

    arguments = parser.parse_args()
    if arguments.command == "collect":
        result = collect_transactional_audit(
            config_path=arguments.config,
            output=arguments.output,
            resume=arguments.resume,
        )
    else:
        result = analyze_transactional_audit(arguments.config, arguments.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    if arguments.command == "collect":
        return 0 if result.get("status") == "complete" else 1
    return 0 if result.get("integrity_passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
