from __future__ import annotations

import argparse
import json

from experiments.stride_transactionalrepair_confirmation import (
    analyze_confirmation,
    collect_confirmation,
    prepare_confirmation_cohort,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the STRIDE TransactionalRepair result-blind confirmation"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--selection", required=True)
    prepare.add_argument("--selection-report", required=True)
    prepare.add_argument("--source-selection-report", required=True)
    prepare.add_argument("--discovery-report", required=True)
    prepare.add_argument("--output", required=True)

    collect = subparsers.add_parser("collect")
    collect.add_argument("--config", required=True)
    collect.add_argument("--output", required=True)
    collect.add_argument("--resume", action="store_true")

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--config", required=True)
    analyze.add_argument("--output", required=True)

    arguments = parser.parse_args()
    if arguments.command == "prepare":
        result = prepare_confirmation_cohort(
            selection_path=arguments.selection,
            selection_report_path=arguments.selection_report,
            source_selection_report_path=arguments.source_selection_report,
            discovery_report_path=arguments.discovery_report,
            output=arguments.output,
        )
        success = result.get("passed") is True
    elif arguments.command == "collect":
        result = collect_confirmation(
            config_path=arguments.config,
            output=arguments.output,
            resume=arguments.resume,
        )
        success = result.get("status") == "complete"
    else:
        result = analyze_confirmation(arguments.config, arguments.output)
        success = result.get("integrity_passed") is True
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
