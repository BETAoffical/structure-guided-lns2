from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.stride_robustaction_da2_source_stability import (  # noqa: E402
    analyze_da2_source_stability_product,
    analyze_da2_source_stability_qualification,
    prepare_da2_source_stability_dataset,
)


DEFAULT_CONFIG = (
    "configs/stride_robustaction_structpool_da2_source_stability_design.json"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare and audit the DA2 source stability-v2 revision."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--config", default=DEFAULT_CONFIG)
    prepare.add_argument("--output", required=True)

    qualify = subparsers.add_parser("analyze-qualification")
    qualify.add_argument("--config", default=DEFAULT_CONFIG)
    qualify.add_argument("--dataset", required=True)
    qualify.add_argument("--qualification", required=True)
    qualify.add_argument("--output", required=True)

    source = subparsers.add_parser("analyze-source")
    source.add_argument("--config", default=DEFAULT_CONFIG)
    source.add_argument("--dataset", required=True)
    source.add_argument("--source", required=True)
    source.add_argument("--output", required=True)

    arguments = parser.parse_args()
    if arguments.command == "prepare":
        report = prepare_da2_source_stability_dataset(
            config_path=arguments.config, output=arguments.output
        )
    elif arguments.command == "analyze-qualification":
        report = analyze_da2_source_stability_qualification(
            config_path=arguments.config,
            dataset=arguments.dataset,
            qualification=arguments.qualification,
            output=arguments.output,
        )
    else:
        report = analyze_da2_source_stability_product(
            config_path=arguments.config,
            dataset=arguments.dataset,
            source=arguments.source,
            output=arguments.output,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if arguments.command == "prepare" or bool(report["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
