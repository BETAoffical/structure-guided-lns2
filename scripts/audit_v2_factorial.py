from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.v2_factorial_audit import run_v2_factorial_audit  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit v2 training-distribution and feature-set factors without deployment."
    )
    parser.add_argument(
        "--legacy-training-index",
        default="build/initlns-policy-visited-natural-v2-training/aggregate_train_index.jsonl",
    )
    parser.add_argument(
        "--legacy-validation-index",
        default="build/initlns-policy-visited-natural-v2-training/validation_index.jsonl",
    )
    parser.add_argument(
        "--high-load-collection",
        default="build/initlns-v3-pilot-v1/collection",
    )
    parser.add_argument(
        "--controller-bundle",
        default="artifacts/initlns-closed-loop-controller-v2",
    )
    parser.add_argument(
        "--portable-bundle",
        default="artifacts/initlns-closed-loop-policy-v1",
    )
    parser.add_argument(
        "--output", default="build/initlns-v2-factorial-audit-v1"
    )
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def main() -> int:
    arguments = parse_arguments()
    report = run_v2_factorial_audit(
        legacy_training_index=resolve(arguments.legacy_training_index),
        legacy_validation_index=resolve(arguments.legacy_validation_index),
        high_load_collection=resolve(arguments.high_load_collection),
        controller_bundle=resolve(arguments.controller_bundle),
        portable_bundle=resolve(arguments.portable_bundle),
        output=resolve(arguments.output),
    )
    print(report["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
