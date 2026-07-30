from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.mixed_full_v2 import export_mixed_full_v2  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train and export the frozen-semantics Mixed Full v2 controller."
    )
    parser.add_argument(
        "--legacy-training-index",
        default="build/initlns-policy-visited-natural-v2-training/aggregate_train_index.jsonl",
    )
    parser.add_argument(
        "--high-load-collection", default="build/initlns-v3-pilot-v1/collection"
    )
    parser.add_argument(
        "--source-portable-bundle",
        default="artifacts/initlns-closed-loop-policy-v1",
    )
    parser.add_argument(
        "--output", default="artifacts/initlns-mixed-full-controller-v2"
    )
    arguments = parser.parse_args()

    def resolve(value: str) -> Path:
        path = Path(value)
        return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()

    report = export_mixed_full_v2(
        legacy_training_index=resolve(arguments.legacy_training_index),
        high_load_collection=resolve(arguments.high_load_collection),
        source_portable_bundle=resolve(arguments.source_portable_bundle),
        output=resolve(arguments.output),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
