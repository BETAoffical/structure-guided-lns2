from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.controller_performance_benchmark import (  # noqa: E402
    run_controller_feature_benchmark,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark reference and incremental feature-v2 extraction."
    )
    parser.add_argument(
        "--collection",
        default="build/initlns-movingai-ood-collection-v2-compact",
    )
    parser.add_argument(
        "--output",
        default="artifacts/initlns-closed-loop-controller-v2/performance_benchmark.json",
    )
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--maximum-decisions", type=int, default=3)
    parser.add_argument(
        "--feature-backend", choices=("auto", "python", "native"), default="auto"
    )
    arguments = parser.parse_args()
    collection = Path(arguments.collection)
    output = Path(arguments.output)
    if not collection.is_absolute():
        collection = PROJECT_ROOT / collection
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    report = run_controller_feature_benchmark(
        collection,
        output,
        repeats=arguments.repeats,
        maximum_decisions=arguments.maximum_decisions,
        feature_backend=arguments.feature_backend,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not report["native_backend_required"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
