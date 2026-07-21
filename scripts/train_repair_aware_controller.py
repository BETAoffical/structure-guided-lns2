from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.repair_aware_training import run_repair_aware_training  # noqa: E402


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train the policy_train-only v2 repair-aware auxiliary bundle."
    )
    parser.add_argument(
        "--feature-index",
        default=(
            "build/initlns-policy-visited-natural-v2-training/"
            "policy_visited_index.jsonl"
        ),
    )
    parser.add_argument(
        "--trial-manifest",
        default=(
            "build/initlns-policy-visited-natural-v2-collection/"
            "evaluation_trial_manifest.jsonl"
        ),
    )
    parser.add_argument(
        "--controller-bundle",
        default="artifacts/initlns-closed-loop-controller-v2",
    )
    parser.add_argument(
        "--output", default="build/initlns-repair-aware-controller-v1"
    )
    arguments = parser.parse_args()
    report = run_repair_aware_training(
        feature_index=_resolve(arguments.feature_index),
        trial_manifest=_resolve(arguments.trial_manifest),
        controller_bundle=_resolve(arguments.controller_bundle),
        output=_resolve(arguments.output),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
