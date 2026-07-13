from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from generators.candidate_ranker import train_candidate_ranker  # noqa: E402
from generators.candidate_retrieval import FEATURE_PROFILES  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train Train-only Stage 5 v3 candidate rankers."
    )
    parser.add_argument("--memory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--feature-profile",
        choices=sorted(FEATURE_PROFILES),
        default="dedup20",
    )
    parser.add_argument(
        "--models",
        default="pairwise_linear",
        help=(
            "Comma-separated models: pairwise_linear, sklearn_logistic, "
            "sklearn_forest, sklearn_gbdt."
        ),
    )
    arguments = parser.parse_args()
    models = [
        value.strip()
        for value in arguments.models.split(",")
        if value.strip()
    ]
    summary = train_candidate_ranker(
        arguments.memory,
        arguments.output,
        feature_profile=arguments.feature_profile,
        models=models,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
