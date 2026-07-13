from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from generators.candidate_retrieval import (  # noqa: E402
    FEATURE_PROFILES,
    build_candidate_index,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the Train-only candidate-aware kNN index."
    )
    parser.add_argument("--memory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--feature-profile",
        choices=sorted(FEATURE_PROFILES),
        default="full",
    )
    arguments = parser.parse_args()
    summary = build_candidate_index(
        arguments.memory,
        arguments.output,
        feature_profile=arguments.feature_profile,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
