from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from generators.candidate_ranker import evaluate_candidate_ranker  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tune Stage 5 v3 candidate rankers on Validation."
    )
    parser.add_argument("--ranker", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--conservative-gates",
        action="store_true",
        help="Tune regular/low-conflict/clustered fallback margins.",
    )
    arguments = parser.parse_args()
    summary = evaluate_candidate_ranker(
        arguments.ranker,
        arguments.queries,
        arguments.output,
        conservative_gates=arguments.conservative_gates,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
