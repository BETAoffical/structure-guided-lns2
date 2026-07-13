from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from generators.candidate_retrieval import (  # noqa: E402
    evaluate_candidate_retrieval,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Tune candidate-aware kNN and the baseline replacement "
            "margin on Validation."
        )
    )
    parser.add_argument("--index", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    summary = evaluate_candidate_retrieval(
        arguments.index, arguments.queries, arguments.output
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
