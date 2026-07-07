from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from generators.candidate_experience import (  # noqa: E402
    build_candidate_experience,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert Trace V4 trials into candidate cases."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--split", choices=("train", "validation"), required=True
    )
    arguments = parser.parse_args()
    summary = build_candidate_experience(
        dataset=arguments.dataset,
        collection=arguments.collection,
        output=arguments.output,
        split=arguments.split,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
