from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from generators.retrieval import build_retrieval_index  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the Train-only Stage 4 retrieval index."
    )
    parser.add_argument("--memory", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    summary = build_retrieval_index(arguments.memory, arguments.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
