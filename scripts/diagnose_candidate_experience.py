from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from generators.candidate_diagnostics import (  # noqa: E402
    build_candidate_diagnostics,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize candidate generators and replan-order noise."
    )
    parser.add_argument("--memory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset")
    arguments = parser.parse_args()
    summary = build_candidate_diagnostics(
        memory=arguments.memory,
        output=arguments.output,
        dataset=arguments.dataset,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
