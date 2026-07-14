from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.context_audit import run_audit  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit whether static context improves InitLNS action ranking."
    )
    parser.add_argument("--collection", required=True)
    parser.add_argument(
        "--dataset",
        help="Pilot dataset root; inferred from collection/run_config.json when omitted.",
    )
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    report = run_audit(arguments.collection, arguments.output, arguments.dataset)
    print(json.dumps(report["acceptance"], indent=2, sort_keys=True))
    return 0 if report["acceptance"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
