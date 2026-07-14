from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.repair_collection import recover_counterfactual_manifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recover the aggregate manifest from complete episode metadata."
    )
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    report = recover_counterfactual_manifest(arguments.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if report["invalid_metadata"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
