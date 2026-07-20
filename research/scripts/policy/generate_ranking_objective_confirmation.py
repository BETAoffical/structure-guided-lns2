from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from research.studies.policy.ranking_objective_confirmation import (  # noqa: E402
    generate_gated_confirmation_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate the 12-map objective confirmation only after its development gate."
    )
    parser.add_argument("--audit-report", required=True)
    parser.add_argument(
        "--config", default="research/configs/policy/ranking_objective_confirmation_dataset.json"
    )
    parser.add_argument("--output")
    arguments = parser.parse_args()
    result = generate_gated_confirmation_dataset(
        arguments.audit_report, arguments.config, arguments.output
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
