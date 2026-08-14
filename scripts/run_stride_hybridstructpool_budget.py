from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stride_hybridstructpool_budget import (  # noqa: E402
    analyze_hybridstructpool_budget,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit outcome-blind HybridStructPool challenger budgets"
    )
    parser.add_argument(
        "--config",
        default=str(
            ROOT / "configs" / "stride_hybridstructpool_budget_v1_registration.json"
        ),
    )
    parser.add_argument(
        "--output", default=str(ROOT / "build" / "stride-hybridstructpool-budget-v1")
    )
    arguments = parser.parse_args()
    report = analyze_hybridstructpool_budget(arguments.config, arguments.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
