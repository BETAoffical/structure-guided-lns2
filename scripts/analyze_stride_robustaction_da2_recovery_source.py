from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.stride_robustaction_da2_recovery_source import (  # noqa: E402
    analyze_da2_recovery_source,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the complete DA2 recovery source product and its combined "
            "result-blind state capacity with frozen source v4."
        )
    )
    parser.add_argument(
        "--config",
        default=(
            "configs/"
            "stride_robustaction_structpool_da2_recovery_source_design.json"
        ),
    )
    parser.add_argument("--source", required=True)
    parser.add_argument(
        "--output",
        default="build/stride-robustaction-da2-recovery-source-analysis-v1",
    )
    arguments = parser.parse_args()
    report = analyze_da2_recovery_source(
        config_path=arguments.config,
        source_root=arguments.source,
        output=arguments.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if bool(report["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
