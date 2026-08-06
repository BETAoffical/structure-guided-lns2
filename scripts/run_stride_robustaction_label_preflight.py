from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.stride_robustaction_preflight import (  # noqa: E402
    run_robustaction_label_preflight,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the preregistered RobustAction StructPool proposal and "
            "124-feature preflight without candidate repairs."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/stride_robustaction_structpool_label_preflight.json",
    )
    parser.add_argument(
        "--output",
        default="build/stride-robustaction-label-preflight-v1",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    report = run_robustaction_label_preflight(
        config_path=arguments.config,
        output=arguments.output,
        workers=arguments.workers,
        resume=arguments.resume,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if bool(report["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
