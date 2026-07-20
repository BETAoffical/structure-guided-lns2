from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from research.studies.context.contextual_repair_order_audit import run_audit  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit contextual PP repair-order selection.")
    parser.add_argument("--config", default="research/configs/context/contextual_repair_order_audit.json")
    parser.add_argument("--output", default="build/initlns-contextual-repair-order-audit-v1")
    parser.add_argument(
        "--phase", choices=("index", "cross-validate", "fit", "confirm", "all"), default="all"
    )
    arguments = parser.parse_args()
    phase = "all" if arguments.phase == "cross-validate" else arguments.phase
    report = run_audit(arguments.config, arguments.output, phase=phase)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
