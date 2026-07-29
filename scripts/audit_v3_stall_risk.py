from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NATIVE_BUILD = PROJECT_ROOT / "build" / "linux" / "project"
sys.path.insert(0, str(PROJECT_ROOT))
if NATIVE_BUILD.is_dir():
    sys.path.insert(0, str(NATIVE_BUILD))

from experiments.stall_risk_audit import audit_frozen_v3_stall_risk  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit a frozen v3 no-progress head against an exact full-pool "
            "stall Oracle without training or changing v2 actions."
        )
    )
    parser.add_argument("--probe", required=True)
    parser.add_argument("--oracle", required=True)
    parser.add_argument("--v3-controller", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    report = audit_frozen_v3_stall_risk(
        arguments.probe,
        arguments.oracle,
        arguments.v3_controller,
        arguments.output,
        resume=arguments.resume,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
