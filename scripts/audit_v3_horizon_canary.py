#!/usr/bin/env python3
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

from experiments._common import resolve_cli_path  # noqa: E402
from experiments.v3_horizon_canary import audit_v3_horizon_canary  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a stratified subset of historical V3-H3 actions with the "
            "current native runtime and compare repair semantics."
        )
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--state-count", type=int, default=24)
    arguments = parser.parse_args()
    report = audit_v3_horizon_canary(
        source=resolve_cli_path(PROJECT_ROOT, arguments.source),
        output=resolve_cli_path(PROJECT_ROOT, arguments.output),
        state_count=arguments.state_count,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
