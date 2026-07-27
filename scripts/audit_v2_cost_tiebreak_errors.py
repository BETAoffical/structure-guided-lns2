#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments._common import resolve_cli_path  # noqa: E402
from experiments.v2_cost_tiebreak_error_audit import (  # noqa: E402
    audit_v2_cost_tiebreak_errors,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit paired PP-seed stability, prediction errors, and a direct "
            "pairwise Top-3 alternative to the frozen v2 winner."
        )
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--policy-audit", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    report = audit_v2_cost_tiebreak_errors(
        source=resolve_cli_path(PROJECT_ROOT, arguments.source),
        policy_audit=resolve_cli_path(PROJECT_ROOT, arguments.policy_audit),
        output=resolve_cli_path(PROJECT_ROOT, arguments.output),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
