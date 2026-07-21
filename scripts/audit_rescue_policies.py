from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments._common import resolve_cli_path  # noqa: E402
from experiments.rescue_policy_audit import audit_rescue_policies  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit deterministic 4/8/16 rescue orders from an existing high-load "
            "pilot without running the solver."
        )
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    report = audit_rescue_policies(
        source=resolve_cli_path(PROJECT_ROOT, arguments.source),
        output=resolve_cli_path(PROJECT_ROOT, arguments.output),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
