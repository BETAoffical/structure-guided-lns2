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
from experiments.v3_value_uncertainty import (  # noqa: E402
    DEFAULT_THRESHOLDS,
    PRIMARY_THRESHOLD,
    audit_value_uncertainty,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether three-seed expected outcomes select an action that "
            "generalizes to a held-out fourth PP seed. No solver is rerun."
        )
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--required-trials", type=int, default=4)
    parser.add_argument(
        "--thresholds",
        default=",".join(map(str, DEFAULT_THRESHOLDS)),
    )
    parser.add_argument(
        "--primary-threshold",
        type=float,
        default=PRIMARY_THRESHOLD,
    )
    arguments = parser.parse_args()
    thresholds = tuple(
        float(value)
        for value in arguments.thresholds.split(",")
        if value.strip()
    )
    report = audit_value_uncertainty(
        source=resolve_cli_path(PROJECT_ROOT, arguments.source),
        output=resolve_cli_path(PROJECT_ROOT, arguments.output),
        required_trials=arguments.required_trials,
        thresholds=thresholds,
        primary_threshold=arguments.primary_threshold,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
