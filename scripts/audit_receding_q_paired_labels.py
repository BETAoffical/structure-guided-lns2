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
from experiments.receding_q_paired_label_audit import (  # noqa: E402
    audit_receding_q_paired_labels,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare paired-seed H3 rank and centered-risk labels on existing "
            "four-seed receding-Q outcomes."
        )
    )
    parser.add_argument("--stability", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    report = audit_receding_q_paired_labels(
        stability=resolve_cli_path(PROJECT_ROOT, arguments.stability),
        output=resolve_cli_path(PROJECT_ROOT, arguments.output),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
