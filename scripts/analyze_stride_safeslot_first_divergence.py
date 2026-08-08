#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.stride_safeslot_first_divergence import (  # noqa: E402
    analyze_safeslot_first_divergence,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the first matched-state divergence of StructPool and "
            "SlotPool from frozen V2."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    report = analyze_safeslot_first_divergence(
        arguments.config, arguments.output
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.get("integrity_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
