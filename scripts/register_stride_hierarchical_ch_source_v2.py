#!/usr/bin/env python3
"""Validate and freeze the static hierarchical C/H source-v2 registration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stride_hierarchical_ch_source_v2_registration import (
    DEFAULT_OUTPUT_NAME,
    run_preflight,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "stride_hierarchical_ch_source_v2_registration.json"),
    )
    parser.add_argument("--output", default=str(ROOT / "build" / DEFAULT_OUTPUT_NAME))
    args = parser.parse_args()
    report = run_preflight(Path(args.config), Path(args.output))
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
