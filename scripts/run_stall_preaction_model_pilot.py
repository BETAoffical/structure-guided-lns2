#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stall_preaction_model_pilot import run_stall_preaction_model_pilot


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run map-group OOF stall-trigger and rescue-ranker pilots."
    )
    parser.add_argument("--features", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = run_stall_preaction_model_pilot(
        args.features, args.output, resume=args.resume
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
