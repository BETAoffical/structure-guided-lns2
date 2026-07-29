#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stall_preaction_features import build_stall_preaction_features


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a leakage-free compact feature matrix for stall pilots."
    )
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--feature-backend", choices=("auto", "python", "native"), default="auto")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = build_stall_preaction_features(
        args.cohort,
        args.labels,
        args.output,
        feature_backend=args.feature_backend,
        resume=args.resume,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
