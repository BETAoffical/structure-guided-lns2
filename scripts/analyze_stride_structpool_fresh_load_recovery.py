#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.stride_structpool_fresh_load_recovery import (  # noqa: E402
    analyze_structpool_fresh_load_recovery,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze reset-only StructPool fresh-map load recovery."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--qualification", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    report = analyze_structpool_fresh_load_recovery(
        config_path=arguments.config,
        dataset=arguments.dataset,
        qualification=arguments.qualification,
        output=arguments.output,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
