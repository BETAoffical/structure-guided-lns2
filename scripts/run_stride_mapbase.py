#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.stride_mapbase import (  # noqa: E402
    audit_mapbase_collection,
    collect_mapbase_trials,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect or audit STRIDE-MapBase fresh-map repair trials."
    )
    parser.add_argument("command", choices=("collect", "audit"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--collection")
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    if arguments.command == "collect":
        report = collect_mapbase_trials(
            arguments.config,
            arguments.output,
            workers=arguments.workers,
            resume=arguments.resume,
        )
        code = 0 if report["complete"] else 1
    else:
        if not arguments.collection:
            parser.error("audit requires --collection")
        report = audit_mapbase_collection(
            arguments.config, arguments.collection, arguments.output
        )
        code = 0 if report["passed"] else 1
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
