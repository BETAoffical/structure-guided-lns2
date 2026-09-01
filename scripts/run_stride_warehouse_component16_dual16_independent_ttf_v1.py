from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
NATIVE = ROOT / "build" / "linux" / "project"
if NATIVE.is_dir() and str(NATIVE) not in sys.path:
    sys.path.insert(0, str(NATIVE))

from experiments.stride_warehouse_component16_dual16_independent_ttf_v1 import (  # noqa: E402
    analyze_collection,
    run_collection,
    run_preflight,
)

DEFAULT_CONFIG = ROOT / "configs" / "stride_warehouse_component16_dual16_independent_ttf_v1.json"
DEFAULT_OUTPUT = ROOT / "build" / "stride-warehouse-component16-dual16-independent-ttf-v1"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Map-disjoint three-arm Component16/Dual16/Official strict-serial TTF test"
    )
    parser.add_argument("command", choices=("preflight", "collect", "analyze"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command != "collect" and args.resume:
        parser.error("--resume is accepted only by collect")
    if args.command == "preflight":
        result = run_preflight(args.config, args.output)
    elif args.command == "collect":
        result = run_collection(args.config, args.output, resume=args.resume)
    else:
        result = analyze_collection(args.config, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
