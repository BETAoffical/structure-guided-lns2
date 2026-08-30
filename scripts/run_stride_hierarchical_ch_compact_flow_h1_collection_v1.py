from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.stride_hierarchical_ch_compact_flow_h1_collection_v1 import (  # noqa: E402
    DEFAULT_OUTPUT,
    run_collection,
    run_preflight,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Preflight or collect the compact-flow hierarchical V/C/H H1 product. "
            "Preflight generates no repair outcomes; collection evaluates only "
            "exact unique agent tuples under paired seeds."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/stride_hierarchical_ch_compact_flow_h1_collection_v1.json",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--phase", choices=("preflight", "collect"), required=True)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.resume and args.dry_run:
        parser.error("--resume and --dry-run are mutually exclusive")
    fn = run_preflight if args.phase == "preflight" else run_collection
    report = fn(
        args.config,
        args.output,
        workers=args.workers,
        resume=args.resume,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
