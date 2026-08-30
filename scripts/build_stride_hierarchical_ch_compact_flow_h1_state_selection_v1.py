from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.stride_hierarchical_ch_compact_flow_h1_state_selection_v1 import (  # noqa: E402
    DEFAULT_OUTPUT_NAME,
    build_state_selection,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify compact-flow source traces and build the result-blind H1 "
            "state selection. This command never invokes a solver."
        )
    )
    parser.add_argument(
        "--config",
        default=(
            "configs/"
            "stride_hierarchical_ch_compact_flow_h1_state_selection_v1.json"
        ),
    )
    parser.add_argument("--output", default=f"build/{DEFAULT_OUTPUT_NAME}")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    report = build_state_selection(
        args.config,
        args.output,
        workers=args.workers,
        project_root=PROJECT_ROOT,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
