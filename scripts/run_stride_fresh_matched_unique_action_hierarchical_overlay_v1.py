from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stride_fresh_matched_unique_action_hierarchical_overlay_v1 import (  # noqa: E402
    build_plan,
    run_analysis,
    run_h1,
    run_selection,
)


DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "stride_fresh_matched_unique_action_hierarchical_overlay_v1.json"
)
DEFAULT_OUTPUT = (
    ROOT / "build" / "stride-fresh-matched-unique-action-hierarchical-overlay-v1"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the read-only unique-action hierarchical H1 overlay; "
            "never mutates the frozen active-supply cohort."
        )
    )
    parser.add_argument("phase", choices=("plan", "select", "h1", "analyze"))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.phase == "plan":
        result = build_plan(args.config)
    elif args.phase == "select":
        result = run_selection(args.config, args.output, dry_run=args.dry_run)
    elif args.phase == "h1":
        result = run_h1(
            args.config,
            args.output,
            resume=args.resume,
            dry_run=args.dry_run,
        )
    else:
        result = run_analysis(args.config, args.output, dry_run=args.dry_run)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
