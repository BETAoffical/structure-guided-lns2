from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stride_fresh_matched_v2_c16_h16_v1 import (  # noqa: E402
    build_plan,
    run_h1,
    run_preflight,
    run_source_collection,
)


DEFAULT_CONFIG = ROOT / "configs" / "stride_fresh_matched_v2_c16_h16_v1.json"
DEFAULT_OUTPUT = ROOT / "build" / "stride-fresh-matched-v2-c16-h16-v1"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the fresh exact-task/solver/state V2 versus C16/H16 matched "
            "H=1 opportunity protocol. H=8 and training are not executed."
        )
    )
    parser.add_argument("phase", choices=("plan", "source", "preflight", "h1"))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    if arguments.phase == "plan":
        if arguments.resume or arguments.dry_run:
            parser.error("plan does not accept --resume or --dry-run")
        report = build_plan(arguments.config)
    elif arguments.phase == "source":
        report = run_source_collection(
            arguments.config,
            arguments.output,
            resume=bool(arguments.resume),
            dry_run=bool(arguments.dry_run),
        )
    elif arguments.phase == "preflight":
        report = run_preflight(
            arguments.config,
            arguments.output,
            resume=bool(arguments.resume),
            dry_run=bool(arguments.dry_run),
        )
    else:
        report = run_h1(
            arguments.config,
            arguments.output,
            resume=bool(arguments.resume),
            dry_run=bool(arguments.dry_run),
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
