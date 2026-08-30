from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.stride_hierarchical_ch_compact_flow_tasks_v1 import (  # noqa: E402
    DEFAULT_CONFIG,
    materialize_compact_flow_tasks,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize the exact hash-registered 192-task compact-flow train/dev "
            "dataset without running a solver."
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--output",
        default=str(
            PROJECT_ROOT / "build" / "stride-hierarchical-ch-compact-flow-source-v1"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.dry_run and args.resume:
        parser.error("--dry-run and --resume are mutually exclusive")
    report = materialize_compact_flow_tasks(
        config_path=args.config,
        output_root=args.output,
        dry_run=args.dry_run,
        resume=args.resume,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
