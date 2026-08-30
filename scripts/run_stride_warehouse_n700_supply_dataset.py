from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stride_warehouse_n700_supply_dataset import (  # noqa: E402
    plan,
    prepare_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the geometry-only four-task w1020a N700 supply dataset. "
            "This command never resets an environment or invokes a controller."
        )
    )
    parser.add_argument("command", choices=("plan", "prepare-q0"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = Path(args.config)
    if args.command == "plan":
        if args.output or args.dry_run:
            parser.error("plan does not accept --output/--dry-run")
        result = plan(config)
    else:
        if not args.output:
            parser.error("--output is required for prepare-q0")
        if args.dry_run:
            result = {
                **plan(config),
                "command": "prepare-q0",
                "dry_run": True,
                "output": str(Path(args.output).resolve()),
                "output_written": False,
            }
        else:
            result = prepare_dataset(config, Path(args.output))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
