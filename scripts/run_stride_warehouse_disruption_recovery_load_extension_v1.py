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

from experiments.stride_warehouse_disruption_recovery_load_extension_v1 import (  # noqa: E402
    analyze_checkpoints, analyze_ttf, collect_ttf, dry_run, generate_fresh_dataset,
    plan, plan_ttf, prepare_checkpoints,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fresh-map three-band Warehouse disruption recovery extension")
    parser.add_argument("command", choices=("dry-run", "generate-dataset", "plan", "prepare-checkpoints", "analyze-checkpoints", "plan-ttf", "collect-ttf", "analyze-ttf"))
    parser.add_argument("--config", required=True); parser.add_argument("--output"); parser.add_argument("--workers", type=int); parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(); config = Path(args.config)
    if args.command == "dry-run":
        if args.output or args.workers is not None or args.resume: parser.error("dry-run accepts only --config")
        result = dry_run(config)
    elif args.command == "generate-dataset":
        if args.workers is not None or args.resume: parser.error("generate-dataset does not accept --workers/--resume")
        result = generate_fresh_dataset(config, output=args.output)
    elif args.command == "plan":
        if args.output or args.workers is not None or args.resume: parser.error("plan accepts only --config")
        result = plan(config)
    else:
        if not args.output: parser.error(f"--output is required for {args.command}")
        output = Path(args.output)
        if args.command == "prepare-checkpoints":
            if args.workers is not None and args.workers <= 0: parser.error("--workers must be positive")
            result = prepare_checkpoints(config, output, workers=args.workers, resume=args.resume)
        elif args.command == "analyze-checkpoints":
            if args.workers is not None or args.resume: parser.error("analyze-checkpoints does not accept --workers/--resume")
            result = analyze_checkpoints(config, output)
        elif args.command == "plan-ttf":
            if args.workers is not None or args.resume: parser.error("plan-ttf does not accept --workers/--resume")
            result = plan_ttf(config, output)
        elif args.command == "collect-ttf":
            if args.workers is not None: parser.error("collect-ttf does not accept --workers")
            result = collect_ttf(config, output, resume=args.resume)
        else:
            if args.workers is not None or args.resume: parser.error("analyze-ttf does not accept --workers/--resume")
            result = analyze_ttf(config, output)
    print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
