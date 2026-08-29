from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
NATIVE = ROOT / "build" / "linux" / "project"
if NATIVE.is_dir() and str(NATIVE) not in sys.path:
    sys.path.insert(0, str(NATIVE))
from experiments.stride_dual16_room_maze_highload_extension_v1 import analyze, collect, plan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("plan", "dry-run", "collect", "analyze"))
    parser.add_argument("--config", required=True); parser.add_argument("--output"); parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command in {"plan", "dry-run"}:
        if args.output or args.resume: parser.error("plan/dry-run accept only --config")
        result = plan(args.config)
    else:
        if not args.output: parser.error("--output is required")
        result = collect(args.config, args.output, resume=args.resume) if args.command == "collect" else analyze(args.config, args.output)
    print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
