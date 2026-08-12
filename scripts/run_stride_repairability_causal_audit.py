from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stride_repairability_causal_audit import (  # noqa: E402
    analyze_causal_audit,
    collect_causal_audit,
    run_worker_preflight,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the preregistered STRIDE Repairability causal audit."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight-workers")
    preflight.add_argument("--config", required=True)
    preflight.add_argument("--output", required=True)
    collect = subparsers.add_parser("collect")
    collect.add_argument("--config", required=True)
    collect.add_argument("--preflight", required=True)
    collect.add_argument("--output", required=True)
    collect.add_argument("--resume", action="store_true")
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--config", required=True)
    analyze.add_argument("--collection", required=True)
    analyze.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "preflight-workers":
        result = run_worker_preflight(config_path=args.config, output=args.output)
    elif args.command == "collect":
        result = collect_causal_audit(
            config_path=args.config,
            preflight=args.preflight,
            output=args.output,
            resume=args.resume,
        )
    else:
        result = analyze_causal_audit(
            config_path=args.config,
            collection=args.collection,
            output=args.output,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

