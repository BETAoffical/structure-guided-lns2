from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stride_safeslot_pair_motif_sequential_ablation import plan, run_ablation


DEFAULT_CONFIG = "configs/stride_safeslot_pair_motif_sequential_ablation_v1.json"
DEFAULT_OUTPUT = "build/stride-safeslot-pair-motif-sequential-ablation-v1"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the solver-free SafeSlot pair-motif sequential ablation"
    )
    parser.add_argument("command", choices=("plan", "run"), nargs="?", default="plan")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.command == "plan":
        if args.dry_run:
            parser.error("plan is already non-executing; --dry-run is only valid with run")
        payload = plan(config_path=args.config)
    else:
        payload = run_ablation(
            config_path=args.config,
            output=args.output,
            dry_run=bool(args.dry_run),
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.command == "run" and not args.dry_run:
        return 0 if bool(payload["offline_passed"]) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
