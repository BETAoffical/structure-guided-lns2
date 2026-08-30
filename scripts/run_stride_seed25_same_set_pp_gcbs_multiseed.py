from __future__ import annotations

import argparse
import json

from experiments.stride_seed25_same_set_pp_gcbs_multiseed import (
    build_plan,
    run_qualification,
    run_screen,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the bounded four-state PP/GCBS same-set multiseed screen."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--config", required=True)
    qualify_parser = subparsers.add_parser("qualify")
    qualify_parser.add_argument("--source-config", required=True)
    qualify_parser.add_argument("--output", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--output", required=True)
    run_parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.command == "plan":
        payload = build_plan(args.config)
    elif args.command == "qualify":
        payload = run_qualification(args.source_config, args.output)
    else:
        payload = run_screen(args.config, args.output, dry_run=bool(args.dry_run))
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
