from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stride_seed25_same_set_pp_gcbs_screen import (  # noqa: E402
    build_plan,
    run_screen,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen seed25 same-set PP versus GCBS one-step diagnostic."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("plan", "run"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--config", required=True)
        if command == "run":
            subparser.add_argument("--output", required=True)
            subparser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.command == "plan":
        result = build_plan(args.config)
    else:
        result = run_screen(args.config, args.output, dry_run=bool(args.dry_run))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
