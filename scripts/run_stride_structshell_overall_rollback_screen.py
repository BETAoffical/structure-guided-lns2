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

from experiments.stride_structshell_overall_rollback_screen import (  # noqa: E402
    analyze,
    run,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the preregistered ten-key StructShell state-bounded rollback "
            "mechanism and cost screen."
        )
    )
    parser.add_argument("command", choices=("run", "analyze"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config, output = Path(args.config), Path(args.output)
    if args.command == "run":
        result = run(config, output, resume=args.resume, dry_run=args.dry_run)
    else:
        if args.dry_run:
            parser.error("--dry-run is only valid with run")
        result = analyze(config, output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
