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

from experiments.stride_structshell_rollback_aware_platform_replay import (  # noqa: E402
    analyze,
    run,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay two registered StructShell platforms with the exact rollback guard."
    )
    parser.add_argument("command", choices=("run", "analyze"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.command == "run":
        result = run(
            Path(args.config),
            Path(args.output),
            resume=args.resume,
            dry_run=args.dry_run,
        )
    else:
        result = analyze(Path(args.config), Path(args.output))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
