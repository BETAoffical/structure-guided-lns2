#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
NATIVE_ROOT = PROJECT_ROOT / "build" / "linux" / "project"
if NATIVE_ROOT.is_dir() and str(NATIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(NATIVE_ROOT))

from experiments._common import resolve_cli_path  # noqa: E402
from experiments.receding_q_stability import (  # noqa: E402
    run_receding_q_stability_followup,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Add PP trials 2/3 only to receding-Q states whose trial-0 and "
            "trial-1 exact winners differed, then audit four-seed stability."
        )
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    report = run_receding_q_stability_followup(
        source=resolve_cli_path(PROJECT_ROOT, arguments.source),
        output=resolve_cli_path(PROJECT_ROOT, arguments.output),
        resume=arguments.resume,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
