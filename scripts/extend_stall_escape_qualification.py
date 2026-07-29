from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NATIVE_BUILD = PROJECT_ROOT / "build" / "linux" / "project"
sys.path.insert(0, str(PROJECT_ROOT))
if NATIVE_BUILD.is_dir():
    sys.path.insert(0, str(NATIVE_BUILD))

from experiments.stall_escape_continuation import (  # noqa: E402
    run_stall_escape_continuation,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Extend only unresolved v2 qualification episodes and require an "
            "exact common prefix before resolving the future observation window."
        )
    )
    parser.add_argument("--qualification", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--future-decisions", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    report = run_stall_escape_continuation(
        arguments.qualification,
        arguments.output,
        future_observation_decisions=arguments.future_decisions,
        workers=arguments.workers,
        resume=arguments.resume,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
