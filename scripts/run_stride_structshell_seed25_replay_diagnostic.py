from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stride_structshell_seed25_replay_diagnostic import (  # noqa: E402
    load_registration,
    run_diagnostic,
)


DEFAULT_CONFIG = ROOT / "configs" / "stride_structshell_seed25_replay_diagnostic_v1.json"
DEFAULT_OUTPUT = ROOT / "build" / "stride-structshell-seed25-replay-diagnostic-v1"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the registered seed-25 restored-state short-horizon mechanism "
            "diagnostic. This is not a TTF experiment."
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        result = run_diagnostic(args.config, args.output, dry_run=True)
    else:
        if not hasattr(signal, "SIGALRM"):
            raise RuntimeError("bounded replay execution requires the registered WSL runtime")
        _path, _root, registration = load_registration(args.config)
        timeout = float(registration["execution"]["total_process_time_limit_seconds"])

        def expired(_signum: int, _frame: object) -> None:
            raise TimeoutError(f"replay diagnostic exceeded its {timeout:g}s process fuse")

        previous = signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, timeout)
        try:
            result = run_diagnostic(
                args.config,
                args.output,
                resume=bool(args.resume),
            )
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, previous)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
