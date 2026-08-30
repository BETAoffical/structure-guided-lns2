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

from experiments.mixed_full_v2_high_load_multiseed_v1 import (  # noqa: E402
    analyze,
    build_plan,
    run,
)


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the deterministic 54-key high-load Mixed/V2/Official TTF expansion."
    )
    parser.add_argument("phase", choices=("plan", "run", "analyze"))
    parser.add_argument(
        "--dataset",
        default="build/initlns-v2-mixed-qualified-compute-load-pool-v6",
    )
    parser.add_argument(
        "--config",
        default="configs/mixed_full_v2_high_load_multiseed_v1.json",
    )
    parser.add_argument(
        "--output",
        default="build/initlns-mixed-full-v2-high-load-multiseed-v1",
    )
    parser.add_argument(
        "--original-bundle", default="artifacts/initlns-closed-loop-controller-v2"
    )
    parser.add_argument(
        "--mixed-bundle", default="artifacts/initlns-mixed-full-controller-v2"
    )
    args = parser.parse_args()
    try:
        if args.phase == "plan":
            result = build_plan(
                _resolve(args.dataset),
                _resolve(args.config),
                _resolve(args.output),
                _resolve(args.original_bundle),
                _resolve(args.mixed_bundle),
            )
        elif args.phase == "run":
            result = run(
                _resolve(args.dataset),
                _resolve(args.config),
                _resolve(args.output),
                _resolve(args.original_bundle),
                _resolve(args.mixed_bundle),
            )
        else:
            result = analyze(_resolve(args.output))
    except (OSError, RuntimeError, ValueError) as error:
        print(json.dumps({"status": "error", "error": str(error)}), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
