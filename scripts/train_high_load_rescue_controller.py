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

from experiments.high_load_rescue_training import (  # noqa: E402
    train_high_load_rescue_controller,
)
from experiments._common import resolve_cli_path  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train the 400/600-agent wall-clock rescue bundle."
    )
    parser.add_argument("--collection", required=True)
    parser.add_argument(
        "--controller-bundle",
        default="artifacts/initlns-closed-loop-controller-v2",
    )
    parser.add_argument(
        "--output", default="build/initlns-high-load-rescue-controller-v1"
    )
    arguments = parser.parse_args()
    collection = resolve_cli_path(PROJECT_ROOT, arguments.collection)
    report = train_high_load_rescue_controller(
        feature_index=collection / "feature_index.jsonl",
        trial_manifest=collection / "trial_manifest.jsonl",
        controller_bundle=resolve_cli_path(PROJECT_ROOT, arguments.controller_bundle),
        output=resolve_cli_path(PROJECT_ROOT, arguments.output),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
