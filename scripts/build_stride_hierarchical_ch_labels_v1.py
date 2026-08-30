#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments._common import sha256_file  # noqa: E402
from experiments.stride_hierarchical_ch_labels_v1 import (  # noqa: E402
    CONFIG_SCHEMA,
    SOURCE_STATE_SCHEMA,
    build_hierarchical_ch_labels,
)


DEFAULT_MANIFEST = (
    "build/stride-fresh-matched-unique-action-hierarchical-overlay-v1/"
    "h1_state_manifest.jsonl"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the labels-only STRIDE hierarchical V2/Component/Hotspot "
            "three-state audit. No model is fitted or exported."
        )
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--source-state-schema", default=SOURCE_STATE_SCHEMA)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--experiment-id", default="stride_hierarchical_ch_labels_v1"
    )
    parser.add_argument("--expected-state-count", type=int, default=96)
    parser.add_argument("--expected-unique-action-count", type=int, default=256)
    parser.add_argument("--expected-trial-count", type=int, default=4096)
    parser.add_argument("--expected-stage1-pair-count", type=int, default=160)
    parser.add_argument("--expected-stage2-state-count", type=int, default=64)
    arguments = parser.parse_args()

    manifest = (PROJECT_ROOT / arguments.manifest).resolve()
    try:
        manifest.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise ValueError("--manifest must stay within the project root") from error
    if not manifest.is_file():
        raise ValueError(f"source manifest is missing: {manifest}")
    config = {
        "schema": CONFIG_SCHEMA,
        "experiment_id": arguments.experiment_id,
        "source": {
            "manifest": {
                "path": manifest.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": sha256_file(manifest),
            },
            "state_schema": arguments.source_state_schema,
        },
        "expected": {
            "state_count": arguments.expected_state_count,
            "unique_action_count": arguments.expected_unique_action_count,
            "trial_count": arguments.expected_trial_count,
            "stage1_pair_count": arguments.expected_stage1_pair_count,
            "stage2_state_count": arguments.expected_stage2_state_count,
        },
    }
    report = build_hierarchical_ch_labels(
        config,
        project_root=PROJECT_ROOT,
        output=arguments.output,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
