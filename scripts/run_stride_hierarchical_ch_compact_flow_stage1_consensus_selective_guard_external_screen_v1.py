from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stride_hierarchical_ch_compact_flow_stage1_consensus_selective_guard_external_screen_v1 import (
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT,
    run_external_screen,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the retrospective fixed Stage1 consensus guard development screen"
        )
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    report = run_external_screen(
        args.config,
        args.output,
        workers=args.workers,
        dry_run=args.dry_run,
    )
    payload = report
    if args.summary_only:
        payload = {
            "schema": report["schema"],
            "experiment_id": report["experiment_id"],
            "scientific_status": report["scientific_status"],
            "screen_status": report["screen_status"],
            "hard_gates_passed": report["hard_gates_passed"],
            "insufficient_support": report["insufficient_support"],
            "primary_selected_action_metrics": report[
                "primary_selected_action_metrics"
            ],
            "gate_checks": report["gate_checks"],
            "model_exported": report["model_exported"],
            "stage2_loaded": report["stage2_loaded"],
            "prospective_or_sealed_final_confirmation": report[
                "prospective_or_sealed_final_confirmation"
            ],
        }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if report["hard_gates_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
