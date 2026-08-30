from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stride_hierarchical_ch_compact_flow_expanded_stage1_two_head_v1 import (
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT,
    run_expanded_stage1_two_head,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Nested 4x3-map OOF evaluation of the expanded Stage1 two-head router"
        )
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    report = run_expanded_stage1_two_head(
        args.config,
        args.output,
        workers=args.workers,
        dry_run=args.dry_run,
    )
    payload = report
    if args.summary_only:
        payload = {
            "schema": report["schema"],
            "scientific_status": report["scientific_status"],
            "hard_gates_passed": report["hard_gates_passed"],
            "nested_cv_prerequisite_checks": report[
                "nested_cv_prerequisite_checks"
            ],
            "primary_state_level_final_action_metrics": report[
                "primary_state_level_final_action_metrics"
            ],
            "secondary_pair_state_weighted_metrics": report[
                "secondary_pair_state_weighted_metrics"
            ],
            "hard_gate_checks": report["hard_gate_checks"],
            "model_exported": report["model_exported"],
            "v2_replaced": report["v2_replaced"],
            "next_decision": report["next_decision"],
        }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if report["hard_gates_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
