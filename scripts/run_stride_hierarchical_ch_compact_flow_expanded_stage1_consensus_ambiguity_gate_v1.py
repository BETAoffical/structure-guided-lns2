from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stride_hierarchical_ch_compact_flow_expanded_stage1_consensus_ambiguity_gate_v1 import (
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT,
    run_consensus_ambiguity_gate,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Nested same-cohort post-hoc control for the consensus ambiguity gate"
        )
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    report = run_consensus_ambiguity_gate(
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
            "study_role": report["study_role"],
            "scientific_status": report["scientific_status"],
            "protocol_valid": report["protocol_valid"],
            "hard_gates_passed": report["hard_gates_passed"],
            "primary_full_pooled_selected_action_metrics": report[
                "primary_full_pooled_selected_action_metrics"
            ],
            "paired_reference_audit": report["predecessor_control_reference"][
                "paired_audit"
            ],
            "final_gate_checks": report["final_gate_checks"],
            "challenger_claimed": report["challenger_claimed"],
            "next_decision": report["next_decision"],
        }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if report["hard_gates_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
