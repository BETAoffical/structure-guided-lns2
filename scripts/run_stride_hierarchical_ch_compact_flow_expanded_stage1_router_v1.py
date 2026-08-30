from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stride_hierarchical_ch_compact_flow_expanded_stage1_router_v1 import (
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT,
    run_expanded_stage1_router,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fit/evaluate the expanded 16-map Stage1-only abstaining router"
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    report = run_expanded_stage1_router(
        args.config,
        args.output,
        workers=args.workers,
        dry_run=args.dry_run,
    )
    payload = report
    if args.summary_only:
        selected = next(
            row
            for row in report["candidate_models"]
            if row["model_id"] == report["selected_model_id"]
        )
        payload = {
            "schema": report["schema"],
            "scientific_status": report["scientific_status"],
            "selected_model_id": report["selected_model_id"],
            "selected_threshold": report["selected_threshold"],
            "hard_gates_passed": report["hard_gates_passed"],
            "oof_metrics": selected["oof_metrics"],
            "hard_gate_checks": selected["hard_gate_checks"],
            "model_exported": report["model_exported"],
            "development_rows_loaded": False,
            "promotion_or_default_exported": False,
            "final_claim_authorized": False,
            "next_decision": report["next_decision"],
        }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if report["hard_gates_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
