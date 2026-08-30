from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stride_hierarchical_ch_compact_flow_expanded_stage2_router_v1 import (
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT,
    run_expanded_stage2_router,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit support, then conditionally fit the expanded-train Stage2 router"
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    report = run_expanded_stage2_router(
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
            "support_audit": report["support_audit"],
            "support_gate_passed_before_model_fit": report[
                "support_gate_passed_before_model_fit"
            ],
            "model_fit_executed": report["model_fit_executed"],
            "selected_model_id": report["selected_model_id"],
            "hard_gates_passed": report["hard_gates_passed"],
            "model_exported": report["model_exported"],
            "next_decision": report["next_decision"],
        }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if report["scientific_status"] == "TRAIN_OOF_CHALLENGER_ONLY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
