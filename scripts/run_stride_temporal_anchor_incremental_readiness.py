from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stride_temporal_anchor_incremental_readiness import (
    run_temporal_anchor_incremental_readiness,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the solver-free temporal-anchor incremental readiness audit"
    )
    parser.add_argument(
        "--config",
        default="configs/stride_temporal_anchor_incremental_readiness_v1.json",
    )
    parser.add_argument(
        "--output",
        default="build/stride-temporal-anchor-incremental-readiness-v1",
    )
    parser.add_argument(
        "--summary-only", action="store_true",
        help="print only the frozen gate outcome; full evidence remains in the report",
    )
    args = parser.parse_args()
    report = run_temporal_anchor_incremental_readiness(
        config_path=args.config, output=args.output
    )
    payload = report if not args.summary_only else {
        "schema": report["schema"],
        "scientific_status": report["scientific_status"],
        "offline_passed": report["offline_passed"],
        "analysis_wall_seconds": report["analysis_wall_seconds"],
        "static_probability_reproduction_error": report["identity_static_baseline"]["probability_reproduction_maximum_absolute_error"],
        "static_history_roc_auc": report["fair_comparison_static_baseline"]["history_candidate_metrics"]["roc_auc"],
        "temporal_history_roc_auc": report["temporal_model"]["history_candidate_metrics"]["roc_auc"],
        "history_roc_auc_gain": report["temporal_model"]["history_roc_auc_gain_over_static"],
        "offline_checks": report["offline_checks"],
        "next_decision": report["next_decision"],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if report["offline_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
