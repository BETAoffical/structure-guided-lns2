from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.closed_loop_trace_conversion import (  # noqa: E402
    convert_closed_loop_collection,
)
from experiments.repair_collection import _write_json  # noqa: E402
from scripts.verify_closed_loop_equivalence import compare_collections  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert full-v1 closed-loop traces to delta-gzip-v2 without rerunning models."
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--minimum-storage-reduction", type=float, default=0.9)
    parser.add_argument("--skip-equivalence", action="store_true")
    arguments = parser.parse_args()
    report = convert_closed_loop_collection(
        arguments.source,
        arguments.output,
        resume=arguments.resume,
        minimum_storage_reduction=arguments.minimum_storage_reduction,
    )
    if report["passed"] and not arguments.skip_equivalence:
        equivalence = compare_collections(
            arguments.source,
            arguments.output,
            progress_path=Path(arguments.output).resolve() / "equivalence_progress.json",
        )
        equivalence["minimum_storage_reduction"] = arguments.minimum_storage_reduction
        equivalence["storage_target_passed"] = (
            float(equivalence["storage"]["reduction_fraction"])
            >= arguments.minimum_storage_reduction
        )
        equivalence["passed"] = bool(
            equivalence["exact"] and equivalence["storage_target_passed"]
        )
        output_root = Path(arguments.output).resolve()
        _write_json(output_root / "equivalence_report.json", equivalence)
        _write_json(
            output_root / "cleanup_candidates.json",
            {
                "schema": "lns2.closed_loop_cleanup_candidates.v1",
                "source_episodes": str(
                    Path(arguments.source).resolve() / "episodes"
                ),
                "candidate_bytes": int(
                    equivalence["storage"]["reference_trace_bytes"]
                ),
                "compact_collection": str(output_root),
                "equivalence_passed": bool(equivalence["passed"]),
                "eligible_for_explicit_deletion": bool(equivalence["passed"]),
                "deletion_authorized": False,
                "requires_explicit_user_approval": True,
            },
        )
        report = {**report, "equivalence": equivalence, "passed": equivalence["passed"]}
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
