from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments._common import resolve_cli_path  # noqa: E402
from experiments.repair_collection import _read_json, _utc_now, _write_json  # noqa: E402
from experiments.v3_s3_collection import revalidate_v3_s3_collection  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild v3-S3 manifests and revalidate an existing collection "
            "without rerunning solver episodes."
        )
    )
    parser.add_argument("--collection", required=True)
    arguments = parser.parse_args()

    collection = resolve_cli_path(PROJECT_ROOT, arguments.collection)
    report = revalidate_v3_s3_collection(collection)
    pipeline_root = collection.parent
    summary = {
        "schema": "lns2.v3_s3_revalidation_summary.v1",
        "collection": str(collection),
        "complete": bool(report["complete"]),
        "requested_state_count": int(report["requested_state_count"]),
        "completed_state_count": int(report["completed_state_count"]),
        "error_state_count": int(report["error_state_count"]),
        "coverage_passed": bool(dict(report["coverage"])["passed"]),
        "coverage_error_count": int(dict(report["coverage"])["error_count"]),
        "strict_retest_passed": bool(dict(report["strict_retest"])["passed"]),
        "manifest_counts": dict(report["manifest_counts"]),
        "revalidation": dict(report["revalidation"]),
    }
    _write_json(pipeline_root / "collection_revalidation_report.json", summary)

    status_path = pipeline_root / "status.json"
    if status_path.is_file():
        status = _read_json(status_path)
        _write_json(
            status_path,
            {
                **status,
                "updated_at": _utc_now(),
                "status": "running" if report["complete"] else "error",
                "phase": (
                    "awaiting-windows-training"
                    if report["complete"]
                    else "collection-revalidation-failed"
                ),
                "completed_states": int(report["completed_state_count"]),
                "total_states": int(report["requested_state_count"]),
                "error_states": int(report["error_state_count"]),
                "revalidation_report": str(
                    pipeline_root / "collection_revalidation_report.json"
                ),
            },
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
