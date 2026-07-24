from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NATIVE_BUILD = PROJECT_ROOT / "build" / "linux" / "project"
sys.path.insert(0, str(PROJECT_ROOT))
if NATIVE_BUILD.is_dir():
    sys.path.insert(0, str(NATIVE_BUILD))

from experiments.parallel_runtime import isolated_lane_cpu_sets  # noqa: E402
from experiments.repair_collection import _read_json  # noqa: E402
from experiments.v3_s3 import balanced_sequence_templates  # noqa: E402
from experiments.v3_s3_collection import (  # noqa: E402
    _qualification_job,
    _sequence_trial,
    _worker_initialize,
    assign_source_strata,
    source_decisions,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay one existing state through v3-S3 qualification and H3 repair."
    )
    parser.add_argument("--source", required=True)
    parser.add_argument(
        "--controller-bundle",
        default="artifacts/initlns-closed-loop-controller-v2",
    )
    parser.add_argument("--minimum-agents", type=int, default=400)
    arguments = parser.parse_args()
    source = Path(arguments.source)
    if not source.is_absolute():
        source = PROJECT_ROOT / source
    controller = Path(arguments.controller_bundle)
    if not controller.is_absolute():
        controller = PROJECT_ROOT / controller
    rows = source_decisions({"policy_train": [source.resolve()], "policy_validation": []})
    assign_source_strata(rows)
    _worker_initialize(
        tuple(isolated_lane_cpu_sets(1)), str(controller.resolve())
    )
    with tempfile.TemporaryDirectory() as directory:
        selected = None
        result = None
        rejected = []
        for index, source_row in enumerate(
            row
            for row in rows
            if int(row["agent_count"]) >= int(arguments.minimum_agents)
        ):
            candidate = dict(source_row)
            candidate["state_id"] = f"v3-s3-runtime-smoke-{index:04d}"
            observed = _qualification_job(
                {
                    "decision": candidate,
                    "state_id": candidate["state_id"],
                    "state_file": str(
                        Path(directory) / f"qualification-{index:04d}.json"
                    ),
                    "run_fingerprint": "v3-s3-runtime-smoke",
                    "resume": False,
                }
            )
            if str(observed.get("status")) in {"ok", "resumed"}:
                selected = candidate
                result = observed
                break
            rejected.append(
                {
                    "state_id": candidate["state_id"],
                    "reason": str(observed.get("rejection_reason")),
                }
            )
        if selected is None or result is None:
            raise RuntimeError(
                f"no replayable v3-S3 smoke state; rejected={rejected}"
            )
        payload = _read_json(Path(result["state_file"]))
        templates = balanced_sequence_templates(selected["state_id"])[0]
        trial = _sequence_trial(payload, templates, 0)
    report = {
        "state_id": selected["state_id"],
        "agent_count": int(selected["agent_count"]),
        "rejected_state_count": len(rejected),
        "template_count": len(payload["template_indices"]),
        "executed_steps": int(trial["executed_steps"]),
        "conflict_trajectory": trial["conflict_trajectory"],
        "total_seconds": float(trial["total_seconds"]),
        "pp_replan_seconds": float(trial["pp_replan_seconds"]),
        "passed": len(payload["template_indices"]) == 18
        and int(trial["executed_steps"]) >= 1,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
