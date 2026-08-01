from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from experiments._common import producer_identity, sha256_file
from experiments.repair_collection import (
    _fingerprint,
    _read_json,
    _write_json,
    _write_jsonl,
)
from experiments.stride_collection import (
    FROZEN_FEATURE_DIMENSION,
    FROZEN_FEATURE_SCHEMA_ID,
    FULL_POOL_PROPOSAL,
    PP_TRIAL_INDICES,
    STRIDE_COLLECTION_PRODUCER_FILES,
    STRIDE_COLLECTION_SCHEMA,
    STRIDE_SELECTION_SCHEMA,
    _state_artifact_valid,
    load_stride_selection,
)


STRIDE_REUSE_SCHEMA = "lns2.stride.collection_reuse.v1"


def _collection_identity(selection_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    selected = load_stride_selection(selection_path)
    project_root = Path(__file__).resolve().parents[1]
    producer = producer_identity(
        project_root=project_root,
        source_files=STRIDE_COLLECTION_PRODUCER_FILES,
        native_required=True,
        package_names=("numpy",),
    )
    source_run_configs = {
        str(Path(str(row["source_root"])).resolve()): sha256_file(
            Path(str(row["source_root"])).resolve() / "run_config.json"
        )
        for row in selected
    }
    identity = {
        "schema": STRIDE_COLLECTION_SCHEMA,
        "schema_version": 1,
        "selection_path": str(selection_path.resolve()),
        "selection_sha256": sha256_file(selection_path),
        "selected_state_ids": [str(row["state_id"]) for row in selected],
        "feature_schema_id": FROZEN_FEATURE_SCHEMA_ID,
        "feature_dimension": FROZEN_FEATURE_DIMENSION,
        "proposal": FULL_POOL_PROPOSAL,
        "source_run_config_sha256": dict(sorted(source_run_configs.items())),
        "pp_trial_indices": list(PP_TRIAL_INDICES),
        "producer": producer,
    }
    return selected, identity, _fingerprint(identity)


def reuse_stride_collection(
    *, selection_path: Path, source: Path, output: Path, preflight_report: Path
) -> dict[str, Any]:
    """Adapt cohort-independent complete state artifacts to a new cohort identity."""

    preflight = _read_json(preflight_report)
    if preflight.get("passed") is not True or int(preflight.get("failed_state_count", -1)) != 0:
        raise ValueError("STRIDE reuse requires a passing zero-failure preflight")
    if str(Path(str(preflight["selection"])).resolve()) != str(selection_path.resolve()):
        raise ValueError("STRIDE preflight belongs to another selection")
    selected, identity, run_fingerprint = _collection_identity(selection_path)
    output = output.resolve()
    if output.exists():
        raise ValueError("STRIDE reuse output already exists")
    output.mkdir(parents=True)
    _write_json(output / "run_config.json", {**identity, "run_fingerprint": run_fingerprint})
    _write_jsonl(
        output / "state_selection.jsonl",
        [{**row, "schema": STRIDE_SELECTION_SCHEMA} for row in selected],
    )
    selected_by_id = {str(row["state_id"]): row for row in selected}
    source_run = _read_json(source / "run_config.json")
    source_fingerprint = str(source_run["run_fingerprint"])
    reused: list[dict[str, Any]] = []
    for source_file in sorted((source / "states").glob("*.json")):
        payload = _read_json(source_file)
        state_id = str(payload.get("state_id", ""))
        decision = selected_by_id.get(state_id)
        if decision is None:
            continue
        if not _state_artifact_valid(
            payload, run_fingerprint=source_fingerprint, state_id=state_id
        ):
            raise ValueError(f"STRIDE source state artifact is invalid: {source_file}")
        stored_decision = payload.get("decision")
        if not isinstance(stored_decision, dict):
            raise ValueError(f"STRIDE source state has no decision: {state_id}")
        for field in (
            "state_id",
            "task_id",
            "source_policy",
            "solver_seed",
            "decision_index",
            "before_fingerprint",
        ):
            if stored_decision.get(field) != decision.get(field):
                raise ValueError(f"STRIDE reused state differs on {field}: {state_id}")
        target_key = _fingerprint(
            {"state_id": state_id, "before": decision["before_fingerprint"]}
        )[:20]
        target_file = output / "states" / f"{target_key}.json"
        target_file.parent.mkdir(parents=True, exist_ok=True)
        adapted = {**payload, "run_fingerprint": run_fingerprint, "decision": decision}
        partial = target_file.with_name(target_file.name + ".partial")
        _write_json(partial, adapted)
        os.replace(partial, target_file)
        if not _state_artifact_valid(
            _read_json(target_file), run_fingerprint=run_fingerprint, state_id=state_id
        ):
            raise RuntimeError(f"STRIDE adapted state failed validation: {state_id}")
        reused.append(
            {
                "state_id": state_id,
                "source_file": str(source_file.resolve()),
                "source_sha256": sha256_file(source_file),
                "target_file": str(target_file.resolve()),
                "target_sha256": sha256_file(target_file),
                "candidate_count": len(payload["candidates"]),
                "trial_count": len(payload["trials"]),
            }
        )
    report = {
        "schema": STRIDE_REUSE_SCHEMA,
        "schema_version": 1,
        "source": str(source.resolve()),
        "source_run_fingerprint": source_fingerprint,
        "target": str(output),
        "target_run_fingerprint": run_fingerprint,
        "selection_sha256": identity["selection_sha256"],
        "preflight_report": str(preflight_report.resolve()),
        "preflight_sha256": sha256_file(preflight_report),
        "selected_state_count": len(selected),
        "reused_state_count": len(reused),
        "pending_state_count": len(selected) - len(reused),
        "reused_trial_count": sum(int(row["trial_count"]) for row in reused),
        "states": reused,
    }
    _write_json(output / "reuse_report.json", report)
    return report
