"""Small synthetic TTF data; no solver or saved experiment outputs are run."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from experiments._common import episode_id, json_fingerprint
from experiments.closed_loop_trace_storage import storage_fingerprint, trace_file_metadata
from experiments.repair_collection import state_fingerprint
from lns2_selector.runtime.fingerprints import repair_structure_fingerprint


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def synthetic_summary(
    checkpoint: dict[str, Any], *, success: bool = True, capped: float = 10.0,
    observed: float = 30.0, budget: float = 120.0, timeout: bool = False,
) -> dict[str, Any]:
    observed = max(observed, budget) if timeout else observed
    return {
        "initial_fingerprint": checkpoint["expected_fingerprint"],
        "initial_conflicts": checkpoint["expected_conflicts"],
        "final_conflicts": 0 if success else checkpoint["expected_conflicts"],
        "success": success,
        "truncated": not success,
        "external_timeout": timeout,
        "stop_reason": "success" if success else "wall_timeout" if timeout else "controller_stalled",
        "capped_wall_time_to_feasible": capped if success else budget,
        "wall_time_to_feasible": capped if success else None,
        "ttf_observed_wall_seconds": observed,
        "episode_observed_wall_seconds": observed,
        "reset_wall_seconds": 0.1,
        "initial_state_elapsed_seconds": 0.1,
        "repair_wall_seconds": 0.2,
        "repair_iterations": 1,
        "transition_elapsed_seconds": [capped if success else observed],
        "ttf_clock_schema": "lns2.ttf.reset_inclusive_wall.v1",
        "wall_time_budget_seconds": budget,
        "invalid_action_count": 0,
        "fingerprint_mismatch_count": 0,
    }


def mocked_validated_lane(lane: Path, manifest_name: str, **_kwargs: Any) -> dict[str, Any] | None:
    """Bypass artifact I/O validation only in high-level report gate tests."""
    path = lane / manifest_name
    if not path.is_file():
        return None
    row = json.loads(path.read_text(encoding="utf-8").strip())
    return {
        "manifest": row, "summary": row["summary"], "producer_fingerprint": "a" * 64,
        "dataset_fingerprint": "d" * 64,
        "evidence": {"run_config_sha256": "b" * 64, "trace_sha256": "c" * 64},
    }


def write_synthetic_lane(lane: Path, *, success: bool = False) -> dict[str, Any]:
    """Write a real full-v1 terminal trace with authenticated file metadata."""
    state = {
        "initialized": True, "initial_solution_complete": True, "feasible": False,
        "done": True, "iteration": 0, "rows": 1, "cols": 3,
        "sum_of_costs": 4, "num_of_colliding_pairs": 1, "low_level": {},
        "obstacles": [0, 0, 0], "conflict_edges": [[0, 1]],
        "agents": [{"id": 0, "path": [0, 1, 2]}, {"id": 1, "path": [2, 1, 0]}],
    }
    if success:
        state.update({"feasible": True, "sum_of_costs": 0, "num_of_colliding_pairs": 0,
                      "conflict_edges": [], "agents": [{"id": 0, "path": [0]}, {"id": 1, "path": [2]}]})
    checkpoint = {
        "checkpoint_id": "checkpoint-0", "checkpoint_identity_sha256": "d" * 64,
        "task_id": "task-0", "map_id": "map-0", "screen_solver_seed": 7,
        "agent_count": 2, "source_kind": "checkpoint_blob_v1",
        "state_blob": "states/source.json.gz", "state_blob_sha256": "e" * 64,
        "expected_fingerprint": state_fingerprint(state), "expected_conflicts": state["num_of_colliding_pairs"],
        "repair_structure_fingerprint": repair_structure_fingerprint(state),
    }
    item = {"task_id": "task-0", "map_id": "map-0", "solver_seed": 7,
            "controller": "official_adaptive", "checkpoint_id": "checkpoint-0",
            "checkpoint_identity_sha256": checkpoint["checkpoint_identity_sha256"]}
    override = {"schema": "lns2.test.checkpoint_override.v1", "state_id": "checkpoint-0",
                "initial_restore": {**checkpoint, "collection_root": str(lane.parent)}}
    configuration = {
        "workers": 1, "task_ids_override": ["task-0"],
        "cohort_job_keys_override": [["task-0", 7]], "controller": "official_adaptive",
        "proposal": {"hybridstructpool": None, "neighborhood_sizes": [4, 8, 16]},
        "environment": {"replan_algorithm": "PP", "destroy_strategy": "Adaptive", "time_limit": 60.0, "max_repair_iterations": 0},
        "wall_time_budget_seconds": 60.0,
        "max_decisions": 0, "metric_iteration_budget": 100, "stopping_rule": "wall-clock",
        "policies": ["official_adaptive"],
        "solver": {"replan_algorithm": "PP"},
        "episode_override_fingerprints": {"task-0::7": json_fingerprint(override)},
    }
    implementation = {"files": {"fixture.py": "f" * 64}, "native_module": {
        "path": "lns2_env.so", "sha256": "1" * 64,
        "repair_timing_schema": "lns2.repair_timing.v2",
        "native_semantics_schema": "lns2.native_semantics.official_step_timed_extension.v3",
    }}
    implementation["sha256"] = json_fingerprint(implementation)
    run = {
        "schema": "lns2.closed_loop_confirmation.v1", "schema_version": 1,
        "dataset_fingerprint": "2" * 64, "configuration": configuration,
        "configuration_fingerprint": json_fingerprint(configuration),
        "frozen_models": {}, "controller_bundle": {}, "v3_s3_bundle": None,
        "controller_implementation": implementation, "controller": "official_adaptive",
        "trace_format": "full-v1", "storage_fingerprint": storage_fingerprint("full-v1"),
    }
    refresh_run_fingerprint(run)
    identifier = episode_id(item, 7, "official_adaptive")
    summary = synthetic_summary(checkpoint, success=success, capped=0.1, budget=60.0, observed=0.2)
    summary.update({"stop_reason": "success" if success else "native_terminal", "repair_wall_seconds": 0.0,
                    "repair_iterations": 0, "controller_mode": "official_adaptive",
                    "conflict_trajectory": [state["num_of_colliding_pairs"]], "transition_elapsed_seconds": [],
                    "conflict_auc": 0.0, "fixed_budget_conflict_auc": 0.0 if success else 100.0,
                    "normalized_fixed_budget_conflict_auc": 0.0 if success else 1.0,
                    "wall_clock_conflict_auc": 0.0 if success else 60.0, "final_sum_of_costs": state["sum_of_costs"],
                    "final_low_level": {}})
    metadata = {"schema": "lns2.closed_loop_episode.v1", "schema_version": 1,
                "run_fingerprint": run["run_fingerprint"], "episode_id": identifier,
                "policy": "official_adaptive", "solver_seed": 7}
    initial = {**metadata, "event": "initial", "state": state,
        "state_fingerprint": checkpoint["expected_fingerprint"], "episode_override": {
            "schema": override["schema"], "state_id": "checkpoint-0",
            "source_kind": "checkpoint_blob_v1", "source_checkpoint_id": "checkpoint-0",
            "source_checkpoint_identity_sha256": checkpoint["checkpoint_identity_sha256"],
            "source_full_fingerprint": checkpoint["expected_fingerprint"],
            "source_repair_fingerprint": checkpoint["repair_structure_fingerprint"],
            "source_checkpoint_file": str(lane.parent / checkpoint["state_blob"]),
            "source_trace_file": None, "source_decision_index": None, "forced_first_action": False}}
    finish = {**metadata, "event": "finish", "success": success,
              "final_fingerprint": checkpoint["expected_fingerprint"], "summary": copy.deepcopy(summary)}
    trace = lane / "traces" / "episode.jsonl"
    write_json(lane / "run_config.json", run)
    write_jsonl(trace, [initial, finish])
    manifest = {"schema": run["schema"], "schema_version": 1,
        "task_id": "task-0", "map_id": "map-0", "solver_seed": 7, "agent_count": 2,
        "policy": "official_adaptive", "episode_id": identifier, "status": "ok",
        "trace_file": "traces/episode.jsonl", "trace_format": "full-v1",
        "storage_fingerprint": storage_fingerprint("full-v1"),
        "trace_event_count": 2, "initial_state_ref": None, "summary": summary,
        **trace_file_metadata(trace)}
    manifest_name = "official_adaptive_manifest.jsonl"
    write_jsonl(lane / manifest_name, [manifest])
    return {"lane": lane, "manifest_name": manifest_name, "item": item,
            "checkpoint": checkpoint, "expected_policy": "official_adaptive",
            "expected_controller": "official_adaptive", "expected_augmentation": None,
            "wall_time_budget_seconds": 60.0, "expected_override": override,
            "expected_runtime": None,
            "expected_configuration": copy.deepcopy(configuration),
            "expected_frozen_models": copy.deepcopy(run["frozen_models"]),
            "expected_controller_bundle": copy.deepcopy(run["controller_bundle"])}


def refresh_run_fingerprint(run: dict[str, Any]) -> None:
    run["configuration_fingerprint"] = json_fingerprint(run["configuration"])
    run["run_fingerprint"] = json_fingerprint({
        "dataset_fingerprint": run["dataset_fingerprint"],
        "configuration_fingerprint": run["configuration_fingerprint"],
        "freeze_manifest": run["frozen_models"], "controller_bundle_manifest": run["controller_bundle"],
        "v3_s3_bundle_manifest": run["v3_s3_bundle"],
        "controller_implementation": run["controller_implementation"],
    })
