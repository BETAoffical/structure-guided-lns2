from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from experiments.closed_loop_trace_storage import trace_file_metadata
from lns2_selector.evaluation.ttf_artifacts import load_ttf_lane, validate_ttf_lane_cohort
from tests.evaluation.ttf_fixtures import (
    refresh_run_fingerprint, write_json, write_jsonl, write_synthetic_lane,
)


def _manifest(arguments: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = arguments["lane"] / arguments["manifest_name"]
    return path, json.loads(path.read_text(encoding="utf-8"))


def _rebind_run_and_trace(arguments: dict[str, Any], run: dict[str, Any]) -> None:
    refresh_run_fingerprint(run)
    write_json(arguments["lane"] / "run_config.json", run)
    manifest_path, row = _manifest(arguments)
    trace = arguments["lane"] / row["trace_file"]
    events = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    for event in events:
        event["run_fingerprint"] = run["run_fingerprint"]
    write_jsonl(trace, events)
    row.update(trace_file_metadata(trace))
    write_jsonl(manifest_path, [row])


def test_real_synthetic_trace_authenticates_without_current_source_hashes(tmp_path: Path) -> None:
    arguments = write_synthetic_lane(tmp_path / "lane")
    result = load_ttf_lane(**arguments)
    assert result is not None
    assert result["summary"]["stop_reason"] == "native_terminal"
    assert result["summary"]["capped_wall_time_to_feasible"] == 60.0
    assert result["manifest"]["trace_event_count"] == 2
    assert result["evidence"]["trace_sha256"] == result["manifest"]["trace_sha256"]
    assert len(result["producer_fingerprint"]) == 64


def test_missing_or_failed_lane_is_incomplete(tmp_path: Path) -> None:
    arguments = write_synthetic_lane(tmp_path / "lane")
    path, row = _manifest(arguments)
    row["status"] = "error"
    write_jsonl(path, [row])
    assert load_ttf_lane(**arguments) is None
    write_jsonl(path, [])
    assert load_ttf_lane(**arguments) is None
    arguments["manifest_name"] = "missing.jsonl"
    assert load_ttf_lane(**arguments) is None


@pytest.mark.parametrize("field", ["run_fingerprint", "configuration_fingerprint", "controller_implementation"])
def test_completed_lane_rejects_changed_fingerprint_and_preserves_file(tmp_path: Path, field: str) -> None:
    arguments = write_synthetic_lane(tmp_path / "lane")
    path = arguments["lane"] / "run_config.json"
    run = json.loads(path.read_text(encoding="utf-8"))
    if field == "controller_implementation":
        run[field]["native_module"]["sha256"] = "9" * 64
    else:
        run[field] = "9" * 64
    write_json(path, run)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="fingerprint"):
        load_ttf_lane(**arguments)
    assert path.read_bytes() == before


@pytest.mark.parametrize(("field", "value"), [
    ("controller", "v2-full"), ("workers", 2), ("workers", True),
    ("task_ids_override", ["another-task"]), ("cohort_job_keys_override", [["task-0", 8]]),
    ("wall_time_budget_seconds", 30.0), ("episode_override_fingerprints", {}),
    ("proposal", {"hybridstructpool": {"runtime_id": "wrong"}}),
    ("stopping_rule", "repair-limit"), ("policies", ["realized_dynamic"]),
])
def test_self_consistent_run_must_still_match_registered_lane(tmp_path: Path, field: str, value: Any) -> None:
    arguments = write_synthetic_lane(tmp_path / "lane")
    path = arguments["lane"] / "run_config.json"
    run = json.loads(path.read_text(encoding="utf-8"))
    run["configuration"][field] = value
    if field == "controller":
        run["controller"] = value
    refresh_run_fingerprint(run)
    write_json(path, run)
    with pytest.raises(ValueError):
        load_ttf_lane(**arguments)


@pytest.mark.parametrize(("field", "value"), [
    ("task_id", "other-task"), ("map_id", "other-map"), ("solver_seed", 8),
    ("solver_seed", True), ("policy", "realized_dynamic"), ("episode_id", "other-episode"),
    ("trace_sha256", "9" * 64), ("trace_bytes", 1), ("trace_event_count", 3),
    ("initial_state_ref", "other-blob"), ("trace_format", "delta-gzip-v2"),
    ("storage_fingerprint", "9" * 64),
])
def test_manifest_identity_and_metadata_must_match_trace(tmp_path: Path, field: str, value: Any) -> None:
    arguments = write_synthetic_lane(tmp_path / "lane")
    path, row = _manifest(arguments)
    row[field] = value
    write_jsonl(path, [row])
    with pytest.raises(ValueError):
        load_ttf_lane(**arguments)


@pytest.mark.parametrize("payload", [None, [], 7, [None], [{}, {}]])
def test_nonobject_or_duplicate_completed_artifacts_are_rejected(tmp_path: Path, payload: Any) -> None:
    arguments = write_synthetic_lane(tmp_path / "lane")
    path = arguments["lane"] / "run_config.json"
    write_json(path, payload)
    with pytest.raises(ValueError):
        load_ttf_lane(**arguments)


def test_duplicate_manifest_rows_are_rejected(tmp_path: Path) -> None:
    arguments = write_synthetic_lane(tmp_path / "lane")
    path, row = _manifest(arguments)
    write_jsonl(path, [row, row])
    with pytest.raises(ValueError, match="exactly one"):
        load_ttf_lane(**arguments)


@pytest.mark.parametrize("row", [None, [], 7, "completed"])
def test_manifest_row_must_be_an_object(tmp_path: Path, row: Any) -> None:
    arguments = write_synthetic_lane(tmp_path / "lane")
    write_json(arguments["lane"] / arguments["manifest_name"], row)
    with pytest.raises(ValueError, match="object"):
        load_ttf_lane(**arguments)


def test_registered_runtime_settings_are_checked(tmp_path: Path) -> None:
    arguments = write_synthetic_lane(tmp_path / "lane")
    arguments["expected_runtime"] = {"feature_backend": "native_v2"}
    with pytest.raises(ValueError, match="runtime feature_backend"):
        load_ttf_lane(**arguments)


@pytest.mark.parametrize("changed", ["frozen_models", "controller_bundle", "proposal", "environment"])
def test_rehashed_run_cannot_substitute_a_different_registered_model_or_algorithm(tmp_path: Path, changed: str) -> None:
    arguments = write_synthetic_lane(tmp_path / "lane")
    run = json.loads((arguments["lane"] / "run_config.json").read_text(encoding="utf-8"))
    if changed in {"frozen_models", "controller_bundle"}:
        run[changed] = {"model_sha256": "9" * 64}
    elif changed == "proposal":
        run["configuration"]["proposal"]["neighborhood_sizes"] = [32]
    else:
        run["configuration"]["environment"]["replan_algorithm"] = "GCBS"
    _rebind_run_and_trace(arguments, run)
    with pytest.raises(ValueError):
        load_ttf_lane(**arguments)


@pytest.mark.parametrize(("field", "value"), [
    ("success", "false"), ("external_timeout", 0), ("truncated", "true"),
    ("capped_wall_time_to_feasible", 1.0), ("wall_time_to_feasible", 0.1),
    ("repair_wall_seconds", -1.0), ("repair_wall_seconds", float("nan")),
    ("episode_observed_wall_seconds", float("inf")),
    ("stop_reason", "success"), ("final_conflicts", 0),
])
def test_manifest_summary_cannot_fabricate_valid_metrics(tmp_path: Path, field: str, value: Any) -> None:
    arguments = write_synthetic_lane(tmp_path / "lane")
    path, row = _manifest(arguments)
    row["summary"][field] = value
    write_jsonl(path, [row])
    with pytest.raises(ValueError):
        load_ttf_lane(**arguments)


def test_manifest_summary_must_equal_authenticated_trace_summary(tmp_path: Path) -> None:
    arguments = write_synthetic_lane(tmp_path / "lane")
    path, row = _manifest(arguments)
    row["summary"]["episode_observed_wall_seconds"] = 0.3
    write_jsonl(path, [row])
    with pytest.raises(ValueError, match="summary mismatch"):
        load_ttf_lane(**arguments)


def test_rehashing_both_summaries_cannot_lower_ttf_below_the_algorithm_clock(tmp_path: Path) -> None:
    arguments = write_synthetic_lane(tmp_path / "lane", success=True)
    assert load_ttf_lane(**arguments)["summary"]["wall_time_to_feasible"] == 0.1
    manifest_path, row = _manifest(arguments)
    trace = arguments["lane"] / row["trace_file"]
    events = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    for summary in (row["summary"], events[-1]["summary"]):
        summary["wall_time_to_feasible"] = 0.05
        summary["capped_wall_time_to_feasible"] = 0.05
    write_jsonl(trace, events)
    row.update(trace_file_metadata(trace))
    write_jsonl(manifest_path, [row])
    with pytest.raises(ValueError):
        load_ttf_lane(**arguments)


@pytest.mark.parametrize("field", ["source_checkpoint_identity_sha256", "source_checkpoint_id", "source_full_fingerprint", "source_repair_fingerprint", "state_id"])
def test_trace_provenance_remains_bound_after_rehashing(tmp_path: Path, field: str) -> None:
    arguments = write_synthetic_lane(tmp_path / "lane")
    path, row = _manifest(arguments)
    trace = arguments["lane"] / row["trace_file"]
    events = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    events[0]["episode_override"][field] = "unrelated-checkpoint"
    write_jsonl(trace, events)
    row.update(trace_file_metadata(trace))
    write_jsonl(path, [row])
    with pytest.raises(ValueError, match="trace .*mismatch"):
        load_ttf_lane(**arguments)


def test_trace_path_must_stay_inside_lane(tmp_path: Path) -> None:
    arguments = write_synthetic_lane(tmp_path / "lane")
    path, row = _manifest(arguments)
    row["trace_file"] = "../borrowed.jsonl"
    write_jsonl(path, [row])
    with pytest.raises(ValueError, match="contained"):
        load_ttf_lane(**arguments)


@pytest.mark.parametrize("field", ["producer_fingerprint", "dataset_fingerprint"])
def test_paired_lanes_cannot_mix_saved_execution_identity(tmp_path: Path, field: str) -> None:
    result = load_ttf_lane(**write_synthetic_lane(tmp_path / "lane"))
    assert result is not None
    validate_ttf_lane_cohort([result, copy.deepcopy(result)])
    changed = {**result, field: "9" * 64}
    with pytest.raises(ValueError, match="mixed lane"):
        validate_ttf_lane_cohort([result, changed])
