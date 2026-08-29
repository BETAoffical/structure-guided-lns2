from __future__ import annotations

import collections
import json
from pathlib import Path
from typing import Any

import pytest

import experiments.repair_collection as repair_collection
import experiments.stride_warehouse_disruption_recovery_boundary as boundary
import experiments.stride_warehouse_disruption_recovery_boundary_ttf as boundary_ttf
from experiments._common import sha256_file
from experiments.warehouse_disruption_checkpoints import (
    compute_checkpoint_identity_sha256,
)


BANDS = {
    "balanced_od_d10": "medium_high",
    "balanced_od_d125": "high",
}


def _dataset_rows() -> list[dict[str, Any]]:
    return [
        {
            "map_id": f"boundary-map-{map_index}",
            "task_id": f"boundary-task-{map_index}-{variant}",
            "task_variant": variant,
            "split": "boundary_confirmation",
            "agent_count": 500 if variant == "balanced_od_d10" else 700,
        }
        for map_index in range(6)
        for variant in boundary.VARIANTS
    ]


def _config(tmp_path: Path) -> dict[str, Any]:
    return {
        "_dataset_root": str(tmp_path / "dataset"),
        "_prior_dataset_root": str(tmp_path / "prior-dataset"),
        "dataset": {"load_bands": BANDS},
        "checkpoint_generation": {
            "expected_candidate_count": 24,
            "minimum_qualified_count": 18,
            "maximum_qualified_count": 24,
            "minimum_qualified_map_count": 6,
            "minimum_qualified_per_load_band": 8,
            "minimum_qualified_per_map_band_cell": 1,
            "incumbent_solver_seed_base": 1000,
            "selection_seed_base": 2000,
            "gate": {},
        },
        "runtime": {"screen_solver_seed_base": 3000},
        "screen_gate": {
            "minimum_paired_win_rate": 0.70,
            "minimum_mean_capped_ttf_improvement": 0.15,
            "minimum_successes_per_hour_improvement": 0.20,
            "maximum_success_rate_loss": 0.0,
            "no_additional_timeout_or_censor": True,
        },
        "claim_boundary": "boundary-only",
    }


def _schedule(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    monkeypatch.setattr(boundary, "_load_dataset_rows", lambda *_args: _dataset_rows())
    return boundary._boundary_schedule(_config(tmp_path), tmp_path / "dataset")


def test_boundary_schedule_is_six_by_two_by_two_with_unique_job_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    schedule = _schedule(tmp_path, monkeypatch)

    assert len(schedule) == 24
    assert len({row["map_id"] for row in schedule}) == 6
    assert collections.Counter(row["load_band"] for row in schedule) == {
        "medium_high": 12,
        "high": 12,
    }
    assert len({(row["map_id"], row["load_band"]) for row in schedule}) == 12
    assert len({(row["task_id"], row["screen_solver_seed"]) for row in schedule}) == 24
    by_task: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in schedule:
        by_task[str(row["task_id"])].append(row)
    assert len(by_task) == 12
    assert all({row["disturbance_replica"] for row in pair} == {0, 1} for pair in by_task.values())
    assert all(len({row["incumbent_solver_seed"] for row in pair}) == 1 for pair in by_task.values())
    assert all(len({row["screen_solver_seed"] for row in pair}) == 2 for pair in by_task.values())


def _checkpoint_row(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "key_index": item["key_index"],
        "checkpoint_id": item["checkpoint_id"],
        "map_id": item["map_id"],
        "task_id": item["task_id"],
        "task_variant": item["task_variant"],
        "agent_count": item["agent_count"],
        "load_band": item["load_band"],
        "disturbance_replica": item["disturbance_replica"],
        "screen_solver_seed": item["screen_solver_seed"],
        "expected_fingerprint": f"fingerprint-{item['key_index']}",
        "expected_conflicts": 20 + int(item["key_index"]),
        "qualification": {"passed": True},
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


@pytest.mark.parametrize("qualified_count", (18, 24))
def test_boundary_checkpoint_gate_accepts_registered_18_to_24_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qualified_count: int
) -> None:
    schedule = _schedule(tmp_path, monkeypatch)
    # All twelve map-band cells once, then add balanced replicas from both bands.
    selected = schedule[::2] + schedule[1::2][: qualified_count - 12]
    rows = [_checkpoint_row(item) for item in selected]
    output = tmp_path / "output"
    checkpoint_root = output / "checkpoints"
    _write_jsonl(checkpoint_root / boundary.MANIFEST_FILENAME, rows)
    failed_indices = sorted(set(range(24)) - {int(row["key_index"]) for row in rows})
    failures = [
        {
            "status": "state_supply_unavailable",
            "error": "incumbent unavailable",
            "key_index": index,
        }
        for index in failed_indices
    ]
    (checkpoint_root / boundary.WORKER_RESULTS_FILENAME).write_text(
        json.dumps({"completed_count": len(rows), "failures": failures}),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.json"
    config_path.write_text("{}\n", encoding="utf-8")
    config = _config(tmp_path)
    monkeypatch.setattr(boundary, "load_config", lambda *_args: (config_path, tmp_path, config))

    report = boundary.analyze_checkpoints(config_path, output)

    assert report["qualified_checkpoint_count"] == qualified_count
    assert report["qualified_map_band_cell_count"] == 12
    assert report["passed"] is True


def _ttf_rows(schedule: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [_checkpoint_row(item) for item in schedule]
    for row in rows:
        row["checkpoint_identity_sha256"] = str(row["key_index"] % 10) * 64
    return rows


def test_boundary_ttf_schedule_has_48_rotating_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = _ttf_rows(_schedule(tmp_path, monkeypatch))
    timed = boundary_ttf._schedule(rows)

    assert len(timed) == 48
    assert [row["schedule_index"] for row in timed] == list(range(48))
    assert len({(row["task_id"], row["solver_seed"], row["controller"]) for row in timed}) == 48
    assert all(
        [row["controller"] for row in timed if row["key_index"] == key_index]
        == (["official_adaptive", "dual16"] if key_index % 2 == 0 else ["dual16", "official_adaptive"])
        for key_index in range(24)
    )


def test_collect_rejects_duplicate_boundary_job_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = _ttf_rows(_schedule(tmp_path, monkeypatch))
    rows[1]["screen_solver_seed"] = rows[0]["screen_solver_seed"]
    inputs = (tmp_path / "config.json", tmp_path, _config(tmp_path), tmp_path / "checkpoints", rows)
    monkeypatch.setattr(boundary_ttf, "_checkpoint_inputs", lambda *_args: inputs)
    monkeypatch.setattr(boundary_ttf, "_ensure_datasets", lambda *_args: tmp_path / "dataset")
    monkeypatch.setattr(boundary_ttf, "_runtime_config", lambda _config, _rows, path: path)

    with pytest.raises(ValueError, match="job keys are not unique"):
        boundary_ttf.collect_ttf(tmp_path / "config.json", tmp_path / "output")


def _identity_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, dict[str, Any], list[dict[str, Any]]]:
    schedule = _schedule(tmp_path, monkeypatch)
    selected = schedule[::2] + schedule[1::2][:6]
    config, dataset_root = _config(tmp_path), tmp_path / "dataset"
    prior_root = tmp_path / "prior-dataset"
    registered, rows = [], []
    for task in _dataset_rows():
        map_path = dataset_root / "boundary_confirmation" / f"{task['map_id']}.map"
        task_path = dataset_root / "boundary_confirmation" / f"{task['task_id']}.json"
        map_path.parent.mkdir(parents=True, exist_ok=True)
        if not map_path.exists():
            map_path.write_bytes(str(task["map_id"]).encode())
        task_path.write_bytes(str(task["task_id"]).encode())
        registered.append({**task, "map_file": map_path.name, "task_file": task_path.name})
    prior_map = prior_root / "development" / "prior.map"
    prior_map.parent.mkdir(parents=True)
    prior_map.write_bytes(b"prior-map")
    prior = [{"split": "development", "map_id": "prior", "map_file": prior_map.name}]
    planned = {int(row["key_index"]): row for row in schedule}
    registered_by_task = {str(row["task_id"]): row for row in registered}
    checkpoint_root = tmp_path / "output" / "checkpoints"
    for item in selected:
        task = registered_by_task[str(item["task_id"])]
        split_root = dataset_root / "boundary_confirmation"
        blob = checkpoint_root / f"states/{item['key_index']}.bin"
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(f"state-{item['key_index']}".encode())
        row = {
            **_checkpoint_row(item),
            "split": "boundary_confirmation",
            "source_kind": "checkpoint_blob_v1",
            "restore_seed": item["selection_seed"],
            "incumbent": {"solver_seed": item["incumbent_solver_seed"]},
            "map_sha256": sha256_file(split_root / str(task["map_file"])),
            "task_sha256": sha256_file(split_root / str(task["task_file"])),
            "state_blob": f"states/{item['key_index']}.bin",
            "state_blob_sha256": sha256_file(blob),
        }
        row["checkpoint_identity_sha256"] = compute_checkpoint_identity_sha256(row)
        rows.append(row)
    _write_jsonl(checkpoint_root / boundary.MANIFEST_FILENAME, rows)
    config_path = tmp_path / "config.json"
    config_path.write_text("{}\n", encoding="utf-8")
    (checkpoint_root / boundary.REPORT_FILENAME).write_text(
        json.dumps({
            "experiment_id": boundary.EXPERIMENT_ID,
            "passed": True,
            "config_sha256": sha256_file(config_path),
            "attempted_candidate_count": 24,
            "completed_checkpoint_count": 18,
            "qualified_checkpoint_count": 18,
            "qualified_map_count": 6,
            "qualified_map_band_cell_count": 12,
            "qualified_checkpoint_ids": [row["checkpoint_id"] for row in rows],
            "state_supply_unavailable_count": 6,
            "execution_failure_count": 0,
            "checkpoint_selection_controller_outcomes_consulted": False,
            "failed_candidate_replacement": False,
        }),
        encoding="utf-8",
    )
    missing_indices = sorted(set(range(24)) - {int(row["key_index"]) for row in rows})
    (checkpoint_root / boundary.WORKER_RESULTS_FILENAME).write_text(
        json.dumps(
            {
                "completed_count": 18,
                "failures": [
                    {
                        "status": "state_supply_unavailable",
                        "key_index": index,
                    }
                    for index in missing_indices
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(boundary_ttf, "load_config", lambda *_args: (config_path, tmp_path, config))
    monkeypatch.setattr(boundary_ttf, "_ensure_datasets", lambda *_args: dataset_root)
    monkeypatch.setattr(boundary_ttf, "_boundary_schedule", lambda *_args: list(planned.values()))
    monkeypatch.setattr(
        repair_collection,
        "_load_dataset_rows",
        lambda root, _splits: registered if Path(root) == dataset_root else prior,
    )
    return config_path, tmp_path / "output", config, rows


def test_boundary_inputs_reject_identity_and_prior_map_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, output, _config_value, rows = _identity_fixture(tmp_path, monkeypatch)
    rows[0]["expected_conflicts"] += 1
    _write_jsonl(output / "checkpoints" / boundary.MANIFEST_FILENAME, rows)
    with pytest.raises(ValueError, match="identity, seed, replica, blob, or registered hash"):
        boundary_ttf._checkpoint_inputs(config_path, output)

    rows[0]["checkpoint_identity_sha256"] = compute_checkpoint_identity_sha256(rows[0])
    boundary_map = tmp_path / "dataset" / "boundary_confirmation" / f"{rows[0]['map_id']}.map"
    prior_map = tmp_path / "prior-dataset" / "development" / "prior.map"
    prior_map.write_bytes(boundary_map.read_bytes())
    _write_jsonl(output / "checkpoints" / boundary.MANIFEST_FILENAME, rows)
    with pytest.raises(ValueError, match="map is not disjoint"):
        boundary_ttf._checkpoint_inputs(config_path, output)


def test_plan_and_mock_collect_boundary_run_48_strict_serial_lanes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = _ttf_rows(_schedule(tmp_path, monkeypatch))
    config_path = tmp_path / "config.json"
    config_path.write_text("{}\n", encoding="utf-8")
    inputs = (config_path, tmp_path, _config(tmp_path), tmp_path / "output/checkpoints", rows)
    monkeypatch.setattr(boundary_ttf, "_checkpoint_inputs", lambda *_args: inputs)
    assert boundary_ttf.plan_ttf(config_path, tmp_path / "output")["episode_count"] == 48
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(boundary_ttf, "_ensure_datasets", lambda *_args: tmp_path / "dataset")
    monkeypatch.setattr(boundary_ttf, "_runtime_config", lambda _config, _rows, path: path)
    monkeypatch.setattr(boundary_ttf, "_common_kwargs", lambda *_args: {})
    monkeypatch.setattr(boundary_ttf, "_controller_kwargs", lambda _root, _config, controller: (controller, {}))
    monkeypatch.setattr(boundary_ttf, "run_closed_loop_collection", lambda *_args, **kwargs: calls.append(dict(kwargs)) or {})
    monkeypatch.setattr(boundary_ttf, "analyze_ttf", lambda *_args: {"mocked": True})

    assert boundary_ttf.collect_ttf(config_path, tmp_path / "output") == {"mocked": True}
    assert len(calls) == 97
    assert len(calls[0]["job_keys"]) == 24
    override_keys = [next(iter(call["episode_overrides"])) for call in calls[1:]]
    assert len(set(override_keys)) == 24
    assert set(collections.Counter(override_keys).values()) == {4}


def test_boundary_analysis_routes_high_only_when_medium_has_additional_censor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = _ttf_rows(_schedule(tmp_path, monkeypatch))
    config_path = tmp_path / "config.json"
    config_path.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "output"
    monkeypatch.setattr(
        boundary_ttf,
        "_checkpoint_inputs",
        lambda *_args: (config_path, tmp_path, _config(tmp_path), output / "checkpoints", rows),
    )
    timed = boundary_ttf._schedule(rows)
    _write_jsonl(output / "ttf/execution_schedule.jsonl", timed)
    censored = False
    by_id = {str(row["checkpoint_id"]): row for row in rows}
    for item in timed:
        checkpoint = by_id[str(item["checkpoint_id"])]
        controller = str(item["controller"])
        censor = (
            not censored
            and item["load_band"] == "medium_high"
            and controller == "dual16"
        )
        censored = censored or censor
        success = not censor
        capped = 20.0 if controller == "official_adaptive" else 10.0
        observed = 72.0 if controller == "official_adaptive" else 30.0
        manifest = boundary_ttf._lane_root(output / "ttf", item) / boundary_ttf.TIMED_MANIFESTS[controller]
        _write_jsonl(manifest, [{
            "task_id": item["task_id"], "solver_seed": item["solver_seed"], "status": "ok",
            "summary": {
                "initial_fingerprint": checkpoint["expected_fingerprint"],
                "initial_conflicts": checkpoint["expected_conflicts"],
                "success": success,
                "capped_wall_time_to_feasible": capped,
                "wall_time_to_feasible": capped if success else None,
                "episode_observed_wall_seconds": observed,
                "external_timeout": False,
                "ttf_clock_schema": "lns2.ttf.reset_inclusive_wall.v1",
                "wall_time_budget_seconds": 120.0,
                "invalid_action_count": 0,
                "fingerprint_mismatch_count": 0,
                "stop_reason": "success" if success else "controller_stalled",
            },
        }])

    report = boundary_ttf.analyze_ttf(config_path, output)

    assert report["completed_episode_count"] == 48
    assert report["analysis"]["medium_high"]["comparison_vs_official"]["additional_censor_count"] == 1
    assert report["band_gate_passed"] == {"medium_high": False, "high": True}
    assert report["route_recommendation"] == "high_only"
