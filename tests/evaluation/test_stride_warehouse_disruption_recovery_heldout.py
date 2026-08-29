from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import experiments.repair_collection as repair_collection
import experiments.stride_warehouse_disruption_recovery_heldout as heldout
import experiments.stride_warehouse_disruption_recovery_heldout_ttf as heldout_ttf
from experiments._common import sha256_file
from experiments.warehouse_disruption_checkpoints import (
    compute_checkpoint_identity_sha256,
)


def _dataset_rows() -> list[dict[str, object]]:
    return [
        {
            "map_id": map_id,
            "task_id": f"{map_id}__{variant}",
            "task_variant": variant,
            "split": "controller_held_out",
            "agent_count": 600,
        }
        for map_id in ("heldout-map-b", "heldout-map-a")
        for variant in ("balanced_od_d125", "station_rush_d10")
    ]


def _schedule_config() -> dict[str, object]:
    return {
        "dataset": {
            "split": "controller_held_out",
            "screen_task_variants": ["station_rush_d10", "balanced_od_d125"],
        },
        "checkpoint_generation": {
            "incumbent_solver_seed_base": 1000,
            "selection_seed_base": 2000,
        },
        "runtime": {"screen_solver_seed_base": 3000},
    }


def test_heldout_schedule_has_four_candidates_on_two_maps(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        repair_collection,
        "_load_dataset_rows",
        lambda _root, _splits: _dataset_rows(),
    )

    schedule = heldout._heldout_schedule(_schedule_config(), tmp_path)

    assert len(schedule) == 4
    assert {row["map_id"] for row in schedule} == {
        "heldout-map-a",
        "heldout-map-b",
    }
    assert [row["key_index"] for row in schedule] == [0, 1, 2, 3]
    assert [row["task_variant"] for row in schedule] == [
        "station_rush_d10",
        "balanced_od_d125",
        "station_rush_d10",
        "balanced_od_d125",
    ]
    assert [row["incumbent_solver_seed"] for row in schedule] == [
        1000,
        1001,
        1002,
        1003,
    ]
    assert [row["selection_seed"] for row in schedule] == [
        2000,
        2001,
        2002,
        2003,
    ]
    assert [row["screen_solver_seed"] for row in schedule] == [
        3000,
        3001,
        3002,
        3003,
    ]


def _checkpoint_row(index: int, *, qualified: bool) -> dict[str, object]:
    return {
        "checkpoint_id": f"checkpoint-{index}",
        "map_id": f"heldout-map-{index % 2}",
        "qualification": {"passed": qualified},
    }


def test_checkpoint_gate_accepts_three_of_four_across_two_maps(
    tmp_path: Path, monkeypatch
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    config = {
        "checkpoint_generation": {
            "expected_candidate_count": 4,
            "minimum_qualified_count": 3,
            "minimum_qualified_map_count": 2,
            "gate": {
                "minimum_conflict_pair_count": 16,
                "minimum_active_conflict_agent_count": 32,
                "minimum_largest_conflict_component_size": 16,
            },
        },
        "claim_boundary": "heldout-only",
    }
    monkeypatch.setattr(
        heldout,
        "load_config",
        lambda _path: (config_path, tmp_path, config),
    )
    checkpoint_root = tmp_path / "output" / "checkpoints"
    checkpoint_root.mkdir(parents=True)
    rows = [
        _checkpoint_row(0, qualified=True),
        _checkpoint_row(1, qualified=True),
        _checkpoint_row(2, qualified=True),
        _checkpoint_row(3, qualified=False),
    ]
    (checkpoint_root / heldout.MANIFEST_FILENAME).write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    (checkpoint_root / "heldout_worker_results.json").write_text(
        json.dumps({"completed_count": 4, "failures": []}),
        encoding="utf-8",
    )

    report = heldout.analyze_checkpoints(config_path, tmp_path / "output")

    assert report["attempted_candidate_count"] == 4
    assert report["qualified_checkpoint_count"] == 3
    assert report["qualified_map_count"] == 2
    assert report["passed"] is True


def _ttf_config(tmp_path: Path) -> dict[str, Any]:
    return {
        "_dataset_root": str(tmp_path / "dataset"),
        "dataset": {
            "screen_task_variants": ["station_rush_d10", "balanced_od_d125"],
        },
        "checkpoint_generation": {
            "expected_candidate_count": 4,
            "minimum_qualified_count": 3,
            "minimum_qualified_map_count": 2,
        },
        "runtime": {
            "execution_order": "rotating_strict_two_controller_serial",
            "timing_boundary": "checkpoint_restore_inclusive_ttf",
            "wall_time_budget_seconds": 120.0,
        },
        "screen_gate": {
            "minimum_paired_win_rate": 0.70,
            "minimum_mean_capped_ttf_improvement": 0.15,
            "minimum_successes_per_hour_improvement": 0.20,
            "maximum_success_rate_loss": 0.0,
            "no_additional_timeout_or_censor": True,
        },
    }


def _ttf_rows(output: Path) -> list[dict[str, Any]]:
    checkpoint_root = output / "checkpoints"
    rows: list[dict[str, Any]] = []
    for index in range(4):
        relative_blob = f"states/checkpoint_{index}.bin"
        blob = checkpoint_root / relative_blob
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(f"heldout-state-{index}".encode())
        row: dict[str, Any] = {
            "key_index": index,
            "checkpoint_id": f"heldout-checkpoint-{index}",
            "source_kind": "checkpoint_blob_v1",
            "state_blob": relative_blob,
            "state_blob_sha256": sha256_file(blob),
            "map_id": f"heldout-map-{index % 2}",
            "task_id": f"heldout-task-{index}",
            "task_variant": (
                "station_rush_d10" if index % 2 == 0 else "balanced_od_d125"
            ),
            "split": "controller_held_out",
            "agent_count": 600,
            "screen_solver_seed": 4000 + index,
            "expected_fingerprint": f"fingerprint-{index}",
            "expected_conflicts": 20 + index,
            "qualification": {"passed": True},
        }
        row["checkpoint_identity_sha256"] = compute_checkpoint_identity_sha256(row)
        rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_checkpoint_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, dict[str, Any], list[dict[str, Any]]]:
    output = tmp_path / "output"
    config_path = tmp_path / "heldout-config.json"
    config_path.write_text("{}\n", encoding="utf-8")
    config = _ttf_config(tmp_path)
    rows = _ttf_rows(output)
    checkpoint_root = output / "checkpoints"
    _write_jsonl(checkpoint_root / heldout.MANIFEST_FILENAME, rows)
    (checkpoint_root / heldout.REPORT_FILENAME).write_text(
        json.dumps(
            {
                "experiment_id": heldout.EXPERIMENT_ID,
                "passed": True,
                "config_sha256": sha256_file(config_path),
                "attempted_candidate_count": 4,
                "qualified_checkpoint_count": 4,
                "execution_failure_count": 0,
                "qualified_checkpoint_ids": [row["checkpoint_id"] for row in rows],
                "checkpoint_selection_controller_outcomes_consulted": False,
                "failed_candidate_replacement": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        heldout_ttf,
        "load_config",
        lambda _path: (config_path, tmp_path, config),
    )
    return config_path, output, config, rows


def test_heldout_ttf_schedule_rotates_four_by_two() -> None:
    rows = [
        {
            "key_index": index,
            "checkpoint_id": f"checkpoint-{index}",
            "checkpoint_identity_sha256": str(index) * 64,
            "map_id": f"heldout-map-{index % 2}",
            "task_id": f"task-{index}",
            "task_variant": "station_rush_d10",
            "agent_count": 600,
            "screen_solver_seed": 5000 + index,
        }
        for index in range(4)
    ]

    schedule = heldout_ttf._schedule(rows)

    assert len(schedule) == 8
    assert [row["schedule_index"] for row in schedule] == list(range(8))
    assert [row["controller"] for row in schedule] == [
        "official_adaptive",
        "dual16",
        "dual16",
        "official_adaptive",
        "official_adaptive",
        "dual16",
        "dual16",
        "official_adaptive",
    ]


def test_heldout_ttf_inputs_reject_identity_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, output, _config, rows = _write_checkpoint_evidence(
        tmp_path, monkeypatch
    )
    dataset_root = tmp_path / "dataset"
    development_map = dataset_root / "development" / "development.map"
    development_map.parent.mkdir(parents=True)
    development_map.write_bytes(b"development-map")
    development_rows = [
        {"map_id": "development-map", "map_file": development_map.name}
    ]
    heldout_rows: list[dict[str, Any]] = []
    scheduled: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        split_root = dataset_root / "controller_held_out"
        map_file = split_root / f"map-{index % 2}.map"
        task_file = split_root / f"task-{index}.json"
        map_file.parent.mkdir(parents=True, exist_ok=True)
        if not map_file.is_file():
            map_file.write_bytes(f"heldout-map-{index % 2}".encode())
        task_file.write_bytes(f"heldout-task-{index}".encode())
        registered = {
            "map_id": row["map_id"],
            "task_id": row["task_id"],
            "task_variant": row["task_variant"],
            "agent_count": row["agent_count"],
            "map_file": map_file.name,
            "task_file": task_file.name,
        }
        heldout_rows.append(registered)
        row.update(
            {
                "restore_seed": 2000 + index,
                "incumbent": {"solver_seed": 1000 + index},
                "map_sha256": sha256_file(map_file),
                "task_sha256": sha256_file(task_file),
            }
        )
        row["checkpoint_identity_sha256"] = compute_checkpoint_identity_sha256(row)
        scheduled.append(
            {
                "task_id": row["task_id"],
                "checkpoint_id": row["checkpoint_id"],
                "key_index": index,
                "screen_solver_seed": row["screen_solver_seed"],
                "selection_seed": row["restore_seed"],
                "incumbent_solver_seed": row["incumbent"]["solver_seed"],
            }
        )
    monkeypatch.setattr(
        repair_collection,
        "_load_dataset_rows",
        lambda _root, splits: (
            development_rows if splits == ["development"] else heldout_rows
        ),
    )
    monkeypatch.setattr(heldout_ttf, "_heldout_schedule", lambda *_args: scheduled)
    rows[0]["expected_conflicts"] += 1
    _write_jsonl(output / "checkpoints" / heldout.MANIFEST_FILENAME, rows)

    with pytest.raises(ValueError, match="checkpoint identity changed"):
        heldout_ttf._checkpoint_inputs(config_path, output)


def test_heldout_ttf_inputs_reject_development_map_overlap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, output, _config, _rows = _write_checkpoint_evidence(
        tmp_path, monkeypatch
    )
    monkeypatch.setattr(
        repair_collection,
        "_load_dataset_rows",
        lambda _root, _splits: [{"map_id": "heldout-map-0"}],
    )

    with pytest.raises(ValueError, match="overlap development maps"):
        heldout_ttf._checkpoint_inputs(config_path, output)


def test_plan_and_collect_heldout_ttf_use_synthetic_serial_schedule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "output"
    config = _ttf_config(tmp_path)
    rows = _ttf_rows(output)
    inputs = (config_path, tmp_path, config, output / "checkpoints", rows)
    monkeypatch.setattr(heldout_ttf, "_checkpoint_inputs", lambda *_args: inputs)

    plan = heldout_ttf.plan_ttf(config_path, output)

    assert plan["checkpoint_count"] == 4
    assert plan["episode_count"] == 8
    assert plan["map_disjoint_from_development"] is True

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(heldout_ttf, "_ensure_dataset", lambda *_args: tmp_path / "dataset")
    monkeypatch.setattr(
        heldout_ttf,
        "_runtime_config",
        lambda _config, _rows, destination: destination,
    )
    monkeypatch.setattr(heldout_ttf, "_common_kwargs", lambda _config: {})
    monkeypatch.setattr(
        heldout_ttf,
        "_controller_kwargs",
        lambda _root, _config, controller: (controller, {}),
    )
    monkeypatch.setattr(
        heldout_ttf,
        "run_closed_loop_collection",
        lambda *_args, **kwargs: calls.append(dict(kwargs)) or {"status": "ok"},
    )
    monkeypatch.setattr(
        heldout_ttf,
        "analyze_ttf",
        lambda *_args: {"synthetic_analysis": True},
    )

    result = heldout_ttf.collect_ttf(config_path, output)

    assert result == {"synthetic_analysis": True}
    assert len(calls) == 17
    assert calls[0]["phase"] == "qualify"


def _write_ttf_rows(
    output: Path, checkpoints: list[dict[str, Any]], *, censor_dual: bool
) -> None:
    by_id = {str(row["checkpoint_id"]): row for row in checkpoints}
    for item in heldout_ttf._schedule(checkpoints):
        checkpoint = by_id[str(item["checkpoint_id"])]
        controller = str(item["controller"])
        success = not (
            censor_dual
            and controller == "dual16"
            and int(item["key_index"]) == 0
        )
        capped = 20.0 if controller == "official_adaptive" else 10.0
        observed = 72.0 if controller == "official_adaptive" else 30.0
        manifest = (
            heldout_ttf._lane_root(output / "ttf", item)
            / heldout_ttf.TIMED_MANIFESTS[controller]
        )
        _write_jsonl(
            manifest,
            [
                {
                    "task_id": item["task_id"],
                    "solver_seed": item["solver_seed"],
                    "status": "ok",
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
                }
            ],
        )


def test_analyze_heldout_ttf_rejects_additional_censor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "output"
    config = _ttf_config(tmp_path)
    rows = _ttf_rows(output)
    monkeypatch.setattr(
        heldout_ttf,
        "_checkpoint_inputs",
        lambda *_args: (config_path, tmp_path, config, output / "checkpoints", rows),
    )
    _write_ttf_rows(output, rows, censor_dual=True)
    _write_jsonl(
        output / "ttf" / "execution_schedule.jsonl",
        heldout_ttf._schedule(rows),
    )

    report = heldout_ttf.analyze_ttf(config_path, output)

    comparison = report["comparison_vs_official"]
    assert report["completed_episode_count"] == 8
    assert report["complete_pairing_and_initial_state_identity"] is True
    assert comparison["additional_censor_count"] == 1
    assert comparison["screen_gate_passed"] is False
    assert report["targeted_expert_confirmation"] is False
