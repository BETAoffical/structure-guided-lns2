from __future__ import annotations

import json
from pathlib import Path

import pytest

import experiments.stride_warehouse_disruption_recovery as subject


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "configs"
    / "stride_warehouse_disruption_recovery_screen_v1.json"
)


def _write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _synthetic_development_rows() -> list[dict]:
    rows = []
    variants = (
        "station_release_d10",
        "balanced_od_d125",
        "station_rush_d10",
        "station_dominant_d125",
    )
    for map_index in (3, 1, 2, 0):
        map_id = f"development_station_centric_{map_index:04d}"
        for variant_index, variant in enumerate(variants):
            rows.append(
                {
                    "split": "development",
                    "map_id": map_id,
                    "task_id": f"{map_id}__task_{variant_index:04d}",
                    "task_variant": variant,
                    "agent_count": 180 + map_index,
                }
            )
    return rows


def test_config_and_controller_contract_are_frozen() -> None:
    _path, _root, config = subject.load_config(CONFIG)

    assert config["schema"] == subject.CONFIG_SCHEMA
    assert config["experiment_id"] == subject.EXPERIMENT_ID
    assert tuple(config["controllers"]) == (
        "official_adaptive",
        "v2_only",
        "mixed_full_v2",
        "dual16",
    )
    assert config["controller_contract"] == {
        "v2_bundle": "artifacts/initlns-closed-loop-controller-v2",
        "v2_manifest_sha256": (
            "1b699182f9890148d0030e691b457c82ac7054760534665ad65484afb0ac82a8"
        ),
        "mixed_bundle": "artifacts/initlns-mixed-full-controller-v2",
        "mixed_manifest_sha256": (
            "b2ec8775ffc589c2fe5cf0553997889d6352f8a354278a1c701734b8827e59c5"
        ),
        "dual16": "stride-structshell-dual16-v1",
        "frozen_v2_and_mixed": True,
        "no_new_ranker": True,
    }
    assert config["checkpoint_generation"]["global_time_rule"] == (
        "earliest_maximum_hotspot_occupancy_after_t0"
    )
    assert config["checkpoint_generation"][
        "selection_blind_to_controller_outcomes"
    ] is True
    assert config["checkpoint_generation"]["no_failed_candidate_replacement"] is True
    assert config["runtime"]["timed_workers"] == 1


def test_load_config_rejects_controller_schema_change(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["controllers"] = [
        "official_adaptive",
        "v2_only",
        "mixed_full_v2",
    ]
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(
        ValueError, match="Warehouse disruption screen contract changed"
    ):
        subject.load_config(changed)


def test_screen_schedule_is_complete_ordered_and_seeded(tmp_path: Path) -> None:
    _path, _root, config = subject.load_config(CONFIG)
    dataset_root = tmp_path / "dataset"
    _write_manifest(
        dataset_root / "development" / "manifest.jsonl",
        _synthetic_development_rows(),
    )

    schedule = subject._screen_schedule(config, dataset_root)

    assert len(schedule) == 8
    assert [row["key_index"] for row in schedule] == list(range(8))
    assert [row["checkpoint_id"] for row in schedule] == [
        f"warehouse-disruption-dev-{index:02d}" for index in range(8)
    ]
    assert [
        (row["map_id"], row["task_variant"]) for row in schedule
    ] == [
        (f"development_station_centric_{map_index:04d}", variant)
        for map_index in range(4)
        for variant in ("station_rush_d10", "balanced_od_d125")
    ]
    assert [row["incumbent_solver_seed"] for row in schedule] == [
        2026082700 + index for index in range(8)
    ]
    assert [row["selection_seed"] for row in schedule] == [
        2026082800 + index for index in range(8)
    ]
    assert [row["screen_solver_seed"] for row in schedule] == [
        31 + index for index in range(8)
    ]
    assert all(row["split"] == "development" for row in schedule)


def test_global_time_uses_earliest_positive_time_at_maximum_occupancy() -> None:
    hotspot_id = 11
    paths = [[hotspot_id, 0, 0, 0, 0, 0, 0] for _ in range(20)]
    for agent in range(4):
        paths[agent][2] = hotspot_id
    for agent in range(4, 8):
        paths[agent][4] = hotspot_id

    selected = subject._select_global_time(
        paths,
        {"cols": 10, "hotspot_cells": [[1, 1]]},
        0.15,
    )

    assert selected == {
        "global_time": 2,
        "eligible_agent_count": 4,
        "required_delayed_agent_count": 3,
    }


def _qualified_checkpoint(index: int) -> dict:
    map_index = index % 4
    return {
        "checkpoint_id": f"warehouse-disruption-dev-{index:02d}",
        "map_id": f"development_station_centric_{map_index:04d}",
        "task_id": f"development_station_centric_{map_index:04d}__task_{index:04d}",
        "task_variant": (
            "station_rush_d10" if index % 2 == 0 else "balanced_od_d125"
        ),
        "expected_conflicts": 20,
        "native_complexity": {
            "active_conflict_agent_count": 40,
            "largest_conflict_component_size": 20,
        },
        "qualification": {"passed": True},
    }


def _write_checkpoint_inputs(
    output: Path, *, failure_status: str, failure_error: str
) -> None:
    checkpoint_root = output / "checkpoints"
    _write_manifest(
        checkpoint_root / subject.MANIFEST_FILENAME,
        [_qualified_checkpoint(index) for index in range(7)],
    )
    failure = {
        "status": failure_status,
        "error": failure_error,
        "state_count": 0,
        "outcome_count": 0,
        "key_index": 7,
        "key_id": "missing-candidate#delay-07",
        "map_id": "development_station_centric_0003",
        "task_id": "development_station_centric_0003__task_missing",
    }
    (checkpoint_root / "worker_results.json").write_text(
        json.dumps({"completed_count": 7, "failures": [failure]}),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("failure_status", "failure_error"),
    (
        (
            "state_supply_unavailable",
            "Official incumbent did not produce a feasible path set",
        ),
        (
            "error",
            "RuntimeError: Official incumbent did not produce a feasible path set",
        ),
    ),
)
def test_qualification_allows_one_explicit_official_state_supply_gap(
    tmp_path: Path,
    failure_status: str,
    failure_error: str,
) -> None:
    _write_checkpoint_inputs(
        tmp_path,
        failure_status=failure_status,
        failure_error=failure_error,
    )

    report = subject.analyze_checkpoints(CONFIG, tmp_path)

    assert report["candidate_count"] == 8
    assert report["attempted_candidate_count"] == 8
    assert report["completed_checkpoint_count"] == 7
    assert report["qualified_checkpoint_count"] == 7
    assert report["qualified_map_count"] == 4
    assert report["state_supply_unavailable_count"] == 1
    assert report["execution_failure_count"] == 0
    assert report["passed"] is True


@pytest.mark.parametrize(
    ("failure_status", "failure_error"),
    (
        ("timeout", "checkpoint worker exceeded its process timeout"),
        ("error", "RuntimeError: unexpected checkpoint worker failure"),
    ),
)
def test_qualification_rejects_timeout_or_unknown_execution_failure(
    tmp_path: Path,
    failure_status: str,
    failure_error: str,
) -> None:
    _write_checkpoint_inputs(
        tmp_path,
        failure_status=failure_status,
        failure_error=failure_error,
    )

    report = subject.analyze_checkpoints(CONFIG, tmp_path)

    assert report["attempted_candidate_count"] == 8
    assert report["qualified_checkpoint_count"] == 7
    assert report["qualified_map_count"] == 4
    assert report["state_supply_unavailable_count"] == 0
    assert report["execution_failure_count"] == 1
    assert report["passed"] is False
