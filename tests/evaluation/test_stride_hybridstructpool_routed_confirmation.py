from __future__ import annotations

from pathlib import Path

from experiments.stride_hybridstructpool_routed_confirmation import (
    CONTROLLERS,
    _bootstrap_improvement,
    load_config,
    schedule,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "stride_hybridstructpool_routed_confirmation_v1.json"


def test_confirmation_config_is_map_disjoint_and_complete() -> None:
    _path, _root, config = load_config(CONFIG)
    groups = list(config["cohort"]["groups"])
    assert len(groups) == 10
    assert {group["family"] for group in groups} >= {
        "maze",
        "room",
        "warehouse",
        "game",
        "dao",
    }
    assert not {group["id"] for group in groups} & set(
        config["cohort"]["development_map_ids"]
    )


def test_confirmation_schedule_is_rotating_serial_180_episode_contract() -> None:
    _path, _root, config = load_config(CONFIG)
    rows = schedule(config)
    assert len(rows) == 180
    assert len({(row["group_id"], row["task_id"], row["solver_seed"]) for row in rows}) == 60
    for offset in range(0, len(rows), 3):
        block = rows[offset : offset + 3]
        assert {row["controller"] for row in block} == set(CONTROLLERS)
        assert [row["within_key_position"] for row in block] == [0, 1, 2]


def test_bootstrap_requires_positive_paired_gain() -> None:
    keys = [("map", f"task-{index}", 1) for index in range(12)]
    baseline = {
        key: {
            "status": "ok",
            "summary": {"success": True, "wall_time_to_feasible": 10.0},
        }
        for key in keys
    }
    challenger = {
        key: {
            "status": "ok",
            "summary": {"success": True, "wall_time_to_feasible": 8.0},
        }
        for key in keys
    }
    report = _bootstrap_improvement(baseline, challenger, keys, 200)
    assert report["relative_improvement"] == 0.2
    assert report["ci95_lower"] == 0.2
