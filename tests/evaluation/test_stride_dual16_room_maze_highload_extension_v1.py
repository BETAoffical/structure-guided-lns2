from __future__ import annotations

import collections

from experiments.stride_dual16_room_maze_highload_extension_v1 import _execution_schedule, _schedule, load_config


CONFIG = "configs/stride_dual16_room_maze_highload_extension_v1.json"


def test_config_is_sequential_diagnostic() -> None:
    _path, _root, config = load_config(CONFIG)
    assert config["scientific_status"] == "sequential_family_diagnostic_not_map_disjoint"
    assert config["global_or_default_promotion_allowed"] is False
    assert config["map_agent_counts"] == {
        "maze-32-32-4": 200,
        "maze-128-128-1": 400,
        "room-64-64-8": 600,
        "room-64-64-16": 600,
    }
    assert config["diagnostic_roles"]["maze-128-128-1"] == "high_conflict_pressure_negative_control"


def test_schedule_has_sixteen_unique_controller_blind_keys(monkeypatch) -> None:
    import experiments.stride_dual16_room_maze_highload_extension_v1 as subject
    rows = {}
    counts = {"maze-a": 200, "maze-b": 400, "room-a": 600, "room-b": 600}
    for family, maps in {"maze": ["maze-a", "maze-b"], "room": ["room-a", "room-b"]}.items():
        for map_id in maps:
            for scenario in (4, 5):
                task = f"{map_id}__random_{scenario:02d}__agents_{counts[map_id]:04d}"
                rows[task] = {"task_id": task, "map_id": map_id, "agent_count": counts[map_id], "scenario_type": f"movingai_random_{scenario}"}
    monkeypatch.setattr(subject, "_dataset_rows", lambda _config: rows)
    config = {"maps": {"maze": ["maze-a", "maze-b"], "room": ["room-a", "room-b"]}, "map_agent_counts": counts,
              "diagnostic_roles": {"maze-b": "high_conflict_pressure_negative_control"}, "solver_seeds": [51, 52]}
    keys = _schedule(config)
    assert len(keys) == len({(r["task_id"], r["solver_seed"]) for r in keys}) == 16
    assert {r["family"] for r in keys} == {"room", "maze"}
    assert collections.Counter(r["agent_count"] for r in keys) == {200: 4, 400: 4, 600: 8}


def test_execution_schedule_rotates_and_is_strictly_paired() -> None:
    keys = [{"key_index": i, "task_id": f"t{i}", "solver_seed": 51, "map_id": "m", "family": "room", "agent_count": 600} for i in range(4)]
    rows = _execution_schedule(keys)
    assert len(rows) == 8
    assert [r["controller"] for r in rows[:4]] == ["official_adaptive", "dual16", "dual16", "official_adaptive"]
    assert all(rows[i]["key_index"] == rows[i + 1]["key_index"] for i in range(0, len(rows), 2))
