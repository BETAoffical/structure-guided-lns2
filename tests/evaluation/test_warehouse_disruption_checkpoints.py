from __future__ import annotations

import copy
import inspect

import pytest

from experiments import warehouse_disruption_checkpoints as checkpoints


def _semantic(rows: int, cols: int, replacements: dict[tuple[int, int], str] | None = None) -> list[str]:
    values = [["." for _ in range(cols)] for _ in range(rows)]
    for (row, col), value in dict(replacements or {}).items():
        values[row][col] = value
    return ["".join(row) for row in values]


def _parallel_paths(agent_count: int = 20, cols: int = 8) -> list[list[int]]:
    return [[row * cols + col for col in range(5)] for row in range(agent_count)]


def test_hotspot_is_station_and_semantic_x_union_with_free_neighborhood() -> None:
    semantic = _semantic(5, 6, {(1, 1): "X", (1, 2): "@", (4, 5): "X"})
    result = checkpoints.warehouse_hotspot_definition(
        {"station_00": [[3, 3], [3, 4]]}, semantic, neighborhood_radius=1
    )

    hotspot = {tuple(cell) for cell in result["hotspot_cells"]}
    assert result["station_zone_cell_count"] == 2
    assert result["semantic_x_cell_count"] == 2
    assert {(3, 3), (3, 4), (1, 1), (4, 5)} <= hotspot
    assert {(0, 1), (1, 0), (2, 1), (3, 2), (4, 3)} <= hotspot
    assert (1, 2) not in hotspot
    assert len(result["hotspot_definition_sha256"]) == 64


def test_delay_injection_is_deterministic_global_and_does_not_mutate_source() -> None:
    paths = _parallel_paths()
    source = copy.deepcopy(paths)
    semantic = _semantic(20, 8)
    station_cells = [[row, 1] for row in range(10)]

    first = checkpoints.inject_global_hotspot_delays(
        paths,
        semantic_cell_types=semantic,
        station_zones={"station_00": station_cells},
        global_time=1,
        delay_ticks=3,
        delayed_agent_fraction=0.10,
        selection_seed=2026082701,
    )
    repeated = checkpoints.inject_global_hotspot_delays(
        paths,
        semantic_cell_types=semantic,
        station_zones={"station_00": station_cells},
        global_time=1,
        delay_ticks=3,
        delayed_agent_fraction=0.10,
        selection_seed=2026082701,
    )

    assert first == repeated
    assert paths == source
    assert first["delayed_agent_count"] == 2
    assert first["delayed_agent_fraction_realized"] == 0.10
    assert first["controller_outcomes_consulted"] is False
    selected = set(first["selected_agent_ids"])
    assert len(selected) == 2
    for agent_id, disturbed in enumerate(first["paths"]):
        if agent_id in selected:
            assert disturbed == source[agent_id][:2] + [source[agent_id][1]] * 3 + source[agent_id][2:]
        else:
            assert disturbed == source[agent_id]


def test_delay_injection_rejects_invalid_registration_or_source_paths() -> None:
    paths = _parallel_paths()
    semantic = _semantic(20, 8)
    common = {
        "semantic_cell_types": semantic,
        "station_zones": {"station_00": [[0, 1]]},
        "global_time": 1,
        "delay_ticks": 2,
        "selection_seed": 7,
    }
    with pytest.raises(ValueError, match="between 0.10 and 0.15"):
        checkpoints.inject_global_hotspot_delays(
            paths, delayed_agent_fraction=0.09, **common
        )
    with pytest.raises(ValueError, match="one of 2, 3, or 4"):
        checkpoints.inject_global_hotspot_delays(
            paths,
            delayed_agent_fraction=0.10,
            **{**common, "delay_ticks": 1},
        )
    with pytest.raises(ValueError, match="fewer hotspot agents"):
        checkpoints.inject_global_hotspot_delays(
            paths, delayed_agent_fraction=0.15, **common
        )

    conflicting = copy.deepcopy(paths)
    conflicting[1] = list(conflicting[0])
    with pytest.raises(ValueError, match="feasible and conflict-free"):
        checkpoints.inject_global_hotspot_delays(
            conflicting, delayed_agent_fraction=0.10, **common
        )


def test_path_conflicts_reconstruct_delay_created_vertex_pair() -> None:
    paths = [
        [0, 1, 2, 3],
        [11, 11, 1, 0],
        *[[row * 10 + 9] for row in range(2, 10)],
    ]
    assert checkpoints.summarize_path_conflicts(paths)["conflict_pair_count"] == 0

    disturbed = copy.deepcopy(paths)
    disturbed[0] = paths[0][:2] + [paths[0][1]] * 2 + paths[0][2:]
    summary = checkpoints.summarize_path_conflicts(disturbed)

    assert summary["conflict_pair_count"] == 1
    assert summary["conflict_event_count"] == 1
    assert summary["vertex_conflict_event_count"] == 1
    assert summary["edge_conflict_event_count"] == 0
    assert summary["active_conflict_agent_count"] == 2
    assert summary["largest_conflict_component_size"] == 2
    assert summary["conflict_edges"] == [[0, 1]]


def test_gate_requires_conflicts_active_agents_and_one_large_component() -> None:
    disjoint = checkpoints.summarize_conflict_graph(
        [[2 * index, 2 * index + 1] for index in range(16)], agent_count=40
    )
    failed = checkpoints.warehouse_checkpoint_gate(disjoint)
    assert failed["passed"] is False
    assert failed["observed"] == {
        "conflict_pair_count": 16,
        "active_conflict_agent_count": 32,
        "largest_conflict_component_size": 2,
    }
    assert failed["failed_metrics"] == ["largest_conflict_component_size"]

    coupled = checkpoints.summarize_conflict_graph(
        [[index, index + 1] for index in range(31)], agent_count=40
    )
    passed = checkpoints.warehouse_checkpoint_gate(coupled)
    assert passed["passed"] is True
    assert passed["failed_metrics"] == []
    assert passed["controller_outcomes_consulted"] is False


def test_checkpoint_manifest_binds_paths_identity_and_controller_blind_gate() -> None:
    paths = _parallel_paths()
    disturbance = checkpoints.inject_global_hotspot_delays(
        paths,
        semantic_cell_types=_semantic(20, 8),
        station_zones={"station_00": [[row, 1] for row in range(10)]},
        global_time=1,
        delay_ticks=4,
        delayed_agent_fraction=0.15,
        selection_seed=2026082702,
    )
    manifest = checkpoints.checkpoint_manifest_fields(
        map_id="warehouse-realistic-01",
        task_id="warehouse-realistic-01__station-rush-01",
        map_sha256="a" * 64,
        task_sha256="b" * 64,
        source_paths=paths,
        disturbance=disturbance,
    )

    assert manifest["schema"] == checkpoints.CHECKPOINT_SCHEMA
    assert manifest["checkpoint_id"].startswith("warehouse-disruption-")
    assert len(manifest["checkpoint_identity_sha256"]) == 64
    assert manifest["checkpoint_identity_sha256"] == checkpoints.compute_checkpoint_identity_sha256(manifest)
    assert manifest["agent_count"] == 20
    assert manifest["disturbance"]["delayed_agent_count"] == 3
    assert manifest["controller_outcomes_consulted"] is False
    assert manifest["native_solver_or_controller_invoked"] is False
    assert manifest["qualification"]["gate_id"] == checkpoints.QUALIFICATION_GATE_ID

    tampered = copy.deepcopy(manifest)
    tampered["task_id"] = "different-task"
    assert checkpoints.compute_checkpoint_identity_sha256(tampered) != manifest["checkpoint_identity_sha256"]


def test_module_is_pure_and_has_no_solver_or_controller_inputs() -> None:
    source = inspect.getsource(checkpoints)
    assert "lns2_env" not in source
    assert "run_closed_loop" not in source
    assert "controller_outcome" not in inspect.signature(
        checkpoints.inject_global_hotspot_delays
    ).parameters
    assert checkpoints.DEFAULT_GATE_THRESHOLDS == {
        "minimum_conflict_pair_count": 16,
        "minimum_active_conflict_agent_count": 32,
        "minimum_largest_conflict_component_size": 16,
    }
