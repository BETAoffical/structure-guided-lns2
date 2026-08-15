from __future__ import annotations

from pathlib import Path

from experiments.stride_hybridstructpool_runtime_pilot import (
    ARMS,
    load_registration,
    schedule,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/stride_hybridstructpool_runtime_pilot_v1_registration.json"
OPTIMIZED_CONFIG = ROOT / (
    "configs/stride_hybridstructpool_runtime_optimization_v2_registration.json"
)
FULL_OPTIMIZED_CONFIG = ROOT / (
    "configs/stride_hybridstructpool_runtime_optimization_v3_registration.json"
)


def test_registration_reuses_the_complete_registered_maze_cohort() -> None:
    _path, _root, config, _source, tasks = load_registration(CONFIG)
    assert len(tasks) == 19
    assert config["execution"]["repair_seed_policy"] == "episode_stream"
    assert config["execution"]["deterministic_pp_replay"] is False
    assert config["execution"]["failure_rescue_enabled"] is False


def test_schedule_is_strictly_paired_and_pool_only() -> None:
    _path, _root, config, _source, tasks = load_registration(CONFIG)
    rows = schedule(tasks, config["execution"]["trial_indices"])
    assert len(rows) == 152
    paired: dict[tuple[str, int], set[str]] = {}
    for row in rows:
        key = (str(row["task_fingerprint"]), int(row["trial_index"]))
        paired.setdefault(key, set()).add(str(row["arm"]))
    assert len(paired) == 76
    assert all(value == set(ARMS) for value in paired.values())


def test_optimized_registration_freezes_engineering_and_quality_gates() -> None:
    _path, _root, config, _source, tasks = load_registration(OPTIMIZED_CONFIG)
    assert len(tasks) == 19
    assert config["runtime_optimization"]["full_union_audit_api_preserved"]
    assert config["runtime_optimization"]["v2_pool_complete"]
    assert config["runtime_optimization"]["causal_frontier_complete"]
    assert len(
        config["runtime_optimization"]["removed_structural_family_size_cells"]
    ) == 12
    engineering = config["engineering_gate"]
    assert engineering["maximum_mean_controller_seconds"] < engineering[
        "registered_full_runtime_baseline"
    ]["mean_controller_seconds"]
    assert engineering["maximum_mean_candidate_generation_seconds"] < engineering[
        "registered_full_runtime_baseline"
    ]["mean_candidate_generation_seconds"]


def test_full_optimized_registration_forbids_candidate_compression() -> None:
    _path, _root, config, _source, tasks = load_registration(FULL_OPTIMIZED_CONFIG)
    assert len(tasks) == 19
    optimization = config["runtime_optimization"]
    assert optimization["full_union_required"]
    assert optimization["complete_structural_family_size_cell_count"] == 24
    assert optimization["structural_filtering_enabled"] is False
    assert config["claim_boundary"]["not_candidate_compression"]
    engineering = config["engineering_gate"]
    assert abs(
        engineering["maximum_mean_hybrid_total_seconds"]
        - 0.75
        * engineering["registered_full_runtime_baseline"][
            "mean_hybrid_total_seconds"
        ]
    ) < 1e-12
