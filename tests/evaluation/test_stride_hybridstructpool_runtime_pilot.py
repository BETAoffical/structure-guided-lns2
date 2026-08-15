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
CAUSAL_HOTLOOP_CONFIG = ROOT / (
    "configs/stride_hybridstructpool_runtime_optimization_v4_registration.json"
)
PARETO_HOTLOOP_CONFIG = ROOT / (
    "configs/stride_hybridstructpool_runtime_optimization_v5_registration.json"
)
AUDIT_INDEX_CONFIG = ROOT / (
    "configs/stride_hybridstructpool_runtime_optimization_v6_registration.json"
)
TEMPORAL_INDEX_CONFIG = ROOT / (
    "configs/stride_hybridstructpool_runtime_optimization_v7_registration.json"
)
PRIMITIVE_EVIDENCE_CONFIG = ROOT / (
    "configs/stride_hybridstructpool_runtime_optimization_v8_registration.json"
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


def test_causal_hotloop_registration_preserves_full_pool_and_requires_speedup() -> None:
    _path, _root, config, _source, tasks = load_registration(CAUSAL_HOTLOOP_CONFIG)
    assert len(tasks) == 19
    optimization = config["runtime_optimization"]
    assert optimization["full_union_required"]
    assert optimization["terminal_wait_paths_materialized_once"]
    assert optimization["integer_contact_evidence_accumulator"]
    engineering = config["engineering_gate"]
    baseline = engineering["registered_full_runtime_baseline"]
    assert engineering["maximum_mean_hybrid_total_seconds"] == 0.9 * baseline[
        "mean_hybrid_total_seconds"
    ]
    assert engineering["maximum_success_rate_decrease"] == 0.0
    assert config["claim_boundary"]["not_default_pool_promotion"]


def test_pareto_hotloop_registration_keeps_v4_engineering_gates() -> None:
    _path, _root, config, _source, tasks = load_registration(PARETO_HOTLOOP_CONFIG)
    assert len(tasks) == 19
    optimization = config["runtime_optimization"]
    assert optimization["full_union_required"]
    assert optimization["pareto_coordinates_extracted_once"]
    assert optimization["redundant_final_candidate_deepcopy_removed"]
    engineering = config["engineering_gate"]
    assert abs(
        engineering["maximum_mean_controller_seconds"]
        - 0.9
        * engineering["registered_full_runtime_baseline"][
            "mean_controller_seconds"
        ]
    ) < 1e-12


def test_audit_index_registration_requires_v5_relative_speedup() -> None:
    _path, _root, config, _source, tasks = load_registration(AUDIT_INDEX_CONFIG)
    assert len(tasks) == 19
    optimization = config["runtime_optimization"]
    assert optimization["full_union_required"]
    assert optimization["shared_topology_candidate_audit_index"]
    assert optimization["event_and_pair_incidence_formula_exact"]
    assert optimization["complete_result_sha256_must_match_v5"]
    assert config["claim_boundary"]["not_candidate_compression"]
    engineering = config["engineering_gate"]
    baseline = engineering["registered_full_runtime_baseline"]
    assert baseline["report_sha256"] == (
        "1c31711820da7cc2bced78240f5ff62bc7288e651629699a83dde7bb63cdf7a0"
    )
    for metric, baseline_metric in (
        ("maximum_mean_controller_seconds", "mean_controller_seconds"),
        (
            "maximum_mean_candidate_generation_seconds",
            "mean_candidate_generation_seconds",
        ),
        ("maximum_mean_hybrid_total_seconds", "mean_hybrid_total_seconds"),
    ):
        assert abs(engineering[metric] - 0.9 * baseline[baseline_metric]) < 1e-12


def test_temporal_index_registration_requires_v6_relative_speedup() -> None:
    _path, _root, config, _source, tasks = load_registration(TEMPORAL_INDEX_CONFIG)
    assert len(tasks) == 19
    optimization = config["runtime_optimization"]
    assert optimization["full_union_required"]
    assert optimization["ordinary_and_bottleneck_temporal_variants_share_one_scan"]
    assert optimization["collision_free_integer_reservation_keys"]
    assert optimization["complete_result_sha256_must_match_v6"]
    assert config["claim_boundary"]["not_candidate_compression"]
    engineering = config["engineering_gate"]
    baseline = engineering["registered_full_runtime_baseline"]
    assert baseline["report_sha256"] == (
        "f89cecfb57e763daa34b4e284db658206ecaf516f57b366ade8914b8e809fce8"
    )
    for metric, baseline_metric in (
        ("maximum_mean_controller_seconds", "mean_controller_seconds"),
        (
            "maximum_mean_candidate_generation_seconds",
            "mean_candidate_generation_seconds",
        ),
        ("maximum_mean_hybrid_total_seconds", "mean_hybrid_total_seconds"),
    ):
        assert abs(engineering[metric] - 0.95 * baseline[baseline_metric]) < 1e-12


def test_primitive_evidence_registration_requires_v7_relative_speedup() -> None:
    _path, _root, config, _source, tasks = load_registration(PRIMITIVE_EVIDENCE_CONFIG)
    assert len(tasks) == 19
    optimization = config["runtime_optimization"]
    assert optimization["full_union_required"]
    assert optimization["primitive_temporal_evidence_cached_until_materialization"]
    assert optimization["contiguous_causal_windows_use_bounded_iteration"]
    assert optimization["complete_result_sha256_must_match_v7"]
    assert config["claim_boundary"]["not_candidate_compression"]
    engineering = config["engineering_gate"]
    baseline = engineering["registered_full_runtime_baseline"]
    assert baseline["report_sha256"] == (
        "c937357929d1edc063f05a0b57bc8cc520f11aeaf14bdd078c0b2ca1cd8f620b"
    )
    for metric, baseline_metric in (
        ("maximum_mean_controller_seconds", "mean_controller_seconds"),
        (
            "maximum_mean_candidate_generation_seconds",
            "mean_candidate_generation_seconds",
        ),
        ("maximum_mean_hybrid_total_seconds", "mean_hybrid_total_seconds"),
    ):
        assert abs(engineering[metric] - 0.95 * baseline[baseline_metric]) < 1e-12
