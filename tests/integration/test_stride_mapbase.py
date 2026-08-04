from __future__ import annotations

import json
from pathlib import Path

from experiments._common import sha256_file
from experiments.repair_collection import _read_jsonl
from experiments.stride_mapbase import (
    audit_mapbase_collection,
    build_mapbase_selection,
    validate_mapbase_config,
)
from experiments.stride_maprank import (
    build_maprank_labels,
    prepare_maprank_selection,
    validate_maprank_design,
    validate_maprank_evaluation_config,
    validate_maprank_training_config,
)
from experiments.stride_maprank_evaluation import _training_evidence
from experiments.stride_maprank_raw_ttf import (
    CONTROLLERS,
    _runtime_transition_signature,
    _schedule,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "stride_mapbase_collection.json"
MAPRANK_DESIGN_PATH = ROOT / "configs" / "stride_maprank_design.json"
MAPRANK_TRAINING_PATH = ROOT / "configs" / "stride_maprank_training.json"
MAPRANK_EVALUATION_PATH = ROOT / "configs" / "stride_maprank_evaluation.json"


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_mapbase_is_base_only_multiseed_training_collection() -> None:
    config = _config()
    validate_mapbase_config(config)
    assert config["collection_id"] == "stride-mapbase-v1"
    assert config["expected_active_state_count"] == 63
    assert config["trial_indices"] == list(range(16))
    assert config["topology_boundary_candidates_allowed"] is False
    assert config["candidate_pool"]["topology_boundary"] == "forbidden"
    assert config["selection"]["research_split"] == "train"


def test_mapbase_registered_inputs_and_selection_match_when_available() -> None:
    config = _config()
    paths = {}
    for name, artifact in config["inputs"].items():
        path = ROOT / artifact["path"]
        if not path.is_file():
            return
        assert sha256_file(path) == artifact["sha256"]
        paths[name] = path
    selection = build_mapbase_selection(
        config,
        recommended_tasks=_read_jsonl(paths["recommended_tasks"]),
        relevance_states=_read_jsonl(paths["relevance_states"]),
        dataset_rows=_read_jsonl(paths["dataset_manifest"]),
        qualification_rows=_read_jsonl(paths["qualification_manifest"]),
    )
    assert len(selection) == 63
    assert len({row["map_id"] for row in selection}) == 8
    assert all(row["before_conflicts"] > 0 for row in selection)
    assert all(row["research_split"] == "train" for row in selection)


def test_completed_mapbase_collection_audits_in_state_id_order_when_available(
    tmp_path: Path,
) -> None:
    collection = ROOT / "build" / "stride-mapbase-collection-v1"
    report_path = collection / "collection_report.json"
    if not report_path.is_file():
        return
    collection_report = json.loads(report_path.read_text(encoding="utf-8"))
    if collection_report.get("complete") is not True:
        return
    report = audit_mapbase_collection(CONFIG_PATH, collection, tmp_path)
    assert report["passed"] is True
    assert report["state_count"] == 63
    assert report["run_fingerprint"]
    assert report["gates"]["consolidated_trials_match"] is True


def test_maprank_successor_design_is_distinct_and_outcome_independent() -> None:
    design = json.loads(MAPRANK_DESIGN_PATH.read_text(encoding="utf-8"))
    validate_maprank_design(design)
    assert design["schema"] == "lns2.stride.maprank_design.v1"
    assert design["controller_id"] == "stride-maprank-v1"
    assert design["label_artifact_id"] == "stride-maprank-labels-v1"
    assert design["scientific_status"] == (
        "registered_before_mapbase_collection_summary_and_label_analysis"
    )
    assert [row["expected_state_count"] for row in design["source_collections"]] == [
        240,
        63,
    ]
    coverage = design["expected_data_coverage"]
    assert coverage == {
        "state_count": 303,
        "train_state_count": 237,
        "legacy_validation_state_count": 66,
        "train_map_count": 24,
        "legacy_validation_map_count": 6,
        "map_level_split_required": True,
        "new_mapbase_maps_are_train_only": True,
    }
    label = design["label_design"]
    assert label["trial_indices"] == list(range(16))
    assert label["runtime_used_in_label"] is False
    assert label["future_trajectory_used_in_label"] is False
    ranker = design["ranking_design"]
    assert ranker["architecture"] == (
        "frozen_v2_anchor_plus_conflict_only_challenger"
    )
    assert ranker["base_feature_dimension"] == 124
    assert ranker["legacy_validation_is_descriptive_only"] is True
    assert design["runtime_primary_metric"] == (
        "mean_raw_run_to_completion_wall_ttf"
    )
    assert design["formal_speed_claim"] is False
    assert design["default_replacement_allowed"] is False


def test_maprank_selection_combines_fresh_train_maps_when_inputs_exist(
    tmp_path: Path,
) -> None:
    design = json.loads(MAPRANK_DESIGN_PATH.read_text(encoding="utf-8"))
    if any(
        not (ROOT / artifact["path"]).is_file()
        for artifact in design["selection_sources"].values()
    ):
        return
    report = prepare_maprank_selection(MAPRANK_DESIGN_PATH, tmp_path)
    assert report["state_count"] == 303
    assert report["split_state_counts"] == {"train": 237, "validation": 66}
    assert report["split_map_counts"] == {"train": 24, "validation": 6}
    assert report["new_map_count"] == 8
    assert report["new_state_count"] == 63
    assert report["repair_outcomes_used"] is False
    assert report["controller_outcomes_used"] is False


def test_maprank_training_registration_matches_frozen_design() -> None:
    config = json.loads(MAPRANK_TRAINING_PATH.read_text(encoding="utf-8"))
    validate_maprank_training_config(config)
    assert config["controller_id"] == "stride-maprank-v1"
    assert config["labels"] == "build/stride-maprank-labels-v1"
    assert config["model_parameters"]["random_state"] == 20260804
    assert config["topology_boundary_threshold"] == 0.06
    changed = json.loads(MAPRANK_TRAINING_PATH.read_text(encoding="utf-8"))
    changed["offline_gates"][
        "minimum_relative_normalized_regret_improvement_over_frozen_v2"
    ] = 0.0
    try:
        validate_maprank_training_config(changed)
    except ValueError as error:
        assert "offline gates changed" in str(error)
    else:
        raise AssertionError("MapRank accepted a relaxed promotion gate")


def test_maprank_label_builder_rejects_unregistered_output(tmp_path: Path) -> None:
    try:
        build_maprank_labels(
            training_config_path=MAPRANK_TRAINING_PATH,
            output=tmp_path / "unregistered-labels",
        )
    except ValueError as error:
        assert "label output differs from registration" in str(error)
    else:
        raise AssertionError("MapRank accepted an unregistered label output")


def test_maprank_evaluation_is_uncapped_paired_and_preregistered() -> None:
    config = json.loads(MAPRANK_EVALUATION_PATH.read_text(encoding="utf-8"))
    validate_maprank_evaluation_config(config)
    assert config["controllers"] == [
        "v2-full",
        "v2-augmented-pool",
        "stride-maprank-v1",
    ]
    assert config["scientific_time_limit_seconds"] is None
    assert config["environment_time_limit_seconds"] is None
    assert config["episode_process_timeout_seconds"] is None
    assert config["high_load_development"]["cohorts"][1]["id"] == "room500"
    assert len(config["fresh_map_raw_ttf"]["cohorts"]) == 6
    changed = json.loads(MAPRANK_EVALUATION_PATH.read_text(encoding="utf-8"))
    changed["high_load_development"]["cohorts"][0]["tasks"][0] = "post-hoc-task"
    try:
        validate_maprank_evaluation_config(changed)
    except ValueError as error:
        assert "high-load cohort changed" in str(error)
    else:
        raise AssertionError("MapRank accepted a post-hoc high-load task")


def test_maprank_runtime_registrations_match_exact_bundle() -> None:
    bundle = (
        ROOT
        / "build"
        / "stride-maprank-training-v1"
        / "stride-maprank-v1"
        / "controller_manifest.json"
    )
    if not bundle.is_file():
        return
    expected = sha256_file(bundle)
    for name in (
        "stride_stage4r_high_load_runtime.json",
        "stride_augcontrol_ood_runtime.json",
    ):
        runtime = json.loads((ROOT / "configs" / name).read_text(encoding="utf-8"))
        registered = runtime["model_registration"][
            "registered_controller_bundles"
        ]["stride-maprank-v1"]["controller_manifest_sha256"]
        assert registered == expected


def test_maprank_shadow_accepts_only_offline_gated_training_report() -> None:
    config = json.loads(MAPRANK_EVALUATION_PATH.read_text(encoding="utf-8"))
    report_path = ROOT / config["training_report"]
    if not report_path.is_file():
        return
    path, report = _training_evidence(ROOT, config)
    assert path == report_path.resolve()
    assert report["controller_id"] == "stride-maprank-v1"
    assert report["fresh_development_eligible"] is True


def test_maprank_raw_ttf_schedule_is_strictly_rotated_and_complete() -> None:
    cohorts = [
        {"id": "maze300", "tasks": ["maze-a", "maze-b"]},
        {"id": "room500", "tasks": ["room-a", "room-b"]},
    ]
    schedule = _schedule(cohorts, (1, 2, 3, 4))
    assert len(schedule) == 48
    assert {row["controller"] for row in schedule} == set(CONTROLLERS)
    keys = {}
    for row in schedule:
        key = (row["cohort_id"], row["task_id"], row["solver_seed"])
        keys.setdefault(key, []).append(row)
    assert len(keys) == 16
    assert all(len(rows) == 3 for rows in keys.values())
    assert {
        tuple(row["controller"] for row in rows)
        for rows in keys.values()
    } == {
        CONTROLLERS,
        CONTROLLERS[1:] + CONTROLLERS[:1],
        CONTROLLERS[2:] + CONTROLLERS[:2],
    }


def test_maprank_runtime_signature_excludes_only_timing_fields() -> None:
    transition = {
        "decision_index": 3,
        "before_fingerprint": "before",
        "after_fingerprint": "after",
        "action": {"agents": [1, 2], "pp_random_seed": 9},
        "metrics": {
            "conflicts_before": 4,
            "conflicts_after": 2,
            "pp_replan_seconds": 1.0,
        },
        "controller": {
            "selected_candidate_id": "candidate-a",
            "candidate_generation_seconds": 1.0,
            "proposal": {
                "topology_boundary_gate_reason": "legacy_unconditional",
                "topology_boundary_dynamic_seconds": 0.5,
            },
        },
    }
    first = _runtime_transition_signature(transition)
    transition["controller"]["candidate_generation_seconds"] = 9.0
    transition["controller"]["proposal"]["topology_boundary_dynamic_seconds"] = 4.0
    transition["metrics"]["pp_replan_seconds"] = 8.0
    second = _runtime_transition_signature(transition)
    assert first == second
    transition["controller"]["selected_candidate_id"] = "candidate-b"
    assert first != _runtime_transition_signature(transition)
