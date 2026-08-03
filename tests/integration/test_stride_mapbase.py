from __future__ import annotations

import json
from pathlib import Path

from experiments._common import sha256_file
from experiments.repair_collection import _read_jsonl
from experiments.stride_mapbase import (
    build_mapbase_selection,
    validate_mapbase_config,
)
from experiments.stride_maprank import (
    prepare_maprank_selection,
    validate_maprank_design,
    validate_maprank_training_config,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "stride_mapbase_collection.json"
MAPRANK_DESIGN_PATH = ROOT / "configs" / "stride_maprank_design.json"
MAPRANK_TRAINING_PATH = ROOT / "configs" / "stride_maprank_training.json"


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
