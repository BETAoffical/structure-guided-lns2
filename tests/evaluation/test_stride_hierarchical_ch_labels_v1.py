from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments._common import sha256_file
from experiments.stride_hierarchical_ch_labels_v1 import (
    ANCHOR_COMPONENT_SHARED,
    ANCHOR_HOTSPOT_SHARED,
    CONFIG_SCHEMA,
    CONSENSUS_STRUCTURAL,
    REPORT_SCHEMA,
    SINGLE_UNIQUE,
    SOURCE_STATE_SCHEMA,
    THREE_UNIQUE,
    _canonical_partition,
    build_hierarchical_ch_labels,
    classify_three_way,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE_MANIFEST = (
    ROOT
    / "build"
    / "stride-fresh-matched-unique-action-hierarchical-overlay-v1"
    / "h1_state_manifest.jsonl"
)
LEGACY_CONSENSUS_MANIFEST = (
    ROOT
    / "build"
    / "stride-hierarchical-ch-legacy-consensus-h1-v1"
    / "h1_state_manifest.jsonl"
)
LEGACY_CONSENSUS_STATE_SCHEMA = (
    "lns2.stride.hierarchical_ch_legacy_consensus_h1_state.v1"
)


def _trials(score: float, *, pp_seconds: float = 0.01) -> list[dict]:
    return [
        {
            "trial_index": index,
            "pp_seed": 1000 + index,
            "normalized_conflict_reduction": score,
            "no_progress": score <= 0.0,
            "rollback": False,
            "time_limit": False,
            "pp_seconds": pp_seconds,
        }
        for index in range(16)
    ]


def _config() -> dict:
    return {
        "schema": CONFIG_SCHEMA,
        "experiment_id": "stride_hierarchical_ch_labels_v1_test",
        "source": {
            "manifest": {
                "path": SOURCE_MANIFEST.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(SOURCE_MANIFEST),
            }
        },
        "expected": {
            "state_count": 96,
            "unique_action_count": 256,
            "trial_count": 4096,
            "stage1_pair_count": 160,
            "stage2_state_count": 64,
        },
    }


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_partition_source(root: Path) -> Path:
    source = root / "source"
    source.mkdir(parents=True)
    partitions = {
        CONSENSUS_STRUCTURAL: {"v2_anchor": "v", "component16": "c", "hotspot16": "c"},
        THREE_UNIQUE: {"v2_anchor": "v", "component16": "c", "hotspot16": "h"},
        ANCHOR_COMPONENT_SHARED: {"v2_anchor": "v", "component16": "v", "hotspot16": "h"},
        ANCHOR_HOTSPOT_SHARED: {"v2_anchor": "v", "component16": "c", "hotspot16": "v"},
        SINGLE_UNIQUE: {"v2_anchor": "v", "component16": "v", "hotspot16": "v"},
    }
    agents = {"v": [1, 2], "c": [3, 4], "h": [5, 6]}
    scores = {"v": 0.20, "c": 0.10, "h": 0.00}
    manifest = []
    for ordinal, (partition, role_map) in enumerate(partitions.items()):
        state_id = f"partition-state-{ordinal}"
        action_ids = list(dict.fromkeys(role_map.values()))
        actions = [
            {"action_id": action_id, "agents": agents[action_id]}
            for action_id in action_ids
        ]
        trials = []
        for action_id in action_ids:
            for index in range(16):
                trials.append(
                    {
                        "trial_identity": f"{state_id}::{action_id}::{index}",
                        "action_id": action_id,
                        "trial_index": index,
                        "pp_seed": 1000 + index,
                        "normalized_conflict_reduction": scores[action_id],
                        "no_progress": False,
                        "rollback": False,
                        "time_limit": False,
                        "pp_seconds": 0.01,
                        "runtime_used_in_label": False,
                    }
                )
        payload = {
            "schema": SOURCE_STATE_SCHEMA,
            "complete": True,
            "state_occurrence_id": state_id,
            "runtime_used_in_label": False,
            "state_row": {
                "state_occurrence_id": state_id,
                "map_id": f"map-{ordinal}",
                "map_family": "synthetic",
                "task_id": f"task-{ordinal}",
                "solver_seed": 41,
                "decision_index": ordinal,
                "research_split": "sequential_development",
                "unique_actions": actions,
                "unique_action_count": len(actions),
                "role_to_action_id": role_map,
                "hierarchical_stratum": partition,
            },
            "trials": trials,
        }
        state_file = source / f"{state_id}.json"
        state_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest.append(
            {
                "state_occurrence_id": state_id,
                "state_file": state_file.name,
                "state_sha256": sha256_file(state_file),
                "unique_action_count": len(actions),
                "trial_count": len(trials),
            }
        )
    manifest_path = source / "manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in manifest),
        encoding="utf-8",
    )
    return manifest_path


def test_bidirectional_h1_produces_three_states_and_ignores_timing() -> None:
    left = _trials(0.10, pp_seconds=100.0)
    right = _trials(0.00, pp_seconds=0.001)
    left_win = classify_three_way(
        left,
        right,
        left_label="component_win",
        right_label="hotspot_win",
    )
    right_win = classify_three_way(
        right,
        left,
        left_label="component_win",
        right_label="hotspot_win",
    )
    ambiguous = classify_three_way(
        _trials(0.05, pp_seconds=0.001),
        _trials(0.05, pp_seconds=100.0),
        left_label="component_win",
        right_label="hotspot_win",
    )

    assert left_win["label"] == "component_win"
    assert right_win["label"] == "hotspot_win"
    assert ambiguous["label"] == "ambiguous"
    assert left_win["runtime_or_pp_seconds_used_in_label"] is False
    with pytest.raises(ValueError, match="complete paired trials"):
        classify_three_way(
            left[:-1],
            right,
            left_label="component_win",
            right_label="hotspot_win",
        )


def test_exact_partition_product_keeps_shared_anchor_stage2_and_excludes_all_equal(
    tmp_path: Path,
) -> None:
    assert _canonical_partition((1,), (2,), (2,)) == CONSENSUS_STRUCTURAL
    assert _canonical_partition((1,), (2,), (3,)) == THREE_UNIQUE
    assert _canonical_partition((1,), (1,), (2,)) == ANCHOR_COMPONENT_SHARED
    assert _canonical_partition((1,), (2,), (1,)) == ANCHOR_HOTSPOT_SHARED
    assert _canonical_partition((1,), (1,), (1,)) == SINGLE_UNIQUE

    manifest = _write_partition_source(tmp_path)
    config = {
        "schema": CONFIG_SCHEMA,
        "experiment_id": "exact-partition-label-test",
        "source": {
            "manifest": {
                "path": manifest.relative_to(tmp_path).as_posix(),
                "sha256": sha256_file(manifest),
            }
        },
        "expected": {
            "state_count": 4,
            "unique_action_count": 9,
            "trial_count": 144,
            "stage1_pair_count": 5,
            "stage2_state_count": 3,
        },
    }
    report = build_hierarchical_ch_labels(
        config,
        project_root=tmp_path,
        output=tmp_path / "labels",
    )

    assert report["canonical_partition_support"] == {
        CONSENSUS_STRUCTURAL: 1,
        THREE_UNIQUE: 1,
        ANCHOR_COMPONENT_SHARED: 1,
        ANCHOR_HOTSPOT_SHARED: 1,
        SINGLE_UNIQUE: 1,
    }
    assert report["partition_exclusion"]["state_count"] == 1
    stage1 = _read_jsonl(tmp_path / "labels" / "stage1_pair_labels.jsonl")
    stage2 = _read_jsonl(tmp_path / "labels" / "stage2_state_labels.jsonl")
    assert all(row["structural_action_id"] != row["v2_action_id"] for row in stage1)
    assert {row["hierarchical_stratum"] for row in stage2} == {
        THREE_UNIQUE,
        ANCHOR_COMPONENT_SHARED,
        ANCHOR_HOTSPOT_SHARED,
    }
    assert all(
        row["component_action_id"] != row["hotspot_action_id"] for row in stage2
    )
    shared = [
        row
        for row in stage2
        if row["hierarchical_stratum"]
        in {ANCHOR_COMPONENT_SHARED, ANCHOR_HOTSPOT_SHARED}
    ]
    assert len(shared) == 2
    assert all(row["any_stage1_structural_win"] is False for row in shared)
    assert report["label_contract"]["stage1_positive_used_to_select_stage2"] is False


def test_current_96_state_h1_reproduces_hierarchical_support(tmp_path: Path) -> None:
    report = build_hierarchical_ch_labels(
        _config(), project_root=ROOT, output=tmp_path
    )

    assert report["schema"] == REPORT_SCHEMA
    assert report["complete"] is True
    assert report["observed"] == {
        "state_count": 96,
        "unique_action_count": 256,
        "trial_count": 4096,
        "stage1_pair_count": 160,
        "stage2_state_count": 64,
    }
    assert report["stage1_pair_support"]["label_counts"] == {
        "structural_win": 57,
        "v2_win": 15,
        "ambiguous": 88,
    }
    assert report["stage1_consensus_pair_support"]["label_counts"] == {
        "structural_win": 8,
        "v2_win": 3,
        "ambiguous": 21,
    }
    assert report["stage2_all_support"]["label_counts"] == {
        "component_win": 22,
        "hotspot_win": 5,
        "ambiguous": 37,
    }
    assert report["stage2_any_conditioned_support"]["label_counts"] == {
        "component_win": 21,
        "hotspot_win": 1,
        "ambiguous": 10,
    }
    assert report["stage2_both_conditioned_support"]["label_counts"] == {
        "component_win": 11,
        "hotspot_win": 0,
        "ambiguous": 6,
    }
    assert report["training_authorized"] is False
    assert report["model_fit_executed"] is False
    assert report["model_exported"] is False
    assert report["runtime_or_ttf_claim_authorized"] is False

    stage1_pairs = _read_jsonl(tmp_path / "stage1_pair_labels.jsonl")
    stage1_states = _read_jsonl(tmp_path / "stage1_state_labels.jsonl")
    stage2_states = _read_jsonl(tmp_path / "stage2_state_labels.jsonl")
    assert len(stage1_pairs) == 160
    assert len(stage1_states) == 96
    assert len(stage2_states) == 64
    consensus = [
        row
        for row in stage1_pairs
        if row["hierarchical_stratum"] == "consensus_structural"
    ]
    assert len(consensus) == 32
    assert all(
        row["structural_role_aliases"] == ["component16", "hotspot16"]
        for row in consensus
    )
    assert all(
        row["hierarchical_stratum"] == "three_unique" for row in stage2_states
    )


def test_source_manifest_is_hash_pinned_and_expected_counts_fail_closed(
    tmp_path: Path,
) -> None:
    changed_hash = _config()
    changed_hash["source"]["manifest"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="manifest SHA256 mismatch"):
        build_hierarchical_ch_labels(
            changed_hash, project_root=ROOT, output=tmp_path / "hash"
        )

    changed_count = _config()
    changed_count["expected"]["stage1_pair_count"] = 159
    with pytest.raises(ValueError, match="observed product counts changed"):
        build_hierarchical_ch_labels(
            changed_count, project_root=ROOT, output=tmp_path / "count"
        )


def test_stage2_count_may_be_zero_for_exact_consensus_only_sources(
    tmp_path: Path,
) -> None:
    config = _config()
    config["source"]["state_schema"] = SOURCE_STATE_SCHEMA
    config["expected"]["stage2_state_count"] = 0
    # The zero is a valid contract value; the real 64-row source must still
    # fail closed on its observed product rather than at config validation.
    with pytest.raises(ValueError, match="observed product counts changed"):
        build_hierarchical_ch_labels(
            config,
            project_root=ROOT,
            output=tmp_path,
        )


def test_completed_legacy_consensus_extension_builds_stage1_only(
    tmp_path: Path,
) -> None:
    config = {
        "schema": CONFIG_SCHEMA,
        "experiment_id": "stride_hierarchical_ch_legacy_consensus_labels_test",
        "source": {
            "manifest": {
                "path": LEGACY_CONSENSUS_MANIFEST.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(LEGACY_CONSENSUS_MANIFEST),
            },
            "state_schema": LEGACY_CONSENSUS_STATE_SCHEMA,
        },
        "expected": {
            "state_count": 60,
            "unique_action_count": 120,
            "trial_count": 1920,
            "stage1_pair_count": 60,
            "stage2_state_count": 0,
        },
    }

    report = build_hierarchical_ch_labels(
        config,
        project_root=ROOT,
        output=tmp_path,
    )

    assert report["observed"] == config["expected"]
    assert report["stage1_consensus_pair_support"]["label_counts"] == {
        "structural_win": 20,
        "v2_win": 2,
        "ambiguous": 38,
    }
    assert report["stage2_all_support"]["row_count"] == 0
    assert report["training_authorized"] is False
