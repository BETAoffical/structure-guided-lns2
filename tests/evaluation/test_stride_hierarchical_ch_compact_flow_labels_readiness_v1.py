from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments._common import sha256_file
from experiments.stride_hierarchical_ch_compact_flow_h1_collection_v1 import (
    H1_COLLECTION_SCHEMA,
    H1_MANIFEST_ROW_SCHEMA,
    H1_STATE_SCHEMA,
    H1_TRIAL_SCHEMA,
    paired_pp_seed,
)
from experiments.stride_hierarchical_ch_compact_flow_labels_readiness_v1 import (
    DEVELOPMENT_MAPS,
    LABEL_DIAGNOSTICS_SCHEMA,
    READINESS_SCHEMA,
    TRAIN_MAP_FOLDS,
    build_compact_flow_labels_readiness,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "configs"
    / "stride_hierarchical_ch_compact_flow_labels_readiness_v1.json"
)
COLLECTION_RELATIVE = Path(
    "build/stride-hierarchical-ch-compact-flow-h1-collection-v1"
)
IDENTITY = "a" * 64


def _json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def _fold(map_id: str) -> str | None:
    return next(
        (fold for fold, maps in TRAIN_MAP_FOLDS.items() if map_id in maps), None
    )


def _action(action_id: str, start: int, aliases: list[str]) -> dict:
    agents = list(range(start, start + 16))
    return {
        "action_id": action_id,
        "agents": agents,
        "actual_size": 16,
        "role_aliases": aliases,
        "candidate_ids_by_role": {role: f"candidate-{role}" for role in aliases},
        "contains_v2_anchor_role": "v2_anchor" in aliases,
        "structural_role_aliases": [
            role for role in aliases if role in {"component16", "hotspot16"}
        ],
    }


def _state(
    state_id: str,
    *,
    split: str,
    map_id: str,
    depth: str,
    partition: str,
) -> tuple[dict, list[dict]]:
    if partition == "consensus_structural":
        actions = [
            _action("v", 0, ["v2_anchor"]),
            _action("c", 16, ["component16", "hotspot16"]),
        ]
        roles = {"v2_anchor": "v", "component16": "c", "hotspot16": "c"}
        scores = {"v": 0.1, "c": 0.5}
    elif partition == "three_unique":
        actions = [
            _action("v", 0, ["v2_anchor"]),
            _action("c", 16, ["component16"]),
            _action("h", 32, ["hotspot16"]),
        ]
        roles = {"v2_anchor": "v", "component16": "c", "hotspot16": "h"}
        scores = {"v": 0.7, "c": 0.1, "h": 0.3}
    elif partition == "anchor_component_shared":
        actions = [
            _action("v", 0, ["v2_anchor", "component16"]),
            _action("h", 16, ["hotspot16"]),
        ]
        roles = {"v2_anchor": "v", "component16": "v", "hotspot16": "h"}
        scores = {"v": 0.1, "h": 0.5}
    elif partition == "anchor_hotspot_shared":
        actions = [
            _action("v", 0, ["v2_anchor", "hotspot16"]),
            _action("c", 16, ["component16"]),
        ]
        roles = {"v2_anchor": "v", "component16": "c", "hotspot16": "v"}
        scores = {"v": 0.1, "c": 0.5}
    elif partition == "single_unique":
        actions = [_action("v", 0, ["v2_anchor", "component16", "hotspot16"])]
        roles = {"v2_anchor": "v", "component16": "v", "hotspot16": "v"}
        scores = {"v": 0.1}
    else:  # pragma: no cover - fixture guard
        raise AssertionError(partition)

    state_row = {
        "schema": "lns2.stride.hierarchical_ch_compact_flow_h1_preflight_state.v1",
        "state_id": state_id,
        "state_occurrence_id": state_id,
        "split": split,
        "map_id": map_id,
        "map_family": "synthetic",
        "task_id": f"task-{state_id}",
        "solver_seed": 41,
        "decision_index": 0,
        "depth_band": depth,
        "train_fold": _fold(map_id) if split == "train" else None,
        "canonical_partition": partition,
        "hierarchical_stratum": partition,
        "unique_actions": actions,
        "unique_action_count": len(actions),
        "role_to_action_id": roles,
        "stage2_eligible": roles["component16"] != roles["hotspot16"],
        "all_equal_excluded": False,
        "runtime_used_in_label": False,
        "sequential_design_only": True,
        "training_authorized": False,
        "final_claim_authorized": False,
        "runtime_claim_authorized": False,
    }
    trials: list[dict] = []
    for action in actions:
        action_id = str(action["action_id"])
        for index in range(16):
            trials.append(
                {
                    "schema": H1_TRIAL_SCHEMA,
                    "trial_identity": f"{state_id}-{action_id}-{index}",
                    "state_occurrence_id": state_id,
                    "action_id": action_id,
                    "trial_index": index,
                    "pp_seed": paired_pp_seed(state_id, index),
                    "normalized_conflict_reduction": scores[action_id],
                    "no_progress": False,
                    "rollback": False,
                    "time_limit": False,
                    "runtime_used_in_label": False,
                    "sequential_design_only": True,
                    "training_authorized": False,
                    "final_claim_authorized": False,
                    "runtime_claim_authorized": False,
                }
            )
    payload = {
        "schema": H1_STATE_SCHEMA,
        "identity": IDENTITY,
        "complete": True,
        "state_occurrence_id": state_id,
        "state_row": state_row,
        "trials": trials,
        "runtime_used_in_label": False,
        "sequential_design_only": True,
        "training_authorized": False,
        "final_claim_authorized": False,
        "runtime_claim_authorized": False,
    }
    return payload, trials


def _fixture_project(tmp_path: Path, *, single_unique: bool = False) -> tuple[Path, Path]:
    project = tmp_path / "project"
    config_path = project / "configs" / CONFIG.name
    config_path.parent.mkdir(parents=True)
    config_path.write_bytes(CONFIG.read_bytes())
    collection = project / COLLECTION_RELATIVE
    specifications = [
        ("s-consensus", "train", "den404d", "d0", "consensus_structural"),
        ("s-three", "train", "lak101d", "d1_3", "three_unique"),
        (
            "s-anchor-component",
            "development",
            DEVELOPMENT_MAPS[0],
            "d0",
            "anchor_component_shared",
        ),
        (
            "s-anchor-hotspot",
            "development",
            DEVELOPMENT_MAPS[1],
            "d4plus",
            "anchor_hotspot_shared",
        ),
    ]
    if single_unique:
        specifications = [
            ("s-all", "train", "den404d", "d0", "single_unique")
        ]

    manifest: list[dict] = []
    raw: list[dict] = []
    action_count = 0
    for state_id, split, map_id, depth, partition in specifications:
        payload, trials = _state(
            state_id,
            split=split,
            map_id=map_id,
            depth=depth,
            partition=partition,
        )
        state_path = collection / "h1_states" / f"{state_id}.json"
        _json(state_path, payload)
        action_count += int(payload["state_row"]["unique_action_count"])
        raw.extend(trials)
        manifest.append(
            {
                "schema": H1_MANIFEST_ROW_SCHEMA,
                "state_occurrence_id": state_id,
                "state_file": f"h1_states/{state_id}.json",
                "state_sha256": sha256_file(state_path),
                "unique_action_count": int(
                    payload["state_row"]["unique_action_count"]
                ),
                "trial_count": len(trials),
                "sequential_design_only": True,
                "training_authorized": False,
                "final_claim_authorized": False,
                "runtime_claim_authorized": False,
            }
        )
    manifest_path = collection / "h1_state_manifest.jsonl"
    raw_path = collection / "raw_unique_action_outcomes.jsonl"
    _jsonl(manifest_path, manifest)
    _jsonl(raw_path, raw)
    report = {
        "schema": H1_COLLECTION_SCHEMA,
        "experiment_id": "stride_hierarchical_ch_compact_flow_h1_collection_v1",
        "status": "complete",
        "complete": True,
        "run_identity": IDENTITY,
        "selected_state_count": len(manifest),
        "excluded_all_equal_state_count": 0,
        "expected_state_count": len(manifest),
        "observed_valid_state_count": len(manifest),
        "expected_unique_action_count": action_count,
        "observed_unique_action_count": action_count,
        "expected_trial_count": len(raw),
        "observed_trial_count": len(raw),
        "invalid_state_count": 0,
        "workers": 16,
        "workers_in_fingerprint": True,
        "artifacts": {
            "label_source_state_manifest": {
                "file": manifest_path.name,
                "sha256": sha256_file(manifest_path),
                "state_schema": H1_STATE_SCHEMA,
            },
            "raw_unique_action_outcomes": {
                "file": raw_path.name,
                "sha256": sha256_file(raw_path),
            },
        },
        "exact_agent_tuple_deduplication": True,
        "stage1_population": "v2_vs_each_distinct_structural_action",
        "stage2_population": "component_vs_hotspot_only_when_exact_sets_differ",
        "all_equal_excluded": True,
        "outcome_based_selection": False,
        "backfill_allowed": False,
        "sequential_design_only": True,
        "training_authorized": False,
        "model_fit_executed": False,
        "final_claim_authorized": False,
        "runtime_or_ttf_claim_authorized": False,
        "map_disjoint_confirmation_required": True,
    }
    _json(collection / "h1_collection_report.json", report)
    return config_path, collection


def test_synthetic_fixture_builds_exact_hierarchy_and_five_partition_report(
    tmp_path: Path,
) -> None:
    config, _collection = _fixture_project(tmp_path)
    output = tmp_path / "labels"
    result = build_compact_flow_labels_readiness(config, output=output, workers=16)

    labels = result["labels"]
    readiness = result["readiness"]
    assert labels["schema"] == LABEL_DIAGNOSTICS_SCHEMA
    assert labels["input_integrity"]["state_count"] == 4
    assert labels["label_counts"] == {
        "stage1_pair_count": 5,
        "stage1_state_count": 4,
        "stage2_state_count": 3,
    }
    assert labels["canonical_partition_support"] == {
        "train": {
            "consensus_structural": 1,
            "three_unique": 1,
            "anchor_component_shared": 0,
            "anchor_hotspot_shared": 0,
            "single_unique": 0,
        },
        "development": {
            "consensus_structural": 0,
            "three_unique": 0,
            "anchor_component_shared": 1,
            "anchor_hotspot_shared": 1,
            "single_unique": 0,
        },
    }
    stage2 = [
        json.loads(line)
        for line in (output / "stage2_state_labels.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {row["state_occurrence_id"] for row in stage2} == {
        "s-three",
        "s-anchor-component",
        "s-anchor-hotspot",
    }
    assert all(
        row["component_action_id"] != row["hotspot_action_id"] for row in stage2
    )
    assert readiness["schema"] == READINESS_SCHEMA
    assert readiness["development_has_training_folds"] is False
    assert readiness["training_authorized"] is False
    assert readiness["model_fit_executed"] is False
    assert readiness["final_claim_authorized"] is False


def test_raw_trial_union_tampering_fails_even_with_rehashed_raw_artifact(
    tmp_path: Path,
) -> None:
    config, collection = _fixture_project(tmp_path)
    raw_path = collection / "raw_unique_action_outcomes.jsonl"
    raw = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]
    raw[0]["normalized_conflict_reduction"] = 0.99
    _jsonl(raw_path, raw)
    report_path = collection / "h1_collection_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["artifacts"]["raw_unique_action_outcomes"]["sha256"] = sha256_file(
        raw_path
    )
    _json(report_path, report)

    with pytest.raises(ValueError, match="raw outcomes differ"):
        build_compact_flow_labels_readiness(
            config, output=tmp_path / "labels", workers=16, dry_run=True
        )


def test_state_payload_hash_and_incomplete_collection_fail_closed(
    tmp_path: Path,
) -> None:
    config, collection = _fixture_project(tmp_path)
    state_path = collection / "h1_states" / "s-consensus.json"
    state_path.write_text(
        state_path.read_text(encoding="utf-8") + " ", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="state payload SHA256 mismatch"):
        build_compact_flow_labels_readiness(
            config, output=tmp_path / "labels", workers=16, dry_run=True
        )

    config2, collection2 = _fixture_project(tmp_path / "incomplete")
    report_path = collection2 / "h1_collection_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["complete"] = False
    report["status"] = "INTEGRITY_FAIL_NO_BACKFILL"
    _json(report_path, report)
    with pytest.raises(ValueError, match="collection trust contract"):
        build_compact_flow_labels_readiness(
            config2, output=tmp_path / "labels2", workers=16, dry_run=True
        )


def test_all_equal_manifest_row_is_rejected_and_workers_are_fingerprinted(
    tmp_path: Path,
) -> None:
    config, _collection = _fixture_project(tmp_path, single_unique=True)
    with pytest.raises(ValueError, match="split/depth/partition changed"):
        build_compact_flow_labels_readiness(
            config, output=tmp_path / "labels", workers=16, dry_run=True
        )

    config2, _collection2 = _fixture_project(tmp_path / "workers")
    with pytest.raises(ValueError, match="workers must match"):
        build_compact_flow_labels_readiness(
            config2, output=tmp_path / "labels2", workers=15, dry_run=True
        )
