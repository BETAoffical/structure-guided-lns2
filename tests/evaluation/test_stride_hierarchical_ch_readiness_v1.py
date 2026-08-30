from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from experiments._common import sha256_file
from experiments.stride_hierarchical_ch_labels_v1 import (
    ANCHOR_COMPONENT_SHARED,
    ANCHOR_HOTSPOT_SHARED,
    CONSENSUS_STRUCTURAL,
    STAGE1_PAIR_SCHEMA,
    STAGE2_STATE_SCHEMA,
    THREE_UNIQUE,
)
from experiments.stride_hierarchical_ch_readiness_v1 import (
    LABEL_AUDIT_SHA256,
    LEGACY_FOLD_CONFIG_SHA256,
    REPORT_SCHEMA,
    _validate_stage1,
    _validate_stage2,
    check_hierarchical_ch_readiness,
)


ROOT = Path(__file__).resolve().parents[2]
LABELS_ROOT = ROOT / "build" / "stride-hierarchical-ch-labels-v1"
FOLD_CONFIG = (
    ROOT
    / "configs"
    / "stride_fresh_matched_unique_action_hierarchical_overlay_v1.json"
)


def _check(output: Path, *, labels_root: Path = LABELS_ROOT) -> dict:
    return check_hierarchical_ch_readiness(
        project_root=ROOT,
        labels_root=labels_root,
        output=output,
        label_audit_sha256=LABEL_AUDIT_SHA256,
        legacy_fold_config=FOLD_CONFIG,
        legacy_fold_config_sha256=LEGACY_FOLD_CONFIG_SHA256,
    )


def _metadata(state_id: str, stratum: str) -> dict:
    return {
        "state_occurrence_id": state_id,
        "map_id": f"map-{state_id}",
        "map_family": "synthetic",
        "task_id": f"task-{state_id}",
        "solver_seed": 41,
        "decision_index": 0,
        "hierarchical_stratum": stratum,
        "research_split": "fresh_matched_development",
    }


def _stage1_pair(
    state_id: str,
    stratum: str,
    structural_id: str,
    v2_id: str,
    aliases: list[str],
) -> dict:
    return {
        "schema": STAGE1_PAIR_SCHEMA,
        **_metadata(state_id, stratum),
        "pair_id": f"{state_id}::{structural_id}::{v2_id}",
        "structural_action_id": structural_id,
        "v2_action_id": v2_id,
        "structural_role_aliases": aliases,
        "label": "v2_win",
        "runtime_or_pp_seconds_used_in_label": False,
    }


def _stage2_state(
    state_id: str,
    stratum: str,
    component_id: str,
    hotspot_id: str,
) -> dict:
    return {
        "schema": STAGE2_STATE_SCHEMA,
        **_metadata(state_id, stratum),
        "component_action_id": component_id,
        "hotspot_action_id": hotspot_id,
        "eligibility_condition": "component_action_id_ne_hotspot_action_id",
        "stage1_state_label": "v2_win",
        "any_stage1_structural_win": False,
        "both_stage1_structural_win": False,
        "label": "ambiguous",
        "runtime_or_pp_seconds_used_in_label": False,
    }


def test_current_labels_are_exact_deterministic_no_go(tmp_path: Path) -> None:
    report = _check(tmp_path / "first")
    repeated = _check(tmp_path / "second")

    assert report == repeated
    assert (
        tmp_path / "first" / "readiness_report.json"
    ).read_bytes() == (
        tmp_path / "second" / "readiness_report.json"
    ).read_bytes()
    assert report["schema"] == REPORT_SCHEMA
    assert report["status"] == "NO_GO"
    assert report["integrity"] == {
        "passed": True,
        "source_state_count": 96,
        "excluded_all_equal_state_count": 0,
        "labeled_source_state_count": 96,
        "stage1_pair_count": 160,
        "stage2_state_count": 64,
        "stage2_conditioned_state_count": 64,
        "source_and_cross_stage_consistent": True,
    }

    stage1 = report["stage1"]
    assert stage1["support"]["label_counts"] == {
        "structural_win": 57,
        "v2_win": 15,
        "ambiguous": 88,
    }
    assert stage1["support"]["by_label"]["structural_win"] == {
        "row_count": 57,
        "state_count": 40,
        "map_count": 6,
        "family_group_count": 3,
    }
    assert stage1["support"]["by_label"]["v2_win"] == {
        "row_count": 15,
        "state_count": 11,
        "map_count": 7,
        "family_group_count": 4,
    }
    assert stage1["fit_support"] == {
        "included_labels": ["structural_win", "v2_win"],
        "ambiguous_excluded": True,
        "row_count": 72,
    }
    assert stage1["threshold_calibration"]["row_count"] == 88
    assert stage1["folds"]["fold0"]["label_counts"] == {
        "structural_win": 0,
        "v2_win": 1,
        "ambiguous": 39,
    }
    assert stage1["failure_reasons"] == [
        "stage1.v2_win.row_count=15<40",
        "stage1.structural_win.map_count=6<8",
        "stage1.v2_win.map_count=7<8",
        "stage1.structural_win.family_group_count=3<4",
        "stage1.fold0.missing=structural_win",
    ]
    assert stage1["readiness_passed"] is False
    assert stage1["training_authorized"] is False

    stage2 = report["stage2"]
    assert stage2["condition"] == "component_action_id_ne_hotspot_action_id"
    assert stage2["all_state_count"] == 64
    assert stage2["conditioned_state_count"] == 64
    assert stage2["unconditioned_state_count"] == 0
    assert stage2["stage1_positive_diagnostic_state_count"] == 32
    assert stage2["stage1_nonpositive_diagnostic_state_count"] == 32
    assert stage2["stage1_positive_is_diagnostic_only"] is True
    assert stage2["support"]["label_counts"] == {
        "component_win": 22,
        "hotspot_win": 5,
        "ambiguous": 37,
    }
    assert stage2["support"]["by_label"]["component_win"] == {
        "row_count": 22,
        "state_count": 22,
        "map_count": 6,
        "family_group_count": 3,
    }
    assert stage2["support"]["by_label"]["hotspot_win"] == {
        "row_count": 5,
        "state_count": 5,
        "map_count": 2,
        "family_group_count": 2,
    }
    assert stage2["fit_support"] == {
        "included_labels": ["component_win", "hotspot_win"],
        "ambiguous_excluded": True,
        "row_count": 27,
    }
    assert stage2["threshold_calibration"]["row_count"] == 37
    assert stage2["failure_reasons"] == [
        "stage2.hotspot_win.row_count=5<16",
        "stage2.hotspot_win.map_count=2<4",
        "stage2.hotspot_win.family_group_count=2<3",
        "stage2.fold0.missing=component_win,hotspot_win",
        "stage2.fold3.missing=hotspot_win",
    ]
    assert stage2["readiness_passed"] is False
    assert stage2["training_authorized"] is False

    assert report["overall"]["readiness_passed"] is False
    assert report["overall"]["training_authorized"] is False
    assert report["training_authorized"] is False
    assert report["model_fit_executed"] is False
    assert report["model_exported"] is False
    assert report["runtime_authorized"] is False
    assert report["runtime_or_ttf_claim_authorized"] is False


def test_readiness_accepts_all_exact_stage2_partitions_without_stage1_win() -> None:
    stage1_rows = [
        _stage1_pair(
            "consensus",
            CONSENSUS_STRUCTURAL,
            "c",
            "v",
            ["component16", "hotspot16"],
        ),
        _stage1_pair("three", THREE_UNIQUE, "c", "v", ["component16"]),
        _stage1_pair("three", THREE_UNIQUE, "h", "v", ["hotspot16"]),
        _stage1_pair(
            "anchor-component",
            ANCHOR_COMPONENT_SHARED,
            "h",
            "v",
            ["hotspot16"],
        ),
        _stage1_pair(
            "anchor-hotspot",
            ANCHOR_HOTSPOT_SHARED,
            "c",
            "v",
            ["component16"],
        ),
    ]
    map_to_fold = {
        str(row["map_id"]): "fold0" for row in stage1_rows
    }
    stage1_by_state = _validate_stage1(stage1_rows, map_to_fold)
    stage2_rows = [
        _stage2_state("three", THREE_UNIQUE, "c", "h"),
        _stage2_state(
            "anchor-component", ANCHOR_COMPONENT_SHARED, "v", "h"
        ),
        _stage2_state("anchor-hotspot", ANCHOR_HOTSPOT_SHARED, "c", "v"),
    ]

    validated = _validate_stage2(stage2_rows, map_to_fold, stage1_by_state)

    assert set(validated) == {"three", "anchor-component", "anchor-hotspot"}
    assert all(
        row["any_stage1_structural_win"] is False
        for row in validated.values()
    )


def test_hash_tampering_fails_closed(tmp_path: Path) -> None:
    labels = tmp_path / "labels"
    shutil.copytree(LABELS_ROOT, labels)
    stage1 = labels / "stage1_pair_labels.jsonl"
    stage1.write_text(stage1.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="stage1_pair_labels SHA256 mismatch"):
        _check(tmp_path / "output", labels_root=labels)

    clean = tmp_path / "clean"
    shutil.copytree(LABELS_ROOT, clean)
    audit = clean / "label_audit_report.json"
    audit.write_text(audit.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="label audit report SHA256 mismatch"):
        _check(tmp_path / "audit-output", labels_root=clean)


def test_legacy_fold_change_fails_even_when_its_new_hash_is_supplied(
    tmp_path: Path,
) -> None:
    changed_path = tmp_path / "changed-folds.json"
    changed = json.loads(FOLD_CONFIG.read_text(encoding="utf-8"))
    changed["map_folds"]["fold0"], changed["map_folds"]["fold1"] = (
        changed["map_folds"]["fold1"],
        changed["map_folds"]["fold0"],
    )
    changed_path.write_text(
        json.dumps(changed, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="legacy map folds changed"):
        check_hierarchical_ch_readiness(
            project_root=ROOT,
            labels_root=LABELS_ROOT,
            output=tmp_path / "output",
            label_audit_sha256=LABEL_AUDIT_SHA256,
            legacy_fold_config=changed_path,
            legacy_fold_config_sha256=sha256_file(changed_path),
        )
