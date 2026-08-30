from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from experiments.stride_hierarchical_ch_combined_readiness_v1 import (
    LEGACY_CONSENSUS_LABEL_AUDIT_SHA256,
    ORIGINAL_LABEL_AUDIT_SHA256,
    REPORT_SCHEMA,
    check_hierarchical_ch_combined_readiness,
)
from experiments.stride_hierarchical_ch_readiness_v1 import (
    LEGACY_FOLD_CONFIG_SHA256,
)


ROOT = Path(__file__).resolve().parents[2]
ORIGINAL_ROOT = ROOT / "build" / "stride-hierarchical-ch-labels-v1"
EXTENSION_ROOT = (
    ROOT / "build" / "stride-hierarchical-ch-legacy-consensus-labels-v1"
)
FOLD_CONFIG = (
    ROOT
    / "configs"
    / "stride_fresh_matched_unique_action_hierarchical_overlay_v1.json"
)


def _check(
    output: Path,
    *,
    original_root: Path = ORIGINAL_ROOT,
    extension_root: Path = EXTENSION_ROOT,
) -> dict:
    return check_hierarchical_ch_combined_readiness(
        project_root=ROOT,
        original_labels_root=original_root,
        legacy_consensus_labels_root=extension_root,
        output=output,
        original_label_audit_sha256=ORIGINAL_LABEL_AUDIT_SHA256,
        legacy_consensus_label_audit_sha256=(
            LEGACY_CONSENSUS_LABEL_AUDIT_SHA256
        ),
        legacy_fold_config=FOLD_CONFIG,
        legacy_fold_config_sha256=LEGACY_FOLD_CONFIG_SHA256,
    )


def test_combined_frozen_labels_are_deterministic_no_go(tmp_path: Path) -> None:
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
        "source_state_disjoint": True,
        "original_source_state_count": 96,
        "legacy_consensus_source_state_count": 60,
        "combined_source_state_count": 156,
        "original_stage1_pair_count": 160,
        "legacy_consensus_stage1_pair_count": 60,
        "combined_stage1_pair_count": 220,
        "stage2_original_only": True,
        "stage2_state_count": 64,
        "stage2_conditioned_state_count": 32,
        "exact_action_pair_labels_validated": True,
    }
    assert report["research_split_boundary"]["all_inputs_training_only"] is True
    assert report["research_split_boundary"][
        "map_disjoint_confirmation_required"
    ] is True

    stage1 = report["stage1"]
    assert stage1["support"]["row_count"] == 220
    assert stage1["support"]["label_counts"] == {
        "structural_win": 77,
        "v2_win": 17,
        "ambiguous": 126,
    }
    assert stage1["fit_support"] == {
        "included_labels": ["structural_win", "v2_win"],
        "ambiguous_excluded": True,
        "row_count": 94,
    }
    assert stage1["threshold_calibration"]["label"] == "ambiguous"
    assert stage1["threshold_calibration"]["use"] == (
        "threshold_calibration_only"
    )
    assert stage1["threshold_calibration"]["fit_included"] is False
    assert stage1["threshold_calibration"]["row_count"] == 126
    assert stage1["readiness_passed"] is False
    assert stage1["training_authorized"] is False

    stage2 = report["stage2"]
    assert stage2["source"] == "original_only_extension_is_consensus_structural"
    assert stage2["condition"] == "any_stage1_structural_win"
    assert stage2["all_state_count"] == 64
    assert stage2["conditioned_state_count"] == 32
    assert stage2["extension_state_count"] == 0
    assert stage2["support"]["label_counts"] == {
        "component_win": 21,
        "hotspot_win": 1,
        "ambiguous": 10,
    }
    assert stage2["fit_support"] == {
        "included_labels": ["component_win", "hotspot_win"],
        "ambiguous_excluded": True,
        "row_count": 22,
    }
    assert stage2["threshold_calibration"]["row_count"] == 10
    assert stage2["readiness_passed"] is False
    assert stage2["training_authorized"] is False

    assert report["overall"]["readiness_passed"] is False
    assert report["overall"]["training_authorized"] is False
    assert report["training_authorized"] is False
    assert report["model_fit_executed"] is False
    assert report["model_exported"] is False
    assert report["runtime_authorized"] is False
    assert report["runtime_or_ttf_claim_authorized"] is False


def test_each_frozen_fold_requires_both_decisive_classes(tmp_path: Path) -> None:
    report = _check(tmp_path / "output")

    assert report["thresholds"]["stage1"][
        "each_frozen_fold_requires_both_decisive_classes"
    ] is True
    assert report["thresholds"]["stage2"][
        "each_frozen_fold_requires_both_decisive_classes"
    ] is True
    assert report["stage1"]["folds"]["fold0"][
        "missing_decisive_classes"
    ] == []
    assert all(
        fold["has_both_decisive_classes"]
        for fold in report["stage1"]["folds"].values()
    )
    assert report["stage2"]["folds"]["fold0"][
        "missing_decisive_classes"
    ] == ["component_win", "hotspot_win"]


def test_either_input_tampering_fails_closed(tmp_path: Path) -> None:
    extension = tmp_path / "extension"
    shutil.copytree(EXTENSION_ROOT, extension)
    labels = extension / "stage1_pair_labels.jsonl"
    labels.write_text(labels.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(
        ValueError,
        match="legacy_consensus_extension stage1_pair_labels SHA256 mismatch",
    ):
        _check(tmp_path / "extension-output", extension_root=extension)

    original = tmp_path / "original"
    shutil.copytree(ORIGINAL_ROOT, original)
    audit = original / "label_audit_report.json"
    audit.write_text(audit.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="original label audit report SHA256 mismatch"):
        _check(tmp_path / "original-output", original_root=original)


def test_registered_input_identities_cannot_be_swapped(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="original label audit report SHA256 mismatch"):
        _check(
            tmp_path / "output",
            original_root=EXTENSION_ROOT,
            extension_root=ORIGINAL_ROOT,
        )
