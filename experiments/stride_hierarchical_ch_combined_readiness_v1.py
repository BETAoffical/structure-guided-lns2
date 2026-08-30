from __future__ import annotations

import collections
from pathlib import Path
from typing import Any, Iterable

from experiments._common import sha256_file
from experiments.repair_collection import _read_json, _read_jsonl, _write_json
from experiments.stride_hierarchical_ch_labels_v1 import (
    REPORT_SCHEMA as LABEL_AUDIT_SCHEMA,
    STAGE1_PAIR_SCHEMA,
    STAGE2_STATE_SCHEMA,
)
from experiments.stride_hierarchical_ch_readiness_v1 import (
    FROZEN_LEGACY_FOLDS,
    LEGACY_FOLD_CONFIG_SHA256,
)


REPORT_SCHEMA = "lns2.stride.hierarchical_ch_combined_readiness.v1"
EXPERIMENT_ID = "stride_hierarchical_ch_combined_readiness_v1"

ORIGINAL = {
    "name": "original",
    "root": "build/stride-hierarchical-ch-labels-v1",
    "experiment_id": "stride_hierarchical_ch_labels_v1",
    "audit_sha256": (
        "57e3ab03275e80bc764cdab7bbcd9cea0e50cee7aac4f5eccd9b5b41a71a670a"
    ),
    "manifest": (
        "build/stride-fresh-matched-unique-action-hierarchical-overlay-v1/"
        "h1_state_manifest.jsonl"
    ),
    "manifest_sha256": (
        "56cb83855afd8d2d30eddeee28e4df074f669311cc2c0205ac757acb73e8abdd"
    ),
    "research_split": "fresh_matched_development",
    "observed": {
        "state_count": 96,
        "unique_action_count": 256,
        "trial_count": 4096,
        "stage1_pair_count": 160,
        "stage2_state_count": 64,
    },
    "artifacts": {
        "stage1_pair_labels": (
            "stage1_pair_labels.jsonl",
            "b725cdee4acaa621ba3f3359dc53007a6a3e7d734e9ab7fcbe38bccc9f76f160",
        ),
        "stage1_state_labels": (
            "stage1_state_labels.jsonl",
            "982622dc89275bfa742fc41645eb7f553b77430fbb92b5eea9bb168d774c3d53",
        ),
        "stage2_state_labels": (
            "stage2_state_labels.jsonl",
            "6217cb9977c06a06af0c45d28272401e8855eef6afb7a0682be8321cdd24c769",
        ),
    },
}

LEGACY_CONSENSUS = {
    "name": "legacy_consensus_extension",
    "root": "build/stride-hierarchical-ch-legacy-consensus-labels-v1",
    "experiment_id": "stride_hierarchical_ch_legacy_consensus_labels_v1",
    "audit_sha256": (
        "88bae30a9824dd231ca8a35d5a3a6a74fd7bc63c837b2cadd2dbd34a78b0423a"
    ),
    "manifest": (
        "build/stride-hierarchical-ch-legacy-consensus-h1-v1/"
        "h1_state_manifest.jsonl"
    ),
    "manifest_sha256": (
        "d962b92744ea69f3b541455bc0e4767491edc105a4a0651378ff9aa188e65472"
    ),
    "research_split": "legacy_sequential_development_train_only",
    "observed": {
        "state_count": 60,
        "unique_action_count": 120,
        "trial_count": 1920,
        "stage1_pair_count": 60,
        "stage2_state_count": 0,
    },
    "artifacts": {
        "stage1_pair_labels": (
            "stage1_pair_labels.jsonl",
            "005ee6b8a5975ea7382dcf8ccce80e5e9c3e5b94a444ebe10ff1d1787d9256c7",
        ),
        "stage1_state_labels": (
            "stage1_state_labels.jsonl",
            "aeef8a9809b95ff365b63d72ca54ecbc086735909aa944a75717590cdc25d728",
        ),
        "stage2_state_labels": (
            "stage2_state_labels.jsonl",
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        ),
    },
}

ORIGINAL_LABEL_AUDIT_SHA256 = str(ORIGINAL["audit_sha256"])
LEGACY_CONSENSUS_LABEL_AUDIT_SHA256 = str(LEGACY_CONSENSUS["audit_sha256"])
STAGE1_LABELS = ("structural_win", "v2_win", "ambiguous")
STAGE2_LABELS = ("component_win", "hotspot_win", "ambiguous")
STAGE1_DECISIVE = ("structural_win", "v2_win")
STAGE2_DECISIVE = ("component_win", "hotspot_win")
EXPECTED_COMBINED_STAGE1_COUNTS = {
    "structural_win": 77,
    "v2_win": 17,
    "ambiguous": 126,
}
EXPECTED_CONDITIONED_STAGE2_COUNTS = {
    "component_win": 21,
    "hotspot_win": 1,
    "ambiguous": 10,
}


def _require_sha256(value: str, *, field: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{field} is not a lowercase SHA256")
    return digest


def _pinned_json(path: Path, expected_sha256: str, *, label: str) -> dict[str, Any]:
    expected = _require_sha256(expected_sha256, field=f"{label}.sha256")
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    if sha256_file(path) != expected:
        raise ValueError(f"{label} SHA256 mismatch")
    return _read_json(path)


def _relative_artifact(
    labels_root: Path,
    audit: dict[str, Any],
    specification: dict[str, Any],
    key: str,
) -> Path:
    registered = dict(dict(audit.get("artifacts") or {}).get(key) or {})
    expected_name, expected_sha256 = specification["artifacts"][key]
    if registered != {"file": expected_name, "sha256": expected_sha256}:
        raise ValueError(f"{specification['name']} {key} contract changed")
    path = (labels_root / expected_name).resolve()
    if path.parent != labels_root.resolve() or not path.is_file():
        raise ValueError(f"{specification['name']} {key} is missing")
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"{specification['name']} {key} SHA256 mismatch")
    return path


def _source_ids(
    audit: dict[str, Any],
    specification: dict[str, Any],
    project_root: Path,
) -> tuple[set[str], dict[str, Any]]:
    source = dict(audit.get("source") or {})
    registered = str(source.get("manifest", "")).replace("\\", "/")
    expected_relative = str(specification["manifest"])
    if not registered.endswith(expected_relative):
        raise ValueError(f"{specification['name']} source manifest identity changed")
    expected_sha256 = str(specification["manifest_sha256"])
    if source.get("manifest_sha256") != expected_sha256:
        raise ValueError(f"{specification['name']} source manifest registry changed")
    path = (project_root / expected_relative).resolve()
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise ValueError(f"{specification['name']} source manifest SHA256 mismatch")

    audit_by_id: dict[str, tuple[str, str]] = {}
    for raw in list(source.get("state_payloads") or ()):
        row = dict(raw)
        state_id = str(row.get("state_occurrence_id", ""))
        file_name = str(row.get("file", ""))
        digest = _require_sha256(
            str(row.get("sha256", "")),
            field=f"{specification['name']}.source.state_sha256",
        )
        if not state_id or not file_name or state_id in audit_by_id:
            raise ValueError(f"{specification['name']} source state registry changed")
        audit_by_id[state_id] = (file_name, digest)

    manifest_by_id: dict[str, tuple[str, str]] = {}
    for raw in _read_jsonl(path):
        row = dict(raw)
        state_id = str(row.get("state_occurrence_id", ""))
        value = (
            str(row.get("state_file", "")),
            _require_sha256(
                str(row.get("state_sha256", "")),
                field=f"{specification['name']}.manifest.state_sha256",
            ),
        )
        if not state_id or state_id in manifest_by_id:
            raise ValueError(f"{specification['name']} manifest states changed")
        if specification is LEGACY_CONSENSUS and (
            row.get("research_split") != specification["research_split"]
            or row.get("training_only") is not True
            or row.get("evaluation_eligible") is not False
            or row.get("training_authorized") is not False
        ):
            raise ValueError("legacy consensus manifest training-only boundary changed")
        manifest_by_id[state_id] = value
    if audit_by_id != manifest_by_id:
        raise ValueError(f"{specification['name']} source payload registry changed")
    return set(audit_by_id), {
        "file": expected_relative,
        "sha256": expected_sha256,
        "state_count": len(audit_by_id),
    }


def _load_input(
    *,
    labels_root: Path,
    project_root: Path,
    specification: dict[str, Any],
    audit_sha256: str,
) -> dict[str, Any]:
    audit_path = labels_root / "label_audit_report.json"
    audit = _pinned_json(
        audit_path,
        audit_sha256,
        label=f"{specification['name']} label audit report",
    )
    if audit_sha256 != specification["audit_sha256"]:
        raise ValueError(f"{specification['name']} registered audit SHA256 changed")
    contract = dict(audit.get("label_contract") or {})
    if (
        audit.get("schema") != LABEL_AUDIT_SCHEMA
        or audit.get("experiment_id") != specification["experiment_id"]
        or audit.get("scientific_status")
        != "sequential_development_label_audit_only"
        or audit.get("complete") is not True
        or audit.get("training_authorized") is not False
        or audit.get("model_fit_executed") is not False
        or audit.get("model_exported") is not False
        or audit.get("runtime_or_ttf_claim_authorized") is not False
        or audit.get("map_disjoint_confirmation_required") is not True
        or contract.get("direction_rule")
        != "frozen_h1_rule_applied_in_both_directions"
        or contract.get("ambiguous_rule")
        != "neither_direction_is_a_robust_win"
        or contract.get("paired_trial_count") != 16
        or contract.get("runtime_or_pp_seconds_used_in_label") is not False
        or contract.get("state_or_pair_is_label_unit") is not True
        or contract.get("trial_is_training_sample") is not False
    ):
        raise ValueError(f"{specification['name']} label audit contract changed")
    if dict(audit.get("observed") or {}) != specification["observed"]:
        raise ValueError(f"{specification['name']} observed counts changed")

    artifacts = {
        key: _relative_artifact(labels_root, audit, specification, key)
        for key in specification["artifacts"]
    }
    source_ids, source_report = _source_ids(audit, specification, project_root)
    return {
        "audit_path": audit_path,
        "audit": audit,
        "artifacts": artifacts,
        "source_ids": source_ids,
        "source_report": source_report,
    }


def _frozen_folds(path: Path, expected_sha256: str) -> dict[str, tuple[str, ...]]:
    config = _pinned_json(path, expected_sha256, label="legacy fold config")
    folds = {
        str(fold): tuple(map(str, maps))
        for fold, maps in dict(config.get("map_folds") or {}).items()
    }
    if folds != FROZEN_LEGACY_FOLDS:
        raise ValueError("legacy map folds changed")
    maps = [map_id for values in folds.values() for map_id in values]
    if len(maps) != len(set(maps)):
        raise ValueError("legacy map folds contain duplicate maps")
    return folds


def _metadata(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("map_id", "")),
        str(row.get("map_family", "")),
        str(row.get("task_id", "")),
        int(row.get("solver_seed", -1)),
        int(row.get("decision_index", -1)),
        str(row.get("hierarchical_stratum", "")),
        str(row.get("research_split", "")),
    )


def _validate_direction(row: dict[str, Any]) -> None:
    left = dict(row.get("left_over_right") or {})
    right = dict(row.get("right_over_left") or {})
    if type(left.get("label")) is not bool or type(right.get("label")) is not bool:
        raise ValueError("stage1 robust direction label is not Boolean")
    if left["label"] and right["label"]:
        raise ValueError("stage1 robust directions are contradictory")
    expected = (
        "structural_win"
        if left["label"]
        else "v2_win"
        if right["label"]
        else "ambiguous"
    )
    if (
        row.get("label") != expected
        or row.get("left_label") != "structural_win"
        or row.get("right_label") != "v2_win"
    ):
        raise ValueError("stage1 exact bidirectional label changed")


def _validate_stage1(
    rows: list[dict[str, Any]],
    *,
    source_ids: set[str],
    research_split: str,
    map_to_fold: dict[str, str],
    consensus_only: bool,
) -> dict[str, list[dict[str, Any]]]:
    by_state: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    pair_ids: set[str] = set()
    for row in rows:
        state_id = str(row.get("state_occurrence_id", ""))
        structural = str(row.get("structural_action_id", ""))
        anchor = str(row.get("v2_action_id", ""))
        pair_id = str(row.get("pair_id", ""))
        stratum = str(row.get("hierarchical_stratum", ""))
        aliases = list(row.get("structural_role_aliases") or ())
        expected_aliases = (
            ["component16", "hotspot16"]
            if stratum == "consensus_structural"
            else None
        )
        if (
            row.get("schema") != STAGE1_PAIR_SCHEMA
            or str(row.get("label", "")) not in STAGE1_LABELS
            or state_id not in source_ids
            or not structural
            or not anchor
            or structural == anchor
            or pair_id != f"{state_id}::{structural}::{anchor}"
            or pair_id in pair_ids
            or row.get("runtime_or_pp_seconds_used_in_label") is not False
            or row.get("research_split") != research_split
            or str(row.get("map_id", "")) not in map_to_fold
            or not str(row.get("map_family", ""))
            or stratum not in {"consensus_structural", "three_unique"}
            or (consensus_only and stratum != "consensus_structural")
            or (
                expected_aliases is not None
                and aliases != expected_aliases
            )
            or (
                stratum == "three_unique"
                and aliases not in (["component16"], ["hotspot16"])
            )
        ):
            raise ValueError("stage1 exact action-pair integrity changed")
        _validate_direction(row)
        pair_ids.add(pair_id)
        by_state[state_id].append(row)

    if set(by_state) != source_ids:
        raise ValueError("stage1 labels disagree with the source state registry")
    for state_rows in by_state.values():
        if len({_metadata(row) for row in state_rows}) != 1:
            raise ValueError("stage1 state metadata is inconsistent")
        stratum = str(state_rows[0]["hierarchical_stratum"])
        expected_count = 1 if stratum == "consensus_structural" else 2
        structural_ids = {
            str(row["structural_action_id"]) for row in state_rows
        }
        anchor_ids = {str(row["v2_action_id"]) for row in state_rows}
        aliases = sorted(
            alias
            for row in state_rows
            for alias in list(row["structural_role_aliases"])
        )
        if (
            len(state_rows) != expected_count
            or len(structural_ids) != expected_count
            or len(anchor_ids) != 1
            or aliases != ["component16", "hotspot16"]
        ):
            raise ValueError("stage1 exact action-pair multiplicity changed")
    return dict(by_state)


def _validate_stage2(
    rows: list[dict[str, Any]],
    *,
    original_stage1: dict[str, list[dict[str, Any]]],
    map_to_fold: dict[str, str],
) -> dict[str, dict[str, Any]]:
    by_state: dict[str, dict[str, Any]] = {}
    for row in rows:
        state_id = str(row.get("state_occurrence_id", ""))
        stage1_rows = original_stage1.get(state_id)
        role_to_action = {
            str(alias): str(stage1_row["structural_action_id"])
            for stage1_row in stage1_rows or ()
            for alias in stage1_row["structural_role_aliases"]
        }
        left = dict(row.get("left_over_right") or {})
        right = dict(row.get("right_over_left") or {})
        expected_label = (
            "component_win"
            if left.get("label") is True
            else "hotspot_win"
            if right.get("label") is True
            else "ambiguous"
        )
        if (
            row.get("schema") != STAGE2_STATE_SCHEMA
            or state_id in by_state
            or not stage1_rows
            or row.get("label") != expected_label
            or row.get("left_label") != "component_win"
            or row.get("right_label") != "hotspot_win"
            or type(left.get("label")) is not bool
            or type(right.get("label")) is not bool
            or (left["label"] and right["label"])
            or row.get("runtime_or_pp_seconds_used_in_label") is not False
            or row.get("research_split") != ORIGINAL["research_split"]
            or row.get("hierarchical_stratum") != "three_unique"
            or not str(row.get("component_action_id", ""))
            or not str(row.get("hotspot_action_id", ""))
            or row.get("component_action_id") == row.get("hotspot_action_id")
            or row.get("component_action_id") != role_to_action.get("component16")
            or row.get("hotspot_action_id") != role_to_action.get("hotspot16")
            or str(row.get("map_id", "")) not in map_to_fold
            or _metadata(row) != _metadata(stage1_rows[0])
        ):
            raise ValueError("stage2 exact action-pair integrity changed")
        stage1_labels = [str(value["label"]) for value in stage1_rows]
        expected_any = "structural_win" in stage1_labels
        expected_both = len(stage1_labels) == 2 and all(
            label == "structural_win" for label in stage1_labels
        )
        expected_state = (
            "structural_win"
            if expected_any
            else "v2_win"
            if all(label == "v2_win" for label in stage1_labels)
            else "ambiguous"
        )
        if (
            row.get("any_stage1_structural_win") is not expected_any
            or row.get("both_stage1_structural_win") is not expected_both
            or row.get("stage1_state_label") != expected_state
        ):
            raise ValueError("stage2 condition disagrees with stage1 labels")
        by_state[state_id] = row
    expected_ids = {
        state_id
        for state_id, state_rows in original_stage1.items()
        if state_rows[0]["hierarchical_stratum"] == "three_unique"
    }
    if set(by_state) != expected_ids:
        raise ValueError("stage2 labels disagree with original three-unique states")
    return by_state


def _support(rows: Iterable[dict[str, Any]], labels: tuple[str, ...]) -> dict[str, Any]:
    values = list(rows)
    counts = collections.Counter(str(row["label"]) for row in values)
    if set(counts) - set(labels):
        raise ValueError("readiness support contains an unknown label")
    return {
        "row_count": len(values),
        "state_count": len({str(row["state_occurrence_id"]) for row in values}),
        "map_count": len({str(row["map_id"]) for row in values}),
        "family_group_count": len({str(row["map_family"]) for row in values}),
        "label_counts": {label: int(counts[label]) for label in labels},
        "by_label": {
            label: {
                "row_count": sum(row["label"] == label for row in values),
                "state_count": len(
                    {
                        str(row["state_occurrence_id"])
                        for row in values
                        if row["label"] == label
                    }
                ),
                "map_count": len(
                    {
                        str(row["map_id"])
                        for row in values
                        if row["label"] == label
                    }
                ),
                "family_group_count": len(
                    {
                        str(row["map_family"])
                        for row in values
                        if row["label"] == label
                    }
                ),
            }
            for label in labels
        },
    }


def _fold_support(
    rows: list[dict[str, Any]],
    folds: dict[str, tuple[str, ...]],
    labels: tuple[str, ...],
    decisive: tuple[str, str],
) -> dict[str, Any]:
    result = {}
    for fold, maps in folds.items():
        selected = [row for row in rows if str(row["map_id"]) in maps]
        counts = collections.Counter(str(row["label"]) for row in selected)
        missing = [label for label in decisive if counts[label] == 0]
        result[fold] = {
            "row_count": len(selected),
            "label_counts": {label: int(counts[label]) for label in labels},
            "missing_decisive_classes": missing,
            "has_both_decisive_classes": not missing,
        }
    return result


def _stage_report(
    *,
    stage: str,
    rows: list[dict[str, Any]],
    folds: dict[str, tuple[str, ...]],
    labels: tuple[str, ...],
    decisive: tuple[str, str],
    minimum_rows: int,
    minimum_maps: int,
    minimum_families: int,
) -> dict[str, Any]:
    support = _support(rows, labels)
    fold_support = _fold_support(rows, folds, labels, decisive)
    gates: dict[str, bool] = {}
    reasons: list[str] = []
    for label in decisive:
        observed = support["by_label"][label]
        for metric, minimum, gate_suffix in (
            ("row_count", minimum_rows, "minimum_rows"),
            ("map_count", minimum_maps, "minimum_maps"),
            ("family_group_count", minimum_families, "minimum_family_groups"),
        ):
            passed = int(observed[metric]) >= minimum
            gates[f"{label}_{gate_suffix}"] = passed
            if not passed:
                reasons.append(
                    f"{stage}.{label}.{metric}={observed[metric]}<{minimum}"
                )
    gates["each_frozen_fold_has_both_decisive_classes"] = all(
        row["has_both_decisive_classes"] for row in fold_support.values()
    )
    for fold, row in fold_support.items():
        if row["missing_decisive_classes"]:
            reasons.append(
                f"{stage}.{fold}.missing="
                + ",".join(row["missing_decisive_classes"])
            )
    readiness = all(gates.values())
    return {
        "readiness_passed": readiness,
        "training_authorized": False,
        "support": support,
        "fit_support": {
            "included_labels": list(decisive),
            "ambiguous_excluded": True,
            "row_count": sum(
                int(support["by_label"][label]["row_count"]) for label in decisive
            ),
        },
        "threshold_calibration": {
            "label": "ambiguous",
            "use": "threshold_calibration_only",
            "fit_included": False,
            **support["by_label"]["ambiguous"],
        },
        "folds": fold_support,
        "gates": gates,
        "failure_reasons": reasons,
    }


def check_hierarchical_ch_combined_readiness(
    *,
    project_root: str | Path,
    original_labels_root: str | Path,
    legacy_consensus_labels_root: str | Path,
    output: str | Path,
    original_label_audit_sha256: str = ORIGINAL_LABEL_AUDIT_SHA256,
    legacy_consensus_label_audit_sha256: str = (
        LEGACY_CONSENSUS_LABEL_AUDIT_SHA256
    ),
    legacy_fold_config: str | Path | None = None,
    legacy_fold_config_sha256: str = LEGACY_FOLD_CONFIG_SHA256,
) -> dict[str, Any]:
    """Combine frozen legacy training labels and emit a no-fit readiness decision."""

    root = Path(project_root).resolve()
    original_root = Path(original_labels_root).resolve()
    extension_root = Path(legacy_consensus_labels_root).resolve()
    output_root = Path(output).resolve()
    for input_root in (original_root, extension_root):
        if output_root == input_root or input_root in output_root.parents:
            raise ValueError("combined readiness output must remain outside label inputs")

    original = _load_input(
        labels_root=original_root,
        project_root=root,
        specification=ORIGINAL,
        audit_sha256=original_label_audit_sha256,
    )
    extension = _load_input(
        labels_root=extension_root,
        project_root=root,
        specification=LEGACY_CONSENSUS,
        audit_sha256=legacy_consensus_label_audit_sha256,
    )
    overlap = original["source_ids"] & extension["source_ids"]
    if overlap:
        raise ValueError("original and legacy-consensus source states overlap")

    fold_path = (
        Path(legacy_fold_config).resolve()
        if legacy_fold_config is not None
        else root
        / "configs"
        / "stride_fresh_matched_unique_action_hierarchical_overlay_v1.json"
    )
    folds = _frozen_folds(fold_path, legacy_fold_config_sha256)
    map_to_fold = {
        map_id: fold for fold, maps in folds.items() for map_id in maps
    }

    original_stage1_rows = _read_jsonl(
        original["artifacts"]["stage1_pair_labels"]
    )
    extension_stage1_rows = _read_jsonl(
        extension["artifacts"]["stage1_pair_labels"]
    )
    original_stage1 = _validate_stage1(
        original_stage1_rows,
        source_ids=original["source_ids"],
        research_split=str(ORIGINAL["research_split"]),
        map_to_fold=map_to_fold,
        consensus_only=False,
    )
    extension_stage1 = _validate_stage1(
        extension_stage1_rows,
        source_ids=extension["source_ids"],
        research_split=str(LEGACY_CONSENSUS["research_split"]),
        map_to_fold=map_to_fold,
        consensus_only=True,
    )
    if set(original_stage1) & set(extension_stage1):
        raise ValueError("combined stage1 state identities overlap")

    original_stage2_rows = _read_jsonl(
        original["artifacts"]["stage2_state_labels"]
    )
    extension_stage2_rows = _read_jsonl(
        extension["artifacts"]["stage2_state_labels"]
    )
    if extension_stage2_rows:
        raise ValueError("legacy consensus extension must not contain stage2 labels")
    _validate_stage2(
        original_stage2_rows,
        original_stage1=original_stage1,
        map_to_fold=map_to_fold,
    )

    combined_stage1_rows = [*original_stage1_rows, *extension_stage1_rows]
    combined_stage1_rows.sort(key=lambda row: str(row["pair_id"]))
    stage1 = _stage_report(
        stage="stage1",
        rows=combined_stage1_rows,
        folds=folds,
        labels=STAGE1_LABELS,
        decisive=STAGE1_DECISIVE,
        minimum_rows=40,
        minimum_maps=8,
        minimum_families=4,
    )
    if stage1["support"]["label_counts"] != EXPECTED_COMBINED_STAGE1_COUNTS:
        raise ValueError("combined stage1 frozen label counts changed")

    conditioned_stage2 = [
        row
        for row in original_stage2_rows
        if row["any_stage1_structural_win"] is True
    ]
    stage2 = _stage_report(
        stage="stage2",
        rows=conditioned_stage2,
        folds=folds,
        labels=STAGE2_LABELS,
        decisive=STAGE2_DECISIVE,
        minimum_rows=16,
        minimum_maps=4,
        minimum_families=3,
    )
    if stage2["support"]["label_counts"] != EXPECTED_CONDITIONED_STAGE2_COUNTS:
        raise ValueError("conditioned stage2 frozen label counts changed")
    stage2.update(
        {
            "source": "original_only_extension_is_consensus_structural",
            "condition": "any_stage1_structural_win",
            "all_state_count": len(original_stage2_rows),
            "conditioned_state_count": len(conditioned_stage2),
            "unconditioned_state_count": len(original_stage2_rows)
            - len(conditioned_stage2),
            "extension_state_count": 0,
        }
    )

    readiness = bool(stage1["readiness_passed"] and stage2["readiness_passed"])
    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "scientific_status": "combined_legacy_labels_only_readiness_gate",
        "status": "LABEL_SUPPORT_READY" if readiness else "NO_GO",
        "inputs": {
            "original": {
                "experiment_id": ORIGINAL["experiment_id"],
                "label_audit_report": {
                    "file": f"{ORIGINAL['root']}/label_audit_report.json",
                    "sha256": sha256_file(original["audit_path"]),
                },
                "artifacts": {
                    key: {
                        "file": f"{ORIGINAL['root']}/{path.name}",
                        "sha256": sha256_file(path),
                    }
                    for key, path in original["artifacts"].items()
                },
                "source_manifest": original["source_report"],
            },
            "legacy_consensus_extension": {
                "experiment_id": LEGACY_CONSENSUS["experiment_id"],
                "label_audit_report": {
                    "file": f"{LEGACY_CONSENSUS['root']}/label_audit_report.json",
                    "sha256": sha256_file(extension["audit_path"]),
                },
                "artifacts": {
                    key: {
                        "file": f"{LEGACY_CONSENSUS['root']}/{path.name}",
                        "sha256": sha256_file(path),
                    }
                    for key, path in extension["artifacts"].items()
                },
                "source_manifest": extension["source_report"],
            },
            "legacy_fold_config": {
                "file": (
                    "configs/"
                    "stride_fresh_matched_unique_action_hierarchical_overlay_v1.json"
                ),
                "sha256": sha256_file(fold_path),
            },
        },
        "integrity": {
            "passed": True,
            "source_state_disjoint": True,
            "original_source_state_count": len(original["source_ids"]),
            "legacy_consensus_source_state_count": len(extension["source_ids"]),
            "combined_source_state_count": len(original["source_ids"])
            + len(extension["source_ids"]),
            "original_stage1_pair_count": len(original_stage1_rows),
            "legacy_consensus_stage1_pair_count": len(extension_stage1_rows),
            "combined_stage1_pair_count": len(combined_stage1_rows),
            "stage2_original_only": True,
            "stage2_state_count": len(original_stage2_rows),
            "stage2_conditioned_state_count": len(conditioned_stage2),
            "exact_action_pair_labels_validated": True,
        },
        "research_split_boundary": {
            "original": {
                "research_split": ORIGINAL["research_split"],
                "role": "sequential_development_training_only",
                "evaluation_eligible": False,
            },
            "legacy_consensus_extension": {
                "research_split": LEGACY_CONSENSUS["research_split"],
                "role": "sequential_development_training_only",
                "evaluation_eligible": False,
            },
            "all_inputs_training_only": True,
            "map_disjoint_confirmation_required": True,
        },
        "thresholds": {
            "stage1": {
                "minimum_rows_per_decisive_class": 40,
                "minimum_maps_per_decisive_class": 8,
                "minimum_family_groups_per_decisive_class": 4,
                "each_frozen_fold_requires_both_decisive_classes": True,
            },
            "stage2": {
                "condition": "any_stage1_structural_win",
                "minimum_rows_per_decisive_class": 16,
                "minimum_maps_per_decisive_class": 4,
                "minimum_family_groups_per_decisive_class": 3,
                "each_frozen_fold_requires_both_decisive_classes": True,
            },
        },
        "legacy_folds": {fold: list(maps) for fold, maps in folds.items()},
        "stage1": stage1,
        "stage2": stage2,
        "overall": {
            "readiness_passed": readiness,
            "failure_reasons": [
                *stage1["failure_reasons"],
                *stage2["failure_reasons"],
            ],
            "training_authorized": False,
        },
        "training_authorized": False,
        "model_fit_executed": False,
        "model_exported": False,
        "runtime_authorized": False,
        "runtime_or_ttf_claim_authorized": False,
        "map_disjoint_confirmation_required": True,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "readiness_report.json", report)
    return report


__all__ = [
    "EXPERIMENT_ID",
    "LEGACY_CONSENSUS_LABEL_AUDIT_SHA256",
    "ORIGINAL_LABEL_AUDIT_SHA256",
    "REPORT_SCHEMA",
    "check_hierarchical_ch_combined_readiness",
]
