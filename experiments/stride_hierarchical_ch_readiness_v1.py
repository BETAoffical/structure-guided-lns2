from __future__ import annotations

import collections
from pathlib import Path
from typing import Any, Iterable

from experiments._common import sha256_file
from experiments.repair_collection import _read_json, _read_jsonl, _write_json
from experiments.stride_hierarchical_ch_labels_v1 import (
    ANCHOR_COMPONENT_SHARED,
    ANCHOR_HOTSPOT_SHARED,
    CONSENSUS_STRUCTURAL,
    REPORT_SCHEMA as LABEL_AUDIT_SCHEMA,
    SINGLE_UNIQUE,
    STAGE1_PAIR_SCHEMA,
    STAGE2_STATE_SCHEMA,
    STAGE2_ELIGIBLE_PARTITIONS,
    THREE_UNIQUE,
)


REPORT_SCHEMA = "lns2.stride.hierarchical_ch_readiness.v1"
EXPERIMENT_ID = "stride_hierarchical_ch_readiness_v1"
LABEL_EXPERIMENT_ID = "stride_hierarchical_ch_labels_v1"
LABEL_AUDIT_SHA256 = (
    "57e3ab03275e80bc764cdab7bbcd9cea0e50cee7aac4f5eccd9b5b41a71a670a"
)
LEGACY_FOLD_CONFIG_SHA256 = (
    "80430e09b0ad88f5116908ff1e51a25fe66c9e9165d25ba6b9c28172d0531768"
)
SOURCE_MANIFEST_RELATIVE = (
    "build/stride-fresh-matched-unique-action-hierarchical-overlay-v1/"
    "h1_state_manifest.jsonl"
)
FROZEN_LEGACY_FOLDS = {
    "fold0": (
        "den520d",
        "maze-128-128-10",
        "warehouse-10-20-10-2-1",
        "maze-128-128-1",
    ),
    "fold1": (
        "lak303d",
        "random-32-32-10",
        "room-32-32-4",
        "random-32-32-20",
    ),
    "fold2": (
        "maze-32-32-2",
        "random-64-64-10",
        "warehouse-20-40-10-2-1",
        "maze-32-32-4",
    ),
    "fold3": (
        "random-64-64-20",
        "room-64-64-8",
        "warehouse-20-40-10-2-2",
        "room-64-64-16",
    ),
}
STAGE1_LABELS = ("structural_win", "v2_win", "ambiguous")
STAGE2_LABELS = ("component_win", "hotspot_win", "ambiguous")
STAGE1_DECISIVE = ("structural_win", "v2_win")
STAGE2_DECISIVE = ("component_win", "hotspot_win")
STAGE1_THRESHOLDS = {
    "minimum_rows_per_decisive_class": 40,
    "minimum_maps_per_decisive_class": 8,
    "minimum_family_groups_per_decisive_class": 4,
    "each_fold_requires_both_decisive_classes": True,
}
STAGE2_THRESHOLDS = {
    "condition": "component_action_id_ne_hotspot_action_id",
    "minimum_rows_per_decisive_class": 16,
    "minimum_maps_per_decisive_class": 4,
    "minimum_family_groups_per_decisive_class": 3,
    "each_fold_requires_both_decisive_classes": True,
}


def _require_sha256(value: str, *, field: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{field} is not a lowercase SHA256")
    return digest


def _read_hash_pinned_json(path: Path, expected_sha256: str, *, label: str) -> dict:
    expected = _require_sha256(expected_sha256, field=f"{label}.sha256")
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    if sha256_file(path) != expected:
        raise ValueError(f"{label} SHA256 mismatch")
    return _read_json(path)


def _artifact_path(labels_root: Path, specification: dict[str, Any], key: str) -> Path:
    if set(specification) != {"file", "sha256"}:
        raise ValueError(f"{key} artifact contract changed")
    name = str(specification["file"])
    if not name or Path(name).name != name or Path(name).is_absolute():
        raise ValueError(f"{key} artifact file must be a contained basename")
    path = (labels_root / name).resolve()
    if path.parent != labels_root.resolve() or not path.is_file():
        raise ValueError(f"{key} artifact is missing")
    expected = _require_sha256(
        str(specification["sha256"]), field=f"artifacts.{key}.sha256"
    )
    if sha256_file(path) != expected:
        raise ValueError(f"{key} SHA256 mismatch")
    return path


def _frozen_folds(path: Path, expected_sha256: str) -> dict[str, tuple[str, ...]]:
    config = _read_hash_pinned_json(
        path, expected_sha256, label="legacy fold config"
    )
    folds = {
        str(fold): tuple(map(str, maps))
        for fold, maps in dict(config.get("map_folds") or {}).items()
    }
    if folds != FROZEN_LEGACY_FOLDS:
        raise ValueError("legacy map folds changed")
    observed_maps = [map_id for maps in folds.values() for map_id in maps]
    if len(observed_maps) != len(set(observed_maps)):
        raise ValueError("legacy map folds contain duplicate maps")
    return folds


def _source_contract(
    audit: dict[str, Any], project_root: Path
) -> tuple[set[str], dict[str, Any]]:
    source = dict(audit.get("source") or {})
    registered_manifest = str(source.get("manifest", "")).replace("\\", "/")
    if not registered_manifest.endswith(SOURCE_MANIFEST_RELATIVE):
        raise ValueError("label audit source manifest identity changed")
    source_manifest = (project_root / SOURCE_MANIFEST_RELATIVE).resolve()
    expected_manifest_sha = _require_sha256(
        str(source.get("manifest_sha256", "")),
        field="label_audit.source.manifest_sha256",
    )
    if not source_manifest.is_file() or sha256_file(source_manifest) != expected_manifest_sha:
        raise ValueError("label audit source manifest SHA256 mismatch")

    source_rows = list(source.get("state_payloads") or ())
    source_by_id: dict[str, tuple[str, str]] = {}
    for raw in source_rows:
        row = dict(raw)
        state_id = str(row.get("state_occurrence_id", ""))
        file_name = str(row.get("file", ""))
        digest = _require_sha256(
            str(row.get("sha256", "")), field="source.state_payload.sha256"
        )
        if not state_id or not file_name or state_id in source_by_id:
            raise ValueError("label audit source state identities changed")
        source_by_id[state_id] = (file_name, digest)

    manifest_rows = _read_jsonl(source_manifest)
    manifest_by_id: dict[str, tuple[str, str]] = {}
    for raw in manifest_rows:
        row = dict(raw)
        state_id = str(row.get("state_occurrence_id", ""))
        value = (
            str(row.get("state_file", "")),
            _require_sha256(
                str(row.get("state_sha256", "")),
                field="source.manifest.state_sha256",
            ),
        )
        if not state_id or state_id in manifest_by_id:
            raise ValueError("source manifest state identities changed")
        manifest_by_id[state_id] = value
    if source_by_id != manifest_by_id:
        raise ValueError("label audit source payload registry changed")
    return set(source_by_id), {
        "file": SOURCE_MANIFEST_RELATIVE,
        "sha256": expected_manifest_sha,
        "state_count": len(source_by_id),
    }


def _row_metadata(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("map_id", "")),
        str(row.get("map_family", "")),
        str(row.get("task_id", "")),
        int(row.get("solver_seed", -1)),
        int(row.get("decision_index", -1)),
        str(row.get("hierarchical_stratum", "")),
        str(row.get("research_split", "")),
    )


def _validate_stage1(
    rows: list[dict[str, Any]], map_to_fold: dict[str, str]
) -> dict[str, list[dict[str, Any]]]:
    by_state: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    pair_ids: set[str] = set()
    for row in rows:
        state_id = str(row.get("state_occurrence_id", ""))
        pair_id = str(row.get("pair_id", ""))
        label = str(row.get("label", ""))
        if (
            row.get("schema") != STAGE1_PAIR_SCHEMA
            or label not in STAGE1_LABELS
            or not state_id
            or not pair_id
            or pair_id in pair_ids
            or row.get("runtime_or_pp_seconds_used_in_label") is not False
            or str(row.get("map_id", "")) not in map_to_fold
            or str(row.get("research_split", "")) != "fresh_matched_development"
            or str(row.get("structural_action_id", ""))
            == str(row.get("v2_action_id", ""))
        ):
            raise ValueError("stage1 pair label integrity changed")
        pair_ids.add(pair_id)
        by_state[state_id].append(row)
    for state_rows in by_state.values():
        if len({_row_metadata(row) for row in state_rows}) != 1:
            raise ValueError("stage1 state metadata is inconsistent")
        stratum = str(state_rows[0]["hierarchical_stratum"])
        expected_aliases = {
            CONSENSUS_STRUCTURAL: {("component16", "hotspot16")},
            THREE_UNIQUE: {("component16",), ("hotspot16",)},
            ANCHOR_COMPONENT_SHARED: {("hotspot16",)},
            ANCHOR_HOTSPOT_SHARED: {("component16",)},
        }.get(stratum)
        observed_aliases = {
            tuple(sorted(map(str, row.get("structural_role_aliases") or ())))
            for row in state_rows
        }
        v2_ids = {str(row.get("v2_action_id", "")) for row in state_rows}
        structural_ids = {
            str(row.get("structural_action_id", "")) for row in state_rows
        }
        if (
            expected_aliases is None
            or observed_aliases != expected_aliases
            or len(state_rows) != len(expected_aliases)
            or len(v2_ids) != 1
            or len(structural_ids) != len(state_rows)
        ):
            raise ValueError("stage1 pair multiplicity changed")
    return dict(by_state)


def _validate_stage2(
    rows: list[dict[str, Any]],
    map_to_fold: dict[str, str],
    stage1_by_state: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    by_state: dict[str, dict[str, Any]] = {}
    for row in rows:
        state_id = str(row.get("state_occurrence_id", ""))
        label = str(row.get("label", ""))
        stage1_rows = stage1_by_state.get(state_id)
        if (
            row.get("schema") != STAGE2_STATE_SCHEMA
            or label not in STAGE2_LABELS
            or not state_id
            or state_id in by_state
            or not stage1_rows
            or row.get("runtime_or_pp_seconds_used_in_label") is not False
            or str(row.get("map_id", "")) not in map_to_fold
            or str(row.get("research_split", "")) != "fresh_matched_development"
            or str(row.get("hierarchical_stratum", ""))
            not in STAGE2_ELIGIBLE_PARTITIONS
            or str(row.get("component_action_id", ""))
            == str(row.get("hotspot_action_id", ""))
            or _row_metadata(row) != _row_metadata(stage1_rows[0])
        ):
            raise ValueError("stage2 state label integrity changed")
        stage1_labels = [str(value["label"]) for value in stage1_rows]
        stratum = str(row["hierarchical_stratum"])
        component_id = str(row["component_action_id"])
        hotspot_id = str(row["hotspot_action_id"])
        v2_ids = {str(value["v2_action_id"]) for value in stage1_rows}
        structural_ids = {
            str(value["structural_action_id"]) for value in stage1_rows
        }
        exact_partition = (
            len(v2_ids) == 1
            and (
                stratum == THREE_UNIQUE
                and structural_ids == {component_id, hotspot_id}
                or stratum == ANCHOR_COMPONENT_SHARED
                and v2_ids == {component_id}
                and structural_ids == {hotspot_id}
                or stratum == ANCHOR_HOTSPOT_SHARED
                and v2_ids == {hotspot_id}
                and structural_ids == {component_id}
            )
        )
        expected_any = "structural_win" in stage1_labels
        expected_both = len(stage1_labels) == 2 and all(
            value == "structural_win" for value in stage1_labels
        )
        expected_state_label = (
            "structural_win"
            if expected_any
            else "v2_win"
            if all(value == "v2_win" for value in stage1_labels)
            else "ambiguous"
        )
        if (
            not exact_partition
            or row.get("eligibility_condition")
            not in {None, "component_action_id_ne_hotspot_action_id"}
            or row.get("any_stage1_structural_win") is not expected_any
            or row.get("both_stage1_structural_win") is not expected_both
            or str(row.get("stage1_state_label", "")) != expected_state_label
        ):
            raise ValueError("stage2 condition is inconsistent with stage1 labels")
        by_state[state_id] = row
    return by_state


def _support(rows: Iterable[dict[str, Any]], labels: tuple[str, ...]) -> dict[str, Any]:
    values = list(rows)
    counts = collections.Counter(str(row["label"]) for row in values)
    return {
        "row_count": len(values),
        "state_count": len({str(row["state_occurrence_id"]) for row in values}),
        "map_count": len({str(row["map_id"]) for row in values}),
        "family_group_count": len({str(row["map_family"]) for row in values}),
        "label_counts": {label: int(counts[label]) for label in labels},
        "by_label": {
            label: {
                "row_count": sum(str(row["label"]) == label for row in values),
                "state_count": len(
                    {
                        str(row["state_occurrence_id"])
                        for row in values
                        if str(row["label"]) == label
                    }
                ),
                "map_count": len(
                    {
                        str(row["map_id"])
                        for row in values
                        if str(row["label"]) == label
                    }
                ),
                "family_group_count": len(
                    {
                        str(row["map_family"])
                        for row in values
                        if str(row["label"]) == label
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
        gates[f"{label}_minimum_rows"] = observed["row_count"] >= minimum_rows
        if observed["row_count"] < minimum_rows:
            reasons.append(
                f"{stage}.{label}.row_count={observed['row_count']}<{minimum_rows}"
            )
    for label in decisive:
        observed = support["by_label"][label]
        gates[f"{label}_minimum_maps"] = observed["map_count"] >= minimum_maps
        if observed["map_count"] < minimum_maps:
            reasons.append(
                f"{stage}.{label}.map_count={observed['map_count']}<{minimum_maps}"
            )
    for label in decisive:
        observed = support["by_label"][label]
        gates[f"{label}_minimum_family_groups"] = (
            observed["family_group_count"] >= minimum_families
        )
        if observed["family_group_count"] < minimum_families:
            reasons.append(
                f"{stage}.{label}.family_group_count="
                f"{observed['family_group_count']}<{minimum_families}"
            )
    gates["each_fold_has_both_decisive_classes"] = all(
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
                support["by_label"][label]["row_count"] for label in decisive
            ),
        },
        "threshold_calibration": {
            "label": "ambiguous",
            **support["by_label"]["ambiguous"],
        },
        "folds": fold_support,
        "gates": gates,
        "failure_reasons": reasons,
    }


def check_hierarchical_ch_readiness(
    *,
    project_root: str | Path,
    labels_root: str | Path,
    output: str | Path,
    label_audit_sha256: str = LABEL_AUDIT_SHA256,
    legacy_fold_config: str | Path | None = None,
    legacy_fold_config_sha256: str = LEGACY_FOLD_CONFIG_SHA256,
) -> dict[str, Any]:
    """Validate frozen labels and write a labels-only readiness decision."""

    root = Path(project_root).resolve()
    label_root = Path(labels_root).resolve()
    output_root = Path(output).resolve()
    if output_root == label_root or label_root in output_root.parents:
        raise ValueError("readiness output must remain outside the label inputs")
    audit_path = label_root / "label_audit_report.json"
    audit = _read_hash_pinned_json(
        audit_path, label_audit_sha256, label="label audit report"
    )
    label_contract = dict(audit.get("label_contract") or {})
    if (
        audit.get("schema") != LABEL_AUDIT_SCHEMA
        or audit.get("experiment_id") != LABEL_EXPERIMENT_ID
        or audit.get("scientific_status")
        != "sequential_development_label_audit_only"
        or audit.get("complete") is not True
        or audit.get("training_authorized") is not False
        or audit.get("model_fit_executed") is not False
        or audit.get("model_exported") is not False
        or audit.get("runtime_or_ttf_claim_authorized") is not False
        or audit.get("map_disjoint_confirmation_required") is not True
        or label_contract.get("runtime_or_pp_seconds_used_in_label") is not False
        or label_contract.get("stage2_population")
        not in {None, "component_action_id_ne_hotspot_action_id"}
        or label_contract.get("stage1_positive_used_to_select_stage2")
        not in {None, False}
    ):
        raise ValueError("label audit scientific contract changed")

    artifacts = dict(audit.get("artifacts") or {})
    stage1_path = _artifact_path(
        label_root, dict(artifacts.get("stage1_pair_labels") or {}),
        "stage1_pair_labels",
    )
    stage2_path = _artifact_path(
        label_root, dict(artifacts.get("stage2_state_labels") or {}),
        "stage2_state_labels",
    )
    source_ids, source_report = _source_contract(audit, root)
    exclusion = dict(audit.get("partition_exclusion") or {})
    excluded_ids = set(map(str, exclusion.get("state_occurrence_ids") or ()))
    if exclusion and (
        exclusion.get("condition")
        != "v2_action_id_eq_component_action_id_eq_hotspot_action_id"
        or exclusion.get("canonical_partition") != SINGLE_UNIQUE
        or int(exclusion.get("state_count", -1)) != len(excluded_ids)
        or exclusion.get("excluded_from_stage1_and_stage2") is not True
        or not excluded_ids.issubset(source_ids)
    ):
        raise ValueError("label audit all-equal partition exclusion changed")
    labeled_source_ids = source_ids - excluded_ids
    fold_path = (
        Path(legacy_fold_config).resolve()
        if legacy_fold_config is not None
        else root
        / "configs"
        / "stride_fresh_matched_unique_action_hierarchical_overlay_v1.json"
    )
    folds = _frozen_folds(fold_path, legacy_fold_config_sha256)
    map_to_fold = {
        map_id: fold for fold, map_ids in folds.items() for map_id in map_ids
    }

    stage1_rows = _read_jsonl(stage1_path)
    stage2_rows = _read_jsonl(stage2_path)
    observed = dict(audit.get("observed") or {})
    if (
        int(observed.get("stage1_pair_count", -1)) != len(stage1_rows)
        or int(observed.get("stage2_state_count", -1)) != len(stage2_rows)
        or int(observed.get("state_count", -1)) != len(labeled_source_ids)
    ):
        raise ValueError("label audit observed counts disagree with artifacts")
    stage1_by_state = _validate_stage1(stage1_rows, map_to_fold)
    stage2_by_state = _validate_stage2(stage2_rows, map_to_fold, stage1_by_state)
    stage1_state_ids = set(stage1_by_state)
    stage2_eligible_ids = {
        state_id
        for state_id, rows in stage1_by_state.items()
        if str(rows[0]["hierarchical_stratum"]) in STAGE2_ELIGIBLE_PARTITIONS
    }
    if (
        stage1_state_ids != labeled_source_ids
        or set(stage2_by_state) != stage2_eligible_ids
    ):
        raise ValueError("label artifacts disagree with the source state registry")

    stage1 = _stage_report(
        stage="stage1",
        rows=stage1_rows,
        folds=folds,
        labels=STAGE1_LABELS,
        decisive=STAGE1_DECISIVE,
        minimum_rows=40,
        minimum_maps=8,
        minimum_families=4,
    )
    # Stage-1 outcomes describe the structural-vs-V2 question only.  Every
    # exact C/H mismatch belongs to Stage 2, even when the sole structural
    # action lost to V2 in Stage 1.
    conditioned_stage2 = list(stage2_rows)
    stage1_positive_diagnostic = [
        row for row in stage2_rows if row["any_stage1_structural_win"] is True
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
    stage2["condition"] = "component_action_id_ne_hotspot_action_id"
    stage2["all_state_count"] = len(stage2_rows)
    stage2["conditioned_state_count"] = len(conditioned_stage2)
    stage2["unconditioned_state_count"] = len(stage2_rows) - len(
        conditioned_stage2
    )
    stage2["stage1_positive_diagnostic_state_count"] = len(
        stage1_positive_diagnostic
    )
    stage2["stage1_nonpositive_diagnostic_state_count"] = len(stage2_rows) - len(
        stage1_positive_diagnostic
    )
    stage2["stage1_positive_is_diagnostic_only"] = True

    readiness_passed = bool(
        stage1["readiness_passed"] and stage2["readiness_passed"]
    )
    overall_reasons = [
        *stage1["failure_reasons"],
        *stage2["failure_reasons"],
    ]
    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "scientific_status": "labels_only_readiness_gate",
        "status": "LABEL_SUPPORT_READY" if readiness_passed else "NO_GO",
        "inputs": {
            "label_audit_report": {
                "file": "build/stride-hierarchical-ch-labels-v1/label_audit_report.json",
                "sha256": sha256_file(audit_path),
            },
            "stage1_pair_labels": {
                "file": "build/stride-hierarchical-ch-labels-v1/stage1_pair_labels.jsonl",
                "sha256": sha256_file(stage1_path),
            },
            "stage2_state_labels": {
                "file": "build/stride-hierarchical-ch-labels-v1/stage2_state_labels.jsonl",
                "sha256": sha256_file(stage2_path),
            },
            "source_manifest": source_report,
            "legacy_fold_config": {
                "file": "configs/stride_fresh_matched_unique_action_hierarchical_overlay_v1.json",
                "sha256": sha256_file(fold_path),
            },
        },
        "integrity": {
            "passed": True,
            "source_state_count": len(source_ids),
            "excluded_all_equal_state_count": len(excluded_ids),
            "labeled_source_state_count": len(labeled_source_ids),
            "stage1_pair_count": len(stage1_rows),
            "stage2_state_count": len(stage2_rows),
            "stage2_conditioned_state_count": len(conditioned_stage2),
            "source_and_cross_stage_consistent": True,
        },
        "thresholds": {
            "stage1": STAGE1_THRESHOLDS,
            "stage2": STAGE2_THRESHOLDS,
        },
        "legacy_folds": {fold: list(maps) for fold, maps in folds.items()},
        "stage1": stage1,
        "stage2": stage2,
        "overall": {
            "readiness_passed": readiness_passed,
            "failure_reasons": overall_reasons,
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
    "FROZEN_LEGACY_FOLDS",
    "LABEL_AUDIT_SHA256",
    "LEGACY_FOLD_CONFIG_SHA256",
    "REPORT_SCHEMA",
    "check_hierarchical_ch_readiness",
]
