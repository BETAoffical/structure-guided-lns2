"""Integrity-first compact-flow H1 labels and sequential readiness.

This adapter consumes only a completed compact-flow H1 collection.  It checks
the collection report, its manifest and raw-outcome digests, every state
payload digest, the complete 16-trial product, and the exact raw/payload trial
union before applying the frozen bidirectional H1 classifier.  It reports
label support; it never fits or exports a model and never reads sealed-final
data.
"""

from __future__ import annotations

import collections
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

from experiments._common import sha256_file
from experiments.repair_collection import (
    _fingerprint,
    _read_json,
    _read_jsonl,
    _write_json,
    _write_jsonl,
)
from experiments.stride_hierarchical_ch_compact_flow_h1_collection_v1 import (
    EXPERIMENT_ID as COLLECTION_EXPERIMENT_ID,
    H1_COLLECTION_SCHEMA,
    H1_MANIFEST_ROW_SCHEMA,
    H1_STATE_SCHEMA,
    H1_TRIAL_SCHEMA,
    TRIAL_INDICES,
    _state_payload_valid,
)
from experiments.stride_hierarchical_ch_labels_v1 import (
    ANCHOR_COMPONENT_SHARED,
    ANCHOR_HOTSPOT_SHARED,
    CANONICAL_PARTITIONS,
    CONSENSUS_STRUCTURAL,
    SINGLE_UNIQUE,
    STAGE1_LABELS,
    STAGE1_PAIR_SCHEMA,
    STAGE1_STATE_SCHEMA,
    STAGE2_LABELS,
    STAGE2_STATE_SCHEMA,
    THREE_UNIQUE,
    _validated_state,
    classify_three_way,
)


CONFIG_SCHEMA = (
    "lns2.stride.hierarchical_ch_compact_flow_labels_readiness_config.v1"
)
LABEL_DIAGNOSTICS_SCHEMA = (
    "lns2.stride.hierarchical_ch_compact_flow_label_diagnostics.v1"
)
READINESS_SCHEMA = (
    "lns2.stride.hierarchical_ch_compact_flow_sequential_readiness.v1"
)
EXPERIMENT_ID = "stride_hierarchical_ch_compact_flow_labels_readiness_v1"
DEFAULT_CONFIG = "configs/stride_hierarchical_ch_compact_flow_labels_readiness_v1.json"
DEFAULT_OUTPUT = "build/stride-hierarchical-ch-compact-flow-labels-readiness-v1"
DEFAULT_WORKERS = 16
MAXIMUM_WORKERS = 20

TRAIN_MAPS = (
    "den404d",
    "lak101d",
    "den201d",
    "hrt002d",
    "den009d",
    "den101d",
    "den203d",
    "den308d",
)
DEVELOPMENT_MAPS = (
    "lak108d",
    "lak110d",
    "ost102d",
    "den408d",
    "den202d",
    "den207d",
    "den998d",
    "den020d",
)
TRAIN_MAP_FOLDS = {
    "fold0": ("den404d", "den009d"),
    "fold1": ("lak101d", "den101d"),
    "fold2": ("den201d", "den203d"),
    "fold3": ("hrt002d", "den308d"),
}
DEPTH_BANDS = ("d0", "d1_3", "d4plus")

STAGE1_DECISIVE = ("structural_win", "v2_win")
STAGE2_DECISIVE = ("component_win", "hotspot_win")
STAGE1_PARTITIONS = (
    CONSENSUS_STRUCTURAL,
    THREE_UNIQUE,
    ANCHOR_COMPONENT_SHARED,
    ANCHOR_HOTSPOT_SHARED,
)
STAGE2_PARTITIONS = (
    THREE_UNIQUE,
    ANCHOR_COMPONENT_SHARED,
    ANCHOR_HOTSPOT_SHARED,
)

READINESS_THRESHOLDS = {
    "train": {
        "stage1": {
            "minimum_rows_per_decisive_direction": 4,
            "minimum_maps_per_decisive_direction": 2,
            "minimum_depth_bands_per_decisive_direction": 2,
            "minimum_active_partitions": 1,
            "each_fold_requires_both_decisive_directions": True,
        },
        "stage2": {
            "minimum_rows_per_decisive_direction": 2,
            "minimum_maps_per_decisive_direction": 2,
            "minimum_depth_bands_per_decisive_direction": 2,
            "minimum_active_partitions": 1,
            "each_fold_requires_both_decisive_directions": True,
        },
    },
    "development": {
        "stage1": {
            "minimum_rows_per_decisive_direction": 2,
            "minimum_maps_per_decisive_direction": 2,
            "minimum_depth_bands_per_decisive_direction": 2,
            "minimum_active_partitions": 1,
        },
        "stage2": {
            "minimum_rows_per_decisive_direction": 2,
            "minimum_maps_per_decisive_direction": 2,
            "minimum_depth_bands_per_decisive_direction": 2,
            "minimum_active_partitions": 1,
        },
    },
}

INPUT_CONTRACT = {
    "collection_root": "build/stride-hierarchical-ch-compact-flow-h1-collection-v1",
    "collection_report": "h1_collection_report.json",
    "collection_report_schema": H1_COLLECTION_SCHEMA,
    "collection_experiment_id": COLLECTION_EXPERIMENT_ID,
    "manifest_file": "h1_state_manifest.jsonl",
    "raw_outcomes_file": "raw_unique_action_outcomes.jsonl",
    "manifest_row_schema": H1_MANIFEST_ROW_SCHEMA,
    "state_schema": H1_STATE_SCHEMA,
    "trial_schema": H1_TRIAL_SCHEMA,
}
EXECUTION_CONTRACT = {
    "workers": DEFAULT_WORKERS,
    "maximum_workers": MAXIMUM_WORKERS,
    "workers_enter_run_fingerprint": True,
    "solver_invocation_allowed": False,
}
SEALED_FINAL_CONTRACT = {
    "status": "SEALED",
    "semantic_access_allowed": False,
    "labeling_allowed": False,
    "readiness_access_allowed": False,
}
CLAIM_BOUNDARY = {
    "sequential_design_only": True,
    "development_is_sequential_diagnostic_only": True,
    "map_disjoint_final_confirmation_required": True,
    "training_authorized": False,
    "model_fit_executed": False,
    "model_exported": False,
    "runtime_or_ttf_claim_authorized": False,
}


def _contained(root: Path, value: str, *, field: str) -> Path:
    portable = PurePosixPath(str(value).replace("\\", "/"))
    if portable.is_absolute() or ".." in portable.parts:
        raise ValueError(f"{field} must be a contained project-relative path")
    resolved = (root / Path(*portable.parts)).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{field} escapes the project root") from error
    return resolved


def _require_sha256(value: Any, *, field: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError(f"{field} is not a lowercase SHA256")
    return digest


def validate_config(config: Mapping[str, Any]) -> None:
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("experiment_id") != EXPERIMENT_ID
    ):
        raise ValueError("compact-flow labels/readiness identity changed")
    if dict(config.get("input") or {}) != INPUT_CONTRACT:
        raise ValueError("compact-flow labels/readiness input contract changed")
    if dict(config.get("map_splits") or {}) != {
        "train": list(TRAIN_MAPS),
        "development": list(DEVELOPMENT_MAPS),
    }:
        raise ValueError("compact-flow labels/readiness map splits changed")
    observed_folds = {
        str(fold): tuple(map(str, maps))
        for fold, maps in dict(config.get("train_map_folds") or {}).items()
    }
    if observed_folds != TRAIN_MAP_FOLDS:
        raise ValueError("compact-flow labels/readiness train folds changed")
    folded = [map_id for maps in observed_folds.values() for map_id in maps]
    if len(folded) != len(set(folded)) or set(folded) != set(TRAIN_MAPS):
        raise ValueError("compact-flow labels/readiness train folds overlap")
    if tuple(map(str, config.get("depth_bands") or ())) != DEPTH_BANDS:
        raise ValueError("compact-flow labels/readiness depth bands changed")
    if dict(config.get("readiness_thresholds") or {}) != READINESS_THRESHOLDS:
        raise ValueError("compact-flow labels/readiness thresholds changed")
    if dict(config.get("execution") or {}) != EXECUTION_CONTRACT:
        raise ValueError("compact-flow labels/readiness execution changed")
    if dict(config.get("sealed_final") or {}) != SEALED_FINAL_CONTRACT:
        raise ValueError("compact-flow labels/readiness sealed-final boundary changed")
    if dict(config.get("claim_boundary") or {}) != CLAIM_BOUNDARY:
        raise ValueError("compact-flow labels/readiness claim boundary changed")


def load_config(path: str | Path) -> tuple[Path, Path, dict[str, Any]]:
    config_path = Path(path).resolve()
    project_root = config_path.parents[1]
    config = _read_json(config_path)
    validate_config(config)
    return config_path, project_root, config


def _worker_count(config: Mapping[str, Any], workers: int | None) -> int:
    execution = dict(config["execution"])
    value = int(execution["workers"] if workers is None else workers)
    maximum = int(execution["maximum_workers"])
    if value <= 0 or value > maximum:
        raise ValueError(f"workers must be in 1..{maximum}")
    return value


def _artifact(
    collection_root: Path,
    registered: Mapping[str, Any],
    *,
    expected_file: str,
    field: str,
) -> Path:
    if str(registered.get("file", "")) != expected_file:
        raise ValueError(f"{field} file identity changed")
    path = (collection_root / expected_file).resolve()
    if path.parent != collection_root.resolve() or not path.is_file():
        raise ValueError(f"{field} is missing")
    expected = _require_sha256(registered.get("sha256"), field=f"{field}.sha256")
    if sha256_file(path) != expected:
        raise ValueError(f"{field} SHA256 mismatch")
    return path


def _collection_inputs(
    project_root: Path, config: Mapping[str, Any]
) -> tuple[Path, Path, Path, Path, dict[str, Any]]:
    inputs = dict(config["input"])
    collection_root = _contained(
        project_root, str(inputs["collection_root"]), field="input.collection_root"
    )
    report_path = (collection_root / str(inputs["collection_report"])).resolve()
    if report_path.parent != collection_root or not report_path.is_file():
        raise ValueError("compact-flow H1 collection report is missing")
    report = _read_json(report_path)
    if (
        report.get("schema") != H1_COLLECTION_SCHEMA
        or report.get("experiment_id") != COLLECTION_EXPERIMENT_ID
        or report.get("status") != "complete"
        or report.get("complete") is not True
        or report.get("workers_in_fingerprint") is not True
        or int(report.get("workers", -1)) != DEFAULT_WORKERS
        or report.get("exact_agent_tuple_deduplication") is not True
        or report.get("stage1_population")
        != "v2_vs_each_distinct_structural_action"
        or report.get("stage2_population")
        != "component_vs_hotspot_only_when_exact_sets_differ"
        or report.get("all_equal_excluded") is not True
        or report.get("outcome_based_selection") is not False
        or report.get("backfill_allowed") is not False
        or report.get("sequential_design_only") is not True
        or report.get("training_authorized") is not False
        or report.get("model_fit_executed") is not False
        or report.get("final_claim_authorized") is not False
        or report.get("runtime_or_ttf_claim_authorized") is not False
        or report.get("map_disjoint_confirmation_required") is not True
    ):
        raise ValueError("compact-flow H1 collection trust contract is invalid")
    run_identity = _require_sha256(
        report.get("run_identity"), field="collection.run_identity"
    )
    if not run_identity:
        raise ValueError("compact-flow H1 collection identity is missing")
    artifacts = dict(report.get("artifacts") or {})
    manifest_path = _artifact(
        collection_root,
        dict(artifacts.get("label_source_state_manifest") or {}),
        expected_file=str(inputs["manifest_file"]),
        field="collection.manifest",
    )
    manifest_spec = dict(artifacts["label_source_state_manifest"])
    if manifest_spec.get("state_schema") != H1_STATE_SCHEMA:
        raise ValueError("compact-flow H1 collection state schema changed")
    raw_path = _artifact(
        collection_root,
        dict(artifacts.get("raw_unique_action_outcomes") or {}),
        expected_file=str(inputs["raw_outcomes_file"]),
        field="collection.raw_outcomes",
    )
    return collection_root, report_path, manifest_path, raw_path, report


def _trial_key(row: Mapping[str, Any]) -> tuple[str, str, int]:
    return (
        str(row.get("state_occurrence_id", "")),
        str(row.get("action_id", "")),
        int(row.get("trial_index", -1)),
    )


def _validate_manifest_row(
    collection_root: Path,
    manifest_row: Mapping[str, Any],
    *,
    run_identity: str,
) -> dict[str, Any]:
    row = dict(manifest_row)
    state_id = str(row.get("state_occurrence_id", ""))
    if (
        row.get("schema") != H1_MANIFEST_ROW_SCHEMA
        or not state_id
        or row.get("sequential_design_only") is not True
        or row.get("training_authorized") is not False
        or row.get("final_claim_authorized") is not False
        or row.get("runtime_claim_authorized") is not False
    ):
        raise ValueError(f"compact-flow H1 manifest row is invalid: {state_id}")
    portable = PurePosixPath(str(row.get("state_file", "")).replace("\\", "/"))
    if portable.is_absolute() or ".." in portable.parts:
        raise ValueError("compact-flow H1 manifest state path escapes collection")
    state_path = (collection_root / Path(*portable.parts)).resolve()
    try:
        state_path.relative_to(collection_root.resolve())
    except ValueError as error:
        raise ValueError("compact-flow H1 state path escapes collection") from error
    if not state_path.is_file():
        raise ValueError(f"compact-flow H1 state payload is missing: {state_id}")
    state_sha = _require_sha256(
        row.get("state_sha256"), field="manifest.state_sha256"
    )
    if sha256_file(state_path) != state_sha:
        raise ValueError(f"compact-flow H1 state payload SHA256 mismatch: {state_id}")
    payload = _read_json(state_path)
    state_row = dict(payload.get("state_row") or {})
    if (
        payload.get("schema") != H1_STATE_SCHEMA
        or payload.get("identity") != run_identity
        or payload.get("complete") is not True
        or payload.get("state_occurrence_id") != state_id
        or payload.get("runtime_used_in_label") is not False
        or payload.get("sequential_design_only") is not True
        or payload.get("training_authorized") is not False
        or payload.get("final_claim_authorized") is not False
        or payload.get("runtime_claim_authorized") is not False
        or not _state_payload_valid(
            payload, identity=run_identity, state_row=state_row
        )
    ):
        raise ValueError(f"compact-flow H1 state payload is incomplete: {state_id}")
    trials = [dict(value) for value in payload.get("trials") or ()]
    if (
        int(row.get("unique_action_count", -1))
        != int(state_row.get("unique_action_count", -2))
        or int(row.get("trial_count", -1)) != len(trials)
        or len(trials) != int(row["unique_action_count"]) * len(TRIAL_INDICES)
    ):
        raise ValueError(f"compact-flow H1 state trial count changed: {state_id}")
    identities: set[str] = set()
    for trial in trials:
        trial_identity = str(trial.get("trial_identity", ""))
        if (
            trial.get("schema") != H1_TRIAL_SCHEMA
            or trial.get("state_occurrence_id") != state_id
            or not trial_identity
            or trial_identity in identities
            or trial.get("runtime_used_in_label") is not False
            or trial.get("sequential_design_only") is not True
            or trial.get("training_authorized") is not False
            or trial.get("final_claim_authorized") is not False
            or trial.get("runtime_claim_authorized") is not False
        ):
            raise ValueError(f"compact-flow H1 trial integrity changed: {state_id}")
        identities.add(trial_identity)

    split = str(state_row.get("split", state_row.get("research_split", "")))
    map_id = str(state_row.get("map_id", ""))
    depth_band = str(state_row.get("depth_band", ""))
    expected_maps = TRAIN_MAPS if split == "train" else DEVELOPMENT_MAPS
    expected_fold = next(
        (fold for fold, maps in TRAIN_MAP_FOLDS.items() if map_id in maps), None
    )
    if (
        split not in {"train", "development"}
        or map_id not in expected_maps
        or depth_band not in DEPTH_BANDS
        or not str(state_row.get("map_family", ""))
        or state_row.get("all_equal_excluded") is not False
        or str(state_row.get("hierarchical_stratum", "")) == SINGLE_UNIQUE
        or (split == "train" and state_row.get("train_fold") != expected_fold)
        or (split == "development" and state_row.get("train_fold") is not None)
    ):
        raise ValueError(f"compact-flow H1 state split/depth/partition changed: {state_id}")

    # Reuse the frozen label module's exact action/paired-seed validation.
    validated = _validated_state(
        collection_root,
        row,
        source_state_schema=H1_STATE_SCHEMA,
    )
    role_map = dict(validated["role_map"])
    actions = dict(validated["actions"])
    if (
        len(actions[str(role_map["component16"])]) != 16
        or len(actions[str(role_map["hotspot16"])]) != 16
        or state_row.get("canonical_partition")
        != state_row.get("hierarchical_stratum")
        or bool(state_row.get("stage2_eligible"))
        is not (role_map["component16"] != role_map["hotspot16"])
    ):
        raise ValueError(f"compact-flow H1 exact C/H contract changed: {state_id}")
    return {
        **validated,
        "manifest_row": row,
        "payload_trials": trials,
        "split": split,
        "map_id": map_id,
        "depth_band": depth_band,
        "train_fold": expected_fold if split == "train" else None,
    }


def _load_validated_states(
    collection_root: Path,
    manifest_path: Path,
    raw_path: Path,
    report: Mapping[str, Any],
    *,
    workers: int,
) -> list[dict[str, Any]]:
    manifest = [dict(row) for row in _read_jsonl(manifest_path)]
    state_ids = [str(row.get("state_occurrence_id", "")) for row in manifest]
    if not manifest or not all(state_ids) or len(state_ids) != len(set(state_ids)):
        raise ValueError("compact-flow H1 manifest is empty or has duplicate states")
    run_identity = str(report["run_identity"])
    with ThreadPoolExecutor(max_workers=workers) as executor:
        states = list(
            executor.map(
                lambda row: _validate_manifest_row(
                    collection_root, row, run_identity=run_identity
                ),
                manifest,
            )
        )
    states.sort(key=lambda state: str(state["state_id"]))

    raw = [dict(row) for row in _read_jsonl(raw_path)]
    raw_keys = [_trial_key(row) for row in raw]
    if len(raw_keys) != len(set(raw_keys)):
        raise ValueError("compact-flow H1 raw outcomes have duplicate trials")
    payload_trials = [
        trial for state in states for trial in state["payload_trials"]
    ]
    raw_by_key = {_trial_key(row): row for row in raw}
    payload_by_key = {_trial_key(row): row for row in payload_trials}
    if raw_by_key != payload_by_key:
        raise ValueError("compact-flow H1 raw outcomes differ from state payload trials")

    observed_actions = sum(len(state["actions"]) for state in states)
    counts = {
        "state_count": len(states),
        "unique_action_count": observed_actions,
        "trial_count": len(raw),
    }
    expected = {
        "state_count": int(report.get("expected_state_count", -1)),
        "unique_action_count": int(report.get("expected_unique_action_count", -1)),
        "trial_count": int(report.get("expected_trial_count", -1)),
    }
    observed_report = {
        "state_count": int(report.get("observed_valid_state_count", -2)),
        "unique_action_count": int(report.get("observed_unique_action_count", -2)),
        "trial_count": int(report.get("observed_trial_count", -2)),
    }
    if counts != expected or counts != observed_report or int(
        report.get("invalid_state_count", -1)
    ) != 0:
        raise ValueError(
            "compact-flow H1 collection counts disagree: "
            f"payload={counts}, expected={expected}, observed={observed_report}"
        )
    if int(report.get("selected_state_count", -1)) != (
        counts["state_count"]
        + int(report.get("excluded_all_equal_state_count", -2))
    ):
        raise ValueError("compact-flow H1 all-equal exclusion count changed")
    return states


def _metadata(state: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(state["state_row"])
    return {
        "state_occurrence_id": str(state["state_id"]),
        "map_id": str(row["map_id"]),
        "map_family": str(row["map_family"]),
        "task_id": str(row["task_id"]),
        "solver_seed": int(row["solver_seed"]),
        "decision_index": int(row["decision_index"]),
        "depth_band": str(row["depth_band"]),
        "hierarchical_stratum": str(state["hierarchical_stratum"]),
        "research_split": str(state["split"]),
        "train_fold": state["train_fold"],
        "sequential_design_only": True,
        "training_authorized": False,
    }


def _label_state(
    state: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any] | None]:
    role_map = dict(state["role_map"])
    anchor_id = str(role_map["v2_anchor"])
    structural_roles: dict[str, list[str]] = collections.defaultdict(list)
    for role in ("component16", "hotspot16"):
        action_id = str(role_map[role])
        if action_id != anchor_id:
            structural_roles[action_id].append(role)
    if not structural_roles:
        raise ValueError("compact-flow H1 all-equal state reached Stage 1")

    pairs: list[dict[str, Any]] = []
    for structural_id, aliases in sorted(structural_roles.items()):
        comparison = classify_three_way(
            state["by_action"][structural_id],
            state["by_action"][anchor_id],
            left_label="structural_win",
            right_label="v2_win",
        )
        pairs.append(
            {
                "schema": STAGE1_PAIR_SCHEMA,
                **_metadata(state),
                "pair_id": f"{state['state_id']}::{structural_id}::{anchor_id}",
                "structural_action_id": structural_id,
                "v2_action_id": anchor_id,
                "structural_role_aliases": sorted(aliases),
                **comparison,
            }
        )
    pair_counts = collections.Counter(str(row["label"]) for row in pairs)
    state_label = (
        "structural_win"
        if pair_counts["structural_win"] > 0
        else "v2_win"
        if pair_counts["v2_win"] == len(pairs)
        else "ambiguous"
    )
    stage1_state = {
        "schema": STAGE1_STATE_SCHEMA,
        **_metadata(state),
        "label": state_label,
        "pair_count": len(pairs),
        "pair_label_counts": {
            label: int(pair_counts[label]) for label in STAGE1_LABELS
        },
        "any_structural_win": pair_counts["structural_win"] > 0,
        "both_structural_win": (
            len(pairs) == 2 and pair_counts["structural_win"] == 2
        ),
        "runtime_or_pp_seconds_used_in_label": False,
    }

    component_id = str(role_map["component16"])
    hotspot_id = str(role_map["hotspot16"])
    stage2 = None
    if component_id != hotspot_id:
        comparison = classify_three_way(
            state["by_action"][component_id],
            state["by_action"][hotspot_id],
            left_label="component_win",
            right_label="hotspot_win",
        )
        stage2 = {
            "schema": STAGE2_STATE_SCHEMA,
            **_metadata(state),
            "component_action_id": component_id,
            "hotspot_action_id": hotspot_id,
            "eligibility_condition": "component_action_id_ne_hotspot_action_id",
            "stage1_state_label": state_label,
            "any_stage1_structural_win": bool(stage1_state["any_structural_win"]),
            "both_stage1_structural_win": bool(stage1_state["both_structural_win"]),
            **comparison,
        }
    return pairs, stage1_state, stage2


def _coverage(
    rows: Iterable[Mapping[str, Any]], labels: tuple[str, ...]
) -> dict[str, Any]:
    values = [dict(row) for row in rows]
    counts = collections.Counter(str(row["label"]) for row in values)
    if set(counts) - set(labels):
        raise ValueError("compact-flow label support contains an unknown direction")

    def support(selected: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "row_count": len(selected),
            "state_count": len(
                {str(row["state_occurrence_id"]) for row in selected}
            ),
            "map_count": len({str(row["map_id"]) for row in selected}),
            "maps": sorted({str(row["map_id"]) for row in selected}),
            "depth_band_count": len(
                {str(row["depth_band"]) for row in selected}
            ),
            "depth_bands": sorted({str(row["depth_band"]) for row in selected}),
            "partition_count": len(
                {str(row["hierarchical_stratum"]) for row in selected}
            ),
            "partitions": sorted(
                {str(row["hierarchical_stratum"]) for row in selected}
            ),
        }

    return {
        **support(values),
        "label_counts": {label: int(counts[label]) for label in labels},
        "by_direction": {
            label: support(
                [row for row in values if str(row["label"]) == label]
            )
            for label in labels
        },
        "by_map": dict(
            sorted(collections.Counter(str(row["map_id"]) for row in values).items())
        ),
        "by_depth_band": {
            band: sum(str(row["depth_band"]) == band for row in values)
            for band in DEPTH_BANDS
        },
        "by_partition": {
            partition: sum(
                str(row["hierarchical_stratum"]) == partition for row in values
            )
            for partition in CANONICAL_PARTITIONS
        },
    }


def _stage_readiness(
    *,
    stage: str,
    split: str,
    rows: list[dict[str, Any]],
    labels: tuple[str, ...],
    decisive: tuple[str, str],
    allowed_partitions: tuple[str, ...],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    coverage = _coverage(rows, labels)
    gates: dict[str, bool] = {}
    failures: list[str] = []
    for direction in decisive:
        observed = dict(coverage["by_direction"][direction])
        for metric, threshold_key in (
            ("row_count", "minimum_rows_per_decisive_direction"),
            ("map_count", "minimum_maps_per_decisive_direction"),
            ("depth_band_count", "minimum_depth_bands_per_decisive_direction"),
        ):
            minimum = int(thresholds[threshold_key])
            passed = int(observed[metric]) >= minimum
            key = f"{direction}_{metric}"
            gates[key] = passed
            if not passed:
                failures.append(
                    f"{split}.{stage}.{direction}.{metric}="
                    f"{observed[metric]}<{minimum}"
                )
    active_partitions = [
        partition
        for partition in allowed_partitions
        if int(coverage["by_partition"][partition]) > 0
    ]
    minimum_partitions = int(thresholds["minimum_active_partitions"])
    gates["minimum_active_partitions"] = len(active_partitions) >= minimum_partitions
    if len(active_partitions) < minimum_partitions:
        failures.append(
            f"{split}.{stage}.active_partitions="
            f"{len(active_partitions)}<{minimum_partitions}"
        )

    fold_support: dict[str, Any] | None = None
    if split == "train":
        fold_support = {}
        for fold, maps in TRAIN_MAP_FOLDS.items():
            selected = [row for row in rows if str(row["map_id"]) in maps]
            fold_counts = collections.Counter(str(row["label"]) for row in selected)
            missing = [direction for direction in decisive if fold_counts[direction] == 0]
            fold_support[fold] = {
                "maps": list(maps),
                "row_count": len(selected),
                "label_counts": {
                    label: int(fold_counts[label]) for label in labels
                },
                "missing_decisive_directions": missing,
                "has_both_decisive_directions": not missing,
            }
        folds_pass = all(
            row["has_both_decisive_directions"] for row in fold_support.values()
        )
        gates["each_fold_has_both_decisive_directions"] = folds_pass
        for fold, row in fold_support.items():
            if row["missing_decisive_directions"]:
                failures.append(
                    f"train.{stage}.{fold}.missing="
                    + ",".join(row["missing_decisive_directions"])
                )

    passed = all(gates.values())
    return {
        "support": coverage,
        "decisive_directions": list(decisive),
        "ambiguous_is_threshold_calibration_only": True,
        "active_eligible_partitions": active_partitions,
        "thresholds": dict(thresholds),
        "folds": fold_support,
        "gates": gates,
        "failure_reasons": failures,
        "label_support_trainable": passed if split == "train" else False,
        "sequential_diagnostic_threshold_passed": (
            passed if split == "development" else None
        ),
        "development_used_for_model_selection_or_final_claim": (
            False if split == "development" else None
        ),
        "training_authorized": False,
    }


def build_compact_flow_labels_readiness(
    config_path: str | Path,
    *,
    output: str | Path = DEFAULT_OUTPUT,
    workers: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Validate a completed H1 collection and build labels/readiness only."""

    path, root, config = load_config(config_path)
    worker_count = _worker_count(config, workers)
    collection_root, report_path, manifest_path, raw_path, collection_report = (
        _collection_inputs(root, config)
    )
    if int(collection_report["workers"]) != worker_count:
        raise ValueError(
            "labels/readiness workers must match the collection fingerprint"
        )
    output_root = Path(output).resolve()
    if output_root == collection_root or collection_root in output_root.parents:
        raise ValueError("labels/readiness output must remain outside H1 collection")

    states = _load_validated_states(
        collection_root,
        manifest_path,
        raw_path,
        collection_report,
        workers=worker_count,
    )
    stage1_pairs: list[dict[str, Any]] = []
    stage1_states: list[dict[str, Any]] = []
    stage2_states: list[dict[str, Any]] = []
    partition_counts: collections.Counter[tuple[str, str]] = collections.Counter()
    for state in states:
        partition = str(state["hierarchical_stratum"])
        if partition == SINGLE_UNIQUE:
            raise ValueError("compact-flow H1 all-equal state was not excluded")
        partition_counts[(str(state["split"]), partition)] += 1
        pairs, state_label, stage2 = _label_state(state)
        stage1_pairs.extend(pairs)
        stage1_states.append(state_label)
        if stage2 is not None:
            stage2_states.append(stage2)

    stage1_pairs.sort(key=lambda row: str(row["pair_id"]))
    stage1_states.sort(key=lambda row: str(row["state_occurrence_id"]))
    stage2_states.sort(key=lambda row: str(row["state_occurrence_id"]))
    run_fingerprint = _fingerprint(
        {
            "namespace": "stride-hierarchical-ch-compact-flow-labels-readiness-v1",
            "config_sha256": sha256_file(path),
            "collection_report_sha256": sha256_file(report_path),
            "manifest_sha256": sha256_file(manifest_path),
            "raw_outcomes_sha256": sha256_file(raw_path),
            "workers": worker_count,
        }
    )

    split_reports: dict[str, Any] = {}
    for split in ("train", "development"):
        split_stage1 = [
            row for row in stage1_pairs if str(row["research_split"]) == split
        ]
        split_stage2 = [
            row for row in stage2_states if str(row["research_split"]) == split
        ]
        thresholds = dict(config["readiness_thresholds"])[split]
        split_reports[split] = {
            "scientific_role": (
                "fixed_four_map_folds_training_support"
                if split == "train"
                else "sequential_development_diagnostic_only"
            ),
            "stage1": _stage_readiness(
                stage="stage1",
                split=split,
                rows=split_stage1,
                labels=STAGE1_LABELS,
                decisive=STAGE1_DECISIVE,
                allowed_partitions=STAGE1_PARTITIONS,
                thresholds=thresholds["stage1"],
            ),
            "stage2": _stage_readiness(
                stage="stage2",
                split=split,
                rows=split_stage2,
                labels=STAGE2_LABELS,
                decisive=STAGE2_DECISIVE,
                allowed_partitions=STAGE2_PARTITIONS,
                thresholds=thresholds["stage2"],
            ),
        }

    labels_diagnostics = {
        "schema": LABEL_DIAGNOSTICS_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": "COMPLETE_SEQUENTIAL_LABELS",
        "complete": True,
        "run_fingerprint": run_fingerprint,
        "workers": worker_count,
        "workers_in_fingerprint": True,
        "input_integrity": {
            "collection_report_complete": True,
            "collection_report_sha256": sha256_file(report_path),
            "manifest_sha256": sha256_file(manifest_path),
            "raw_outcomes_sha256": sha256_file(raw_path),
            "every_state_payload_sha256_verified": True,
            "every_state_payload_complete": True,
            "raw_trials_equal_state_payload_trial_union": True,
            "complete_paired_trial_count_per_unique_action": len(TRIAL_INDICES),
            "state_count": len(states),
            "unique_action_count": sum(len(state["actions"]) for state in states),
            "trial_count": sum(len(state["payload_trials"]) for state in states),
        },
        "label_counts": {
            "stage1_pair_count": len(stage1_pairs),
            "stage1_state_count": len(stage1_states),
            "stage2_state_count": len(stage2_states),
        },
        "canonical_partition_support": {
            split: {
                partition: int(partition_counts[(split, partition)])
                for partition in CANONICAL_PARTITIONS
            }
            for split in ("train", "development")
        },
        "all_equal_exclusion": {
            "canonical_partition": SINGLE_UNIQUE,
            "excluded_before_h1_collection": True,
            "excluded_state_count": int(
                collection_report.get("excluded_all_equal_state_count", 0)
            ),
            "present_in_label_source_manifest": False,
        },
        "label_contract": {
            "classifier": "experiments.stride_hierarchical_ch_labels_v1.classify_three_way",
            "paired_trial_count": len(TRIAL_INDICES),
            "direction_rule": "frozen_h1_rule_applied_in_both_directions",
            "ambiguous_rule": "neither_direction_is_a_robust_win",
            "stage1_population": "v2_vs_each_distinct_structural_action",
            "stage2_population": "component_vs_hotspot_only_when_exact_sets_differ",
            "stage1_positive_used_to_select_stage2": False,
            "runtime_or_pp_seconds_used_in_label": False,
            "trial_is_training_sample": False,
        },
        "training_authorized": False,
        "model_fit_executed": False,
        "model_exported": False,
        "final_claim_authorized": False,
        "runtime_or_ttf_claim_authorized": False,
        "map_disjoint_confirmation_required": True,
    }

    train_ready = bool(
        split_reports["train"]["stage1"]["label_support_trainable"]
        and split_reports["train"]["stage2"]["label_support_trainable"]
    )
    development_ready = bool(
        split_reports["development"]["stage1"][
            "sequential_diagnostic_threshold_passed"
        ]
        and split_reports["development"]["stage2"][
            "sequential_diagnostic_threshold_passed"
        ]
    )
    readiness = {
        "schema": READINESS_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "scientific_status": "sequential_labels_only_readiness",
        "status": (
            "LABEL_SUPPORT_READY"
            if train_ready and development_ready
            else "NO_GO_LABEL_SUPPORT"
        ),
        "complete": True,
        "run_fingerprint": run_fingerprint,
        "workers": worker_count,
        "workers_in_fingerprint": True,
        "fixed_train_map_folds": {
            fold: list(maps) for fold, maps in TRAIN_MAP_FOLDS.items()
        },
        "development_has_training_folds": False,
        "splits": split_reports,
        "overall": {
            "train_label_support_trainable": train_ready,
            "development_sequential_threshold_passed": development_ready,
            "sequential_pipeline_label_support_ready": (
                train_ready and development_ready
            ),
            "training_authorized": False,
        },
        "development_is_sequential_diagnostic_only": True,
        "training_authorized": False,
        "model_fit_executed": False,
        "model_exported": False,
        "final_claim_authorized": False,
        "runtime_or_ttf_claim_authorized": False,
        "map_disjoint_confirmation_required": True,
    }

    if dry_run:
        return {
            "labels": labels_diagnostics,
            "readiness": readiness,
            "dry_run": True,
            "artifacts_written": False,
        }

    output_root.mkdir(parents=True, exist_ok=True)
    targets = {
        "stage1_pair_labels": output_root / "stage1_pair_labels.jsonl",
        "stage1_state_labels": output_root / "stage1_state_labels.jsonl",
        "stage2_state_labels": output_root / "stage2_state_labels.jsonl",
        "label_diagnostics": output_root / "label_diagnostics.json",
        "readiness_report": output_root / "readiness_report.json",
    }
    if any(path.is_file() for path in targets.values()):
        raise ValueError("compact-flow labels/readiness output already exists")
    _write_jsonl(targets["stage1_pair_labels"], stage1_pairs)
    _write_jsonl(targets["stage1_state_labels"], stage1_states)
    _write_jsonl(targets["stage2_state_labels"], stage2_states)
    labels_diagnostics["artifacts"] = {
        key: {"file": value.name, "sha256": sha256_file(value)}
        for key, value in targets.items()
        if key in {
            "stage1_pair_labels",
            "stage1_state_labels",
            "stage2_state_labels",
        }
    }
    _write_json(targets["label_diagnostics"], labels_diagnostics)
    readiness["inputs"] = {
        "label_diagnostics": {
            "file": targets["label_diagnostics"].name,
            "sha256": sha256_file(targets["label_diagnostics"]),
        },
        **dict(labels_diagnostics["artifacts"]),
    }
    _write_json(targets["readiness_report"], readiness)
    return {"labels": labels_diagnostics, "readiness": readiness}


__all__ = [
    "CONFIG_SCHEMA",
    "DEFAULT_CONFIG",
    "DEFAULT_OUTPUT",
    "DEVELOPMENT_MAPS",
    "EXPERIMENT_ID",
    "LABEL_DIAGNOSTICS_SCHEMA",
    "READINESS_SCHEMA",
    "TRAIN_MAP_FOLDS",
    "TRAIN_MAPS",
    "build_compact_flow_labels_readiness",
    "load_config",
    "validate_config",
]
