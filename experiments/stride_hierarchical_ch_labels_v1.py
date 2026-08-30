from __future__ import annotations

import collections
import math
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from experiments._common import sha256_file
from experiments.repair_collection import (
    _read_json,
    _read_jsonl,
    _write_json,
    _write_jsonl,
)
from experiments.stride_fresh_matched_v2_c16_h16_v1 import (
    TRIAL_INDICES,
    classify_h1_challenger,
)


CONFIG_SCHEMA = "lns2.stride.hierarchical_ch_labels_config.v1"
REPORT_SCHEMA = "lns2.stride.hierarchical_ch_label_audit.v1"
STAGE1_PAIR_SCHEMA = "lns2.stride.hierarchical_ch_stage1_pair_label.v1"
STAGE1_STATE_SCHEMA = "lns2.stride.hierarchical_ch_stage1_state_label.v1"
STAGE2_STATE_SCHEMA = "lns2.stride.hierarchical_ch_stage2_state_label.v1"
SOURCE_STATE_SCHEMA = (
    "lns2.stride.fresh_matched_unique_action_hierarchical_h1_state.v1"
)
ROLES = ("v2_anchor", "component16", "hotspot16")
STAGE1_LABELS = ("structural_win", "v2_win", "ambiguous")
STAGE2_LABELS = ("component_win", "hotspot_win", "ambiguous")
CONSENSUS_STRUCTURAL = "consensus_structural"
THREE_UNIQUE = "three_unique"
ANCHOR_COMPONENT_SHARED = "anchor_component_shared"
ANCHOR_HOTSPOT_SHARED = "anchor_hotspot_shared"
SINGLE_UNIQUE = "single_unique"
CANONICAL_PARTITIONS = (
    CONSENSUS_STRUCTURAL,
    THREE_UNIQUE,
    ANCHOR_COMPONENT_SHARED,
    ANCHOR_HOTSPOT_SHARED,
    SINGLE_UNIQUE,
)
LABELED_PARTITIONS = tuple(
    partition for partition in CANONICAL_PARTITIONS if partition != SINGLE_UNIQUE
)
STAGE2_ELIGIBLE_PARTITIONS = (
    THREE_UNIQUE,
    ANCHOR_COMPONENT_SHARED,
    ANCHOR_HOTSPOT_SHARED,
)


def _strict_positive_int(value: Any, *, field: str) -> int:
    if type(value) is not int or int(value) <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return int(value)


def _strict_nonnegative_int(value: Any, *, field: str) -> int:
    if type(value) is not int or int(value) < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return int(value)


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unexpected hierarchical C/H label config schema")
    if not str(config.get("experiment_id", "")).strip():
        raise ValueError("hierarchical C/H label experiment_id is missing")
    source = dict(config.get("source") or {})
    if set(source) - {"manifest", "state_schema"}:
        raise ValueError("hierarchical C/H source contract has unknown fields")
    manifest = dict(source.get("manifest") or {})
    if not str(manifest.get("path", "")).strip():
        raise ValueError("hierarchical C/H source manifest path is missing")
    digest = manifest.get("sha256")
    if digest is not None and (
        len(str(digest)) != 64
        or any(character not in "0123456789abcdef" for character in str(digest).lower())
    ):
        raise ValueError("hierarchical C/H source manifest SHA256 is invalid")
    state_schema = source.get("state_schema", SOURCE_STATE_SCHEMA)
    if not isinstance(state_schema, str) or not state_schema.strip():
        raise ValueError("hierarchical C/H source state schema is invalid")
    expected = dict(config.get("expected") or {})
    required = {
        "state_count",
        "unique_action_count",
        "trial_count",
        "stage1_pair_count",
        "stage2_state_count",
    }
    if set(expected) != required:
        raise ValueError("hierarchical C/H expected product contract changed")
    for field in sorted(required - {"stage2_state_count"}):
        _strict_positive_int(expected[field], field=f"expected.{field}")
    _strict_nonnegative_int(
        expected["stage2_state_count"], field="expected.stage2_state_count"
    )


def _contained_path(root: Path, value: str, *, field: str) -> Path:
    portable = PurePosixPath(str(value).replace("\\", "/"))
    if portable.is_absolute() or ".." in portable.parts:
        raise ValueError(f"{field} must be a contained relative path")
    path = (root / Path(*portable.parts)).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{field} escapes its registered root") from error
    return path


def _registered_manifest(config: dict[str, Any], project_root: Path) -> Path:
    specification = dict(dict(config["source"])["manifest"])
    path = _contained_path(
        project_root, str(specification["path"]), field="source.manifest.path"
    )
    if not path.is_file():
        raise ValueError(f"hierarchical C/H source manifest is missing: {path}")
    expected = specification.get("sha256")
    if expected is not None and sha256_file(path) != str(expected).lower():
        raise ValueError("hierarchical C/H source manifest SHA256 mismatch")
    return path


def _action_agents(state_row: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    actions: dict[str, tuple[int, ...]] = {}
    for raw in list(state_row.get("unique_actions") or ()):
        action = dict(raw)
        action_id = str(action.get("action_id", ""))
        agents = tuple(sorted(map(int, action.get("agents") or ())))
        if (
            not action_id
            or not agents
            or len(agents) != len(set(agents))
            or action_id in actions
            or agents in actions.values()
        ):
            raise ValueError("hierarchical C/H source has invalid unique actions")
        actions[action_id] = agents
    return actions


def _canonical_partition(
    anchor: tuple[int, ...],
    component: tuple[int, ...],
    hotspot: tuple[int, ...],
) -> str:
    """Return the exact V/C/H set partition without outcome information."""

    if component == hotspot != anchor:
        return CONSENSUS_STRUCTURAL
    if len({anchor, component, hotspot}) == 3:
        return THREE_UNIQUE
    if anchor == component != hotspot:
        return ANCHOR_COMPONENT_SHARED
    if anchor == hotspot != component:
        return ANCHOR_HOTSPOT_SHARED
    if anchor == component == hotspot:
        return SINGLE_UNIQUE
    raise ValueError("hierarchical C/H exact-set partition is impossible")


def _state_source(
    manifest_root: Path,
    manifest_row: dict[str, Any],
    *,
    source_state_schema: str,
) -> tuple[Path, dict[str, Any]]:
    path = _contained_path(
        manifest_root,
        str(manifest_row.get("state_file", "")),
        field="manifest.state_file",
    )
    if not path.is_file() or sha256_file(path) != str(
        manifest_row.get("state_sha256", "")
    ).lower():
        raise ValueError(f"hierarchical C/H state payload changed: {path}")
    payload = _read_json(path)
    state_id = str(manifest_row.get("state_occurrence_id", ""))
    if (
        payload.get("schema") != source_state_schema
        or payload.get("complete") is not True
        or str(payload.get("state_occurrence_id", "")) != state_id
        or payload.get("runtime_used_in_label") is not False
    ):
        raise ValueError(f"hierarchical C/H state payload is invalid: {state_id}")
    return path, payload


def _validated_state(
    manifest_root: Path,
    manifest_row: dict[str, Any],
    *,
    source_state_schema: str,
) -> dict[str, Any]:
    path, payload = _state_source(
        manifest_root,
        manifest_row,
        source_state_schema=source_state_schema,
    )
    state_row = dict(payload.get("state_row") or {})
    state_id = str(payload["state_occurrence_id"])
    if str(state_row.get("state_occurrence_id", "")) != state_id:
        raise ValueError("hierarchical C/H embedded state identity changed")
    role_map = {
        str(role): str(action_id)
        for role, action_id in dict(state_row.get("role_to_action_id") or {}).items()
    }
    if set(role_map) != set(ROLES):
        raise ValueError("hierarchical C/H role map is incomplete")
    actions = _action_agents(state_row)
    if (
        int(state_row.get("unique_action_count", -1)) != len(actions)
        or int(manifest_row.get("unique_action_count", -1)) != len(actions)
        or any(action_id not in actions for action_id in role_map.values())
    ):
        raise ValueError("hierarchical C/H unique action count changed")
    anchor = actions[role_map["v2_anchor"]]
    component = actions[role_map["component16"]]
    hotspot = actions[role_map["hotspot16"]]
    expected_stratum = _canonical_partition(anchor, component, hotspot)
    observed_stratum = str(
        state_row.get(
            "hierarchical_stratum",
            state_row.get("canonical_partition", ""),
        )
    )
    if observed_stratum != expected_stratum:
        raise ValueError("hierarchical C/H exact-set stratum changed")

    trials = list(payload.get("trials") or ())
    if int(manifest_row.get("trial_count", -1)) != len(trials) or len(trials) != (
        16 * len(actions)
    ):
        raise ValueError("hierarchical C/H trial count changed")
    by_action: dict[str, dict[int, dict[str, Any]]] = {
        action_id: {} for action_id in actions
    }
    seen_trial_ids: set[str] = set()
    for raw in trials:
        trial = dict(raw)
        action_id = str(trial.get("action_id", ""))
        trial_index = int(trial.get("trial_index", -1))
        trial_identity = str(trial.get("trial_identity", ""))
        if (
            action_id not in by_action
            or trial_index not in TRIAL_INDICES
            or trial_index in by_action[action_id]
            or not trial_identity
            or trial_identity in seen_trial_ids
            or trial.get("runtime_used_in_label") is not False
        ):
            raise ValueError("hierarchical C/H paired trial identity changed")
        if not math.isfinite(float(trial.get("normalized_conflict_reduction", math.nan))):
            raise ValueError("hierarchical C/H trial score is non-finite")
        by_action[action_id][trial_index] = trial
        seen_trial_ids.add(trial_identity)
    if any(set(rows) != set(TRIAL_INDICES) for rows in by_action.values()):
        raise ValueError("hierarchical C/H action lacks 16 paired trials")
    for trial_index in TRIAL_INDICES:
        seeds = {
            int(rows[trial_index]["pp_seed"]) for rows in by_action.values()
        }
        if len(seeds) != 1:
            raise ValueError("hierarchical C/H actions do not share paired PP seeds")
    return {
        "path": path,
        "payload": payload,
        "state_row": state_row,
        "state_id": state_id,
        "role_map": role_map,
        "actions": actions,
        "by_action": {
            action_id: [rows[index] for index in TRIAL_INDICES]
            for action_id, rows in by_action.items()
        },
        "hierarchical_stratum": observed_stratum,
    }


def classify_three_way(
    left_trials: Iterable[dict[str, Any]],
    right_trials: Iterable[dict[str, Any]],
    *,
    left_label: str,
    right_label: str,
) -> dict[str, Any]:
    left = list(left_trials)
    right = list(right_trials)
    left_over_right = classify_h1_challenger(right, left, distinct_from_anchor=True)
    right_over_left = classify_h1_challenger(left, right, distinct_from_anchor=True)
    if left_over_right.get("label") is None or right_over_left.get("label") is None:
        raise ValueError("hierarchical C/H three-way label requires complete paired trials")
    left_win = left_over_right.get("label") is True
    right_win = right_over_left.get("label") is True
    if left_win and right_win:
        raise ValueError("hierarchical C/H robust directions are contradictory")
    label = left_label if left_win else right_label if right_win else "ambiguous"
    return {
        "label": label,
        "left_label": left_label,
        "right_label": right_label,
        "left_over_right": left_over_right,
        "right_over_left": right_over_left,
        "runtime_or_pp_seconds_used_in_label": False,
    }


def _metadata(state: dict[str, Any]) -> dict[str, Any]:
    row = state["state_row"]
    return {
        "state_occurrence_id": state["state_id"],
        "map_id": str(row.get("map_id", "")),
        "map_family": str(row.get("map_family", row.get("layout_mode", "unknown"))),
        "task_id": str(row.get("task_id", "")),
        "solver_seed": int(row.get("solver_seed", 0)),
        "decision_index": int(row.get("decision_index", 0)),
        "hierarchical_stratum": str(state["hierarchical_stratum"]),
        "research_split": str(row.get("research_split", row.get("split", ""))),
    }


def _stage1_rows(state: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    role_map = state["role_map"]
    anchor_id = role_map["v2_anchor"]
    structural_roles: dict[str, list[str]] = collections.defaultdict(list)
    for role in ("component16", "hotspot16"):
        action_id = role_map[role]
        if action_id != anchor_id:
            structural_roles[action_id].append(role)
    if not structural_roles:
        raise ValueError("hierarchical C/H state lacks a structural challenger")
    pairs = []
    for action_id, aliases in sorted(structural_roles.items()):
        comparison = classify_three_way(
            state["by_action"][action_id],
            state["by_action"][anchor_id],
            left_label="structural_win",
            right_label="v2_win",
        )
        pairs.append(
            {
                "schema": STAGE1_PAIR_SCHEMA,
                **_metadata(state),
                "pair_id": f"{state['state_id']}::{action_id}::{anchor_id}",
                "structural_action_id": action_id,
                "v2_action_id": anchor_id,
                "structural_role_aliases": sorted(aliases),
                **comparison,
            }
        )
    counts = collections.Counter(str(row["label"]) for row in pairs)
    state_label = (
        "structural_win"
        if counts["structural_win"] > 0
        else "v2_win"
        if counts["v2_win"] == len(pairs)
        else "ambiguous"
    )
    state_row = {
        "schema": STAGE1_STATE_SCHEMA,
        **_metadata(state),
        "label": state_label,
        "pair_count": len(pairs),
        "pair_label_counts": {
            label: int(counts[label]) for label in STAGE1_LABELS
        },
        "any_structural_win": counts["structural_win"] > 0,
        "both_structural_win": (
            len(pairs) == 2 and counts["structural_win"] == 2
        ),
        "runtime_or_pp_seconds_used_in_label": False,
    }
    return pairs, state_row


def _stage2_row(
    state: dict[str, Any], stage1_state: dict[str, Any]
) -> dict[str, Any] | None:
    role_map = state["role_map"]
    component_id = role_map["component16"]
    hotspot_id = role_map["hotspot16"]
    # Stage 2 is a C-vs-H problem whenever, and only whenever, their exact
    # canonical actions differ.  Stage-1 outcomes remain diagnostics and do
    # not select the Stage-2 training population.
    if component_id == hotspot_id:
        return None
    comparison = classify_three_way(
        state["by_action"][component_id],
        state["by_action"][hotspot_id],
        left_label="component_win",
        right_label="hotspot_win",
    )
    return {
        "schema": STAGE2_STATE_SCHEMA,
        **_metadata(state),
        "component_action_id": component_id,
        "hotspot_action_id": hotspot_id,
        "eligibility_condition": "component_action_id_ne_hotspot_action_id",
        "stage1_state_label": stage1_state["label"],
        "any_stage1_structural_win": bool(stage1_state["any_structural_win"]),
        "both_stage1_structural_win": bool(stage1_state["both_structural_win"]),
        **comparison,
    }


def _support(rows: Iterable[dict[str, Any]], labels: tuple[str, ...]) -> dict[str, Any]:
    materialized = list(rows)
    counts = collections.Counter(str(row["label"]) for row in materialized)
    unknown = set(counts) - set(labels)
    if unknown:
        raise ValueError(f"hierarchical C/H support has unknown labels: {sorted(unknown)}")
    by_label = {}
    for label in labels:
        selected = [row for row in materialized if str(row["label"]) == label]
        by_label[label] = {
            "row_count": len(selected),
            "state_count": len(
                {str(row["state_occurrence_id"]) for row in selected}
            ),
            "map_count": len({str(row["map_id"]) for row in selected}),
            "family_count": len({str(row["map_family"]) for row in selected}),
        }
    return {
        "row_count": len(materialized),
        "state_count": len(
            {str(row["state_occurrence_id"]) for row in materialized}
        ),
        "map_count": len({str(row["map_id"]) for row in materialized}),
        "family_count": len({str(row["map_family"]) for row in materialized}),
        "label_counts": {label: int(counts[label]) for label in labels},
        "by_label": by_label,
    }


def build_hierarchical_ch_labels(
    config: dict[str, Any],
    *,
    project_root: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    validate_config(config)
    root = Path(project_root).resolve()
    manifest_path = _registered_manifest(config, root)
    source_state_schema = str(
        dict(config["source"]).get("state_schema", SOURCE_STATE_SCHEMA)
    )
    manifest = _read_jsonl(manifest_path)
    if not manifest:
        raise ValueError("hierarchical C/H source manifest is empty")
    state_ids = [str(row.get("state_occurrence_id", "")) for row in manifest]
    if not all(state_ids) or len(state_ids) != len(set(state_ids)):
        raise ValueError("hierarchical C/H source manifest has duplicate states")

    stage1_pairs: list[dict[str, Any]] = []
    stage1_states: list[dict[str, Any]] = []
    stage2_states: list[dict[str, Any]] = []
    unique_action_count = 0
    trial_count = 0
    source_state_hashes = []
    partition_counts: collections.Counter[str] = collections.Counter()
    excluded_all_equal_state_ids: list[str] = []
    for manifest_row in sorted(manifest, key=lambda row: str(row["state_occurrence_id"])):
        state = _validated_state(
            manifest_path.parent,
            dict(manifest_row),
            source_state_schema=source_state_schema,
        )
        source_state_hashes.append(
            {
                "state_occurrence_id": state["state_id"],
                "file": str(manifest_row["state_file"]),
                "sha256": str(manifest_row["state_sha256"]),
            }
        )
        partition = str(state["hierarchical_stratum"])
        partition_counts[partition] += 1
        if partition == SINGLE_UNIQUE:
            excluded_all_equal_state_ids.append(state["state_id"])
            continue
        unique_action_count += len(state["actions"])
        trial_count += len(state["payload"]["trials"])
        pairs, stage1_state = _stage1_rows(state)
        stage2_state = _stage2_row(state, stage1_state)
        stage1_pairs.extend(pairs)
        stage1_states.append(stage1_state)
        if stage2_state is not None:
            stage2_states.append(stage2_state)

    expected = dict(config["expected"])
    observed = {
        "state_count": len(stage1_states),
        "unique_action_count": unique_action_count,
        "trial_count": trial_count,
        "stage1_pair_count": len(stage1_pairs),
        "stage2_state_count": len(stage2_states),
    }
    if observed != expected:
        raise ValueError(
            f"hierarchical C/H observed product counts changed: "
            f"expected={expected}, observed={observed}"
        )

    stage1_pairs.sort(key=lambda row: str(row["pair_id"]))
    stage1_states.sort(key=lambda row: str(row["state_occurrence_id"]))
    stage2_states.sort(key=lambda row: str(row["state_occurrence_id"]))
    consensus_pairs = [
        row
        for row in stage1_pairs
        if row["hierarchical_stratum"] == "consensus_structural"
    ]
    any_conditioned = [
        row for row in stage2_states if bool(row["any_stage1_structural_win"])
    ]
    both_conditioned = [
        row for row in stage2_states if bool(row["both_stage1_structural_win"])
    ]

    output_root = Path(output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stage1_pair_path = output_root / "stage1_pair_labels.jsonl"
    stage1_state_path = output_root / "stage1_state_labels.jsonl"
    stage2_state_path = output_root / "stage2_state_labels.jsonl"
    _write_jsonl(stage1_pair_path, stage1_pairs)
    _write_jsonl(stage1_state_path, stage1_states)
    _write_jsonl(stage2_state_path, stage2_states)

    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": str(config["experiment_id"]),
        "scientific_status": "sequential_development_label_audit_only",
        "complete": True,
        "source": {
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "state_schema": source_state_schema,
            "state_payloads": source_state_hashes,
            "source_state_count": len(manifest),
        },
        "observed": observed,
        "canonical_partition_support": {
            partition: int(partition_counts[partition])
            for partition in CANONICAL_PARTITIONS
        },
        "partition_exclusion": {
            "condition": "v2_action_id_eq_component_action_id_eq_hotspot_action_id",
            "canonical_partition": SINGLE_UNIQUE,
            "state_count": len(excluded_all_equal_state_ids),
            "state_occurrence_ids": sorted(excluded_all_equal_state_ids),
            "excluded_from_stage1_and_stage2": True,
        },
        "stage1_pair_support": _support(stage1_pairs, STAGE1_LABELS),
        "stage1_consensus_pair_support": _support(consensus_pairs, STAGE1_LABELS),
        "stage1_state_support": _support(stage1_states, STAGE1_LABELS),
        "stage2_all_support": _support(stage2_states, STAGE2_LABELS),
        "stage2_any_conditioned_support": _support(
            any_conditioned, STAGE2_LABELS
        ),
        "stage2_both_conditioned_support": _support(
            both_conditioned, STAGE2_LABELS
        ),
        "label_contract": {
            "paired_trial_count": len(TRIAL_INDICES),
            "direction_rule": "frozen_h1_rule_applied_in_both_directions",
            "ambiguous_rule": "neither_direction_is_a_robust_win",
            "runtime_or_pp_seconds_used_in_label": False,
            "trial_is_training_sample": False,
            "state_or_pair_is_label_unit": True,
            "stage1_population": (
                "one_pair_per_unique_structural_action_with_action_id_ne_v2_action_id"
            ),
            "stage2_population": "component_action_id_ne_hotspot_action_id",
            "stage1_positive_used_to_select_stage2": False,
            "stage1_positive_is_diagnostic_only": True,
            "all_equal_partition_excluded": True,
        },
        "artifacts": {
            "stage1_pair_labels": {
                "file": stage1_pair_path.name,
                "sha256": sha256_file(stage1_pair_path),
            },
            "stage1_state_labels": {
                "file": stage1_state_path.name,
                "sha256": sha256_file(stage1_state_path),
            },
            "stage2_state_labels": {
                "file": stage2_state_path.name,
                "sha256": sha256_file(stage2_state_path),
            },
        },
        "training_authorized": False,
        "model_fit_executed": False,
        "model_exported": False,
        "runtime_or_ttf_claim_authorized": False,
        "map_disjoint_confirmation_required": True,
    }
    report_path = output_root / "label_audit_report.json"
    _write_json(report_path, report)
    return report


__all__ = [
    "ANCHOR_COMPONENT_SHARED",
    "ANCHOR_HOTSPOT_SHARED",
    "CANONICAL_PARTITIONS",
    "CONFIG_SCHEMA",
    "CONSENSUS_STRUCTURAL",
    "LABELED_PARTITIONS",
    "REPORT_SCHEMA",
    "SINGLE_UNIQUE",
    "STAGE1_PAIR_SCHEMA",
    "STAGE1_STATE_SCHEMA",
    "STAGE2_STATE_SCHEMA",
    "STAGE2_ELIGIBLE_PARTITIONS",
    "THREE_UNIQUE",
    "build_hierarchical_ch_labels",
    "classify_three_way",
    "validate_config",
]
