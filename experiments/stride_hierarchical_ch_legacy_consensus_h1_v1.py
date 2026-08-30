from __future__ import annotations

import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from experiments._common import registered_input, sha256_file
from experiments.repair_collection import (
    _fingerprint,
    _plain,
    _read_json,
    _read_jsonl,
    _run_jobs,
    _write_json,
    _write_jsonl,
    state_fingerprint,
)
from experiments.stride_collection import (
    _paired_action,
    _replay_job,
    _validate_native_repair,
)
from experiments.stride_repairability_collection import (
    _source_target_state,
    repairability_restore_seed,
)
from experiments.trace_replay import restore_repair_state
from lns2_selector.runtime.fingerprints import repair_structure_fingerprint
from lns2_selector.runtime.unique_action_hierarchy import (
    COMPONENT_ROLE,
    CONSENSUS_STRUCTURAL,
    HOTSPOT_ROLE,
    ROLE_ORDER,
    V2_ANCHOR_ROLE,
    CanonicalRoleActions,
    canonicalize_role_actions,
)


CONFIG_SCHEMA = "lns2.stride.hierarchical_ch_legacy_consensus_h1_config.v1"
PLAN_SCHEMA = "lns2.stride.hierarchical_ch_legacy_consensus_h1_plan.v1"
SELECTION_SCHEMA = "lns2.stride.hierarchical_ch_legacy_consensus_selection.v1"
SELECTION_REPORT_SCHEMA = (
    "lns2.stride.hierarchical_ch_legacy_consensus_selection_report.v1"
)
H1_TRIAL_SCHEMA = "lns2.stride.hierarchical_ch_legacy_consensus_h1_trial.v1"
H1_STATE_SCHEMA = "lns2.stride.hierarchical_ch_legacy_consensus_h1_state.v1"
H1_MANIFEST_ROW_SCHEMA = (
    "lns2.stride.hierarchical_ch_legacy_consensus_h1_manifest_row.v1"
)
H1_COLLECTION_SCHEMA = (
    "lns2.stride.hierarchical_ch_legacy_consensus_h1_collection.v1"
)
EXPERIMENT_ID = "stride_hierarchical_ch_legacy_consensus_h1_v1"
DEFAULT_OUTPUT_NAME = "stride-hierarchical-ch-legacy-consensus-h1-v1"
RESEARCH_SPLIT = "legacy_sequential_development_train_only"
TRIAL_INDICES = tuple(range(16))

EXPECTED_PREFLIGHT = {
    "path": "build/stride-fresh-matched-v2-active-supply-overlay-v1/preflight_rows.jsonl",
    "sha256": "c588cfbda699b782667d1bd7e85742b97bc04910e156716923d39c59a367b1dc",
}
EXPECTED_PRIOR_SELECTION = {
    "path": "build/stride-fresh-matched-unique-action-hierarchical-overlay-v1/selected_states.jsonl",
    "sha256": "3948ff19e3c5cdbeacc36d133fece7d63245dbd552ebfc670957df3f4b8055c8",
}

FORBIDDEN_OUTCOME_KEYS = {
    "after",
    "after_conflicts",
    "conflicts_after",
    "final_success",
    "native_step_seconds",
    "outcome",
    "pp_seconds",
    "pp_time",
    "repair_outcome",
    "repair_status",
    "replan_success",
    "success",
    "target",
    "target_outcome",
    "target_result",
    "trials",
    "ttf",
}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    _write_json(partial, payload)
    os.replace(partial, path)


def _strict_selection_contract() -> dict[str, Any]:
    return {
        "source_preaction_state_count": 380,
        "source_consensus_structural_state_count": 92,
        "prior_selected_state_count": 96,
        "prior_selected_consensus_structural_state_count": 32,
        "target_remaining_consensus_state_count": 60,
        "expected_unique_action_count": 120,
        "expected_logical_trial_count": 1920,
        "required_partition": CONSENSUS_STRUCTURAL,
        "exclusion_identity": "origin_source_episode_decision_before_fingerprint",
        "rank_rule": "all_remaining_exact_identity_sha256_ascending",
        "deduplicate_by_exact_sorted_agent_set": True,
        "execute_each_unique_set_once": True,
        "preserve_all_role_aliases": True,
        "outcome_fields_read": False,
        "reserve_or_backfill_allowed": False,
    }


def _strict_h1_contract() -> dict[str, Any]:
    return {
        "trial_indices": list(TRIAL_INDICES),
        "first_fixed_half": list(range(8)),
        "second_fixed_half": list(range(8, 16)),
        "expected_selected_state_count": 60,
        "expected_unique_action_count": 120,
        "logical_trial_count": 1920,
        "workers": 16,
        "per_action_time_limit_seconds": 5.0,
        "per_state_process_fuse_seconds": 420.0,
        "same_state_trial_seed_across_unique_actions": True,
        "trial_level_atomic_checkpoint": True,
        "timed_pp_failure_requires_atomic_rollback": True,
        "runtime_or_pp_seconds_used_in_label": False,
    }


def _strict_claim_boundary() -> dict[str, Any]:
    return {
        "legacy_sequential_development_only": True,
        "training_pool_only": True,
        "development_or_final_evaluation_allowed": False,
        "map_disjoint_claim_allowed": False,
        "input_artifact_mutation_allowed": False,
        "outcome_based_state_selection_allowed": False,
        "training_authorized": False,
        "runtime_authorized": False,
        "ttf_or_speed_claim_allowed": False,
    }


def validate_config(
    config: Mapping[str, Any], *, project_root: Path | None = None
) -> None:
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("experiment_id") != EXPERIMENT_ID
        or config.get("scientific_status")
        != "legacy_sequential_development_training_only_extension"
        or config.get("research_split") != RESEARCH_SPLIT
    ):
        raise ValueError("legacy consensus H1 experiment identity changed")
    inputs = dict(config.get("inputs") or {})
    if inputs != {
        "active_supply_preflight_rows": EXPECTED_PREFLIGHT,
        "prior_selected_states": EXPECTED_PRIOR_SELECTION,
        "mutation_allowed": False,
    }:
        raise ValueError("legacy consensus H1 frozen inputs changed")
    if dict(config.get("selection") or {}) != _strict_selection_contract():
        raise ValueError("legacy consensus H1 selection contract changed")
    if dict(config.get("h1") or {}) != _strict_h1_contract():
        raise ValueError("legacy consensus H1 execution contract changed")
    if dict(config.get("claim_boundary") or {}) != _strict_claim_boundary():
        raise ValueError("legacy consensus H1 claim boundary changed")
    if project_root is not None:
        _registered_inputs(project_root.resolve(), config)


def _registered_inputs(
    root: Path, config: Mapping[str, Any]
) -> tuple[Path, Path]:
    inputs = dict(config["inputs"])
    preflight = registered_input(
        root,
        dict(inputs["active_supply_preflight_rows"]),
        label="legacy consensus active-supply preflight rows",
    )
    prior = registered_input(
        root,
        dict(inputs["prior_selected_states"]),
        label="legacy consensus prior selected states",
    )
    return preflight, prior


def load_config(path: str | Path) -> tuple[Path, Path, dict[str, Any]]:
    config_path = Path(path).resolve()
    root = config_path.parents[1]
    config = _read_json(config_path)
    validate_config(config, project_root=root)
    return config_path, root, config


def _output_root(
    root: Path, config: Mapping[str, Any], output: str | Path
) -> Path:
    target = Path(output).resolve()
    protected = {
        (root / str(spec["path"])).resolve().parent
        for spec in (
            config["inputs"]["active_supply_preflight_rows"],
            config["inputs"]["prior_selected_states"],
        )
    }
    if any(target == item or item in target.parents for item in protected):
        raise ValueError("legacy consensus output overlaps a frozen input product")
    return target


def _identity_tuple(row: Mapping[str, Any]) -> tuple[str, str, str, int, str]:
    try:
        return (
            str(row["origin"]),
            str(row["source_id"]),
            str(row["episode_id"]),
            int(row["decision_index"]),
            str(row["before_fingerprint"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid legacy consensus source-state identity") from error


def _identity_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    origin, source_id, episode_id, decision_index, before_fingerprint = (
        _identity_tuple(row)
    )
    return {
        "namespace": "stride-hierarchical-ch-legacy-consensus-state-v1",
        "origin": origin,
        "source_id": source_id,
        "episode_id": episode_id,
        "task_id": str(row["task_id"]),
        "solver_seed": int(row["solver_seed"]),
        "decision_index": decision_index,
        "before_fingerprint": before_fingerprint,
    }


def _selection_rank(row: Mapping[str, Any]) -> str:
    return _fingerprint(
        {
            **_identity_payload(row),
            "namespace": "stride-hierarchical-ch-legacy-consensus-rank-v1",
        }
    )


def _forbidden_outcome_paths(
    value: Any, path: tuple[str, ...] = ()
) -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, nested in value.items():
            key = str(raw_key)
            nested_path = (*path, key)
            if key in FORBIDDEN_OUTCOME_KEYS:
                found.append(".".join(nested_path))
            found.extend(_forbidden_outcome_paths(nested, nested_path))
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            found.extend(_forbidden_outcome_paths(nested, (*path, str(index))))
    return found


def _canonical(row: Mapping[str, Any]) -> CanonicalRoleActions:
    agent_count = row.get("agent_count")
    if type(agent_count) is not int or int(agent_count) <= 0:
        raise ValueError("invalid legacy consensus agent_count")
    return canonicalize_role_actions(
        dict(row.get("arms") or {}), agent_count=int(agent_count)
    )


def _preaction_row_valid(row: Mapping[str, Any]) -> bool:
    try:
        _identity_tuple(row)
        canonical = _canonical(row)
    except (KeyError, TypeError, ValueError):
        return False
    required_strings = (
        "task_id",
        "map_id",
        "map_family",
        "source_trace_file",
        "source_trace_path",
        "source_trace_sha256",
        "before_repair_fingerprint",
    )
    return (
        not _forbidden_outcome_paths(row)
        and all(isinstance(row.get(field), str) and row[field] for field in required_strings)
        and type(row.get("solver_seed")) is int
        and type(row.get("decision_index")) is int
        and 0 <= int(row["decision_index"]) < 12
        and type(row.get("before_conflicts")) is int
        and int(row["before_conflicts"]) > 0
        and row.get("target_outcome_fields_read") is False
        and row.get("candidate_repair_actions_executed") is False
        and row.get("repair_fingerprint_preserved") is True
        and canonical.partition == CONSENSUS_STRUCTURAL
        and canonical.unique_action_count == 2
        and canonical.action_index(COMPONENT_ROLE)
        == canonical.action_index(HOTSPOT_ROLE)
        and canonical.action_index(V2_ANCHOR_ROLE)
        != canonical.action_index(COMPONENT_ROLE)
        and len(canonical.agents_for_role(COMPONENT_ROLE)) == 16
    )


def _source_equivalent(
    source: Mapping[str, Any], prior: Mapping[str, Any]
) -> bool:
    try:
        source_canonical = _canonical(source)
        prior_canonical = _canonical(prior)
    except (TypeError, ValueError):
        return False
    return (
        _identity_tuple(source) == _identity_tuple(prior)
        and source_canonical == prior_canonical
        and source.get("task_id") == prior.get("task_id")
        and source.get("solver_seed") == prior.get("solver_seed")
        and source.get("source_trace_file") == prior.get("source_trace_file")
        and source.get("source_trace_path") == prior.get("source_trace_path")
        and source.get("source_trace_sha256") == prior.get("source_trace_sha256")
        and source.get("before_repair_fingerprint")
        == prior.get("before_repair_fingerprint")
        and source.get("before_conflicts") == prior.get("before_conflicts")
    )


def _action_product(
    row: Mapping[str, Any], state_occurrence_id: str
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    canonical = _canonical(row)
    if canonical.partition != CONSENSUS_STRUCTURAL:
        raise ValueError("legacy consensus action product is not C=H!=V")
    actions: list[dict[str, Any]] = []
    action_ids: list[str] = []
    for action in canonical.actions:
        action_id = "legacy-consensus-action-" + _fingerprint(
            {
                "namespace": "stride-hierarchical-ch-legacy-consensus-action-v1",
                "state_occurrence_id": state_occurrence_id,
                "agents": list(action.agents),
            }
        )[:24]
        action_ids.append(action_id)
        aliases = list(action.role_aliases)
        actions.append(
            {
                "action_id": action_id,
                "agents": list(action.agents),
                "actual_size": action.actual_size,
                "role_aliases": aliases,
                "candidate_ids_by_role": dict(action.candidate_ids_by_role),
                "contains_v2_anchor_role": V2_ANCHOR_ROLE in aliases,
                "structural_role_aliases": [
                    role for role in aliases if role in {COMPONENT_ROLE, HOTSPOT_ROLE}
                ],
            }
        )
    role_to_action = {
        role: action_ids[canonical.action_index(role)] for role in ROLE_ORDER
    }
    return actions, role_to_action


def select_legacy_consensus_states(
    preflight_rows: Iterable[dict[str, Any]],
    prior_selected_states: Iterable[dict[str, Any]],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if dict(config.get("selection") or {}) != _strict_selection_contract():
        raise ValueError("legacy consensus selection contract changed")
    rows = [dict(row) for row in preflight_rows]
    prior = [dict(row) for row in prior_selected_states]
    failures: list[str] = []

    source_keys = [_identity_tuple(row) for row in rows]
    prior_keys = [_identity_tuple(row) for row in prior]
    if len(source_keys) != len(set(source_keys)):
        failures.append("duplicate_source_state_identity")
    if len(prior_keys) != len(set(prior_keys)):
        failures.append("duplicate_prior_selected_state_identity")
    source_by_key = {_identity_tuple(row): row for row in rows}
    if any(
        key not in source_by_key
        or not _source_equivalent(source_by_key[key], prior_row)
        for key, prior_row in zip(prior_keys, prior)
    ):
        failures.append("prior_selection_not_exact_subset_of_source")
    if any(_forbidden_outcome_paths(row) for row in rows):
        failures.append("forbidden_source_outcome_field")
    if any(row.get("target_outcome_fields_read") is not False for row in rows):
        failures.append("source_target_outcome_read")

    consensus = [row for row in rows if _preaction_row_valid(row)]
    prior_consensus_count = 0
    for row in prior:
        try:
            prior_consensus_count += int(
                _canonical(row).partition == CONSENSUS_STRUCTURAL
            )
        except (TypeError, ValueError):
            failures.append("invalid_prior_canonical_action_product")
    prior_key_set = set(prior_keys)
    remaining = [
        row for row in consensus if _identity_tuple(row) not in prior_key_set
    ]
    remaining.sort(key=lambda row: (_selection_rank(row), _identity_tuple(row)))

    if len(rows) != 380:
        failures.append("exact_380_source_preaction_states")
    if len(prior) != 96:
        failures.append("exact_96_prior_selected_states")
    if len(consensus) != 92:
        failures.append("exact_92_source_consensus_states")
    if prior_consensus_count != 32:
        failures.append("exact_32_prior_consensus_states")
    if len(remaining) != 60:
        failures.append("exact_60_remaining_consensus_states")

    selected: list[dict[str, Any]] = []
    if not failures:
        for row in remaining:
            state_id = "legacy-consensus-state-" + _fingerprint(
                _identity_payload(row)
            )[:24]
            actions, role_to_action = _action_product(row, state_id)
            selected.append(
                {
                    **row,
                    "schema": SELECTION_SCHEMA,
                    "state_occurrence_id": state_id,
                    "selection_rank_sha256": _selection_rank(row),
                    "canonical_partition": CONSENSUS_STRUCTURAL,
                    "unique_actions": actions,
                    "role_to_action_id": role_to_action,
                    "unique_action_count": len(actions),
                    "source_research_split": row.get("research_split"),
                    "research_split": RESEARCH_SPLIT,
                    "legacy_sequential_development": True,
                    "training_only": True,
                    "evaluation_eligible": False,
                    "legacy_extension_eligible": True,
                    "training_authorized": False,
                    "selection_outcome_fields_read": False,
                    "target_outcome_fields_read": False,
                }
            )
        selected.sort(key=lambda row: str(row["state_occurrence_id"]))

    action_count = sum(int(row["unique_action_count"]) for row in selected)
    if not failures and action_count != 120:
        failures.append("exact_120_unique_actions")
        selected = []
        action_count = 0
    selected_keys = {_identity_tuple(row) for row in selected}
    if selected_keys & prior_key_set:
        failures.append("prior_selection_exclusion_failed")
        selected = []
        action_count = 0
    aliases_ok = all(
        len(row["unique_actions"]) == 2
        and row["role_to_action_id"][COMPONENT_ROLE]
        == row["role_to_action_id"][HOTSPOT_ROLE]
        and row["role_to_action_id"][V2_ANCHOR_ROLE]
        != row["role_to_action_id"][COMPONENT_ROLE]
        for row in selected
    )
    if not failures and not aliases_ok:
        failures.append("exact_dedup_or_role_alias_integrity")
        selected = []
        action_count = 0

    report = {
        "schema": SELECTION_REPORT_SCHEMA,
        "status": "ok" if not failures else "STATE_SUPPLY_FAIL",
        "failure_reasons": sorted(set(failures)),
        "source_preaction_state_count": len(rows),
        "source_consensus_structural_state_count": len(consensus),
        "prior_selected_state_count": len(prior),
        "prior_selected_consensus_structural_state_count": prior_consensus_count,
        "remaining_consensus_state_count": len(remaining),
        "selected_state_count": len(selected),
        "selected_unique_action_count": action_count,
        "logical_h1_trial_count": action_count * len(TRIAL_INDICES),
        "selected_map_counts": dict(
            sorted(Counter(str(row["map_id"]) for row in selected).items())
        ),
        "selected_family_counts": dict(
            sorted(Counter(str(row["map_family"]) for row in selected).items())
        ),
        "prior_selected_overlap_count": len(selected_keys & prior_key_set),
        "canonical_partition": CONSENSUS_STRUCTURAL,
        "unique_action_count_per_state": 2,
        "component_hotspot_share_action": aliases_ok,
        "outcome_fields_read": False,
        "reserve_or_backfill_used": False,
        "research_split": RESEARCH_SPLIT,
        "training_only": True,
        "evaluation_eligible": False,
        "training_authorized": False,
    }
    return selected, report


def _selection_preview(
    root: Path, config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    preflight_path, prior_path = _registered_inputs(root, config)
    return select_legacy_consensus_states(
        _read_jsonl(preflight_path), _read_jsonl(prior_path), config
    )


def build_plan(config_path: str | Path) -> dict[str, Any]:
    path, root, config = load_config(config_path)
    selected, selection = _selection_preview(root, config)
    return {
        "schema": PLAN_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "config_path": str(path),
        "config_sha256": sha256_file(path),
        "selection_status": selection["status"],
        "selected_state_count": len(selected),
        "selected_unique_action_count": selection["selected_unique_action_count"],
        "logical_h1_trial_count": selection["logical_h1_trial_count"],
        "trial_indices": list(TRIAL_INDICES),
        "fixed_seed_halves": [list(range(8)), list(range(8, 16))],
        "workers": 16,
        "research_split": RESEARCH_SPLIT,
        "collection_executed_by_plan": False,
        "training_or_runtime_or_ttf_executed": False,
        "training_authorized": False,
    }


def run_selection(
    config_path: str | Path,
    output: str | Path,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    path, root, config = load_config(config_path)
    output_root = _output_root(root, config, output)
    selected, report = _selection_preview(root, config)
    report.update(
        {
            "experiment_id": EXPERIMENT_ID,
            "config_sha256": sha256_file(path),
            "input_preflight_rows_sha256": EXPECTED_PREFLIGHT["sha256"],
            "input_prior_selected_states_sha256": EXPECTED_PRIOR_SELECTION[
                "sha256"
            ],
            "dry_run": bool(dry_run),
            "input_artifacts_mutated": False,
        }
    )
    if dry_run:
        return report
    if report["status"] != "ok" or len(selected) != 60:
        _write_jsonl(output_root / "selected_states.jsonl", [])
        _atomic_json(output_root / "selection_report.json", report)
        return report
    _write_jsonl(output_root / "selected_states.jsonl", selected)
    report["selected_states_sha256"] = sha256_file(
        output_root / "selected_states.jsonl"
    )
    _atomic_json(output_root / "selection_report.json", report)
    return report


def matched_legacy_consensus_pp_seed(
    state_occurrence_id: str, trial_index: int
) -> int:
    if not state_occurrence_id or trial_index not in TRIAL_INDICES:
        raise ValueError("invalid legacy consensus paired PP seed identity")
    digest = _fingerprint(
        {
            "namespace": "stride-hierarchical-ch-legacy-consensus-paired-pp-v1",
            "state_occurrence_id": state_occurrence_id,
            "trial_index": int(trial_index),
            "step_index": 0,
        }
    )
    return int(digest[:16], 16) % (2**31)


def _h1_run_identity(
    *, config_sha256: str, selection_identity: str, h1: Mapping[str, Any]
) -> str:
    return _fingerprint(
        {
            "namespace": "stride-hierarchical-ch-legacy-consensus-h1-run-v1",
            "experiment_id": EXPERIMENT_ID,
            "config_sha256": config_sha256,
            "selection_identity": selection_identity,
            "h1": dict(h1),
            "research_split": RESEARCH_SPLIT,
        }
    )


def _expected_state_actions(
    state_row: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    return _action_product(state_row, str(state_row["state_occurrence_id"]))


def _trial_valid(
    row: Mapping[str, Any],
    state_id: str,
    actions: Mapping[str, Mapping[str, Any]],
    expected_restore_seed: int,
) -> bool:
    action_id = str(row.get("action_id"))
    trial_index = int(row.get("trial_index", -1))
    action = actions.get(action_id)
    if action is None or trial_index not in TRIAL_INDICES:
        return False
    expected_seed = matched_legacy_consensus_pp_seed(state_id, trial_index)
    repair_order_count = int(row.get("repair_order_count", -1))
    expected_applied = expected_seed if repair_order_count > 0 else -1
    if (
        row.get("schema") != H1_TRIAL_SCHEMA
        or row.get("state_occurrence_id") != state_id
        or row.get("trial_identity")
        != _fingerprint(
            {
                "namespace": "stride-hierarchical-ch-legacy-consensus-trial-v1",
                "state_occurrence_id": state_id,
                "action_id": action_id,
                "trial_index": trial_index,
            }
        )
        or row.get("role_aliases") != action["role_aliases"]
        or row.get("candidate_ids_by_role") != action["candidate_ids_by_role"]
        or row.get("agents") != action["agents"]
        or int(row.get("actual_size", -1)) != int(action["actual_size"])
        or int(row.get("pp_seed", -1)) != expected_seed
        or int(row.get("requested_random_seed", -1)) != expected_seed
        or int(row.get("requested_pp_random_seed", -1)) != expected_seed
        or int(row.get("applied_pp_random_seed", -2)) != expected_applied
        or int(row.get("restore_seed", -1)) != expected_restore_seed
        or int(row.get("step_index", -1)) != 0
        or row.get("action_valid") is not True
        or row.get("generated") is not True
        or row.get("fresh_independent_environment_restore") is not True
        or row.get("integrity_ok") is not True
        or int(row.get("before_conflicts", -1)) <= 0
        or int(row.get("before_sum_of_costs", -1)) < 0
        or int(row.get("conflicts_after", -1)) < 0
        or int(row.get("after_sum_of_costs", -1)) < 0
        or not isinstance(row.get("before_repair_fingerprint"), str)
        or not row.get("before_repair_fingerprint")
        or not isinstance(row.get("after_repair_fingerprint"), str)
        or not row.get("after_repair_fingerprint")
        or type(row.get("replan_success")) is not bool
        or row.get("runtime_used_in_label") is not False
        or row.get("research_split") != RESEARCH_SPLIT
        or row.get("training_only") is not True
        or row.get("evaluation_eligible") is not False
        or not math.isclose(
            float(row.get("requested_pp_time_limit_seconds", math.nan)),
            5.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        return False
    success = bool(row["replan_success"])
    expected_reduction = (
        int(row["before_conflicts"]) - int(row["conflicts_after"])
    ) / max(1, int(row["before_conflicts"]))
    if not math.isclose(
        float(row.get("normalized_conflict_reduction", math.nan)),
        expected_reduction,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        return False
    if success and int(row["conflicts_after"]) > int(row["before_conflicts"]):
        return False
    if success and row.get("atomic_rollback") is not False:
        return False
    if not success and not (
        row.get("rollback") is True
        and row.get("atomic_rollback") is True
        and row.get("before_repair_fingerprint")
        == row.get("after_repair_fingerprint")
        and int(row["before_conflicts"]) == int(row["conflicts_after"])
        and int(row["before_sum_of_costs"]) == int(row["after_sum_of_costs"])
    ):
        return False
    is_time_limit = str(row.get("failure_reason")) == "time_limit"
    if bool(row.get("time_limit")) != (is_time_limit and not success):
        return False
    if bool(row.get("no_progress")) != (
        bool(row.get("time_limit"))
        or int(row["conflicts_after"]) >= int(row["before_conflicts"])
    ):
        return False
    return not bool(row.get("time_limit")) or (
        row.get("rollback") is True and row.get("no_progress") is True
    )


def _state_payload_valid(
    payload: Mapping[str, Any],
    *,
    identity: str,
    state_row: dict[str, Any],
    complete: bool,
    source_snapshot: Mapping[str, Any] | None = None,
) -> bool:
    state_id = str(state_row["state_occurrence_id"])
    expected_actions, expected_roles = _expected_state_actions(state_row)
    if (
        payload.get("schema") != H1_STATE_SCHEMA
        or payload.get("identity") != identity
        or payload.get("state_occurrence_id") != state_id
        or payload.get("state_row") != state_row
        or payload.get("complete") is not complete
        or state_row.get("unique_actions") != expected_actions
        or state_row.get("role_to_action_id") != expected_roles
        or payload.get("research_split") != RESEARCH_SPLIT
        or payload.get("training_only") is not True
        or payload.get("evaluation_eligible") is not False
        or payload.get("training_authorized") is not False
        or int(payload.get("before_conflicts", -1)) <= 0
        or int(payload.get("before_conflicts", -1))
        != int(state_row.get("before_conflicts", -2))
        or int(payload.get("before_sum_of_costs", -1)) < 0
        or not isinstance(payload.get("before_repair_fingerprint"), str)
        or not payload.get("before_repair_fingerprint")
        or payload.get("before_repair_fingerprint")
        != state_row.get("before_repair_fingerprint")
        or payload.get("source_trace_file") != state_row.get("source_trace_file")
        or payload.get("source_trace_path") != state_row.get("source_trace_path")
        or payload.get("source_trace_sha256") != state_row.get("source_trace_sha256")
        or payload.get("runtime_used_in_label") is not False
    ):
        return False
    actions = {str(action["action_id"]): action for action in expected_actions}
    expected_restore_seed = repairability_restore_seed(
        str(state_row["before_repair_fingerprint"])
    )
    if int(payload.get("restore_seed", -1)) != expected_restore_seed:
        return False
    if source_snapshot is not None and any(
        payload.get(field) != source_snapshot.get(field)
        for field in (
            "source_trace_file",
            "source_trace_path",
            "source_trace_sha256",
            "before_repair_fingerprint",
            "before_conflicts",
            "before_sum_of_costs",
            "restore_seed",
        )
    ):
        return False
    trials = payload.get("trials")
    if not isinstance(trials, list):
        return False
    expected_product = {
        (action_id, trial_index)
        for action_id in actions
        for trial_index in TRIAL_INDICES
    }
    product = Counter(
        (str(row.get("action_id")), int(row.get("trial_index", -1)))
        for row in trials
    )
    product_ok = (
        set(product) == expected_product
        if complete
        else set(product) <= expected_product
    )
    return (
        product_ok
        and all(count == 1 for count in product.values())
        and all(
            _trial_valid(dict(row), state_id, actions, expected_restore_seed)
            for row in trials
        )
        and all(
            row.get("before_repair_fingerprint")
            == payload.get("before_repair_fingerprint")
            and int(row.get("before_conflicts", -1))
            == int(payload.get("before_conflicts", -1))
            and int(row.get("before_sum_of_costs", -1))
            == int(payload.get("before_sum_of_costs", -1))
            for row in trials
        )
    )


def _validated_source_snapshot(state_row: dict[str, Any]) -> dict[str, Any]:
    source_state, source_manifest, source_trace_path = _source_target_state(state_row)
    observed_trace_sha256 = sha256_file(source_trace_path)
    before_repair = repair_structure_fingerprint(source_state)
    snapshot = {
        "source_state": source_state,
        "source_trace_file": str(source_manifest["trace_file"]),
        "source_trace_path": str(source_trace_path),
        "source_trace_sha256": observed_trace_sha256,
        "before_repair_fingerprint": before_repair,
        "before_conflicts": int(source_state["num_of_colliding_pairs"]),
        "before_sum_of_costs": int(source_state["sum_of_costs"]),
        "restore_seed": repairability_restore_seed(before_repair),
    }
    if (
        observed_trace_sha256 != str(source_manifest.get("trace_sha256"))
        or snapshot["source_trace_file"] != state_row.get("source_trace_file")
        or snapshot["source_trace_path"] != state_row.get("source_trace_path")
        or observed_trace_sha256 != state_row.get("source_trace_sha256")
        or state_fingerprint(source_state) != state_row.get("before_fingerprint")
        or before_repair != state_row.get("before_repair_fingerprint")
        or snapshot["before_conflicts"] != int(state_row.get("before_conflicts", -1))
        or snapshot["before_conflicts"] <= 0
        or snapshot["before_sum_of_costs"] < 0
    ):
        raise RuntimeError("legacy consensus H1 source trace or state changed")
    return snapshot


def _h1_state_worker(job: dict[str, Any]) -> dict[str, Any]:
    state_row = dict(job["state_row"])
    state_id = str(state_row["state_occurrence_id"])
    output_path = Path(str(job["output_path"]))
    partial_path = output_path.with_name(output_path.name + ".partial")
    identity = str(job["identity"])
    if output_path.exists() and not bool(job["resume"]):
        raise ValueError(f"legacy consensus H1 output exists; pass --resume: {output_path}")
    if partial_path.exists() and not bool(job["resume"]):
        raise ValueError(
            f"legacy consensus H1 partial exists; pass --resume: {partial_path}"
        )

    expected_actions, expected_roles = _expected_state_actions(state_row)
    if (
        state_row.get("schema") != SELECTION_SCHEMA
        or state_row.get("canonical_partition") != CONSENSUS_STRUCTURAL
        or state_row.get("unique_actions") != expected_actions
        or state_row.get("role_to_action_id") != expected_roles
        or len(expected_actions) != 2
        or expected_roles[COMPONENT_ROLE] != expected_roles[HOTSPOT_ROLE]
        or expected_roles[V2_ANCHOR_ROLE] == expected_roles[COMPONENT_ROLE]
        or state_row.get("research_split") != RESEARCH_SPLIT
        or state_row.get("training_only") is not True
        or state_row.get("evaluation_eligible") is not False
        or state_row.get("training_authorized") is not False
    ):
        raise RuntimeError("legacy consensus frozen state product changed")

    source_snapshot = _validated_source_snapshot(state_row)
    source_state = dict(source_snapshot["source_state"])
    before_repair = str(source_snapshot["before_repair_fingerprint"])
    before_conflicts = int(source_snapshot["before_conflicts"])
    before_soc = int(source_snapshot["before_sum_of_costs"])
    restore_seed = int(source_snapshot["restore_seed"])
    if bool(job["resume"]) and output_path.is_file():
        payload = _read_json(output_path)
        if _state_payload_valid(
            payload,
            identity=identity,
            state_row=state_row,
            complete=True,
            source_snapshot=source_snapshot,
        ):
            return {
                "status": "resumed",
                "job_id": job["job_id"],
                "state_occurrence_id": state_id,
                "output_path": str(output_path),
                "trial_count": len(payload["trials"]),
            }
        raise ValueError(f"invalid legacy consensus H1 artifact: {output_path}")

    replay = _replay_job(state_row)
    time_limit = float(job["per_action_time_limit_seconds"])
    partial_payload = {
        "schema": H1_STATE_SCHEMA,
        "identity": identity,
        "complete": False,
        "state_occurrence_id": state_id,
        "state_row": state_row,
        "source_trace_file": str(source_snapshot["source_trace_file"]),
        "source_trace_path": str(source_snapshot["source_trace_path"]),
        "source_trace_sha256": str(source_snapshot["source_trace_sha256"]),
        "before_repair_fingerprint": before_repair,
        "before_conflicts": before_conflicts,
        "before_sum_of_costs": before_soc,
        "restore_seed": restore_seed,
        "trials": [],
        "runtime_used_in_label": False,
        "research_split": RESEARCH_SPLIT,
        "training_only": True,
        "evaluation_eligible": False,
        "training_authorized": False,
    }
    if bool(job["resume"]) and partial_path.is_file():
        partial_payload = _read_json(partial_path)
        if not _state_payload_valid(
            partial_payload,
            identity=identity,
            state_row=state_row,
            complete=False,
            source_snapshot=source_snapshot,
        ):
            raise ValueError(f"invalid legacy consensus H1 partial: {partial_path}")
    trials = list(partial_payload["trials"])
    completed = {
        (str(row["action_id"]), int(row["trial_index"])) for row in trials
    }
    for action in expected_actions:
        action_id = str(action["action_id"])
        agents = list(map(int, action["agents"]))
        if not agents or len(agents) != len(set(agents)):
            raise RuntimeError("legacy consensus H1 action is illegal")
        if action["structural_role_aliases"] and len(agents) != 16:
            raise RuntimeError("legacy consensus structural action is not size16")
        for trial_index in TRIAL_INDICES:
            if (action_id, trial_index) in completed:
                continue
            environment, branch = restore_repair_state(
                replay, source_state, seed=restore_seed
            )
            if (
                repair_structure_fingerprint(branch) != before_repair
                or int(branch["num_of_colliding_pairs"]) != before_conflicts
                or int(branch["sum_of_costs"]) != before_soc
            ):
                raise RuntimeError("legacy consensus H1 branch restore changed")
            pp_seed = matched_legacy_consensus_pp_seed(state_id, trial_index)
            result = _plain(
                environment.step_with_time_limit(
                    _paired_action(agents, pp_seed), time_limit
                )
            )
            after, metrics = _validate_native_repair(
                result, expected_agents=agents, expected_seed=pp_seed
            )
            after_repair = repair_structure_fingerprint(after)
            conflicts_after = int(after["num_of_colliding_pairs"])
            after_soc = int(after["sum_of_costs"])
            success = bool(metrics["replan_success"])
            rolled_back = bool(metrics.get("pp_rolled_back", False))
            failure_reason = str(metrics.get("pp_failure_reason", ""))
            atomic_rollback = (
                rolled_back
                and after_repair == before_repair
                and conflicts_after == before_conflicts
                and after_soc == before_soc
            )
            if not success and not atomic_rollback:
                raise RuntimeError("legacy consensus timed PP failure was not atomic")
            if success and conflicts_after > before_conflicts:
                raise RuntimeError("legacy consensus successful PP increased conflicts")
            repair_order = metrics.get("repair_order")
            if not isinstance(repair_order, list):
                raise RuntimeError("legacy consensus timed PP omitted repair order")
            if (
                int(metrics.get("requested_random_seed", -1)) != pp_seed
                or metrics.get("action_valid") is not True
                or metrics.get("generated") is not True
            ):
                raise RuntimeError("legacy consensus timed PP identity changed")
            requested_seed = int(metrics.get("requested_pp_random_seed", -1))
            applied_seed = int(metrics.get("applied_pp_random_seed", -2))
            time_limit_rollback = (
                failure_reason == "time_limit" and not success and atomic_rollback
            )
            trials.append(
                {
                    "schema": H1_TRIAL_SCHEMA,
                    "trial_identity": _fingerprint(
                        {
                            "namespace": "stride-hierarchical-ch-legacy-consensus-trial-v1",
                            "state_occurrence_id": state_id,
                            "action_id": action_id,
                            "trial_index": trial_index,
                        }
                    ),
                    "state_occurrence_id": state_id,
                    "action_id": action_id,
                    "role_aliases": list(action["role_aliases"]),
                    "candidate_ids_by_role": dict(action["candidate_ids_by_role"]),
                    "agents": agents,
                    "actual_size": len(agents),
                    "trial_index": trial_index,
                    "step_index": 0,
                    "restore_seed": restore_seed,
                    "fresh_independent_environment_restore": True,
                    "pp_seed": pp_seed,
                    "requested_random_seed": int(metrics["requested_random_seed"]),
                    "requested_pp_random_seed": requested_seed,
                    "applied_pp_random_seed": applied_seed,
                    "repair_order_count": len(repair_order),
                    "before_conflicts": before_conflicts,
                    "conflicts_after": conflicts_after,
                    "before_sum_of_costs": before_soc,
                    "after_sum_of_costs": after_soc,
                    "normalized_conflict_reduction": (
                        before_conflicts - conflicts_after
                    )
                    / max(1, before_conflicts),
                    "replan_success": success,
                    "rollback": rolled_back,
                    "atomic_rollback": atomic_rollback if not success else False,
                    "failure_reason": failure_reason,
                    "time_limit": time_limit_rollback,
                    "no_progress": (
                        time_limit_rollback or conflicts_after >= before_conflicts
                    ),
                    "before_repair_fingerprint": before_repair,
                    "after_repair_fingerprint": after_repair,
                    "native_step_seconds": float(
                        metrics.get("native_step_seconds", 0.0)
                    ),
                    "pp_seconds": float(
                        metrics.get(
                            "pp_replan_seconds",
                            metrics.get("native_replan_seconds", 0.0),
                        )
                    ),
                    "requested_pp_time_limit_seconds": time_limit,
                    "action_valid": True,
                    "generated": True,
                    "runtime_used_in_label": False,
                    "integrity_ok": True,
                    "research_split": RESEARCH_SPLIT,
                    "training_only": True,
                    "evaluation_eligible": False,
                }
            )
            partial_payload["trials"] = trials
            _atomic_json(partial_path, partial_payload)
    payload = {**partial_payload, "complete": True, "trials": trials}
    if not _state_payload_valid(
        payload,
        identity=identity,
        state_row=state_row,
        complete=True,
        source_snapshot=source_snapshot,
    ):
        raise RuntimeError("legacy consensus H1 state artifact is invalid")
    _atomic_json(output_path, payload)
    partial_path.unlink(missing_ok=True)
    return {
        "status": "ok",
        "job_id": job["job_id"],
        "state_occurrence_id": state_id,
        "output_path": str(output_path),
        "trial_count": len(trials),
    }


def _h1_failure(job: dict[str, Any], status: str, message: str) -> dict[str, Any]:
    return {
        "status": status,
        "error": message,
        "job_id": str(job["job_id"]),
        "state_occurrence_id": str(job["state_row"]["state_occurrence_id"]),
        "output_path": str(job["output_path"]),
        "trial_count": 0,
    }


def _frozen_selection(
    root: Path,
    config_path: Path,
    config: Mapping[str, Any],
    output_root: Path,
    *,
    allow_preview: bool,
) -> tuple[list[dict[str, Any]], str, str]:
    expected, expected_report = _selection_preview(root, config)
    if expected_report.get("status") != "ok" or len(expected) != 60:
        raise ValueError("legacy consensus outcome-blind selection is not viable")
    selection_path = output_root / "selected_states.jsonl"
    report_path = output_root / "selection_report.json"
    if not selection_path.is_file() or not report_path.is_file():
        if allow_preview:
            return expected, _fingerprint(expected), "in_memory_outcome_blind_preview"
        raise ValueError("legacy consensus collection requires frozen selection artifacts")
    selected = _read_jsonl(selection_path)
    report = _read_json(report_path)
    if (
        report.get("schema") != SELECTION_REPORT_SCHEMA
        or report.get("experiment_id") != EXPERIMENT_ID
        or report.get("status") != "ok"
        or report.get("dry_run") is not False
        or report.get("config_sha256") != sha256_file(config_path)
        or report.get("selected_states_sha256") != sha256_file(selection_path)
        or report.get("input_preflight_rows_sha256") != EXPECTED_PREFLIGHT["sha256"]
        or report.get("input_prior_selected_states_sha256")
        != EXPECTED_PRIOR_SELECTION["sha256"]
        or report.get("input_artifacts_mutated") is not False
        or report.get("training_authorized") is not False
        or selected != expected
    ):
        raise ValueError("legacy consensus frozen selection product changed")
    return selected, sha256_file(selection_path), "frozen_selection_artifact"


def run_collection(
    config_path: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    path, root, config = load_config(config_path)
    output_root = _output_root(root, config, output)
    selected, selection_identity, selection_source = _frozen_selection(
        root,
        path,
        config,
        output_root,
        allow_preview=dry_run,
    )
    action_count = sum(int(row["unique_action_count"]) for row in selected)
    trial_count = action_count * len(TRIAL_INDICES)
    if len(selected) != 60 or action_count != 120 or trial_count != 1920:
        raise ValueError("legacy consensus H1 product count changed")
    identity = _h1_run_identity(
        config_sha256=sha256_file(path),
        selection_identity=selection_identity,
        h1=dict(config["h1"]),
    )
    jobs = [
        {
            "job_id": str(row["state_occurrence_id"]),
            "state_row": row,
            "identity": identity,
            "per_action_time_limit_seconds": float(
                config["h1"]["per_action_time_limit_seconds"]
            ),
            "output_path": str(
                output_root / "h1_states" / f"{row['state_occurrence_id']}.json"
            ),
            "resume": bool(resume),
        }
        for row in selected
    ]
    if dry_run:
        return {
            "schema": H1_COLLECTION_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "status": "dry_run",
            "run_identity": identity,
            "state_job_count": len(jobs),
            "unique_action_count": action_count,
            "logical_trial_count": trial_count,
            "trial_indices": list(TRIAL_INDICES),
            "fixed_seed_halves": [list(range(8)), list(range(8, 16))],
            "workers": 16,
            "per_state_process_fuse_seconds": 420.0,
            "selection_source": selection_source,
            "research_split": RESEARCH_SPLIT,
            "training_only": True,
            "evaluation_eligible": False,
            "collection_executed": False,
            "training_or_runtime_or_ttf_executed": False,
            "training_authorized": False,
        }
    results = _run_jobs(
        _h1_state_worker,
        jobs,
        int(config["h1"]["workers"]),
        phase="legacy-consensus-h1",
        output_root=output_root / "h1_progress",
        run_fingerprint=identity,
        timeout_seconds=float(config["h1"]["per_state_process_fuse_seconds"]),
        failure_result=_h1_failure,
    )
    successful = [
        result for result in results if result.get("status") in {"ok", "resumed"}
    ]
    selected_by_id = {str(row["state_occurrence_id"]): row for row in selected}
    manifest: list[dict[str, Any]] = []
    invalid_count = 0
    observed_trials = 0
    for result in successful:
        state_id = str(result["state_occurrence_id"])
        state_row = selected_by_id.get(state_id)
        try:
            payload = _read_json(Path(result["output_path"]))
            source_snapshot = (
                _validated_source_snapshot(state_row)
                if state_row is not None
                else None
            )
        except (OSError, RuntimeError, ValueError):
            invalid_count += 1
            continue
        if state_row is None or not _state_payload_valid(
            payload,
            identity=identity,
            state_row=state_row,
            complete=True,
            source_snapshot=source_snapshot,
        ):
            invalid_count += 1
            continue
        count = len(payload["trials"])
        observed_trials += count
        manifest.append(
            {
                "schema": H1_MANIFEST_ROW_SCHEMA,
                "state_occurrence_id": state_id,
                "state_file": str(Path(result["output_path"]).relative_to(output_root)),
                "state_sha256": sha256_file(Path(result["output_path"])),
                "unique_action_count": int(state_row["unique_action_count"]),
                "trial_count": count,
                "research_split": RESEARCH_SPLIT,
                "training_only": True,
                "evaluation_eligible": False,
                "training_authorized": False,
            }
        )
    manifest.sort(key=lambda row: row["state_occurrence_id"])
    expected_state_ids = Counter(selected_by_id.keys())
    successful_state_ids = Counter(
        str(result.get("state_occurrence_id")) for result in successful
    )
    manifest_state_ids = Counter(str(row["state_occurrence_id"]) for row in manifest)
    complete = (
        len(successful) == 60
        and successful_state_ids == expected_state_ids
        and invalid_count == 0
        and len(manifest) == 60
        and manifest_state_ids == expected_state_ids
        and sum(int(row["unique_action_count"]) for row in manifest) == 120
        and observed_trials == 1920
    )
    _write_jsonl(output_root / "h1_state_manifest.jsonl", manifest)
    report = {
        "schema": H1_COLLECTION_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": "complete" if complete else "INTEGRITY_FAIL",
        "complete": complete,
        "run_identity": identity,
        "config_sha256": sha256_file(path),
        "selection_identity": selection_identity,
        "selection_source": selection_source,
        "expected_state_count": 60,
        "observed_valid_state_count": len(manifest),
        "expected_unique_action_count": 120,
        "observed_unique_action_count": sum(
            int(row["unique_action_count"]) for row in manifest
        ),
        "expected_trial_count": 1920,
        "observed_trial_count": observed_trials,
        "invalid_state_count": invalid_count,
        "worker_results": results,
        "research_split": RESEARCH_SPLIT,
        "training_only": True,
        "evaluation_eligible": False,
        "training_or_runtime_or_ttf_executed": False,
        "training_authorized": False,
    }
    _atomic_json(output_root / "h1_collection_report.json", report)
    return report


__all__ = [
    "CONFIG_SCHEMA",
    "EXPERIMENT_ID",
    "H1_COLLECTION_SCHEMA",
    "H1_STATE_SCHEMA",
    "H1_TRIAL_SCHEMA",
    "PLAN_SCHEMA",
    "RESEARCH_SPLIT",
    "SELECTION_REPORT_SCHEMA",
    "SELECTION_SCHEMA",
    "TRIAL_INDICES",
    "build_plan",
    "load_config",
    "matched_legacy_consensus_pp_seed",
    "run_collection",
    "run_selection",
    "select_legacy_consensus_states",
    "validate_config",
]
