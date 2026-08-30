from __future__ import annotations

import math
import os
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from experiments._common import contained_file, registered_input, sha256_file
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
from experiments.stride_fresh_matched_v2_active_supply_overlay_v1 import (
    FOLDS,
    _depth_band,
    _q0_row_eligibility,
    _selection_rank,
    _validate_wave_source_report,
    load_config as load_active_supply_config,
)
from experiments.stride_fresh_matched_v2_c16_h16_v1 import (
    ARMS,
    TRIAL_INDICES,
    classify_h1_challenger,
)
from experiments.stride_repairability_collection import (
    _source_target_state,
    repairability_restore_seed,
)
from experiments.trace_replay import restore_repair_state
from lns2_selector.runtime.fingerprints import repair_structure_fingerprint


CONFIG_SCHEMA = "lns2.stride.fresh_matched_unique_action_hierarchical_overlay_config.v1"
PLAN_SCHEMA = "lns2.stride.fresh_matched_unique_action_hierarchical_overlay_plan.v1"
SELECTION_SCHEMA = "lns2.stride.fresh_matched_unique_action_hierarchical_selection.v1"
SELECTION_REPORT_SCHEMA = (
    "lns2.stride.fresh_matched_unique_action_hierarchical_selection_report.v1"
)
H1_TRIAL_SCHEMA = "lns2.stride.fresh_matched_unique_action_hierarchical_h1_trial.v1"
H1_STATE_SCHEMA = "lns2.stride.fresh_matched_unique_action_hierarchical_h1_state.v1"
H1_COLLECTION_SCHEMA = (
    "lns2.stride.fresh_matched_unique_action_hierarchical_h1_collection.v1"
)
H1_REPORT_SCHEMA = "lns2.stride.fresh_matched_unique_action_hierarchical_h1_report.v1"
EXPERIMENT_ID = "stride_fresh_matched_unique_action_hierarchical_overlay_v1"
ACTIVE_EXPERIMENT_ID = "stride_fresh_matched_v2_active_supply_overlay_v1"
DEFAULT_OUTPUT_NAME = "stride-fresh-matched-unique-action-hierarchical-overlay-v1"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    _write_json(partial, payload)
    os.replace(partial, path)


def _registered(root: Path, specification: dict[str, Any], label: str) -> Path:
    return registered_input(root, specification, label=label)


def _strict_selection_contract() -> dict[str, Any]:
    return {
        "target_state_count": 96,
        "states_per_fold": 24,
        "maximum_states_per_episode": 4,
        "maximum_states_per_map": 16,
        "minimum_selected_map_count": 10,
        "minimum_selected_family_count": 4,
        "structural_required_actual_size": 16,
        "minimum_unique_action_count": 2,
        "require_structural_action_distinct_from_v2": True,
        "deduplicate_by_exact_sorted_agent_set": True,
        "execute_each_unique_set_once": True,
        "preserve_all_role_aliases": True,
        "rank_rule": "reuse_stride_active_supply_q0_preaction_sha256_ascending",
        "depth_bands": {"d0": [0, 0], "d1_3": [1, 3], "d4_plus": [4, 11]},
        "minimum_states_per_depth_band_per_fold": 4,
        "expected_eligible_preaction_count": 379,
        "expected_eligible_strata_supply": {
            "consensus_structural": 92,
            "three_unique": 286,
            "anchor_component_shared": 1,
        },
        "selected_state_strata_per_fold": {
            "consensus_structural": 8,
            "three_unique": 16,
        },
        "expected_selected_state_strata": {
            "consensus_structural": 32,
            "three_unique": 64,
        },
        "expected_selected_unique_action_count_distribution": {"2": 32, "3": 64},
        "require_all_distinct_support_maps": True,
        "expected_distinct_support_map_count": 7,
        "expected_distinct_support_family_count": 3,
        "failure_action": "STATE_SUPPLY_FAIL_h1_zero_no_reserve",
        "target_outcome_fields_read": False,
    }


def validate_config(config: dict[str, Any], *, project_root: Path | None = None) -> None:
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("experiment_id") != EXPERIMENT_ID
        or config.get("scientific_status")
        != "post_q0_sequential_preregistration_before_any_h1_outcome"
        or config.get("freshness")
        != "read_only_overlay_on_frozen_active_supply_preaction_cohort"
        or config.get("research_split") != "fresh_matched_development"
    ):
        raise ValueError("unique-action overlay identity changed")
    source = dict(config.get("active_supply_read_only") or {})
    if (
        set(source)
        != {
            "config",
            "output_root",
            "wave_source_report",
            "preflight_report",
            "preflight_rows",
            "source_episode_audit",
            "wave_manifests",
            "trace_registry_sha256",
            "expected_source_episode_count",
            "expected_preaction_row_count",
            "expected_prior_distinct_eligible_count",
            "expected_prior_status",
            "mutation_allowed",
        }
        or source.get("output_root")
        != "build/stride-fresh-matched-v2-active-supply-overlay-v1"
        or source.get("config")
        != {
            "path": "configs/stride_fresh_matched_v2_active_supply_overlay_v1.json",
            "sha256": "ac309b23b0500c373289f01d20db526d416a1c888ad405bf865fe317d4596137",
        }
        or source.get("wave_source_report", {}).get("sha256")
        != "9d4950ee4a5dc36c58688b5785a9706a4337420dff86588ad7ef1d2c9297dfbf"
        or source.get("preflight_report", {}).get("sha256")
        != "2cff7940d382dabc28da5dd5756e597a5a98610f66edc0edce56546001ca02f7"
        or source.get("preflight_rows", {}).get("sha256")
        != "c588cfbda699b782667d1bd7e85742b97bc04910e156716923d39c59a367b1dc"
        or source.get("source_episode_audit", {}).get("sha256")
        != "d7fe33118660876601cc6e0fabd7fc4f7d61a90d7eec79d575658037037418f5"
        or source.get("wave_manifests")
        != {
            "balanced_wall_clock": {
                "path": "build/stride-fresh-matched-v2-active-supply-overlay-v1/wave_a/source/balanced_wall_clock/realized_dynamic_manifest.jsonl",
                "sha256": "37e443ec7a73878eb60560bdcf74dd701a733dd02a9a7a450f91ffc938f490fd",
            },
            "movingai_ood": {
                "path": "build/stride-fresh-matched-v2-active-supply-overlay-v1/wave_a/source/movingai_ood/realized_dynamic_manifest.jsonl",
                "sha256": "01c04f58d2e45b52311a87d99c75af5c3c00a83ef6c202219e6eb344fb135d70",
            },
        }
        or source.get("trace_registry_sha256")
        != "c320c466dbdfb7f4a38e501b68bc0e2b046e19df5e05bc822202d0b210eeced9"
        or int(source.get("expected_source_episode_count", -1)) != 68
        or int(source.get("expected_preaction_row_count", -1)) != 380
        or int(source.get("expected_prior_distinct_eligible_count", -1)) != 286
        or source.get("expected_prior_status") != "STATE_SUPPLY_FAIL"
        or source.get("mutation_allowed") is not False
    ):
        raise ValueError("unique-action read-only source contract changed")
    if dict(config.get("state_selection") or {}) != _strict_selection_contract():
        raise ValueError("unique-action state-selection contract changed")
    folds = {
        key: tuple(value) for key, value in dict(config.get("map_folds") or {}).items()
    }
    if folds != FOLDS:
        raise ValueError("unique-action folds changed")
    h1 = dict(config.get("h1") or {})
    if h1 != {
        "trial_indices": list(TRIAL_INDICES),
        "first_fixed_half": list(range(8)),
        "second_fixed_half": list(range(8, 16)),
        "expected_selected_state_count": 96,
        "expected_unique_action_count": 256,
        "logical_trial_count": 4096,
        "workers": 16,
        "per_action_time_limit_seconds": 5.0,
        "per_state_process_fuse_seconds": 420.0,
        "same_state_trial_seed_across_unique_actions": True,
        "trial_level_atomic_checkpoint": True,
        "runtime_or_pp_seconds_used_in_label": False,
    }:
        raise ValueError("unique-action H1 execution contract changed")
    label = dict(config.get("h1_label") or {})
    if (
        label.get("stage1_unit")
        != "unique_structural_action_vs_unique_v2_anchor_action"
        or label.get("stage2_unit")
        != "direct_component_vs_hotspot_on_three_unique_states"
        or label.get("role_aliases_share_one_action_result") is not True
        or label.get("duplicate_role_of_anchor_label") is not None
        or label.get("per_seed_score")
        != "normalized_current_step_conflict_reduction"
        or int(label.get("minimum_strict_paired_wins", -1)) != 12
        or float(label.get("minimum_mean_delta", -1)) != 0.02
        or label.get("require_positive_first_half_mean_delta") is not True
        or label.get("require_positive_second_half_mean_delta") is not True
        or label.get("require_no_progress_rate_noninferiority") is not True
        or label.get("require_rollback_rate_noninferiority") is not True
        or label.get("require_time_limit_rate_noninferiority") is not True
        or float(label.get("tie_epsilon", -1)) != 1e-12
    ):
        raise ValueError("unique-action H1 label contract changed")
    gates = dict(config.get("h1_gates") or {})
    if gates != {
        "expected_state_count": 96,
        "expected_unique_action_count": 256,
        "expected_trial_count": 4096,
        "minimum_opportunity_state_count": 20,
        "minimum_opportunity_map_count": 8,
        "minimum_opportunity_family_count": 4,
        "each_fold_requires_opportunity_and_nonopportunity": True,
        "unique_candidate_labels_require_both_classes": True,
        "all_integrity_gates_required": True,
        "map_folds": {key: list(value) for key, value in FOLDS.items()},
        "stage2": {
            "conditional_claim": "conditional_on_observed_C_ne_H_support",
            "expected_state_count": 64,
            "states_per_fold": 16,
            "expected_support_map_count": 7,
            "expected_support_family_count": 3,
            "minimum_decisive_state_count": 20,
            "minimum_decisive_support_map_count": 6,
            "required_decisive_support_family_count": 3,
            "each_fold_requires_decisive_state": True,
            "require_component_win_and_hotspot_win": True,
            "training_or_runtime_authorized": False,
        },
    }:
        raise ValueError("unique-action H1 gates changed")
    boundary = dict(config.get("claim_boundary") or {})
    if boundary != {
        "unique_action_h1_opportunity_only": True,
        "input_source_or_preflight_mutation_allowed": False,
        "outcome_based_state_deletion_allowed": False,
        "reserve_backfill_allowed": False,
        "h8_executed": False,
        "training_allowed": False,
        "ttf_or_speed_claim_allowed": False,
        "conditional_on_observed_C_ne_H_support": True,
        "legacy_rescue_fallback_or_warehouse_only_route_used": False,
        "safeslot_teacher_model_feature_or_label_used": False,
    }:
        raise ValueError("unique-action claim boundary changed")
    if project_root is not None:
        _validate_read_only_inputs(project_root.resolve(), config)


def _validate_read_only_inputs(root: Path, config: dict[str, Any]) -> None:
    source = dict(config["active_supply_read_only"])
    paths = {
        name: _registered(root, dict(source[name]), f"unique-action {name}")
        for name in (
            "config",
            "wave_source_report",
            "preflight_report",
            "preflight_rows",
            "source_episode_audit",
        )
    }
    active_path, active_root, active_config = load_active_supply_config(paths["config"])
    if active_config.get("experiment_id") != ACTIVE_EXPERIMENT_ID or active_root != root:
        raise ValueError("active-supply input identity changed")
    wave_report = _read_json(paths["wave_source_report"])
    active_output = (root / str(source["output_root"])).resolve()
    _validate_wave_source_report(active_config, active_path, active_output, wave_report)
    if wave_report.get("trace_registry_sha256") != source["trace_registry_sha256"]:
        raise ValueError("active-supply trace registry changed")
    manifest_hashes = {
        source_id: sha256_file(
            _registered(root, dict(spec), f"unique-action Wave {source_id}")
        )
        for source_id, spec in dict(source["wave_manifests"]).items()
    }
    if wave_report.get("manifest_sha256") != manifest_hashes:
        raise ValueError("active-supply Wave manifests changed")
    preflight = _read_json(paths["preflight_report"])
    rows = _read_jsonl(paths["preflight_rows"])
    audits = _read_jsonl(paths["source_episode_audit"])
    if (
        preflight.get("status") != "STATE_SUPPLY_FAIL"
        or int(preflight.get("preaction_state_count", -1)) != 380
        or int(preflight.get("eligible_preaction_state_count", -1)) != 286
        or preflight.get("preflight_rows_sha256") != sha256_file(paths["preflight_rows"])
        or preflight.get("source_episode_audit_sha256")
        != sha256_file(paths["source_episode_audit"])
        or preflight.get("wave_a_source_report_sha256")
        != sha256_file(paths["wave_source_report"])
        or len(rows) != 380
        or len(audits) != 68
        or sum(bool(row.get("eligible")) for row in rows) != 286
        or any(row.get("target_outcome_fields_read") is not False for row in rows)
    ):
        raise ValueError("active-supply frozen preflight product changed")


def load_config(path: str | Path) -> tuple[Path, Path, dict[str, Any]]:
    config_path = Path(path).resolve()
    root = config_path.parents[1]
    config = _read_json(config_path)
    validate_config(config, project_root=root)
    return config_path, root, config


def _output_root(root: Path, config: dict[str, Any], output: str | Path) -> Path:
    target = Path(output).resolve()
    read_only = (
        root / str(config["active_supply_read_only"]["output_root"])
    ).resolve()
    active_config = _read_json(
        root / str(config["active_supply_read_only"]["config"]["path"])
    )
    v1_read_only = (
        root / str(active_config["v1_read_only"]["output_root"])
    ).resolve()
    if any(
        target == protected or protected in target.parents
        for protected in (read_only, v1_read_only)
    ):
        raise ValueError("unique-action output must remain outside read-only inputs")
    return target


def _input_rows(root: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    path = _registered(
        root,
        dict(config["active_supply_read_only"]["preflight_rows"]),
        "unique-action preflight rows",
    )
    return [dict(row) for row in _read_jsonl(path)]


def _agent_set(candidate: dict[str, Any]) -> tuple[int, ...]:
    return tuple(sorted(map(int, candidate["agents"])))


def _unique_row_eligibility(row: dict[str, Any]) -> tuple[bool, list[str]]:
    _active_ok, active_reasons = _q0_row_eligibility(row)
    declared = set(map(str, row.get("ineligibility_reasons") or ()))
    may_ignore_distinct = not declared or declared == {
        "arm_agent_sets_not_pairwise_distinct"
    }
    reasons = [
        reason
        for reason in active_reasons
        if not (
            may_ignore_distinct
            and reason
            in {
                "arm_agent_sets_not_pairwise_distinct",
                "preflight_declared_ineligible",
            }
        )
    ]
    arms = dict(row.get("arms") or {})
    if set(arms) == set(ARMS) and all(isinstance(arms[role], dict) for role in ARMS):
        anchor = _agent_set(arms["v2_anchor"])
        structural = [_agent_set(arms[role]) for role in ("component16", "hotspot16")]
        unique_sets = {anchor, *structural}
        if len(unique_sets) < 2 or not any(agents != anchor for agents in structural):
            reasons.append("no_structural_action_distinct_from_v2")
    else:
        reasons.append("missing_unique_action_arms")
    return not reasons, sorted(set(reasons))


def _state_identity(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "namespace": "stride-fresh-matched-unique-action-state-v1",
        "origin": str(row["origin"]),
        "source_id": str(row["source_id"]),
        "episode_id": str(row["episode_id"]),
        "task_id": str(row["task_id"]),
        "solver_seed": int(row["solver_seed"]),
        "decision_index": int(row["decision_index"]),
        "before_fingerprint": str(row["before_fingerprint"]),
    }


def _unique_stratum(row: dict[str, Any]) -> str:
    arms = dict(row["arms"])
    anchor = _agent_set(arms["v2_anchor"])
    component = _agent_set(arms["component16"])
    hotspot = _agent_set(arms["hotspot16"])
    if component == hotspot != anchor:
        return "consensus_structural"
    if len({anchor, component, hotspot}) == 3:
        return "three_unique"
    if anchor == component != hotspot:
        return "anchor_component_shared"
    if anchor == hotspot != component:
        return "anchor_hotspot_shared"
    return "single_unique"


def _canonical_unique_actions(
    row: dict[str, Any], state_occurrence_id: str
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    grouped: dict[tuple[int, ...], list[str]] = {}
    for role in ARMS:
        agents = _agent_set(dict(row["arms"][role]))
        grouped.setdefault(agents, []).append(role)
    actions = []
    role_to_action: dict[str, str] = {}
    for agents, aliases in grouped.items():
        action_id = "unique-action-" + _fingerprint(
            {
                "namespace": "stride-fresh-matched-unique-action-v1",
                "state_occurrence_id": state_occurrence_id,
                "agents": list(agents),
            }
        )[:24]
        for role in aliases:
            role_to_action[role] = action_id
        actions.append(
            {
                "action_id": action_id,
                "agents": list(agents),
                "actual_size": len(agents),
                "role_aliases": aliases,
                "candidate_ids_by_role": {
                    role: str(row["arms"][role]["candidate_id"]) for role in aliases
                },
                "contains_v2_anchor_role": "v2_anchor" in aliases,
                "structural_role_aliases": [
                    role for role in aliases if role in {"component16", "hotspot16"}
                ],
            }
        )
    actions.sort(key=lambda action: tuple(ARMS.index(role) for role in action["role_aliases"]))
    return actions, role_to_action


def select_unique_action_states(
    rows: Iterable[dict[str, Any]], config: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    all_rows = [dict(row) for row in rows]
    eligibility = [_unique_row_eligibility(row) for row in all_rows]
    eligible = [
        row for row, (passed, _reasons) in zip(all_rows, eligibility) if passed
    ]
    supply_strata = Counter(_unique_stratum(row) for row in eligible)
    selectable = [
        row
        for row in eligible
        if _unique_stratum(row) in {"consensus_structural", "three_unique"}
    ]
    distinct_support_maps = {
        str(row["map_id"])
        for row in selectable
        if _unique_stratum(row) == "three_unique"
    }
    distinct_support_families = {
        str(row["map_family"])
        for row in selectable
        if _unique_stratum(row) == "three_unique"
    }
    map_to_fold = {
        map_id: fold for fold, map_ids in FOLDS.items() for map_id in map_ids
    }
    episode_counts: Counter[tuple[str, str, str]] = Counter()
    map_counts: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    selected_keys: set[tuple[str, str, str, int, str]] = set()
    failures: list[str] = [
        reason
        for _passed, reasons in eligibility
        for reason in reasons
        if reason.startswith("forbidden_outcome_field:")
    ]

    def episode_key(row: dict[str, Any]) -> tuple[str, str, str]:
        return str(row["origin"]), str(row["source_id"]), str(row["episode_id"])

    def row_key(row: dict[str, Any]) -> tuple[str, str, str, int, str]:
        return (
            str(row["origin"]),
            str(row["source_id"]),
            str(row["episode_id"]),
            int(row["decision_index"]),
            str(row["before_fingerprint"]),
        )

    def can_add(row: dict[str, Any]) -> bool:
        return (
            episode_counts[episode_key(row)] < 4
            and map_counts[str(row["map_id"])] < 16
            and row_key(row) not in selected_keys
        )

    def add(row: dict[str, Any]) -> None:
        selected.append(row)
        selected_keys.add(row_key(row))
        episode_counts[episode_key(row)] += 1
        map_counts[str(row["map_id"])] += 1

    fold_selected_rows: dict[str, list[dict[str, Any]]] = {}
    fold_stratum_counts: dict[str, Counter[str]] = {}
    for fold in FOLDS:
        candidates = sorted(
            (
                row
                for row in selectable
                if map_to_fold.get(str(row["map_id"])) == fold
            ),
            key=lambda row: (_selection_rank(row), int(row["decision_index"])),
        )
        fold_selected: list[dict[str, Any]] = []
        represented: set[str] = set()
        stratum_counts: Counter[str] = Counter()
        stratum_quotas = {"consensus_structural": 8, "three_unique": 16}

        def can_add_fold(row: dict[str, Any]) -> bool:
            stratum = _unique_stratum(row)
            return (
                can_add(row)
                and stratum in stratum_quotas
                and stratum_counts[stratum] < stratum_quotas[stratum]
            )

        def add_fold(row: dict[str, Any]) -> None:
            add(row)
            fold_selected.append(row)
            represented.add(str(row["map_id"]))
            stratum_counts[_unique_stratum(row)] += 1

        # Result-blind support coverage is fixed before depth/quota filling.
        for map_id in FOLDS[fold]:
            map_rows = [row for row in candidates if str(row["map_id"]) == map_id]
            if not map_rows:
                continue
            required_stratum = (
                "three_unique"
                if map_id in distinct_support_maps
                else "consensus_structural"
            )
            representative = next(
                (
                    row
                    for row in map_rows
                    if _unique_stratum(row) == required_stratum
                    and can_add_fold(row)
                ),
                None,
            )
            if representative is None:
                failures.append(f"{fold}:{map_id}_support")
            else:
                add_fold(representative)

        for band in ("d0", "d1_3", "d4_plus"):
            band_rows = [
                row
                for row in candidates
                if _depth_band(int(row["decision_index"])) == band
            ]
            for prefer_new_map in (True, False):
                for row in band_rows:
                    if sum(
                        _depth_band(int(item["decision_index"])) == band
                        for item in fold_selected
                    ) >= 4:
                        break
                    if prefer_new_map and str(row["map_id"]) in represented:
                        continue
                    if can_add_fold(row):
                        add_fold(row)
            if sum(
                _depth_band(int(item["decision_index"])) == band
                for item in fold_selected
            ) < 4:
                failures.append(f"{fold}:{band}_supply")
        for stratum in ("consensus_structural", "three_unique"):
            for row in candidates:
                if stratum_counts[stratum] >= stratum_quotas[stratum]:
                    break
                if _unique_stratum(row) == stratum and can_add_fold(row):
                    add_fold(row)
        fold_selected_rows[fold] = fold_selected
        fold_stratum_counts[fold] = stratum_counts
        if len(fold_selected) != 24:
            failures.append(f"{fold}:exact_24")
        if stratum_counts != Counter(stratum_quotas):
            failures.append(f"{fold}:exact_hierarchical_strata")

    selected_maps = {str(row["map_id"]) for row in selected}
    selected_families = {str(row["map_family"]) for row in selected}
    if len(all_rows) != 380:
        failures.append("exact_380_frozen_preaction_rows")
    if len(eligible) != 379:
        failures.append("exact_379_unique_action_eligible_rows")
    if supply_strata != Counter(
        {
            "consensus_structural": 92,
            "three_unique": 286,
            "anchor_component_shared": 1,
        }
    ):
        failures.append("unique_action_supply_strata_changed")
    if len(selected) != 96:
        failures.append("exact_96")
    if len(selected_maps) < 10:
        failures.append("minimum_10_maps")
    if len(selected_families) < 4:
        failures.append("minimum_4_families")
    selected_distinct_maps = {
        str(row["map_id"])
        for row in selected
        if _unique_stratum(row) == "three_unique"
    }
    selected_distinct_families = {
        str(row["map_family"])
        for row in selected
        if _unique_stratum(row) == "three_unique"
    }
    if (
        len(distinct_support_maps) != 7
        or selected_distinct_maps != distinct_support_maps
        or len(distinct_support_families) != 3
        or selected_distinct_families != distinct_support_families
    ):
        failures.append("distinct_support_coverage")
    if max(episode_counts.values(), default=0) > 4:
        failures.append("episode_cap")
    if max(map_counts.values(), default=0) > 16:
        failures.append("map_cap")

    frozen = []
    action_distribution: Counter[int] = Counter()
    if not failures:
        for row in selected:
            state_id = "unique-action-state-" + _fingerprint(_state_identity(row))[:24]
            actions, role_to_action = _canonical_unique_actions(row, state_id)
            action_distribution[len(actions)] += 1
            frozen.append(
                {
                    **row,
                    "schema": SELECTION_SCHEMA,
                    "state_occurrence_id": state_id,
                    "selection_rank_sha256": _selection_rank(row),
                    "depth_band": _depth_band(int(row["decision_index"])),
                    "unique_actions": actions,
                    "role_to_action_id": role_to_action,
                    "unique_action_count": len(actions),
                    "hierarchical_stratum": _unique_stratum(row),
                    "target_outcome_fields_read": False,
                }
            )
        frozen.sort(key=lambda row: str(row["state_occurrence_id"]))
        if action_distribution != Counter({2: 32, 3: 64}):
            failures.append("selected_unique_action_distribution_changed")
            frozen = []
    if failures:
        action_distribution = Counter()
    fold_counts = {
        fold: len(rows_in_fold) for fold, rows_in_fold in fold_selected_rows.items()
    }
    fold_depth_counts = {
        fold: dict(
            Counter(
                _depth_band(int(row["decision_index"])) for row in rows_in_fold
            )
        )
        for fold, rows_in_fold in fold_selected_rows.items()
    }
    return frozen, {
        "schema": SELECTION_REPORT_SCHEMA,
        "status": "ok" if not failures else "STATE_SUPPLY_FAIL",
        "failure_reasons": sorted(set(failures)),
        "preaction_state_count": len(all_rows),
        "prior_distinct_eligible_count": sum(bool(row.get("eligible")) for row in all_rows),
        "unique_action_eligible_count": len(eligible),
        "hierarchical_selectable_count": len(selectable),
        "eligible_strata_supply": dict(sorted(supply_strata.items())),
        "unique_action_ineligibility_reason_counts": dict(
            sorted(
                Counter(
                    reason
                    for passed, reasons in eligibility
                    if not passed
                    for reason in reasons
                ).items()
            )
        ),
        "selected_state_count": len(frozen),
        "selected_unique_action_count_distribution": {
            str(key): value for key, value in sorted(action_distribution.items())
        },
        "selected_unique_action_count": sum(
            key * value for key, value in action_distribution.items()
        ),
        "logical_h1_trial_count": (
            sum(key * value for key, value in action_distribution.items()) * 16
        ),
        "selected_map_count": len(selected_maps),
        "selected_family_count": len(selected_families),
        "selected_state_strata": dict(
            sorted(Counter(_unique_stratum(row) for row in selected).items())
        ),
        "fold_state_strata": {
            fold: dict(sorted(counts.items()))
            for fold, counts in fold_stratum_counts.items()
        },
        "distinct_support_map_count": len(selected_distinct_maps),
        "distinct_support_family_count": len(selected_distinct_families),
        "maximum_episode_count": max(episode_counts.values(), default=0),
        "maximum_map_count": max(map_counts.values(), default=0),
        "fold_counts": fold_counts,
        "fold_depth_counts": fold_depth_counts,
        "outcome_fields_read": False,
        "reserve_backfill_used": False,
    }


def _selection_preview(
    root: Path, config: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return select_unique_action_states(_input_rows(root, config), config)


def build_plan(config_path: str | Path) -> dict[str, Any]:
    path, root, config = load_config(config_path)
    selected, selection = _selection_preview(root, config)
    return {
        "schema": PLAN_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "config_path": str(path),
        "config_sha256": sha256_file(path),
        "read_only_preaction_row_count": 380,
        "unique_action_eligible_count": selection["unique_action_eligible_count"],
        "selected_state_count": len(selected),
        "selected_unique_action_count_distribution": selection[
            "selected_unique_action_count_distribution"
        ],
        "selected_unique_action_count": selection["selected_unique_action_count"],
        "logical_h1_trial_count": selection["logical_h1_trial_count"],
        "workers": 16,
        "h1_executed_by_plan": False,
        "h8_or_training_or_ttf_executed": False,
        "selection_status": selection["status"],
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
            "input_preflight_rows_sha256": config["active_supply_read_only"][
                "preflight_rows"
            ]["sha256"],
            "input_preflight_report_sha256": config["active_supply_read_only"][
                "preflight_report"
            ]["sha256"],
            "input_wave_source_report_sha256": config["active_supply_read_only"][
                "wave_source_report"
            ]["sha256"],
            "dry_run": bool(dry_run),
            "input_artifacts_mutated": False,
        }
    )
    if dry_run:
        return report
    if report["status"] != "ok" or len(selected) != 96:
        _atomic_json(output_root / "selection_report.json", report)
        _write_jsonl(output_root / "selected_states.jsonl", [])
        return report
    _write_jsonl(output_root / "selected_states.jsonl", selected)
    report["selected_states_sha256"] = sha256_file(
        output_root / "selected_states.jsonl"
    )
    _atomic_json(output_root / "selection_report.json", report)
    return report


def matched_unique_pp_seed(state_occurrence_id: str, trial_index: int) -> int:
    if not state_occurrence_id or trial_index not in TRIAL_INDICES:
        raise ValueError("invalid unique-action paired PP seed identity")
    digest = _fingerprint(
        {
            "namespace": "stride-fresh-matched-unique-action-paired-pp-v1",
            "state_occurrence_id": state_occurrence_id,
            "trial_index": int(trial_index),
            "step_index": 0,
        }
    )
    return int(digest[:16], 16) % (2**31)


def _h1_run_identity(
    *, config_sha256: str, selection_identity: str, h1: dict[str, Any]
) -> str:
    return _fingerprint(
        {
            "namespace": "stride-fresh-matched-unique-action-h1-v1",
            "config_sha256": config_sha256,
            "selection_identity": selection_identity,
            "h1": h1,
        }
    )


def _trial_valid(
    row: dict[str, Any],
    state_id: str,
    actions: dict[str, dict[str, Any]],
    expected_restore_seed: int,
) -> bool:
    action_id = str(row.get("action_id"))
    trial_index = int(row.get("trial_index", -1))
    action = actions.get(action_id)
    if action is None or trial_index not in TRIAL_INDICES:
        return False
    expected_seed = matched_unique_pp_seed(state_id, trial_index)
    repair_order_count = int(row.get("repair_order_count", -1))
    expected_applied = expected_seed if repair_order_count > 0 else -1
    if (
        row.get("schema") != H1_TRIAL_SCHEMA
        or row.get("state_occurrence_id") != state_id
        or row.get("trial_identity")
        != _fingerprint(
            {
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
    payload: dict[str, Any],
    *,
    identity: str,
    state_row: dict[str, Any],
    complete: bool,
    source_snapshot: dict[str, Any] | None = None,
) -> bool:
    state_id = str(state_row["state_occurrence_id"])
    expected_actions, expected_roles = _canonical_unique_actions(state_row, state_id)
    if (
        payload.get("schema") != H1_STATE_SCHEMA
        or payload.get("identity") != identity
        or payload.get("state_occurrence_id") != state_id
        or payload.get("state_row") != state_row
        or payload.get("complete") is not complete
        or state_row.get("unique_actions") != expected_actions
        or state_row.get("role_to_action_id") != expected_roles
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
    if complete:
        product_ok = set(product) == expected_product
    else:
        product_ok = set(product) <= expected_product
    return (
        product_ok
        and all(count == 1 for count in product.values())
        and all(
            _trial_valid(
                dict(row), state_id, actions, expected_restore_seed
            )
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
    before_conflicts = int(source_state["num_of_colliding_pairs"])
    before_soc = int(source_state["sum_of_costs"])
    snapshot = {
        "source_state": source_state,
        "source_trace_file": str(source_manifest["trace_file"]),
        "source_trace_path": str(source_trace_path),
        "source_trace_sha256": observed_trace_sha256,
        "before_repair_fingerprint": before_repair,
        "before_conflicts": before_conflicts,
        "before_sum_of_costs": before_soc,
        "restore_seed": repairability_restore_seed(before_repair),
    }
    if (
        observed_trace_sha256 != str(source_manifest.get("trace_sha256"))
        or snapshot["source_trace_file"] != state_row.get("source_trace_file")
        or snapshot["source_trace_path"] != state_row.get("source_trace_path")
        or observed_trace_sha256 != state_row.get("source_trace_sha256")
        or state_fingerprint(source_state) != state_row.get("before_fingerprint")
        or before_repair != state_row.get("before_repair_fingerprint")
        or before_conflicts != int(state_row.get("before_conflicts", -1))
        or before_conflicts <= 0
        or before_soc < 0
    ):
        raise RuntimeError("unique-action H1 source identity changed")
    return snapshot


def _h1_state_worker(job: dict[str, Any]) -> dict[str, Any]:
    state_row = dict(job["state_row"])
    state_id = str(state_row["state_occurrence_id"])
    output_path = Path(str(job["output_path"]))
    partial_path = output_path.with_name(output_path.name + ".partial")
    identity = str(job["identity"])
    if output_path.exists() and not bool(job["resume"]):
        raise ValueError(f"unique-action H1 output exists; pass --resume: {output_path}")
    if partial_path.exists() and not bool(job["resume"]):
        raise ValueError(
            f"unique-action H1 partial exists; pass --resume: {partial_path}"
        )

    expected_actions, expected_roles = _canonical_unique_actions(state_row, state_id)
    if (
        state_row.get("schema") != SELECTION_SCHEMA
        or state_row.get("unique_actions") != expected_actions
        or state_row.get("role_to_action_id") != expected_roles
        or state_row.get("hierarchical_stratum")
        not in {"consensus_structural", "three_unique"}
        or len(expected_actions) not in {2, 3}
    ):
        raise RuntimeError("unique-action frozen state product changed")
    if state_row["hierarchical_stratum"] == "consensus_structural" and not (
        len(expected_actions) == 2
        and expected_roles["component16"] == expected_roles["hotspot16"]
        and expected_roles["v2_anchor"] != expected_roles["component16"]
    ):
        raise RuntimeError("unique-action consensus stratum changed")
    if state_row["hierarchical_stratum"] == "three_unique" and len(
        set(expected_roles.values())
    ) != 3:
        raise RuntimeError("unique-action three-unique stratum changed")

    source_snapshot = _validated_source_snapshot(state_row)
    source_state = dict(source_snapshot["source_state"])
    observed_trace_sha256 = str(source_snapshot["source_trace_sha256"])
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
        raise ValueError(f"invalid unique-action H1 artifact: {output_path}")
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
        "source_trace_sha256": observed_trace_sha256,
        "before_repair_fingerprint": before_repair,
        "before_conflicts": before_conflicts,
        "before_sum_of_costs": before_soc,
        "restore_seed": restore_seed,
        "trials": [],
        "runtime_used_in_label": False,
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
            raise ValueError(f"invalid unique-action H1 partial: {partial_path}")
    trials = list(partial_payload["trials"])
    completed = {
        (str(row["action_id"]), int(row["trial_index"])) for row in trials
    }
    for action in expected_actions:
        action_id = str(action["action_id"])
        agents = list(map(int, action["agents"]))
        if not agents or len(agents) != len(set(agents)):
            raise RuntimeError("unique-action H1 action is illegal")
        if action["structural_role_aliases"] and len(agents) != 16:
            raise RuntimeError("unique-action structural action is not size16")
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
                raise RuntimeError("unique-action H1 branch restore changed")
            pp_seed = matched_unique_pp_seed(state_id, trial_index)
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
                raise RuntimeError("unique-action timed PP failure was not atomic")
            if success and conflicts_after > before_conflicts:
                raise RuntimeError("unique-action successful PP increased conflicts")
            repair_order = metrics.get("repair_order")
            if not isinstance(repair_order, list):
                raise RuntimeError("unique-action timed PP omitted repair order")
            if (
                int(metrics.get("requested_random_seed", -1)) != pp_seed
                or metrics.get("action_valid") is not True
                or metrics.get("generated") is not True
            ):
                raise RuntimeError("unique-action timed PP identity changed")
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
        raise RuntimeError("unique-action H1 state artifact is invalid")
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
    config: dict[str, Any],
    output_root: Path,
    *,
    allow_preview: bool,
) -> tuple[list[dict[str, Any]], str, str]:
    expected, expected_report = _selection_preview(root, config)
    if expected_report.get("status") != "ok" or len(expected) != 96:
        raise ValueError("unique-action result-blind selection is not viable")
    selection_path = output_root / "selected_states.jsonl"
    report_path = output_root / "selection_report.json"
    if not selection_path.is_file() or not report_path.is_file():
        if allow_preview:
            return expected, _fingerprint(expected), "in_memory_result_blind_preview"
        raise ValueError("unique-action H1 requires frozen selection artifacts")
    selected = _read_jsonl(selection_path)
    report = _read_json(report_path)
    if (
        report.get("schema") != SELECTION_REPORT_SCHEMA
        or report.get("experiment_id") != EXPERIMENT_ID
        or report.get("status") != "ok"
        or report.get("dry_run") is not False
        or report.get("config_sha256") != sha256_file(config_path)
        or report.get("selected_states_sha256") != sha256_file(selection_path)
        or report.get("input_artifacts_mutated") is not False
        or report.get("input_preflight_rows_sha256")
        != config["active_supply_read_only"]["preflight_rows"]["sha256"]
        or selected != expected
    ):
        raise ValueError("unique-action frozen selection product changed")
    return selected, sha256_file(selection_path), "frozen_selection_artifact"


def run_h1(
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
    if len(selected) != 96 or action_count != 256 or trial_count != 4096:
        raise ValueError("unique-action H1 product count changed")
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
            "status": "dry_run",
            "state_job_count": len(jobs),
            "unique_action_count": action_count,
            "logical_trial_count": trial_count,
            "workers": 16,
            "per_state_process_fuse_seconds": 420.0,
            "selection_source": selection_source,
            "h1_executed": False,
            "h8_or_training_or_ttf_executed": False,
        }
    results = _run_jobs(
        _h1_state_worker,
        jobs,
        int(config["h1"]["workers"]),
        phase="unique-action-h1",
        output_root=output_root / "h1_progress",
        run_fingerprint=identity,
        timeout_seconds=float(config["h1"]["per_state_process_fuse_seconds"]),
        failure_result=_h1_failure,
    )
    successful = [
        result for result in results if result.get("status") in {"ok", "resumed"}
    ]
    selected_by_id = {
        str(row["state_occurrence_id"]): row for row in selected
    }
    manifest = []
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
                "state_occurrence_id": state_id,
                "state_file": str(Path(result["output_path"]).relative_to(output_root)),
                "state_sha256": sha256_file(Path(result["output_path"])),
                "unique_action_count": int(state_row["unique_action_count"]),
                "trial_count": count,
            }
        )
    manifest.sort(key=lambda row: row["state_occurrence_id"])
    # Counter(mapping) treats mapping values as counts; count the selected IDs
    # themselves so the parent collection identity check matches worker rows.
    expected_state_ids = Counter(selected_by_id.keys())
    successful_state_ids = Counter(
        str(result.get("state_occurrence_id")) for result in successful
    )
    manifest_state_ids = Counter(
        str(row["state_occurrence_id"]) for row in manifest
    )
    complete = (
        len(successful) == 96
        and successful_state_ids == expected_state_ids
        and invalid_count == 0
        and len(manifest) == 96
        and manifest_state_ids == expected_state_ids
        and sum(int(row["unique_action_count"]) for row in manifest) == 256
        and observed_trials == 4096
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
        "expected_state_count": 96,
        "observed_valid_state_count": len(manifest),
        "expected_unique_action_count": 256,
        "observed_unique_action_count": sum(
            int(row["unique_action_count"]) for row in manifest
        ),
        "expected_trial_count": 4096,
        "observed_trial_count": observed_trials,
        "invalid_state_count": invalid_count,
        "worker_results": results,
        "h8_or_training_or_ttf_executed": False,
    }
    _atomic_json(output_root / "h1_collection_report.json", report)
    return report


def analyze_payloads(
    config: dict[str, Any],
    payloads: Iterable[dict[str, Any]],
    *,
    identity: str,
    selected: list[dict[str, Any]],
    error_count: int = 0,
) -> dict[str, Any]:
    payload_list = list(payloads)
    selected_by_id = {
        str(row["state_occurrence_id"]): row for row in selected
    }
    integrity_ok = error_count == 0
    state_results = []
    all_trial_ids: list[str] = []
    observed_actions = 0
    observed_trials = 0
    unique_candidate_labels: list[bool] = []
    contradictory_family_head_count = 0
    for payload in payload_list:
        state_id = str(payload.get("state_occurrence_id"))
        state_row = selected_by_id.get(state_id)
        if state_row is None or not _state_payload_valid(
            payload,
            identity=identity,
            state_row=state_row,
            complete=True,
        ):
            integrity_ok = False
            continue
        actions = {
            str(action["action_id"]): dict(action)
            for action in state_row["unique_actions"]
        }
        trials = list(payload["trials"])
        observed_actions += len(actions)
        observed_trials += len(trials)
        all_trial_ids.extend(str(row["trial_identity"]) for row in trials)
        by_action = {
            action_id: [
                row for row in trials if str(row["action_id"]) == action_id
            ]
            for action_id in actions
        }
        role_map = dict(state_row["role_to_action_id"])
        anchor_id = str(role_map["v2_anchor"])
        structural_ids = []
        for role in ("component16", "hotspot16"):
            action_id = str(role_map[role])
            if action_id != anchor_id and action_id not in structural_ids:
                structural_ids.append(action_id)
        comparisons = {}
        for action_id in structural_ids:
            comparison = classify_h1_challenger(
                by_action[anchor_id],
                by_action[action_id],
                distinct_from_anchor=True,
                tie_epsilon=float(config["h1_label"]["tie_epsilon"]),
            )
            comparisons[action_id] = comparison
            if comparison.get("label") is not None:
                unique_candidate_labels.append(bool(comparison["label"]))
        role_labels = {}
        for role in ("component16", "hotspot16"):
            action_id = str(role_map[role])
            if action_id == anchor_id:
                role_labels[role] = {
                    "label": None,
                    "label_reason": "duplicate_role_of_anchor",
                    "action_id": action_id,
                }
            else:
                role_labels[role] = {
                    **comparisons[action_id],
                    "action_id": action_id,
                    "shared_action_role_aliases": actions[action_id]["role_aliases"],
                }
        opportunity = any(
            comparison.get("label") is True for comparison in comparisons.values()
        )
        direct = None
        if state_row["hierarchical_stratum"] == "three_unique":
            component_id = str(role_map["component16"])
            hotspot_id = str(role_map["hotspot16"])
            component_over_hotspot = classify_h1_challenger(
                by_action[hotspot_id],
                by_action[component_id],
                distinct_from_anchor=True,
                tie_epsilon=float(config["h1_label"]["tie_epsilon"]),
            )
            hotspot_over_component = classify_h1_challenger(
                by_action[component_id],
                by_action[hotspot_id],
                distinct_from_anchor=True,
                tie_epsilon=float(config["h1_label"]["tie_epsilon"]),
            )
            component_win = component_over_hotspot.get("label") is True
            hotspot_win = hotspot_over_component.get("label") is True
            mutually_exclusive = not (component_win and hotspot_win)
            if not mutually_exclusive:
                contradictory_family_head_count += 1
                integrity_ok = False
            result = (
                "component_win"
                if component_win and not hotspot_win
                else (
                    "hotspot_win"
                    if hotspot_win and not component_win
                    else "ambiguous"
                )
            )
            direct = {
                "result": result,
                "decisive": result in {"component_win", "hotspot_win"},
                "mutually_exclusive": mutually_exclusive,
                "component_over_hotspot": component_over_hotspot,
                "hotspot_over_component": hotspot_over_component,
                "paired_seed_identity": all(
                    int(left["pp_seed"]) == int(right["pp_seed"])
                    for left, right in zip(
                        sorted(
                            by_action[component_id],
                            key=lambda row: int(row["trial_index"]),
                        ),
                        sorted(
                            by_action[hotspot_id],
                            key=lambda row: int(row["trial_index"]),
                        ),
                    )
                ),
            }
        state_results.append(
            {
                "state_occurrence_id": state_id,
                "source_id": str(state_row["source_id"]),
                "map_id": str(state_row["map_id"]),
                "map_family": str(state_row["map_family"]),
                "task_id": str(state_row["task_id"]),
                "solver_seed": int(state_row["solver_seed"]),
                "hierarchical_stratum": str(state_row["hierarchical_stratum"]),
                "unique_action_count": int(state_row["unique_action_count"]),
                "role_to_action_id": role_map,
                "unique_structural_comparisons": comparisons,
                "role_labels": role_labels,
                "opportunity": opportunity,
                "component_vs_hotspot": direct,
            }
        )
    if len(all_trial_ids) != len(set(all_trial_ids)):
        integrity_ok = False
    if set(selected_by_id) != {
        str(row["state_occurrence_id"]) for row in state_results
    }:
        integrity_ok = False

    opportunities = [row for row in state_results if row["opportunity"]]
    opportunity_maps = {row["map_id"] for row in opportunities}
    opportunity_families = {row["map_family"] for row in opportunities}
    by_map_opportunity = Counter(row["map_id"] for row in opportunities)
    by_map_nonopportunity = Counter(
        row["map_id"] for row in state_results if not row["opportunity"]
    )
    stage1_fold_gates = {}
    for fold, map_ids in FOLDS.items():
        has_opportunity = any(by_map_opportunity[map_id] for map_id in map_ids)
        has_nonopportunity = any(
            by_map_nonopportunity[map_id] for map_id in map_ids
        )
        stage1_fold_gates[fold] = {
            "has_opportunity": has_opportunity,
            "has_nonopportunity": has_nonopportunity,
        }
    stage1_gates = {
        "exact_96_states": len(state_results) == 96,
        "exact_256_unique_actions": observed_actions == 256,
        "exact_4096_unique_trials": (
            observed_trials == 4096
            and len(all_trial_ids) == len(set(all_trial_ids))
        ),
        "all_integrity_gates": integrity_ok,
        "minimum_20_opportunity_states": len(opportunities) >= 20,
        "minimum_8_opportunity_maps": len(opportunity_maps) >= 8,
        "minimum_4_opportunity_families": len(opportunity_families) >= 4,
        "each_fold_has_opportunity_and_nonopportunity": all(
            values["has_opportunity"] and values["has_nonopportunity"]
            for values in stage1_fold_gates.values()
        ),
        "unique_candidate_labels_have_both_classes": (
            any(unique_candidate_labels)
            and any(not value for value in unique_candidate_labels)
        ),
    }
    stage1_passed = all(stage1_gates.values())

    distinct_rows = [
        row for row in state_results if row["hierarchical_stratum"] == "three_unique"
    ]
    decisive = [
        row
        for row in distinct_rows
        if dict(row.get("component_vs_hotspot") or {}).get("decisive") is True
    ]
    support_maps = {row["map_id"] for row in distinct_rows}
    support_families = {row["map_family"] for row in distinct_rows}
    decisive_maps = {row["map_id"] for row in decisive}
    decisive_families = {row["map_family"] for row in decisive}
    direction_counts = Counter(
        row["component_vs_hotspot"]["result"] for row in distinct_rows
    )
    stage2_fold_counts = {
        fold: sum(row["map_id"] in map_ids for row in distinct_rows)
        for fold, map_ids in FOLDS.items()
    }
    stage2_fold_decisive = {
        fold: any(row["map_id"] in map_ids for row in decisive)
        for fold, map_ids in FOLDS.items()
    }
    stage2_gates = {
        "exact_64_three_unique_states": len(distinct_rows) == 64,
        "exact_16_three_unique_states_per_fold": all(
            count == 16 for count in stage2_fold_counts.values()
        ),
        "paired_seed_integrity": all(
            row["component_vs_hotspot"]["paired_seed_identity"]
            for row in distinct_rows
        ),
        "mutually_exclusive_family_head_directions": (
            contradictory_family_head_count == 0
        ),
        "all_7_support_maps_represented": len(support_maps) == 7,
        "all_3_support_families_represented": len(support_families) == 3,
        "minimum_20_decisive_states": len(decisive) >= 20,
        "minimum_6_decisive_support_maps": len(decisive_maps) >= 6,
        "all_3_decisive_support_families": len(decisive_families) == 3,
        "each_fold_has_decisive_state": all(stage2_fold_decisive.values()),
        "both_component_and_hotspot_win_directions_present": (
            direction_counts["component_win"] > 0
            and direction_counts["hotspot_win"] > 0
        ),
    }
    stage2_raw_passed = integrity_ok and all(stage2_gates.values())
    stage2_interpretable = stage1_passed
    stage2_readiness_passed = stage1_passed and stage2_raw_passed
    state_results.sort(key=lambda row: row["state_occurrence_id"])
    return {
        "schema": H1_REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "complete": integrity_ok and len(state_results) == 96,
        "passed": stage1_passed,
        "state_count": len(state_results),
        "observed_unique_action_count": observed_actions,
        "observed_trial_count": observed_trials,
        "stage1_structural_vs_v2": {
            "passed": stage1_passed,
            "opportunity_state_count": len(opportunities),
            "opportunity_map_count": len(opportunity_maps),
            "opportunity_family_count": len(opportunity_families),
            "folds": stage1_fold_gates,
            "gates": stage1_gates,
        },
        "stage2_component_vs_hotspot": {
            "claim_boundary": "conditional_on_observed_C_ne_H_support",
            "interpretable_only_if_stage1_passes": True,
            "interpretable": stage2_interpretable,
            "raw_gates_passed": stage2_raw_passed,
            "readiness_passed": stage2_readiness_passed,
            "three_unique_state_count": len(distinct_rows),
            "support_map_count": len(support_maps),
            "support_family_count": len(support_families),
            "decisive_state_count": len(decisive),
            "decisive_map_count": len(decisive_maps),
            "decisive_family_count": len(decisive_families),
            "direction_counts": dict(sorted(direction_counts.items())),
            "contradictory_direction_state_count": (
                contradictory_family_head_count
            ),
            "fold_state_counts": stage2_fold_counts,
            "fold_has_decisive": stage2_fold_decisive,
            "gates": stage2_gates,
            "training_or_runtime_authorized": False,
        },
        "state_results": state_results,
        "h8_executed": False,
        "training_authorized": False,
        "runtime_or_ttf_claim_authorized": False,
        "runtime_or_pp_seconds_used_in_label": False,
    }


def run_analysis(
    config_path: str | Path,
    output: str | Path,
    *,
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
    if dry_run:
        return {
            "schema": H1_REPORT_SCHEMA,
            "status": "dry_run",
            "expected_state_count": 96,
            "expected_unique_action_count": 256,
            "expected_trial_count": 4096,
            "expected_stage2_three_unique_state_count": 64,
            "selection_source": selection_source,
            "analysis_executed": False,
            "training_or_runtime_or_ttf_executed": False,
        }
    collection_path = output_root / "h1_collection_report.json"
    manifest_path = output_root / "h1_state_manifest.jsonl"
    if not collection_path.is_file() or not manifest_path.is_file():
        raise ValueError("unique-action analysis requires complete H1 collection")
    collection = _read_json(collection_path)
    manifest = _read_jsonl(manifest_path)
    expected_identity = _h1_run_identity(
        config_sha256=sha256_file(path),
        selection_identity=selection_identity,
        h1=dict(config["h1"]),
    )
    identity = str(collection.get("run_identity"))
    if (
        collection.get("schema") != H1_COLLECTION_SCHEMA
        or collection.get("experiment_id") != EXPERIMENT_ID
        or collection.get("complete") is not True
        or collection.get("status") != "complete"
        or collection.get("config_sha256") != sha256_file(path)
        or collection.get("selection_identity") != selection_identity
        or identity != expected_identity
        or int(collection.get("observed_valid_state_count", -1)) != 96
        or int(collection.get("observed_unique_action_count", -1)) != 256
        or int(collection.get("observed_trial_count", -1)) != 4096
        or len(manifest) != 96
    ):
        raise ValueError("unique-action H1 collection trust chain changed")
    selected_by_id = {
        str(row["state_occurrence_id"]): row for row in selected
    }
    payloads = []
    errors = 0
    seen_ids = set()
    for row in manifest:
        state_id = str(row.get("state_occurrence_id"))
        state_row = selected_by_id.get(state_id)
        try:
            state_path = contained_file(
                output_root, row.get("state_file"), field="unique-action H1 state"
            )
            payload = _read_json(state_path)
            source_snapshot = (
                _validated_source_snapshot(state_row)
                if state_row is not None
                else None
            )
        except (OSError, RuntimeError, ValueError):
            errors += 1
            continue
        if (
            state_id in seen_ids
            or state_row is None
            or sha256_file(state_path) != str(row.get("state_sha256"))
            or int(row.get("unique_action_count", -1))
            != int(state_row.get("unique_action_count", -2))
            or int(row.get("trial_count", -1)) != len(payload.get("trials", []))
            or not _state_payload_valid(
                payload,
                identity=identity,
                state_row=state_row,
                complete=True,
                source_snapshot=source_snapshot,
            )
        ):
            errors += 1
            continue
        seen_ids.add(state_id)
        payloads.append(payload)
    report = analyze_payloads(
        config,
        payloads,
        identity=identity,
        selected=selected,
        error_count=errors,
    )
    report.update(
        {
            "config_sha256": sha256_file(path),
            "selection_identity": selection_identity,
            "h1_collection_report_sha256": sha256_file(collection_path),
            "h1_state_manifest_sha256": sha256_file(manifest_path),
        }
    )
    _atomic_json(output_root / "h1_analysis_report.json", report)
    _write_jsonl(output_root / "h1_state_labels.jsonl", report["state_results"])
    return report
