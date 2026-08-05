from __future__ import annotations

import math
import statistics
from pathlib import Path
from typing import Any

from experiments._common import sha256_file
from experiments.repair_collection import (
    _fingerprint,
    _plain,
    _read_json,
    _read_jsonl,
    _write_json,
    _write_jsonl,
    state_fingerprint,
)
from experiments.state_analysis import analyze_state
from experiments.stride_collection import (
    FULL_POOL_PROPOSAL,
    _paired_action,
    _replay_job,
    _validate_native_repair,
)
from experiments.stride_repairability_collection import (
    _artifact_valid as repairability_artifact_valid,
    _source_target_state,
    repairability_pp_seed,
    repairability_restore_seed,
)
from experiments.trace_replay import restore_repair_state
from lns2_selector.runtime.fingerprints import repair_structure_fingerprint
from lns2_selector.runtime.online_selection import generate_online_candidates
from lns2_selector.runtime.topology_candidates import (
    generate_structpool_candidates,
    generate_topology_boundary_candidates,
    merge_structpool_candidates,
    merge_topology_anchor_candidates,
)


CONFIG_SCHEMA = "lns2.stride.structpool_headroom_config.v1"
STATE_SCHEMA = "lns2.stride.structpool_headroom_state.v1"
TRIAL_SCHEMA = "lns2.stride.structpool_headroom_trial.v1"
REPORT_SCHEMA = "lns2.stride.structpool_headroom_report.v1"


def _mean(values: list[float]) -> float:
    return math.fsum(values) / len(values) if values else 0.0


def _registered_path(project_root: Path, spec: dict[str, Any]) -> Path:
    path = (project_root / str(spec["path"])).resolve()
    if not path.is_file() or sha256_file(path) != str(spec["sha256"]):
        raise ValueError(f"StructPool headroom input changed: {spec['path']}")
    return path


def validate_structpool_headroom_config(config: dict[str, Any]) -> None:
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unexpected StructPool headroom config")
    if (
        config.get("scientific_status") != "paired_current_step_quality_pilot_no_timing"
        or config.get("pilot_id") != "stride-structpool-headroom-v1"
        or config.get("candidate_generator_id") != "stride-structpool-v1"
        or config.get("incumbent_pool_id") != "v2-plus-stride-topoboundary-v1"
        or bool(config.get("formal_speed_claim"))
        or bool(config.get("default_replacement_allowed"))
        or bool(config.get("runtime_export_allowed"))
        or bool(config.get("formal_ood_allowed"))
        or bool(config.get("controller_actions_allowed"))
        or bool(config.get("future_trajectory_allowed"))
        or bool(config.get("timing_fields_allowed"))
    ):
        raise ValueError("StructPool headroom evidence boundary changed")
    if list(config.get("implementation_amendments") or ()) != [
        {
            "parent_commit": "5b09f15",
            "reason": (
                "path_overlap_candidate_must_touch_current_conflict_graph_for_"
                "native_explicit_action"
            ),
            "change": (
                "reserve_one_currently_conflicting_agent_as_path_overlap_anchor_"
                "and_add_proposal_legality_gate"
            ),
            "completed_scientific_state_count_before_stop": 0,
            "repair_quality_outcomes_used_for_change": False,
            "cohort_label_and_thresholds_changed": False,
        }
    ]:
        raise ValueError("StructPool headroom implementation amendment changed")
    if set(config.get("inputs") or {}) != {
        "design",
        "coverage_report",
        "coverage_states",
        "repairability_run_config",
        "repairability_report",
        "state_selection",
    }:
        raise ValueError("StructPool headroom input registry changed")
    if dict(config.get("selection") or {}) != {
        "rule": "one_per_map_with_alternating_source_policy_then_map_round_robin",
        "expected_state_count": 16,
        "minimum_distinct_map_count": 13,
        "required_source_policies": ["official_adaptive", "v2-full"],
        "selection_fields": ["state_id", "map_id", "source_policy"],
        "repair_outcome_used_for_selection": False,
    }:
        raise ValueError("StructPool headroom outcome-blind cohort changed")
    if list(config.get("trial_indices") or ()) != list(range(16)):
        raise ValueError("StructPool headroom requires trial indices 0..15")
    if dict(config.get("label") or {}) != {
        "id": "mean_current_step_normalized_conflict_reduction",
        "formula": "mean_seed((conflicts_before-conflicts_after)/max(conflicts_before,1))",
        "aggregation_unit": "candidate_within_state",
        "paired_seed_rule": "same_repair_state_fingerprint_and_trial_index",
        "single_seed_winner_used": False,
        "timing_used": False,
        "future_state_used": False,
    }:
        raise ValueError("StructPool headroom label changed")
    if dict(config.get("candidate_space") or {}) != {
        "neighborhood_sizes": [8, 16, 24, 32],
        "maximum_added_candidates": 6,
        "maximum_jaccard_similarity": 0.8,
        "minimum_novel_candidates_per_state": 4,
    }:
        raise ValueError("StructPool headroom candidate space changed")
    if dict(config.get("gates") or {}) != {
        "minimum_state_fraction_with_novel_expected_gain": 0.2,
        "minimum_mean_best_expected_gain_over_incumbent_pool": 0.01,
        "minimum_gain_for_state_opportunity": 0.02,
        "require_complete_16_seed_product": True,
        "require_exact_incumbent_artifact_reproduction": True,
        "require_paired_seed_identity": True,
        "require_zero_errors": True,
    }:
        raise ValueError("StructPool headroom gates changed")


def select_headroom_states(
    coverage_rows: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    """Freeze a 16-state map/policy-balanced cohort before reading outcomes."""

    by_map: dict[str, dict[str, list[dict[str, Any]]]] = {}
    required = list(config["selection"]["required_source_policies"])
    for row in coverage_rows:
        by_map.setdefault(str(row["map_id"]), {}).setdefault(
            str(row["source_policy"]), []
        ).append(row)
    for policies in by_map.values():
        for values in policies.values():
            values.sort(key=lambda row: str(row["state_id"]))
    selected: list[dict[str, Any]] = []
    maps = sorted(by_map)
    round_index = 0
    expected = int(config["selection"]["expected_state_count"])
    while len(selected) < expected:
        progressed = False
        for map_index, map_id in enumerate(maps):
            if len(selected) >= expected:
                break
            preferred = required[(map_index + round_index) % len(required)]
            fallback = required[(map_index + round_index + 1) % len(required)]
            choices = by_map[map_id]
            policy = preferred if choices.get(preferred) else fallback
            if choices.get(policy):
                selected.append(choices[policy].pop(0))
                progressed = True
        if not progressed:
            break
        round_index += 1
    if len(selected) != expected:
        raise ValueError("StructPool headroom source has insufficient states")
    if len({str(row["map_id"]) for row in selected}) < int(
        config["selection"]["minimum_distinct_map_count"]
    ):
        raise ValueError("StructPool headroom cohort has insufficient maps")
    if {str(row["source_policy"]) for row in selected} != set(required):
        raise ValueError("StructPool headroom cohort has insufficient source policies")
    return selected


def _candidate_signature(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["candidate_id"]),
        tuple(sorted(map(int, row["agents"]))),
        tuple(sorted(map(str, row.get("selection_families") or ()))),
    )


def _aggregate_candidate(
    candidate: dict[str, Any], trials: list[dict[str, Any]], *, before_conflicts: int,
    candidate_kind: str,
) -> dict[str, Any]:
    values = [
        (before_conflicts - int(row["conflicts_after"])) / max(before_conflicts, 1)
        for row in sorted(trials, key=lambda row: int(row["trial_index"]))
    ]
    lower_half = sorted(values)[: len(values) // 2]
    return {
        "candidate_id": str(candidate["candidate_id"]),
        "candidate_kind": candidate_kind,
        "agents": sorted(map(int, candidate["agents"])),
        "actual_size": len(candidate["agents"]),
        "selection_families": sorted(
            map(str, candidate.get("selection_families") or ())
        ),
        "structpool_family_groups": sorted(
            map(str, candidate.get("structpool_family_groups") or ())
        ),
        "trial_count": len(values),
        "mean_normalized_conflict_reduction": _mean(values),
        "standard_deviation_normalized_conflict_reduction": (
            statistics.pstdev(values) if len(values) > 1 else 0.0
        ),
        "lower_half_mean_normalized_conflict_reduction": _mean(lower_half),
        "minimum_normalized_conflict_reduction": min(values, default=0.0),
        "maximum_normalized_conflict_reduction": max(values, default=0.0),
    }


def _state_output_valid(payload: dict[str, Any], config: dict[str, Any]) -> bool:
    trial_indices = set(map(int, config["trial_indices"]))
    trials = list(payload.get("novel_trials") or ())
    aggregates = list(payload.get("candidate_aggregates") or ())
    novel_ids = {
        str(row["candidate_id"])
        for row in aggregates
        if row.get("candidate_kind") == "novel"
    }
    return (
        payload.get("schema") == STATE_SCHEMA
        and payload.get("complete") is True
        and len(novel_ids) >= int(
            config["candidate_space"]["minimum_novel_candidates_per_state"]
        )
        and {
            (str(row.get("candidate_id")), int(row.get("trial_index", -1)))
            for row in trials
        }
        == {(candidate_id, trial_index) for candidate_id in novel_ids for trial_index in trial_indices}
        and all("native_step_seconds" not in row for row in trials)
    )


def collect_structpool_headroom(
    config_path: str | Path, output: str | Path, *, resume: bool = False
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    project_root = config_path.parents[1]
    config = _read_json(config_path)
    validate_structpool_headroom_config(config)
    inputs = {
        name: _registered_path(project_root, dict(spec))
        for name, spec in dict(config["inputs"]).items()
    }
    coverage_report = _read_json(inputs["coverage_report"])
    if coverage_report.get("passed") is not True:
        raise ValueError("StructPool headroom requires passed proposal coverage")
    selected_coverage = select_headroom_states(
        _read_jsonl(inputs["coverage_states"]), config
    )
    decisions = {
        str(row["state_id"]): row for row in _read_jsonl(inputs["state_selection"])
    }
    repair_run = _read_json(inputs["repairability_run_config"])
    repair_report = _read_json(inputs["repairability_report"])
    if (
        repair_report.get("complete") is not True
        or int(repair_report.get("error_state_count", -1)) != 0
        or repair_report.get("run_fingerprint") != repair_run.get("run_fingerprint")
    ):
        raise ValueError("StructPool incumbent repair collection is incomplete")

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    state_output_root = output / "states"
    state_output_root.mkdir(parents=True, exist_ok=True)
    old_state_root = (project_root / str(config["state_artifact_root"])).resolve()
    state_payloads: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for selected in selected_coverage:
        state_id = str(selected["state_id"])
        decision = decisions.get(state_id)
        if decision is None:
            raise ValueError(f"StructPool headroom state decision is missing: {state_id}")
        key = _fingerprint(
            {"state_id": state_id, "before": decision["before_fingerprint"]}
        )[:20]
        output_path = state_output_root / f"{key}.json"
        if resume and output_path.is_file():
            existing = _read_json(output_path)
            if _state_output_valid(existing, config):
                state_payloads.append(existing)
                continue
            raise ValueError(f"invalid resumable StructPool headroom state: {state_id}")
        try:
            old_path = old_state_root / f"{key}.json"
            old = _read_json(old_path)
            if not repairability_artifact_valid(
                old,
                run_fingerprint=str(repair_run["run_fingerprint"]),
                state_id=state_id,
                trial_indices=tuple(map(int, config["trial_indices"])),
                decision=decision,
            ):
                raise RuntimeError("incumbent state artifact failed semantic validation")

            replay = _replay_job(decision)
            state, _manifest, _trace = _source_target_state(decision)
            before_fingerprint = state_fingerprint(state)
            repair_fingerprint = repair_structure_fingerprint(state)
            restore_seed = repairability_restore_seed(repair_fingerprint)
            environment, restored = restore_repair_state(
                replay, state, seed=restore_seed
            )
            if (
                before_fingerprint != str(decision["before_fingerprint"])
                or repair_structure_fingerprint(restored) != repair_fingerprint
            ):
                raise RuntimeError("StructPool headroom state restore changed")
            before_conflicts = int(state["num_of_colliding_pairs"])
            proposal = {**dict(replay["proposal"]), **FULL_POOL_PROPOSAL}
            proposal.pop("topology_boundary", None)
            base, _generation = generate_online_candidates(
                environment,
                state,
                task_id=str(decision["task_id"]),
                solver_seed=int(decision["solver_seed"]),
                decision_index=int(decision["decision_index"]),
                proposal_config=proposal,
                state_hash=before_fingerprint,
                verify_full_state=False,
                proposal_backend="optimized",
                shadow_validation=False,
            )
            analysis = analyze_state(state)
            old_boundaries = generate_topology_boundary_candidates(
                state, analysis, neighborhood_size=16, core_budget=4
            )
            reproduced_incumbent = merge_topology_anchor_candidates(
                base, old_boundaries
            )
            old_candidates = list(old["candidates"])
            if sorted(map(_candidate_signature, reproduced_incumbent)) != sorted(
                map(_candidate_signature, old_candidates)
            ):
                raise RuntimeError("incumbent candidate artifact did not reproduce")

            additions = generate_structpool_candidates(state, analysis)
            full_pool = merge_structpool_candidates(base, additions)
            incumbent_ids = {str(row["candidate_id"]) for row in old_candidates}
            novel = [
                row for row in full_pool if str(row["candidate_id"]) not in incumbent_ids
            ]
            if len(novel) < int(
                config["candidate_space"]["minimum_novel_candidates_per_state"]
            ):
                raise RuntimeError("StructPool headroom state has too few novel actions")

            old_trials_by_candidate: dict[str, list[dict[str, Any]]] = {}
            for row in old["trials"]:
                old_trials_by_candidate.setdefault(str(row["candidate_id"]), []).append(row)
            candidate_aggregates = [
                _aggregate_candidate(
                    candidate,
                    old_trials_by_candidate[str(candidate["candidate_id"])],
                    before_conflicts=before_conflicts,
                    candidate_kind="incumbent",
                )
                for candidate in old_candidates
            ]

            novel_trials: list[dict[str, Any]] = []
            for candidate in novel:
                agents = sorted(map(int, candidate["agents"]))
                candidate_id = str(candidate["candidate_id"])
                candidate_trials = []
                for trial_index in map(int, config["trial_indices"]):
                    branch_environment, branch_state = restore_repair_state(
                        replay, state, seed=restore_seed
                    )
                    if repair_structure_fingerprint(branch_state) != repair_fingerprint:
                        raise RuntimeError("paired StructPool branch restore changed")
                    pp_seed = repairability_pp_seed(repair_fingerprint, trial_index)
                    result = _plain(
                        branch_environment.step(_paired_action(agents, pp_seed))
                    )
                    after, _metrics = _validate_native_repair(
                        result, expected_agents=agents, expected_seed=pp_seed
                    )
                    trial = {
                        "schema": TRIAL_SCHEMA,
                        "state_id": state_id,
                        "candidate_id": candidate_id,
                        "trial_index": trial_index,
                        "pp_seed": pp_seed,
                        "before_conflicts": before_conflicts,
                        "conflicts_after": int(after["num_of_colliding_pairs"]),
                        "replan_success": bool(result["metrics"]["replan_success"]),
                    }
                    candidate_trials.append(trial)
                    novel_trials.append(trial)
                candidate_aggregates.append(
                    _aggregate_candidate(
                        candidate,
                        candidate_trials,
                        before_conflicts=before_conflicts,
                        candidate_kind="novel",
                    )
                )

            incumbent_best = max(
                row["mean_normalized_conflict_reduction"]
                for row in candidate_aggregates
                if row["candidate_kind"] == "incumbent"
            )
            novel_best = max(
                row["mean_normalized_conflict_reduction"]
                for row in candidate_aggregates
                if row["candidate_kind"] == "novel"
            )
            best_gain = max(0.0, novel_best - incumbent_best)
            payload = {
                "schema": STATE_SCHEMA,
                "complete": True,
                "state_id": state_id,
                "map_id": str(decision["map_id"]),
                "source_policy": str(decision["source_policy"]),
                "solver_seed": int(decision["solver_seed"]),
                "before_conflicts": before_conflicts,
                "before_fingerprint": before_fingerprint,
                "before_repair_fingerprint": repair_fingerprint,
                "restore_seed": restore_seed,
                "incumbent_reproduced": True,
                "paired_seed_identity": True,
                "incumbent_candidate_count": len(old_candidates),
                "novel_candidate_count": len(novel),
                "candidate_aggregates": candidate_aggregates,
                "novel_trials": novel_trials,
                "incumbent_best_expected_score": incumbent_best,
                "novel_best_expected_score": novel_best,
                "best_expected_gain_over_incumbent": best_gain,
                "timing_fields_stored": False,
                "future_trajectory_stored": False,
            }
            if not _state_output_valid(payload, config):
                raise RuntimeError("StructPool headroom state product is incomplete")
            _write_json(output_path, payload)
            state_payloads.append(payload)
        except Exception as exc:
            errors.append({"state_id": state_id, "error": str(exc)})
            break
        finally:
            _write_json(
                output / "collection_status.json",
                {
                    "schema": REPORT_SCHEMA,
                    "requested_state_count": len(selected_coverage),
                    "completed_state_count": len(state_payloads),
                    "error_state_count": len(errors),
                    "status": "failed" if errors else "running",
                    "errors": errors,
                },
            )

    if errors:
        raise RuntimeError(errors[0]["error"])
    gains = [float(row["best_expected_gain_over_incumbent"]) for row in state_payloads]
    opportunity_threshold = float(config["gates"]["minimum_gain_for_state_opportunity"])
    opportunity_fraction = (
        sum(gain >= opportunity_threshold for gain in gains) / len(gains)
        if gains
        else 0.0
    )
    gates = {
        "expected_state_count": len(state_payloads)
        == int(config["selection"]["expected_state_count"]),
        "minimum_state_fraction_with_novel_expected_gain": opportunity_fraction
        >= float(config["gates"]["minimum_state_fraction_with_novel_expected_gain"]),
        "minimum_mean_best_expected_gain_over_incumbent_pool": _mean(gains)
        >= float(config["gates"]["minimum_mean_best_expected_gain_over_incumbent_pool"]),
        "complete_16_seed_product": all(_state_output_valid(row, config) for row in state_payloads),
        "exact_incumbent_artifact_reproduction": all(
            bool(row["incumbent_reproduced"]) for row in state_payloads
        ),
        "paired_seed_identity": all(bool(row["paired_seed_identity"]) for row in state_payloads),
        "zero_errors": not errors,
    }
    passed = all(gates.values())
    trial_rows = [row for state in state_payloads for row in state["novel_trials"]]
    aggregate_rows = [
        {"state_id": state["state_id"], **row}
        for state in state_payloads
        for row in state["candidate_aggregates"]
    ]
    trials_path = output / "novel_trials.jsonl"
    aggregates_path = output / "candidate_aggregates.jsonl"
    _write_jsonl(trials_path, trial_rows)
    _write_jsonl(aggregates_path, aggregate_rows)
    report = {
        "schema": REPORT_SCHEMA,
        "scientific_status": "paired_current_step_quality_pilot_no_timing",
        "state_count": len(state_payloads),
        "map_count": len({str(row["map_id"]) for row in state_payloads}),
        "novel_trial_count": len(trial_rows),
        "mean_best_expected_gain_over_incumbent_pool": _mean(gains),
        "state_fraction_with_novel_expected_gain": opportunity_fraction,
        "state_opportunity_count": sum(
            gain >= opportunity_threshold for gain in gains
        ),
        "minimum_state_gain": min(gains, default=0.0),
        "maximum_state_gain": max(gains, default=0.0),
        "candidate_repair_timing_used": False,
        "controller_actions_executed": False,
        "future_trajectory_read": False,
        "formal_speed_claim": False,
        "errors": errors,
        "gates": gates,
        "passed": passed,
        "next_decision": config[
            "next_decision_on_pass" if passed else "next_decision_on_failure"
        ],
        "inputs": {
            "config_sha256": sha256_file(config_path),
            **{
                f"{name}_sha256": sha256_file(path)
                for name, path in sorted(inputs.items())
            },
        },
        "artifacts": {
            "novel_trials_sha256": sha256_file(trials_path),
            "candidate_aggregates_sha256": sha256_file(aggregates_path),
        },
    }
    _write_json(output / "structpool_headroom_report.json", report)
    _write_json(
        output / "collection_status.json",
        {
            "schema": REPORT_SCHEMA,
            "requested_state_count": len(selected_coverage),
            "completed_state_count": len(state_payloads),
            "error_state_count": 0,
            "status": "complete",
            "passed": passed,
            "errors": [],
        },
    )
    return report


__all__ = [
    "analyze_structpool_headroom",
    "collect_structpool_headroom",
    "select_headroom_states",
    "validate_structpool_headroom_config",
]


def analyze_structpool_headroom(
    config: dict[str, Any], state_payloads: list[dict[str, Any]]
) -> dict[str, Any]:
    """Small pure helper retained for unit-level gate checks."""

    gains = [float(row["best_expected_gain_over_incumbent"]) for row in state_payloads]
    threshold = float(config["gates"]["minimum_gain_for_state_opportunity"])
    fraction = sum(value >= threshold for value in gains) / len(gains) if gains else 0.0
    gates = {
        "expected_state_count": len(state_payloads)
        == int(config["selection"]["expected_state_count"]),
        "minimum_state_fraction_with_novel_expected_gain": fraction
        >= float(config["gates"]["minimum_state_fraction_with_novel_expected_gain"]),
        "minimum_mean_best_expected_gain_over_incumbent_pool": _mean(gains)
        >= float(config["gates"]["minimum_mean_best_expected_gain_over_incumbent_pool"]),
    }
    return {
        "mean_gain": _mean(gains),
        "opportunity_fraction": fraction,
        "gates": gates,
        "passed": all(gates.values()),
    }
