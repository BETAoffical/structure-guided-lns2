from __future__ import annotations

import collections
import math
from pathlib import Path
from typing import Any, Iterable

from experiments._common import closed_loop_producer_identity, registered_input, sha256_file
from experiments.closed_loop_confirmation import run_closed_loop_collection
from experiments.closed_loop_trace_storage import read_trace_events
from experiments.repair_collection import _fingerprint, _read_json, _read_jsonl, _write_json
from experiments.run_output_guard import load_completed_report, prepare_resumable_output
from experiments.stride_collection import _paired_action
from experiments.stride_maze_tail_state_collection import (
    _fused_controller_kwargs,
    load_maze_tail_state_collection_config,
)
from experiments.stride_repairability_collection import repairability_restore_seed
from experiments.trace_replay import target_state_from_trace
from lns2_selector.runtime.fingerprints import repair_structure_fingerprint


CONFIG_SCHEMA = "lns2.stride.tailswitch_registration.v1"
STATUS_SCHEMA = "lns2.stride.tailswitch_status.v1"
REPORT_SCHEMA = "lns2.stride.tailswitch_report.v1"
OVERRIDE_SCHEMA = "lns2.stride.tailswitch_episode_override.v1"
POLICIES = (
    "v2-then-v2",
    "struct-then-v2",
    "v2-then-struct",
    "struct-then-struct",
)
STATUS_FILENAME = "tailswitch_status.json"
REPORT_FILENAME = "tailswitch_report.json"


def _mean(values: Iterable[float]) -> float:
    rows = list(values)
    return sum(rows) / len(rows) if rows else 0.0


def _registered(root: Path, specification: dict[str, Any]) -> Path:
    return registered_input(root, specification, label="TailSwitch")


def load_tailswitch_config(
    path: str | Path,
) -> tuple[Path, Path, dict[str, Any], dict[str, Path], dict[str, Any]]:
    path = Path(path).resolve()
    root = path.parent.parent
    config = _read_json(path)
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("scientific_status")
        != "preregistered_two_by_two_first_action_and_continuation_causal_diagnostic"
        or config.get("experiment_id") != "stride-tailswitch-v1"
        or config.get("pre_registration_parent_commit")
        != "62ca3f4bef4a51d55e687f725916f70a1687c733"
    ):
        raise ValueError("TailSwitch registration identity changed")
    expected_inputs = {
        "state_selection",
        "action_replay_report",
        "state_collection_config",
        "runtime_config",
        "source_report",
        "v2_manifest",
        "structpool_manifest",
        "slotpool_manifest",
    }
    if set(config.get("inputs") or {}) != expected_inputs:
        raise ValueError("TailSwitch input registry changed")
    inputs = {
        name: _registered(root, dict(specification))
        for name, specification in dict(config["inputs"]).items()
    }
    parent_path, parent_root, parent = load_maze_tail_state_collection_config(
        inputs["state_collection_config"]
    )
    if parent_root != root or parent_path != inputs["state_collection_config"]:
        raise ValueError("TailSwitch parent configuration root changed")
    if tuple(map(str, config["cohort"]["challengers"])) != (
        "v2-plus-structpool",
        "v2-plus-slotpool",
    ):
        raise ValueError("TailSwitch challengers changed")
    if int(config["cohort"]["state_count"]) != 66:
        raise ValueError("TailSwitch state count changed")
    if tuple(str(value["id"]) for value in config["factorial_policies"]) != POLICIES:
        raise ValueError("TailSwitch factorial policies changed")
    execution = dict(config["execution"])
    if execution != {
        "state_restore_contract": "lns2.trace_replay.target-path-restore.v1",
        "first_action_forced_exactly_once": True,
        "first_action_pp_seed": "source_paired_pp_seed",
        "continuation_begins_after_first_action": True,
        "strict_four_policy_rotation_within_state": True,
        "workers": 1,
        "maximum_repair_decisions_from_restored_state": 200,
        "fixed_metric_horizon": 200,
        "wall_time_fuse_seconds": 300.0,
        "process_timeout_seconds": 360.0,
        "right_censoring_is_valid_state_evidence": True,
        "right_censoring_is_not_an_execution_error": True,
    }:
        raise ValueError("TailSwitch execution contract changed")
    if dict(config["claim_boundary"]) != {
        "causal_policy_switch_diagnostic_only": True,
        "model_training_allowed": False,
        "ttf_improvement_claim": False,
        "generalization_claim": False,
        "default_controller_replacement_allowed": False,
        "no_result_based_exclusion": True,
    }:
        raise ValueError("TailSwitch claim boundary changed")
    action_report = _read_json(inputs["action_replay_report"])
    if (
        action_report.get("action_stability_passed") is not True
        or action_report.get("integrity_passed") is not True
        or int(action_report.get("state_count", -1)) != 66
    ):
        raise ValueError("TailSwitch requires the completed action replay")
    return path, root, config, inputs, parent


def tailswitch_schedule(selection: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for state_position, row in enumerate(selection):
        for policy_position, policy in enumerate(POLICIES):
            rows.append(
                {
                    "state_id": str(row["state_id"]),
                    "task_id": str(row["task_id"]),
                    "solver_seed": int(row["solver_seed"]),
                    "challenger": str(row["challenger"]),
                    "policy": policy,
                    "state_position": state_position,
                    "policy_position": policy_position,
                }
            )
    return rows


def _state_directory(output: Path, state_id: str) -> Path:
    return output / "states" / _fingerprint({"state_id": state_id})[:20]


def _manifest_path(output: Path, item: dict[str, Any]) -> Path:
    return (
        _state_directory(output, str(item["state_id"]))
        / str(item["policy"])
        / "realized_dynamic_manifest.jsonl"
    )


def _producer(root: Path, *, native_required: bool = True) -> dict[str, Any]:
    return closed_loop_producer_identity(
        project_root=root,
        source_files=(
            "experiments/stride_tailswitch.py",
            "experiments/closed_loop_confirmation.py",
            "experiments/trace_replay.py",
            "experiments/stride_maze_tail_state_collection.py",
        ),
        native_required=native_required,
    )


def _source_root(path: Path) -> Path:
    return path.parent


def _episode_override(
    row: dict[str, Any],
    *,
    source_root: Path,
    use_struct_action: bool,
) -> dict[str, Any]:
    source, _trace = target_state_from_trace(
        source_root,
        dict(row["v2_manifest"]),
        decision_index=int(row["decision_index"]),
        expected_fingerprint=str(row["before_fingerprint"]),
    )
    repair_fingerprint = repair_structure_fingerprint(source)
    candidate = dict(row["challenger_action"] if use_struct_action else row["v2_action"])
    pp_seed = int(row["source_paired_pp_seed"])
    return {
        "schema": OVERRIDE_SCHEMA,
        "state_id": str(row["state_id"]),
        "initial_restore": {
            "collection_root": str(source_root),
            "manifest": dict(row["v2_manifest"]),
            "decision_index": int(row["decision_index"]),
            "expected_fingerprint": str(row["before_fingerprint"]),
            "repair_structure_fingerprint": repair_fingerprint,
            "expected_conflicts": int(row["before_conflicts"]),
            "restore_seed": repairability_restore_seed(repair_fingerprint),
        },
        "forced_first_action": _paired_action(
            list(map(int, candidate["agents"])), pp_seed
        ),
        "forced_candidate_id": str(candidate["candidate_id"]),
        "forced_candidate_role": str(candidate["role"]),
        "forced_selection_families": list(
            map(str, candidate.get("selection_families") or ())
        ),
    }


def _controller_for_policy(policy: str, challenger: str) -> str:
    return challenger if policy.endswith("struct") else "v2-full"


def _completed_counts(output: Path, schedule: list[dict[str, Any]]) -> dict[str, int]:
    counts = {policy: 0 for policy in POLICIES}
    for item in schedule:
        path = _manifest_path(output, item)
        if path.is_file() and any(
            row.get("status") in {"ok", "error", "timeout"}
            for row in _read_jsonl(path)
        ):
            counts[str(item["policy"])] += 1
    return counts


def run_tailswitch(
    config_path: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
    limit_states: int | None = None,
) -> dict[str, Any]:
    path, root, config, inputs, parent = load_tailswitch_config(config_path)
    selection = _read_jsonl(inputs["state_selection"])
    if len(selection) != int(config["cohort"]["state_count"]):
        raise ValueError("TailSwitch frozen state selection count changed")
    if len({str(row["state_id"]) for row in selection}) != len(selection):
        raise ValueError("TailSwitch state ids are not unique")
    selected = selection if limit_states is None else selection[: int(limit_states)]
    schedule = tailswitch_schedule(selected)
    if dry_run:
        return {
            "schema": STATUS_SCHEMA,
            "state_count": len(selected),
            "schedule_entry_count": len(schedule),
            "schedule_sha256": _fingerprint(schedule),
        }
    output = Path(output).resolve()
    prepared = prepare_resumable_output(
        output,
        status_filename=STATUS_FILENAME,
        status_schema=STATUS_SCHEMA,
        config_path=path,
        schedule=schedule,
        producer=_producer(root),
        resume=resume,
        report_filename=REPORT_FILENAME,
        report_schema=REPORT_SCHEMA,
        label="TailSwitch",
    )
    if prepared.completed_report is not None:
        return prepared.completed_report
    status_base = prepared.base_status
    dataset = (root / str(parent["cohort"]["dataset"])).resolve()
    runtime = inputs["runtime_config"]
    source_root = _source_root(inputs["v2_manifest"])
    row_by_state = {str(row["state_id"]): row for row in selected}
    registered_job_keys = {
        (str(row["task_id"]), int(row["solver_seed"])) for row in selection
    }
    qualification_root = output / "qualification"
    qualification_kwargs = _fused_controller_kwargs(root, parent, "v2-full")
    run_closed_loop_collection(
        dataset,
        runtime,
        qualification_root,
        phase="qualify",
        workers=1,
        resume=(
            prepared.resumed
            and qualification_root.joinpath("run_config.json").is_file()
        ),
        cohort_job_keys=registered_job_keys,
        job_keys=registered_job_keys,
        **qualification_kwargs,
    )
    overrides: dict[tuple[str, bool], dict[str, Any]] = {}
    completed = 0
    for item in schedule:
        row = row_by_state[str(item["state_id"])]
        use_struct_action = str(item["policy"]).startswith("struct-")
        override_key = (str(item["state_id"]), use_struct_action)
        if override_key not in overrides:
            overrides[override_key] = _episode_override(
                row,
                source_root=source_root,
                use_struct_action=use_struct_action,
            )
        key = (str(item["task_id"]), int(item["solver_seed"]))
        collection = _state_directory(output, str(item["state_id"])) / str(
            item["policy"]
        )
        manifest = collection / "realized_dynamic_manifest.jsonl"
        done = bool(
            manifest.is_file()
            and any(
                value.get("status") in {"ok", "error", "timeout"}
                for value in _read_jsonl(manifest)
            )
        )
        controller = _controller_for_policy(
            str(item["policy"]), str(item["challenger"])
        )
        kwargs = _fused_controller_kwargs(root, parent, controller)
        episode_overrides = {key: overrides[override_key]}
        if not done:
            run_closed_loop_collection(
                dataset,
                runtime,
                collection,
                phase="qualify",
                workers=1,
                resume=(prepared.resumed and collection.joinpath("run_config.json").is_file()),
                cohort_job_keys=registered_job_keys,
                job_keys=registered_job_keys,
                qualification_source=qualification_root,
                episode_overrides=episode_overrides,
                **kwargs,
            )
            run_closed_loop_collection(
                dataset,
                runtime,
                collection,
                phase="realized_dynamic",
                workers=1,
                resume=True,
                cohort_job_keys=registered_job_keys,
                job_keys={key},
                episode_overrides=episode_overrides,
                **kwargs,
            )
        completed += 1
        _write_json(
            output / STATUS_FILENAME,
            {
                **status_base,
                "completed_schedule_entries": completed,
                "total_schedule_entries": len(schedule),
                "policy_completed_episode_counts": _completed_counts(
                    output, schedule
                ),
                "current": item,
                "complete": False,
            },
        )
    report = analyze_tailswitch(path, output, expected_states=len(selected))
    _write_json(
        output / STATUS_FILENAME,
        {
            **status_base,
            "completed_schedule_entries": completed,
            "total_schedule_entries": len(schedule),
            "policy_completed_episode_counts": _completed_counts(output, schedule),
            "complete": True,
            "report_sha256": sha256_file(output / REPORT_FILENAME),
        },
    )
    return report


def _episode_summary(row: dict[str, Any]) -> dict[str, Any]:
    summary = dict(row.get("summary") or {})
    return {
        "status": str(row.get("status")),
        "success": bool(summary.get("success")),
        "stop_reason": str(summary.get("stop_reason") or "execution_error"),
        "repair_iterations": int(summary.get("repair_iterations", 0)),
        "initial_conflicts": int(summary.get("initial_conflicts", 0)),
        "final_conflicts": int(summary.get("final_conflicts", 0)),
        "normalized_fixed_auc": float(
            summary.get("normalized_fixed_budget_conflict_auc", math.inf)
        ),
        "invalid_action_count": int(summary.get("invalid_action_count", 0)),
        "fingerprint_mismatch_count": int(
            summary.get("fingerprint_mismatch_count", 0)
        ),
        "forced_first_action_count": int(
            dict(summary.get("controller_totals") or {}).get(
                "forced_first_action_count", 0
            )
        ),
        "ttf_observed_wall_seconds": float(
            summary.get("ttf_observed_wall_seconds", 0.0)
        ),
    }


def classify_adverse_pair(
    reference: dict[str, Any], treatment: dict[str, Any], rule: dict[str, Any]
) -> dict[str, Any]:
    auc_delta = float(treatment["normalized_fixed_auc"]) - float(
        reference["normalized_fixed_auc"]
    )
    final_delta = int(treatment["final_conflicts"]) - int(
        reference["final_conflicts"]
    )
    if bool(reference["success"]) and not bool(treatment["success"]):
        classification = "adverse"
        reason = "treatment_censored_reference_complete"
    elif not bool(reference["success"]) and bool(treatment["success"]):
        classification = "beneficial"
        reason = "treatment_complete_reference_censored"
    elif (
        auc_delta >= float(rule["normalized_auc_delta_at_least"])
        or final_delta >= int(rule["or_final_conflict_delta_at_least"])
    ):
        classification = "adverse"
        reason = "fixed_horizon_degradation"
    elif (
        auc_delta <= -float(rule["normalized_auc_delta_at_least"])
        or final_delta <= -int(rule["or_final_conflict_delta_at_least"])
    ):
        classification = "beneficial"
        reason = "fixed_horizon_improvement"
    else:
        classification = "neutral"
        reason = "below_registered_effect_threshold"
    return {
        "classification": classification,
        "reason": reason,
        "normalized_auc_delta": auc_delta,
        "final_conflict_delta": final_delta,
    }


def _contrast_gate(
    rows: list[dict[str, Any]], gate: dict[str, Any]
) -> dict[str, Any]:
    adverse = [row for row in rows if row["classification"] == "adverse"]
    by_challenger = collections.Counter(str(row["challenger"]) for row in adverse)
    conditions = {
        "minimum_adverse_fraction": (
            len(adverse) / len(rows) if rows else 0.0
        )
        >= float(gate["minimum_adverse_fraction"]),
        "minimum_adverse_comparison_count": len(adverse)
        >= int(gate["minimum_adverse_comparison_count"]),
        "minimum_map_count": len({str(row["map_id"]) for row in adverse})
        >= int(gate["minimum_map_count"]),
        "minimum_task_count": len({str(row["task_id"]) for row in adverse})
        >= int(gate["minimum_task_count"]),
        "minimum_solver_seed_count": len(
            {int(row["solver_seed"]) for row in adverse}
        )
        >= int(gate["minimum_solver_seed_count"]),
        "minimum_per_challenger_count": all(
            by_challenger[value] >= int(gate["minimum_per_challenger_count"])
            for value in ("v2-plus-structpool", "v2-plus-slotpool")
        ),
    }
    return {
        "passed": all(conditions.values()),
        "conditions": conditions,
        "comparison_count": len(rows),
        "adverse_count": len(adverse),
        "adverse_fraction": len(adverse) / len(rows) if rows else 0.0,
        "beneficial_count": sum(
            row["classification"] == "beneficial" for row in rows
        ),
        "neutral_count": sum(row["classification"] == "neutral" for row in rows),
        "adverse_by_challenger": dict(sorted(by_challenger.items())),
        "mean_normalized_auc_delta": _mean(
            float(row["normalized_auc_delta"]) for row in rows
        ),
        "mean_final_conflict_delta": _mean(
            float(row["final_conflict_delta"]) for row in rows
        ),
    }


def analyze_tailswitch(
    config_path: str | Path,
    output: str | Path,
    *,
    expected_states: int | None = None,
) -> dict[str, Any]:
    path, _root, config, inputs, _parent = load_tailswitch_config(config_path)
    output = Path(output).resolve()
    completed = load_completed_report(
        output,
        status_filename=STATUS_FILENAME,
        status_schema=STATUS_SCHEMA,
        report_filename=REPORT_FILENAME,
        report_schema=REPORT_SCHEMA,
    )
    if completed is not None:
        return completed
    selection = _read_jsonl(inputs["state_selection"])
    if expected_states is not None:
        selection = selection[: int(expected_states)]
    action_report = _read_json(inputs["action_replay_report"])
    replay_by_state = {
        str(row["state_id"]): row for row in action_report["states"]
    }
    errors: list[str] = []
    episodes: dict[str, dict[str, dict[str, Any]]] = {}
    manifest_hashes: dict[str, str] = {}
    for source in selection:
        state_id = str(source["state_id"])
        episodes[state_id] = {}
        for policy in POLICIES:
            item = {"state_id": state_id, "policy": policy}
            manifest_path = _manifest_path(output, item)
            if not manifest_path.is_file():
                errors.append(f"{state_id}/{policy}: missing manifest")
                continue
            manifest_hashes[f"{state_id}/{policy}"] = sha256_file(manifest_path)
            rows = _read_jsonl(manifest_path)
            if len(rows) != 1:
                errors.append(f"{state_id}/{policy}: manifest coverage changed")
                continue
            row = rows[0]
            summary = _episode_summary(row)
            trace_path = manifest_path.parent / str(row.get("trace_file", ""))
            if row.get("status") != "ok" or not trace_path.is_file():
                errors.append(f"{state_id}/{policy}: execution failed")
                episodes[state_id][policy] = summary
                continue
            events = read_trace_events(trace_path)
            transitions = events[1:-1]
            if not transitions:
                errors.append(f"{state_id}/{policy}: no forced transition")
            else:
                expected_action = dict(
                    source[
                        "challenger_action"
                        if policy.startswith("struct-")
                        else "v2_action"
                    ]
                )
                first = transitions[0]
                action = dict(first.get("action") or {})
                controller = dict(first.get("controller") or {})
                if (
                    list(map(int, action.get("agents") or ()))
                    != list(map(int, expected_action["agents"]))
                    or int(action.get("pp_random_seed", -1))
                    != int(source["source_paired_pp_seed"])
                    or controller.get("forced_first_action") is not True
                    or str(controller.get("selected_candidate_id"))
                    != str(expected_action["candidate_id"])
                ):
                    errors.append(f"{state_id}/{policy}: forced action changed")
                if any(
                    dict(event.get("controller") or {}).get("forced_first_action")
                    is True
                    for event in transitions[1:]
                ):
                    errors.append(f"{state_id}/{policy}: first action forced repeatedly")
            if (
                summary["forced_first_action_count"] != 1
                or summary["invalid_action_count"] != 0
                or summary["fingerprint_mismatch_count"] != 0
                or summary["stop_reason"]
                not in {"success", "repair_limit", "wall_timeout"}
                or summary["repair_iterations"] > 200
                or summary["initial_conflicts"] != int(source["before_conflicts"])
            ):
                errors.append(f"{state_id}/{policy}: runtime integrity changed")
            episodes[state_id][policy] = summary
    contrasts: dict[str, list[dict[str, Any]]] = {
        "first_action_under_v2_continuation": [],
        "first_action_under_struct_continuation": [],
        "struct_continuation_after_v2": [],
        "struct_continuation_after_struct": [],
        "sustained_interaction": [],
    }
    rule = dict(config["adverse_pair_rule"])
    complete_states: list[dict[str, Any]] = []
    for source in selection:
        state_id = str(source["state_id"])
        policy_rows = episodes.get(state_id, {})
        if set(policy_rows) != set(POLICIES):
            continue
        metadata = {
            key: source[key]
            for key in (
                "state_id",
                "map_id",
                "task_id",
                "solver_seed",
                "challenger",
                "tail_category",
                "conflict_band",
            )
        }
        metadata["robust_first_action_winner"] = str(
            replay_by_state[state_id]["robust_winner"]
        )
        pairs = {
            "first_action_under_v2_continuation": (
                "v2-then-v2",
                "struct-then-v2",
            ),
            "first_action_under_struct_continuation": (
                "v2-then-struct",
                "struct-then-struct",
            ),
            "struct_continuation_after_v2": (
                "v2-then-v2",
                "v2-then-struct",
            ),
            "struct_continuation_after_struct": (
                "struct-then-v2",
                "struct-then-struct",
            ),
        }
        state_contrasts: dict[str, Any] = {}
        for name, (reference, treatment) in pairs.items():
            result = {
                **metadata,
                **classify_adverse_pair(
                    policy_rows[reference], policy_rows[treatment], rule
                ),
            }
            contrasts[name].append(result)
            state_contrasts[name] = result
        left = state_contrasts["struct_continuation_after_struct"]
        right = state_contrasts["first_action_under_struct_continuation"]
        sustained = {
            **metadata,
            "classification": (
                "adverse"
                if left["classification"] == "adverse"
                and right["classification"] == "adverse"
                else "neutral"
            ),
            "reason": "full_structural_adverse_against_both_switches",
            "normalized_auc_delta": min(
                float(left["normalized_auc_delta"]),
                float(right["normalized_auc_delta"]),
            ),
            "final_conflict_delta": min(
                int(left["final_conflict_delta"]),
                int(right["final_conflict_delta"]),
            ),
        }
        contrasts["sustained_interaction"].append(sustained)
        interaction_auc = (
            policy_rows["struct-then-struct"]["normalized_fixed_auc"]
            - policy_rows["struct-then-v2"]["normalized_fixed_auc"]
            - policy_rows["v2-then-struct"]["normalized_fixed_auc"]
            + policy_rows["v2-then-v2"]["normalized_fixed_auc"]
        )
        complete_states.append(
            {
                **metadata,
                "policies": policy_rows,
                "contrasts": state_contrasts,
                "factorial_interaction_normalized_auc": interaction_auc,
            }
        )
    gate = dict(config["mechanism_evidence_gate"])
    contrast_reports = {
        name: _contrast_gate(rows, gate) for name, rows in contrasts.items()
    }
    passed = [name for name, row in contrast_reports.items() if row["passed"]]
    if not passed:
        conclusion = "inconclusive"
    elif "first_action_under_v2_continuation" in passed:
        conclusion = "first_action_mechanism"
    elif "struct_continuation_after_v2" in passed:
        conclusion = "later_exposure_sufficient"
    elif "struct_continuation_after_struct" in passed:
        conclusion = "continuation_feedback_mechanism"
    elif "sustained_interaction" in passed:
        conclusion = "sustained_interaction"
    else:
        conclusion = "mixed_mechanism"
    policy_summaries = {}
    for policy in POLICIES:
        rows = [state["policies"][policy] for state in complete_states]
        policy_summaries[policy] = {
            "episode_count": len(rows),
            "success_count": sum(bool(row["success"]) for row in rows),
            "repair_limit_count": sum(
                row["stop_reason"] == "repair_limit" for row in rows
            ),
            "wall_timeout_count": sum(
                row["stop_reason"] == "wall_timeout" for row in rows
            ),
            "mean_repair_iterations": _mean(
                float(row["repair_iterations"]) for row in rows
            ),
            "mean_normalized_fixed_auc": _mean(
                float(row["normalized_fixed_auc"]) for row in rows
            ),
            "mean_final_conflicts": _mean(
                float(row["final_conflicts"]) for row in rows
            ),
        }
    integrity = {
        "exact_frozen_state_coverage": len(complete_states) == len(selection),
        "four_policies_per_state": all(
            set(episodes.get(str(row["state_id"]), {})) == set(POLICIES)
            for row in selection
        ),
        "forced_first_action_exactly_once": not any(
            "forced" in error for error in errors
        ),
        "zero_execution_errors": not any(
            "execution failed" in error for error in errors
        ),
        "zero_runtime_integrity_errors": not any(
            "runtime integrity" in error for error in errors
        ),
        "no_result_based_exclusion": len(selection)
        == int(config["cohort"]["state_count"])
        or expected_states is not None,
    }
    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": "stride-tailswitch-v1",
        "scientific_status": (
            "causal_policy_switch_diagnostic_complete"
            if not errors
            else "causal_policy_switch_diagnostic_integrity_failed"
        ),
        "integrity_passed": not errors and all(integrity.values()),
        "integrity_gates": integrity,
        "errors": errors,
        "state_count": len(selection),
        "complete_state_count": len(complete_states),
        "episode_count": sum(len(value) for value in episodes.values()),
        "policy_summaries": policy_summaries,
        "registered_contrast_reports": contrast_reports,
        "passed_mechanism_gates": passed,
        "causal_conclusion": conclusion,
        "mean_factorial_interaction_normalized_auc": _mean(
            float(row["factorial_interaction_normalized_auc"])
            for row in complete_states
        ),
        "states": complete_states,
        "claim_boundary": dict(config["claim_boundary"]),
        "producer_identity": (
            dict(_read_json(output / STATUS_FILENAME).get("producer_identity") or {})
            if (output / STATUS_FILENAME).is_file()
            else None
        ),
        "inputs": {
            "config_sha256": sha256_file(path),
            "state_selection_sha256": sha256_file(inputs["state_selection"]),
            "action_replay_report_sha256": sha256_file(
                inputs["action_replay_report"]
            ),
            "manifest_sha256": manifest_hashes,
        },
    }
    _write_json(output / REPORT_FILENAME, report)
    return report


__all__ = [
    "POLICIES",
    "analyze_tailswitch",
    "classify_adverse_pair",
    "load_tailswitch_config",
    "run_tailswitch",
    "tailswitch_schedule",
]
