from __future__ import annotations

import os
import statistics
from pathlib import Path
from typing import Any, Iterable

from experiments._common import contained_file, sha256_file
from experiments.repair_collection import (
    _fingerprint,
    _plain,
    _read_json,
    _read_jsonl,
    _write_json,
    _write_jsonl,
    state_fingerprint,
)
from experiments.stride_collection import (
    _paired_action,
    _replay_job,
    _validate_native_repair,
)
from experiments.stride_lns import post_structure_metrics
from experiments.stride_stage4r_quick import _mean, _project_path
from experiments.trace_replay import decision_rows, replay_prefix
from lns2_selector.runtime.fingerprints import repair_structure_fingerprint
from lns2_selector.runtime.repair_outcomes import classify_repair_outcome


STRIDE_STAGE4R_PP_REPLAY_REGISTRATION_SCHEMA = (
    "lns2.stride.stage4r_pp_replay_registration.v1"
)
STRIDE_STAGE4R_PP_REPLAY_SELECTION_SCHEMA = (
    "lns2.stride.stage4r_pp_replay_selection.v1"
)
STRIDE_STAGE4R_PP_REPLAY_STATE_SCHEMA = "lns2.stride.stage4r_pp_replay_state.v1"
STRIDE_STAGE4R_PP_REPLAY_REPORT_SCHEMA = "lns2.stride.stage4r_pp_replay_report.v1"
ACTION_LABELS = ("v2-full", "stride-quality-v1")
PP_TRIAL_INDICES = tuple(range(16))


def _checked_file(
    project_root: Path,
    raw_path: str | Path,
    expected_sha256: str,
    label: str,
) -> Path:
    path = _project_path(project_root, raw_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256_file(path) != str(expected_sha256).lower():
        raise ValueError(f"Stage 4R PP replay source hash differs: {label}")
    return path


def validate_pp_replay_config(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("schema") != STRIDE_STAGE4R_PP_REPLAY_REGISTRATION_SCHEMA:
        raise ValueError("unexpected Stage 4R PP replay registration schema")
    if (
        config.get("scientific_status") != "diagnostic_only"
        or bool(config.get("formal_speed_claim"))
        or bool(config.get("formal_ood_data_allowed"))
    ):
        raise ValueError("Stage 4R PP replay must remain diagnostic-only")
    if tuple(config.get("action_labels") or ()) != ACTION_LABELS:
        raise ValueError("Stage 4R PP replay action labels differ")
    if tuple(map(int, config.get("solver_seeds") or ())) != (1, 2, 3, 4):
        raise ValueError("Stage 4R PP replay solver seeds differ")
    if int(config.get("decision_index", -1)) != 0:
        raise ValueError("Stage 4R PP replay must use the first decision")
    if int(config.get("pp_trial_count", 0)) != len(PP_TRIAL_INDICES):
        raise ValueError("Stage 4R PP replay trial count differs")
    if int(config.get("workers", 0)) != 1:
        raise ValueError("Stage 4R PP replay must execute serially")
    if config.get("execution_order") != "paired_seed_alternating_action_order":
        raise ValueError("Stage 4R PP replay execution order differs")
    registered = tuple(map(str, config.get("registered_task_ids") or ()))
    if len(registered) != 4 or len(set(registered)) != 4:
        raise ValueError("Stage 4R PP replay requires four unique tasks")
    sources = dict(config.get("sources") or {})
    if set(sources) != set(ACTION_LABELS):
        raise ValueError("Stage 4R PP replay source registration differs")
    thresholds = dict(config.get("decision_thresholds") or {})
    expected = {
        "minimum_different_action_states": 4,
        "winner_flip_state_fraction": 0.25,
        "minimum_mean_winner_stability": 0.70,
        "robust_action_minimum_wins": 10,
    }
    if thresholds != expected:
        raise ValueError("Stage 4R PP replay decision thresholds differ")
    return config


def stage4r_paired_pp_seed(
    state_repair_fingerprint: str, trial_index: int
) -> int:
    if trial_index not in PP_TRIAL_INDICES:
        raise ValueError("Stage 4R PP replay trial index must be between 0 and 15")
    return int(
        _fingerprint(
            {
                "namespace": "stride-stage4r-same-state-paired-pp-v1",
                "repair_state": str(state_repair_fingerprint),
                "trial_index": int(trial_index),
            }
        )[:16],
        16,
    ) % (2**31)


def _source_evidence(
    project_root: Path, config: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for label in ACTION_LABELS:
        source = dict(config["sources"][label])
        root = _project_path(project_root, str(source["collection_root"]))
        run_config = _checked_file(
            project_root,
            str(source["run_config"]),
            str(source["run_config_sha256"]),
            f"{label}:run_config",
        )
        manifest = _checked_file(
            project_root,
            str(source["manifest"]),
            str(source["manifest_sha256"]),
            f"{label}:manifest",
        )
        result[label] = {
            "root": root,
            "run_config": run_config,
            "manifest": manifest,
            "rows": {
                (str(row["task_id"]), int(row["solver_seed"])): row
                for row in _read_jsonl(manifest)
            },
        }
    return result


def prepare_pp_replay_selection(
    config_path: str | Path,
    selection_path: str | Path,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    project_root = config_path.parents[1]
    config = validate_pp_replay_config(_read_json(config_path))
    prior_report_path = _checked_file(
        project_root,
        str(config["source_report"]),
        str(config["source_report_sha256"]),
        "source_report",
    )
    prior_report = _read_json(prior_report_path)
    if (
        not bool(prior_report.get("passed"))
        or prior_report.get("next_decision")
        != "run_same_state_paired_pp_repair_replay_before_label_change"
        or bool(prior_report.get("formal_speed_claim"))
    ):
        raise ValueError("source diagnostic does not authorize paired PP replay")
    evidence = _source_evidence(project_root, config)
    expected_keys = {
        (task_id, seed)
        for task_id in map(str, config["registered_task_ids"])
        for seed in map(int, config["solver_seeds"])
    }
    if any(set(evidence[label]["rows"]) != expected_keys for label in ACTION_LABELS):
        raise ValueError("source controller manifests do not cover registered states")

    selection: list[dict[str, Any]] = []
    for index, (task_id, solver_seed) in enumerate(sorted(expected_keys)):
        source_decisions = {}
        source_trace = {}
        for label in ACTION_LABELS:
            source = evidence[label]
            manifest_row = dict(source["rows"][(task_id, solver_seed)])
            if manifest_row.get("status") != "ok":
                raise ValueError(f"source episode is not successful: {label}/{task_id}")
            trace_path = contained_file(
                source["root"], manifest_row.get("trace_file"), field="trace_file"
            )
            trace_sha = sha256_file(trace_path)
            if trace_sha != str(manifest_row.get("trace_sha256", "")).lower():
                raise ValueError(f"source trace hash differs: {label}/{task_id}")
            decisions, _events = decision_rows(source["root"], manifest_row)
            if not decisions or int(decisions[0]["decision_index"]) != 0:
                raise ValueError(f"source trace lacks decision zero: {label}/{task_id}")
            first = dict(decisions[0])
            if first["prefix_actions"]:
                raise ValueError("decision-zero PP replay unexpectedly has a prefix")
            source_decisions[label] = first
            source_trace[label] = {
                "trace_file": str(manifest_row["trace_file"]),
                "trace_sha256": trace_sha,
            }

        reference = source_decisions["v2-full"]
        challenger = source_decisions["stride-quality-v1"]
        for field in ("before_fingerprint", "before_repair_fingerprint", "before_conflicts"):
            if reference[field] != challenger[field]:
                raise ValueError(f"source first-decision states differ: {field}/{task_id}")
        actions = {}
        for label in ACTION_LABELS:
            agents = sorted(map(int, source_decisions[label]["replay_action"]["agents"]))
            actions[label] = {
                "action_id": _fingerprint({"agents": agents}),
                "agents": agents,
                "size": len(agents),
            }
        state_id = _fingerprint(
            {
                "task_id": task_id,
                "solver_seed": solver_seed,
                "before_repair_fingerprint": reference[
                    "before_repair_fingerprint"
                ],
            }
        )
        selection.append(
            {
                "schema": STRIDE_STAGE4R_PP_REPLAY_SELECTION_SCHEMA,
                "state_id": state_id,
                "artifact_file": f"state_{index:03d}_{state_id[:12]}.json",
                "task_id": task_id,
                "map_id": str(evidence["v2-full"]["rows"][(task_id, solver_seed)]["map_id"]),
                "split": str(evidence["v2-full"]["rows"][(task_id, solver_seed)]["split"]),
                "layout_mode": str(
                    evidence["v2-full"]["rows"][(task_id, solver_seed)]["layout_mode"]
                ),
                "agent_count": int(
                    evidence["v2-full"]["rows"][(task_id, solver_seed)]["agent_count"]
                ),
                "solver_seed": solver_seed,
                "decision_index": 0,
                "before_fingerprint": str(reference["before_fingerprint"]),
                "before_repair_fingerprint": str(
                    reference["before_repair_fingerprint"]
                ),
                "before_conflicts": int(reference["before_conflicts"]),
                "actions": actions,
                "same_action": actions[ACTION_LABELS[0]]["action_id"]
                == actions[ACTION_LABELS[1]]["action_id"],
                "source_trace": source_trace,
                "source_outcomes_used_for_selection": False,
            }
        )

    same_count = sum(bool(row["same_action"]) for row in selection)
    different_count = len(selection) - same_count
    if same_count != int(config["expected_same_action_states"]) or different_count != int(
        config["expected_different_action_states"]
    ):
        raise ValueError("source action identity counts differ from registration")
    selection_path = Path(selection_path).resolve()
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    if selection_path.is_file():
        if _read_jsonl(selection_path) != selection:
            raise ValueError("existing Stage 4R PP replay selection differs")
    else:
        _write_jsonl(selection_path, selection)
    report = {
        "schema": STRIDE_STAGE4R_PP_REPLAY_SELECTION_SCHEMA,
        "scientific_status": "diagnostic_only",
        "formal_speed_claim": False,
        "state_count": len(selection),
        "same_action_state_count": same_count,
        "different_action_state_count": different_count,
        "trial_count_per_action": len(PP_TRIAL_INDICES),
        "planned_trial_count": len(selection) * len(ACTION_LABELS) * len(PP_TRIAL_INDICES),
        "source_outcomes_used_for_selection": False,
        "test_data_read": False,
        "formal_ood_data_read": False,
        "inputs": {
            "config_sha256": sha256_file(config_path),
            "source_report_sha256": sha256_file(prior_report_path),
            "source_manifest_sha256": {
                label: sha256_file(evidence[label]["manifest"])
                for label in ACTION_LABELS
            },
        },
        "selection_sha256": sha256_file(selection_path),
    }
    _write_json(selection_path.parent / "selection_report.json", report)
    return report


def _artifact_valid(
    payload: dict[str, Any],
    *,
    identity: str,
    selection: dict[str, Any],
) -> bool:
    if (
        payload.get("schema") != STRIDE_STAGE4R_PP_REPLAY_STATE_SCHEMA
        or payload.get("identity") != identity
        or payload.get("state_id") != selection["state_id"]
        or payload.get("complete") is not True
    ):
        return False
    trials = payload.get("trials")
    if not isinstance(trials, list):
        return False
    expected = {
        (label, trial_index)
        for label in ACTION_LABELS
        for trial_index in PP_TRIAL_INDICES
    }
    observed = {
        (str(row.get("action_label")), int(row.get("trial_index", -1)))
        for row in trials
        if isinstance(row, dict)
    }
    if observed != expected or len(trials) != len(expected):
        return False
    repair_fingerprint = str(selection["before_repair_fingerprint"])
    return all(
        int(row.get("pp_seed", -1))
        == stage4r_paired_pp_seed(repair_fingerprint, int(row["trial_index"]))
        and str(row.get("action_id"))
        == str(selection["actions"][str(row["action_label"])]["action_id"])
        for row in trials
    )


def _collect_state(
    *,
    config: dict[str, Any],
    project_root: Path,
    selection: dict[str, Any],
    output_path: Path,
    identity: str,
    resume: bool,
) -> dict[str, Any]:
    if resume and output_path.is_file():
        existing = _read_json(output_path)
        if _artifact_valid(existing, identity=identity, selection=selection):
            return existing
        raise ValueError(f"invalid completed PP replay artifact: {output_path}")

    canonical_root = _project_path(
        project_root, str(config["sources"]["v2-full"]["collection_root"])
    )
    decision = {
        "source_root": str(canonical_root),
        "split": str(selection["split"]),
        "task_id": str(selection["task_id"]),
        "solver_seed": int(selection["solver_seed"]),
        "prefix_actions": [],
    }
    replay = _replay_job(decision)
    environment, initial = replay_prefix(replay, [])
    if state_fingerprint(initial) != str(selection["before_fingerprint"]):
        raise RuntimeError("Stage 4R PP replay initial state fingerprint changed")
    before_repair_fingerprint = repair_structure_fingerprint(initial)
    if before_repair_fingerprint != str(selection["before_repair_fingerprint"]):
        raise RuntimeError("Stage 4R PP replay repair state fingerprint changed")
    if int(initial["num_of_colliding_pairs"]) != int(selection["before_conflicts"]):
        raise RuntimeError("Stage 4R PP replay initial conflict count changed")
    paths = [list(agent["path"]) for agent in initial["agents"]]
    trials: list[dict[str, Any]] = []
    ordinal = 0
    for trial_index in PP_TRIAL_INDICES:
        order = ACTION_LABELS if trial_index % 2 == 0 else ACTION_LABELS[::-1]
        pp_seed = stage4r_paired_pp_seed(before_repair_fingerprint, trial_index)
        for action_position, label in enumerate(order):
            restored = _plain(environment.restore_paths(paths, seed=pp_seed))
            restored_repair_fingerprint = repair_structure_fingerprint(restored)
            if restored_repair_fingerprint != before_repair_fingerprint:
                raise RuntimeError("Stage 4R PP replay restored repair state changed")
            if int(restored["num_of_colliding_pairs"]) != int(
                selection["before_conflicts"]
            ):
                raise RuntimeError("Stage 4R PP replay restored conflicts changed")
            action = dict(selection["actions"][label])
            agents = list(map(int, action["agents"]))
            result = _plain(environment.step(_paired_action(agents, pp_seed)))
            after, metrics = _validate_native_repair(
                result, expected_agents=agents, expected_seed=pp_seed
            )
            conflicts_before = int(selection["before_conflicts"])
            conflicts_after = int(after["num_of_colliding_pairs"])
            after_repair_fingerprint = repair_structure_fingerprint(after)
            trials.append(
                {
                    "ordinal": ordinal,
                    "action_position": action_position,
                    "action_label": label,
                    "action_id": str(action["action_id"]),
                    "trial_index": trial_index,
                    "pp_seed": pp_seed,
                    "before_repair_fingerprint": before_repair_fingerprint,
                    "replan_success": bool(metrics["replan_success"]),
                    "repair_outcome": classify_repair_outcome(
                        before_fingerprint=before_repair_fingerprint,
                        after_fingerprint=after_repair_fingerprint,
                        replan_success=bool(metrics["replan_success"]),
                        conflicts_before=conflicts_before,
                        conflicts_after=conflicts_after,
                        feasible=bool(after.get("feasible")),
                    ),
                    "feasible": bool(after.get("feasible")),
                    "conflicts_before": conflicts_before,
                    "conflicts_after": conflicts_after,
                    "conflict_reduction": conflicts_before - conflicts_after,
                    "no_progress": conflicts_after >= conflicts_before,
                    "after_repair_fingerprint": after_repair_fingerprint,
                    "repair_order": list(map(int, metrics.get("repair_order") or [])),
                    "post_structure": post_structure_metrics(after),
                    "pp_replan_seconds": float(metrics.get("pp_replan_seconds", 0.0)),
                    "native_step_seconds": float(metrics["native_step_seconds"]),
                    "low_level": dict(after.get("low_level") or {}),
                }
            )
            ordinal += 1

    payload = {
        "schema": STRIDE_STAGE4R_PP_REPLAY_STATE_SCHEMA,
        "identity": identity,
        "complete": True,
        "state_id": str(selection["state_id"]),
        "task_id": str(selection["task_id"]),
        "map_id": str(selection["map_id"]),
        "solver_seed": int(selection["solver_seed"]),
        "before_fingerprint": str(selection["before_fingerprint"]),
        "before_repair_fingerprint": before_repair_fingerprint,
        "before_conflicts": int(selection["before_conflicts"]),
        "actions": dict(selection["actions"]),
        "same_action": bool(selection["same_action"]),
        "trials": trials,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(output_path.name + ".partial")
    _write_json(partial, payload)
    os.replace(partial, output_path)
    return payload


def run_pp_replay(
    config_path: str | Path,
    selection_path: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    project_root = config_path.parents[1]
    config = validate_pp_replay_config(_read_json(config_path))
    selection_path = Path(selection_path).resolve()
    prepare_pp_replay_selection(config_path, selection_path)
    selection = _read_jsonl(selection_path)
    if len(selection) != 16:
        raise ValueError("Stage 4R PP replay selection must contain 16 states")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    identity = _fingerprint(
        {
            "schema": STRIDE_STAGE4R_PP_REPLAY_STATE_SCHEMA,
            "config_sha256": sha256_file(config_path),
            "selection_sha256": sha256_file(selection_path),
        }
    )
    registration = {
        "schema": STRIDE_STAGE4R_PP_REPLAY_STATE_SCHEMA,
        "identity": identity,
        "scientific_status": "diagnostic_only",
        "formal_speed_claim": False,
        "state_count": len(selection),
        "trial_count_per_state": len(ACTION_LABELS) * len(PP_TRIAL_INDICES),
        "total_trial_count": len(selection) * len(ACTION_LABELS) * len(PP_TRIAL_INDICES),
        "config_sha256": sha256_file(config_path),
        "selection_sha256": sha256_file(selection_path),
        "complete": False,
    }
    if dry_run:
        return registration
    status_path = output / "pp_replay_status.json"
    if status_path.is_file():
        existing = _read_json(status_path)
        if existing.get("identity") != identity:
            raise ValueError("existing Stage 4R PP replay output identity differs")
        if not resume:
            raise ValueError("Stage 4R PP replay output exists; pass resume")
    _write_json(status_path, registration)

    manifest = []
    try:
        for state_index, row in enumerate(selection):
            artifact_path = output / "states" / str(row["artifact_file"])
            payload = _collect_state(
                config=config,
                project_root=project_root,
                selection=row,
                output_path=artifact_path,
                identity=identity,
                resume=resume,
            )
            manifest.append(
                {
                    "state_id": str(row["state_id"]),
                    "task_id": str(row["task_id"]),
                    "solver_seed": int(row["solver_seed"]),
                    "state_file": artifact_path.relative_to(output).as_posix(),
                    "state_sha256": sha256_file(artifact_path),
                    "trial_count": len(payload["trials"]),
                    "status": "ok",
                }
            )
            _write_json(
                status_path,
                {
                    **registration,
                    "completed_state_count": state_index + 1,
                    "completed_trial_count": (state_index + 1)
                    * len(ACTION_LABELS)
                    * len(PP_TRIAL_INDICES),
                    "current": {
                        "state_id": str(row["state_id"]),
                        "task_id": str(row["task_id"]),
                        "solver_seed": int(row["solver_seed"]),
                    },
                    "complete": False,
                },
            )
    except Exception as error:
        _write_json(
            status_path,
            {
                **registration,
                "completed_state_count": len(manifest),
                "completed_trial_count": len(manifest)
                * len(ACTION_LABELS)
                * len(PP_TRIAL_INDICES),
                "complete": False,
                "failed": True,
                "error_kind": type(error).__name__,
                "error": str(error),
            },
        )
        raise

    _write_jsonl(output / "artifact_manifest.jsonl", manifest)
    report = analyze_pp_replay(config_path, selection_path, output)
    _write_json(
        status_path,
        {
            **registration,
            "completed_state_count": len(selection),
            "completed_trial_count": registration["total_trial_count"],
            "complete": True,
            "artifact_manifest_sha256": sha256_file(
                output / "artifact_manifest.jsonl"
            ),
            "report_sha256": sha256_file(output / "pp_replay_report.json"),
        },
    )
    return report


def _summary(trials: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(trials)
    reductions = [int(row["conflict_reduction"]) for row in rows]
    return {
        "trial_count": len(rows),
        "mean_conflict_reduction": _mean(reductions),
        "median_conflict_reduction": (
            statistics.median(reductions) if reductions else 0.0
        ),
        "minimum_conflict_reduction": min(reductions) if reductions else 0,
        "maximum_conflict_reduction": max(reductions) if reductions else 0,
        "no_progress_count": sum(bool(row["no_progress"]) for row in rows),
        "no_progress_rate": (
            sum(bool(row["no_progress"]) for row in rows) / len(rows)
            if rows
            else 0.0
        ),
        "replan_failure_count": sum(not bool(row["replan_success"]) for row in rows),
        "mean_pp_replan_seconds": _mean(row["pp_replan_seconds"] for row in rows),
        "mean_native_step_seconds": _mean(row["native_step_seconds"] for row in rows),
    }


def _state_comparison(
    payload: dict[str, Any], thresholds: dict[str, Any]
) -> dict[str, Any]:
    indexed = {
        (str(row["action_label"]), int(row["trial_index"])): row
        for row in payload["trials"]
    }
    pairs = [
        (
            indexed[("v2-full", trial_index)],
            indexed[("stride-quality-v1", trial_index)],
        )
        for trial_index in PP_TRIAL_INDICES
    ]
    deterministic_equal = True
    if bool(payload["same_action"]):
        for left, right in pairs:
            left_signature = (
                bool(left["replan_success"]),
                int(left["conflicts_after"]),
                str(left["after_repair_fingerprint"]),
                tuple(map(int, left["repair_order"])),
                dict(left["low_level"]),
            )
            right_signature = (
                bool(right["replan_success"]),
                int(right["conflicts_after"]),
                str(right["after_repair_fingerprint"]),
                tuple(map(int, right["repair_order"])),
                dict(right["low_level"]),
            )
            deterministic_equal = deterministic_equal and left_signature == right_signature

    v2_wins = sum(
        int(left["conflict_reduction"]) > int(right["conflict_reduction"])
        for left, right in pairs
    )
    quality_wins = sum(
        int(right["conflict_reduction"]) > int(left["conflict_reduction"])
        for left, right in pairs
    )
    ties = len(pairs) - v2_wins - quality_wins
    non_ties = v2_wins + quality_wins
    winner_stability = max(v2_wins, quality_wins) / non_ties if non_ties else 1.0
    v2_trials = [left for left, _right in pairs]
    quality_trials = [right for _left, right in pairs]
    v2_summary = _summary(v2_trials)
    quality_summary = _summary(quality_trials)
    mean_delta = float(quality_summary["mean_conflict_reduction"]) - float(
        v2_summary["mean_conflict_reduction"]
    )
    minimum_wins = int(thresholds["robust_action_minimum_wins"])
    if bool(payload["same_action"]):
        robust_preference = "same_action"
    elif (
        quality_wins >= minimum_wins
        and mean_delta > 0.0
        and float(quality_summary["no_progress_rate"])
        <= float(v2_summary["no_progress_rate"])
    ):
        robust_preference = "stride-quality-v1"
    elif (
        v2_wins >= minimum_wins
        and mean_delta < 0.0
        and float(v2_summary["no_progress_rate"])
        <= float(quality_summary["no_progress_rate"])
    ):
        robust_preference = "v2-full"
    else:
        robust_preference = "inconclusive"
    return {
        "state_id": str(payload["state_id"]),
        "task_id": str(payload["task_id"]),
        "solver_seed": int(payload["solver_seed"]),
        "same_action": bool(payload["same_action"]),
        "same_action_deterministic_equal": deterministic_equal,
        "v2": v2_summary,
        "quality": quality_summary,
        "quality_minus_v2_mean_conflict_reduction": mean_delta,
        "v2_win_count": v2_wins,
        "quality_win_count": quality_wins,
        "tie_count": ties,
        "winner_flip": v2_wins > 0 and quality_wins > 0,
        "winner_stability": winner_stability,
        "robust_preference": robust_preference,
    }


def analyze_pp_replay(
    config_path: str | Path,
    selection_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = validate_pp_replay_config(_read_json(config_path))
    selection_path = Path(selection_path).resolve()
    selection = _read_jsonl(selection_path)
    output = Path(output).resolve()
    identity = _fingerprint(
        {
            "schema": STRIDE_STAGE4R_PP_REPLAY_STATE_SCHEMA,
            "config_sha256": sha256_file(config_path),
            "selection_sha256": sha256_file(selection_path),
        }
    )
    errors = []
    artifacts = []
    artifact_hashes = {}
    for row in selection:
        path = output / "states" / str(row["artifact_file"])
        if not path.is_file():
            errors.append(f"missing state artifact: {row['state_id']}")
            continue
        payload = _read_json(path)
        if not _artifact_valid(payload, identity=identity, selection=row):
            errors.append(f"invalid state artifact: {row['state_id']}")
            continue
        artifacts.append(payload)
        artifact_hashes[str(row["state_id"])] = sha256_file(path)

    thresholds = dict(config["decision_thresholds"])
    comparisons = [_state_comparison(payload, thresholds) for payload in artifacts]
    different = [row for row in comparisons if not bool(row["same_action"])]
    same = [row for row in comparisons if bool(row["same_action"])]
    all_trials = [trial for payload in artifacts for trial in payload["trials"]]
    action_summaries = {
        label: _summary(
            trial for trial in all_trials if str(trial["action_label"]) == label
        )
        for label in ACTION_LABELS
    }
    flip_count = sum(bool(row["winner_flip"]) for row in different)
    flip_fraction = flip_count / len(different) if different else 0.0
    mean_stability = _mean(row["winner_stability"] for row in different)
    pp_seed_material = (
        flip_fraction >= float(thresholds["winner_flip_state_fraction"])
        or mean_stability < float(thresholds["minimum_mean_winner_stability"])
    )
    quality_robust = sum(
        row["robust_preference"] == "stride-quality-v1" for row in different
    )
    v2_robust = sum(row["robust_preference"] == "v2-full" for row in different)
    gates = {
        "complete_state_coverage": len(artifacts) == len(selection) == 16,
        "complete_trial_coverage": len(all_trials) == 512,
        "minimum_different_action_states": len(different)
        >= int(thresholds["minimum_different_action_states"]),
        "paired_pp_seeds": all(
            len(
                {
                    int(trial["pp_seed"])
                    for trial in payload["trials"]
                    if int(trial["trial_index"]) == trial_index
                }
            )
            == 1
            for payload in artifacts
            for trial_index in PP_TRIAL_INDICES
        ),
        "same_action_exact_outcomes": all(
            bool(row["same_action_deterministic_equal"]) for row in same
        ),
        "zero_artifact_errors": not errors,
    }
    passed = all(gates.values())
    if not passed:
        next_decision = "repair_pp_replay_before_scientific_conclusion"
    elif pp_seed_material:
        next_decision = "retain_multiseed_label_and_model_action_uncertainty"
    elif v2_robust > quality_robust:
        next_decision = "revise_current_step_quality_label_or_generalization"
    elif quality_robust > v2_robust:
        next_decision = "study_residual_state_hardness_after_quality_action"
    else:
        next_decision = "collect_more_same_state_actions_before_label_change"

    report = {
        "schema": STRIDE_STAGE4R_PP_REPLAY_REPORT_SCHEMA,
        "scientific_status": "diagnostic_only",
        "formal_speed_claim": False,
        "primary_metric": "paired_current_step_conflict_reduction",
        "state_count": len(artifacts),
        "trial_count": len(all_trials),
        "same_action_state_count": len(same),
        "different_action_state_count": len(different),
        "action_summaries": action_summaries,
        "state_comparisons": comparisons,
        "diagnosis": {
            "winner_flip_state_count": flip_count,
            "winner_flip_state_fraction": flip_fraction,
            "mean_winner_stability": mean_stability,
            "pp_seed_material": pp_seed_material,
            "quality_robust_better_state_count": quality_robust,
            "v2_robust_better_state_count": v2_robust,
            "inconclusive_different_action_state_count": len(different)
            - quality_robust
            - v2_robust,
        },
        "next_decision": next_decision,
        "gates": gates,
        "passed": passed,
        "errors": errors,
        "source_outcomes_used_for_selection": False,
        "test_data_read": False,
        "formal_ood_data_read": False,
        "inputs": {
            "config_sha256": sha256_file(config_path),
            "selection_sha256": sha256_file(selection_path),
            "state_artifact_sha256": artifact_hashes,
        },
    }
    _write_json(output / "pp_replay_report.json", report)
    return report


__all__ = [
    "ACTION_LABELS",
    "PP_TRIAL_INDICES",
    "STRIDE_STAGE4R_PP_REPLAY_REPORT_SCHEMA",
    "analyze_pp_replay",
    "prepare_pp_replay_selection",
    "run_pp_replay",
    "stage4r_paired_pp_seed",
    "validate_pp_replay_config",
]
