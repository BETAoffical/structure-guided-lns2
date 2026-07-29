from __future__ import annotations

import collections
import csv
import hashlib
import json
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

from experiments._common import atomic_write_csv, sha256_file
from experiments.compact_controller_model import load_controller_bundle
from experiments.lns2_bottleneck import validate_manifest_trace
from experiments.repair_aware import classify_repair_outcome
from experiments.repair_collection import (
    _fingerprint,
    _load_dataset_rows,
    _low_level_delta,
    _plain,
    _read_json,
    _read_jsonl,
    _write_json,
    state_fingerprint,
)
from experiments.run_output_guard import prepare_run_output
from experiments.stall_guard import repair_structure_fingerprint
from experiments.stall_shadow import _ranked_candidates
from experiments.stalled_state_probe import (
    _candidate_pool_signature,
    _finite_nonnegative,
    _json_int_list,
    _model_candidates,
    _optional_native_seed,
    _restore_trial_state,
    _source_v2_selection_ids,
    _strict_int,
)
from experiments.trace_replay import decision_rows, replay_prefix


STALL_TRIGGER_COUNTERFACTUAL_SCHEMA = "lns2.stall_trigger_counterfactual.v1"
STALL_TRIGGER_COUNTERFACTUAL_VERSION = 1


def paired_continuation_seed(
    before_repair_fingerprint: str, trial_index: int, step_index: int
) -> int:
    if not before_repair_fingerprint:
        raise ValueError("counterfactual seed requires a repair fingerprint")
    if trial_index < 0 or step_index < 0:
        raise ValueError("counterfactual seed indices must be non-negative")
    digest = hashlib.sha256(
        (
            f"stall-trigger-counterfactual-v1:{before_repair_fingerprint}:"
            f"{trial_index}:{step_index}"
        ).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def ordered_trial_branches(
    branches: Iterable[dict[str, Any]],
    *,
    state_anchor_fingerprint: str,
    trial_index: int,
) -> list[dict[str, Any]]:
    """Return a deterministic, paired order with opposite order on odd trials.

    Counterfactual branches run in independent environments, but their wall times can
    still inherit process and filesystem warm-up effects.  A state-specific hash order
    prevents one policy from always running first; reversing it on the paired trial
    places every branch on the opposite side of the warm/cold ordering.
    """

    if not state_anchor_fingerprint:
        raise ValueError("branch ordering requires a state anchor fingerprint")
    if trial_index < 0:
        raise ValueError("branch ordering trial index must be non-negative")
    ordered = sorted(
        (dict(branch) for branch in branches),
        key=lambda branch: hashlib.sha256(
            (
                f"stall-trigger-branch-order-v1:{state_anchor_fingerprint}:"
                f"{branch.get('branch', '')}"
            ).encode("utf-8")
        ).hexdigest(),
    )
    if trial_index % 2:
        ordered.reverse()
    return ordered


def _json_list(value: str, *, field: str) -> list[Any]:
    try:
        result = json.loads(value)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"{field} is not valid JSON") from error
    if not isinstance(result, list):
        raise ValueError(f"{field} must be a JSON list")
    return result


def load_trigger_plan(path: str | Path) -> list[dict[str, Any]]:
    trigger_path = Path(path).resolve()
    with trigger_path.open("r", encoding="utf-8", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    rows: list[dict[str, Any]] = []
    keys: set[tuple[str, int, int]] = set()
    for raw in raw_rows:
        task_id = str(raw.get("task_id") or "")
        solver_seed = int(raw.get("solver_seed", -1))
        decision_index = int(raw.get("trigger_decision_index", -1))
        resolution = str(raw.get("resolution") or "")
        fingerprint = str(raw.get("state_anchor_fingerprint") or "")
        candidate_ids = [
            str(value)
            for value in _json_list(
                str(raw.get("suggested_rescue_candidate_ids") or ""),
                field="suggested rescue candidate ids",
            )
        ]
        candidate_ranks = [
            int(value)
            for value in _json_list(
                str(raw.get("suggested_rescue_ranks") or ""),
                field="suggested rescue ranks",
            )
        ]
        if (
            not task_id
            or solver_seed < 0
            or decision_index < 0
            or resolution not in {"confirmed_stall", "premature_trigger"}
            or not fingerprint
            or not candidate_ids
            or len(candidate_ids) != len(candidate_ranks)
            or candidate_ranks != sorted(set(candidate_ranks))
            or any(rank < 2 or rank > 8 for rank in candidate_ranks)
        ):
            raise ValueError("stall trigger plan row is invalid")
        key = (task_id, solver_seed, decision_index)
        if key in keys:
            raise ValueError("stall trigger plan repeats a trigger key")
        keys.add(key)
        rows.append(
            {
                "task_id": task_id,
                "solver_seed": solver_seed,
                "decision_index": decision_index,
                "resolution": resolution,
                "state_anchor_fingerprint": fingerprint,
                "suggested_rescue_candidate_ids": candidate_ids,
                "suggested_rescue_ranks": candidate_ranks,
            }
        )
    if not rows:
        raise ValueError("stall trigger plan is empty")
    return sorted(
        rows,
        key=lambda row: (
            str(row["task_id"]),
            int(row["solver_seed"]),
            int(row["decision_index"]),
        ),
    )


def _job_id(index: int, row: dict[str, Any]) -> str:
    safe_task = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in str(row["task_id"])
    )
    return (
        f"trigger-{index:03d}-d{int(row['decision_index']):04d}-"
        f"s{int(row['solver_seed']):04d}-{safe_task}"
    )


def _mean(values: Iterable[float]) -> float | None:
    materialized = list(map(float, values))
    return statistics.fmean(materialized) if materialized else None


def _source_context(
    source_root: Path, trigger: dict[str, Any]
) -> dict[str, Any]:
    source_run = _read_json(source_root / "run_config.json")
    if str(source_run.get("controller")) != "v2-full":
        raise ValueError("counterfactual source must be a v2-full collection")
    configuration = dict(source_run.get("configuration") or {})
    manifests = [
        dict(row)
        for row in _read_jsonl(source_root / "realized_dynamic_manifest.jsonl")
        if str(row.get("task_id")) == str(trigger["task_id"])
        and int(row.get("solver_seed", -1)) == int(trigger["solver_seed"])
        and str(row.get("status")) in {"ok", "resumed"}
    ]
    if len(manifests) != 1:
        raise ValueError("counterfactual source episode coverage is invalid")
    manifest = manifests[0]
    _path, events, _blob = validate_manifest_trace(
        source_root,
        manifest,
        run_fingerprint=str(source_run["run_fingerprint"]),
        expected_policy="realized_dynamic",
    )
    decisions, _unused = decision_rows(source_root, manifest)
    matches = [
        dict(row)
        for row in decisions
        if int(row["decision_index"]) == int(trigger["decision_index"])
    ]
    transitions = [
        event
        for event in events
        if str(event.get("event")) == "transition"
        and int(event.get("decision_index", -1)) == int(trigger["decision_index"])
    ]
    if len(matches) != 1 or len(transitions) != 1:
        raise ValueError("counterfactual trigger decision is absent from source")
    decision = matches[0]
    event = transitions[0]
    if (
        str(decision.get("before_repair_fingerprint"))
        != str(trigger["state_anchor_fingerprint"])
    ):
        raise ValueError("counterfactual trigger repair fingerprint mismatch")
    controller = event.get("controller")
    if not isinstance(controller, dict):
        raise ValueError("counterfactual source lacks controller evidence")
    source_pool = controller.get("candidate_pool")
    if not isinstance(source_pool, list) or not source_pool:
        raise ValueError("counterfactual source lacks its candidate pool")
    dataset_root = Path(str(source_run["dataset"])).resolve()
    dataset_rows = {
        str(row["task_id"]): row
        for row in _load_dataset_rows(dataset_root, [str(configuration["split"])])
    }
    if str(trigger["task_id"]) not in dataset_rows:
        raise ValueError("counterfactual task is missing from its dataset")
    job = {
        "dataset_root": str(dataset_root),
        "row": dataset_rows[str(trigger["task_id"])],
        "environment": dict(configuration["environment"]),
        "solver_seed": int(trigger["solver_seed"]),
    }
    environment, before = replay_prefix(job, decision["prefix_actions"])
    if (
        state_fingerprint(before) != str(decision["before_fingerprint"])
        or repair_structure_fingerprint(before)
        != str(trigger["state_anchor_fingerprint"])
    ):
        raise RuntimeError("counterfactual prefix replay did not reproduce the trigger")
    bundle = load_controller_bundle(Path(str(configuration["controller_bundle"])))
    source_bundle = source_run.get("controller_bundle")
    if not isinstance(source_bundle, dict) or _fingerprint(bundle.manifest) != _fingerprint(
        source_bundle
    ):
        raise ValueError("counterfactual controller bundle identity mismatch")
    model = bundle.main_models["realized_dynamic"]
    model_ranges = bundle.main_ranges["realized_dynamic"]
    candidates, scores, selection = _model_candidates(
        environment,
        before,
        task_id=str(trigger["task_id"]),
        solver_seed=int(trigger["solver_seed"]),
        decision_index=int(trigger["decision_index"]),
        configuration=configuration,
        model=model,
        model_ranges=model_ranges,
    )
    if _candidate_pool_signature(source_pool) != _candidate_pool_signature(
        candidates, scores
    ):
        raise ValueError("counterfactual current candidate pool differs from source")
    selected = int(selection["selected_index"])
    current_selected_id = str(candidates[selected]["candidate_id"])
    source_base_id, source_selected_id = _source_v2_selection_ids(controller)
    if current_selected_id != source_base_id or current_selected_id != source_selected_id:
        raise ValueError("counterfactual current v2 winner differs from source")
    ranked = _ranked_candidates(candidates, scores)
    by_id = {str(row["candidate_id"]): dict(row) for row in candidates}
    ranked_by_id = {str(row["candidate_id"]): dict(row) for row in ranked}
    rescue_ids = list(trigger["suggested_rescue_candidate_ids"])
    rescue_ranks = list(trigger["suggested_rescue_ranks"])
    if any(candidate_id not in by_id for candidate_id in rescue_ids):
        raise ValueError("counterfactual rescue candidate is absent from current pool")
    if any(
        int(ranked_by_id[candidate_id]["rank"]) != rank
        for candidate_id, rank in zip(rescue_ids, rescue_ranks)
    ):
        raise ValueError("counterfactual rescue candidate rank mismatch")
    branches = [
        {
            "branch": "v2_rank1",
            "candidate_rank": 1,
            "candidate": dict(candidates[selected]),
        }
    ]
    branches.extend(
        {
            "branch": f"rescue_rank{rank}",
            "candidate_rank": rank,
            "candidate": by_id[candidate_id],
        }
        for candidate_id, rank in zip(rescue_ids, rescue_ranks)
    )
    return {
        "source_run": source_run,
        "configuration": configuration,
        "manifest": manifest,
        "decision": decision,
        "source_state": before,
        "job": job,
        "model": model,
        "model_ranges": model_ranges,
        "initial_selection_seconds": float(selection["selection_seconds"]),
        "candidate_pool_fingerprint": _fingerprint(
            _candidate_pool_signature(candidates, scores)
        ),
        "branches": branches,
    }


def _branch_rollout(
    *,
    trigger: dict[str, Any],
    context: dict[str, Any],
    branch: dict[str, Any],
    trial_index: int,
    horizon: int,
) -> dict[str, Any]:
    before_repair_fingerprint = str(trigger["state_anchor_fingerprint"])
    initial_seed = paired_continuation_seed(
        before_repair_fingerprint, trial_index, 0
    )
    environment, state, restore = _restore_trial_state(
        context["job"], context["source_state"], seed=initial_seed
    )
    if repair_structure_fingerprint(state) != before_repair_fingerprint:
        raise RuntimeError("counterfactual branch restore changed the repair state")
    initial_conflicts = _strict_int(
        state["num_of_colliding_pairs"], field="initial conflicts", minimum=0
    )
    conflicts = [initial_conflicts]
    steps: list[dict[str, Any]] = []
    total_selection = 0.0
    total_repair = 0.0
    total_pp = 0.0
    total_conflict_auc = 0.0
    feasible = initial_conflicts == 0
    time_to_feasible: float | None = 0.0 if feasible else None
    for step_index in range(horizon):
        if feasible:
            break
        if step_index == 0:
            candidate = dict(branch["candidate"])
            selection_seconds = float(context["initial_selection_seconds"])
            candidate_rank = int(branch["candidate_rank"])
            shared_initial_selection = True
        else:
            candidates, _scores, selection = _model_candidates(
                environment,
                state,
                task_id=str(trigger["task_id"]),
                solver_seed=int(trigger["solver_seed"]),
                decision_index=int(trigger["decision_index"]) + step_index,
                configuration=context["configuration"],
                model=context["model"],
                model_ranges=context["model_ranges"],
            )
            candidate = dict(candidates[int(selection["selected_index"])])
            selection_seconds = float(selection["selection_seconds"])
            candidate_rank = 1
            shared_initial_selection = False
        seed = paired_continuation_seed(
            before_repair_fingerprint, trial_index, step_index
        )
        action = {
            "mode": "explicit_neighborhood",
            "agents": list(map(int, candidate["agents"])),
            "random_seed": seed,
            "pp_random_seed": seed,
        }
        conflicts_before = _strict_int(
            state["num_of_colliding_pairs"],
            field="step conflicts before",
            minimum=0,
        )
        repair_started = time.perf_counter()
        result = _plain(environment.step(action))
        repair_seconds = time.perf_counter() - repair_started
        after = dict(result["observation"])
        metrics = dict(result["metrics"])
        requested_seed = _strict_int(
            metrics.get("requested_pp_random_seed"),
            field="requested PP seed",
            minimum=0,
        )
        if requested_seed != seed:
            raise RuntimeError("counterfactual branch did not retain its paired PP seed")
        repair_order = _json_int_list(
            metrics.get("repair_order", []), field="repair order"
        )
        applied_seed = _optional_native_seed(metrics.get("applied_pp_random_seed"))
        if (repair_order and applied_seed != seed) or (
            not repair_order and applied_seed is not None
        ):
            raise RuntimeError("counterfactual branch PP seed application mismatch")
        native_neighborhood = sorted(
            _json_int_list(metrics.get("neighborhood", []), field="neighborhood")
        )
        if native_neighborhood != sorted(map(int, candidate["agents"])):
            raise RuntimeError("counterfactual native neighborhood differs from action")
        conflicts_after = _strict_int(
            after["num_of_colliding_pairs"],
            field="step conflicts after",
            minimum=0,
        )
        terminated = bool(result.get("terminated"))
        if terminated != (conflicts_after == 0):
            raise RuntimeError("counterfactual termination differs from conflicts")
        pp_seconds = _finite_nonnegative(
            metrics.get("pp_replan_seconds", 0.0), field="PP replan seconds"
        )
        duration = selection_seconds + repair_seconds
        total_selection += selection_seconds
        total_repair += repair_seconds
        total_pp += pp_seconds
        total_conflict_auc += conflicts_before * duration
        low_level = _low_level_delta(state, after)
        before_structure = repair_structure_fingerprint(state)
        after_structure = repair_structure_fingerprint(after)
        replan_success = bool(metrics.get("replan_success"))
        outcome = classify_repair_outcome(
            before_fingerprint=before_structure,
            after_fingerprint=after_structure,
            replan_success=replan_success,
            conflicts_before=conflicts_before,
            conflicts_after=conflicts_after,
            feasible=terminated,
        )
        steps.append(
            {
                "step_index": step_index,
                "candidate_id": str(candidate["candidate_id"]),
                "candidate_rank": candidate_rank,
                "candidate_size": int(candidate["actual_size"]),
                "random_seed": seed,
                "shared_initial_selection": shared_initial_selection,
                "selection_seconds": selection_seconds,
                "repair_wall_seconds": repair_seconds,
                "pp_replan_seconds": pp_seconds,
                "conflicts_before": conflicts_before,
                "conflicts_after": conflicts_after,
                "repair_outcome": outcome,
                "replan_success": replan_success,
                "expanded": int(low_level.get("expanded", 0)),
                "generated": int(low_level.get("generated", 0)),
                "reopened": int(low_level.get("reopened", 0)),
                "after_repair_fingerprint": after_structure,
            }
        )
        conflicts.append(conflicts_after)
        state = after
        feasible = terminated
        if feasible:
            time_to_feasible = total_selection + total_repair
    return {
        "schema": STALL_TRIGGER_COUNTERFACTUAL_SCHEMA,
        "schema_version": STALL_TRIGGER_COUNTERFACTUAL_VERSION,
        "complete": True,
        "task_id": trigger["task_id"],
        "solver_seed": int(trigger["solver_seed"]),
        "trigger_decision_index": int(trigger["decision_index"]),
        "trigger_resolution": trigger["resolution"],
        "before_repair_fingerprint": before_repair_fingerprint,
        "branch": branch["branch"],
        "first_candidate_id": str(dict(branch["candidate"])["candidate_id"]),
        "first_candidate_rank": int(branch["candidate_rank"]),
        "first_candidate_size": int(dict(branch["candidate"])["actual_size"]),
        "trial_index": trial_index,
        "horizon": horizon,
        "executed_repairs": len(steps),
        "initial_conflicts": initial_conflicts,
        "final_conflicts": conflicts[-1],
        "conflict_delta": initial_conflicts - conflicts[-1],
        "conflict_trajectory": conflicts,
        "feasible": feasible,
        "time_to_feasible": time_to_feasible,
        "selection_seconds": total_selection,
        "repair_wall_seconds": total_repair,
        "pp_replan_seconds": total_pp,
        "total_decision_seconds": total_selection + total_repair,
        "wall_conflict_auc": total_conflict_auc,
        "steps": steps,
        "restore_evidence": restore,
    }


def compare_trial_branches(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int, int], list[dict[str, Any]]] = (
        collections.defaultdict(list)
    )
    for row in rows:
        grouped[
            (
                str(row["task_id"]),
                int(row["solver_seed"]),
                int(row["trigger_decision_index"]),
                int(row["trial_index"]),
            )
        ].append(row)
    comparisons: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items()):
        baseline_rows = [row for row in values if str(row["branch"]) == "v2_rank1"]
        rescue_rows = [row for row in values if str(row["branch"]) != "v2_rank1"]
        if len(baseline_rows) != 1 or not rescue_rows:
            raise ValueError("counterfactual trial lacks baseline or rescue coverage")
        baseline = baseline_rows[0]
        first = min(rescue_rows, key=lambda row: int(row["first_candidate_rank"]))
        oracle = min(
            rescue_rows,
            key=lambda row: (
                not bool(row["feasible"]),
                int(row["final_conflicts"]),
                float(row["total_decision_seconds"]),
                int(row["first_candidate_rank"]),
            ),
        )
        for policy, rescue in (("first_suggestion", first), ("oracle_rescue", oracle)):
            time_delta = float(rescue["total_decision_seconds"]) - float(
                baseline["total_decision_seconds"]
            )
            conflict_advantage = int(baseline["final_conflicts"]) - int(
                rescue["final_conflicts"]
            )
            rescue_dominates = bool(
                int(rescue["final_conflicts"]) <= int(baseline["final_conflicts"])
                and float(rescue["total_decision_seconds"])
                <= float(baseline["total_decision_seconds"])
                and (
                    int(rescue["final_conflicts"]) < int(baseline["final_conflicts"])
                    or float(rescue["total_decision_seconds"])
                    < float(baseline["total_decision_seconds"])
                )
            )
            comparisons.append(
                {
                    "task_id": key[0],
                    "solver_seed": key[1],
                    "trigger_decision_index": key[2],
                    "trial_index": key[3],
                    "trigger_resolution": baseline["trigger_resolution"],
                    "policy": policy,
                    "rescue_branch": rescue["branch"],
                    "rescue_rank": int(rescue["first_candidate_rank"]),
                    "baseline_final_conflicts": int(baseline["final_conflicts"]),
                    "rescue_final_conflicts": int(rescue["final_conflicts"]),
                    "rescue_conflict_advantage": conflict_advantage,
                    "baseline_total_seconds": float(
                        baseline["total_decision_seconds"]
                    ),
                    "rescue_total_seconds": float(rescue["total_decision_seconds"]),
                    "rescue_time_delta_seconds": time_delta,
                    "baseline_selection_seconds": float(
                        baseline["selection_seconds"]
                    ),
                    "rescue_selection_seconds": float(rescue["selection_seconds"]),
                    "baseline_pp_seconds": float(baseline["pp_replan_seconds"]),
                    "rescue_pp_seconds": float(rescue["pp_replan_seconds"]),
                    "baseline_feasible": bool(baseline["feasible"]),
                    "rescue_feasible": bool(rescue["feasible"]),
                    "rescue_dominates": rescue_dominates,
                }
            )
    return comparisons


def run_counterfactual_job(
    source_v2: str | Path,
    trigger_csv: str | Path,
    output: str | Path,
    *,
    job_index: int,
    trials: int = 2,
    horizon: int = 3,
    resume: bool = False,
) -> dict[str, Any]:
    if trials <= 0 or horizon <= 0:
        raise ValueError("counterfactual trials and horizon must be positive")
    source_root = Path(source_v2).resolve()
    trigger_path = Path(trigger_csv).resolve()
    output_root = Path(output).resolve()
    triggers = load_trigger_plan(trigger_path)
    if job_index < 0 or job_index >= len(triggers):
        raise ValueError("counterfactual job index is out of range")
    trigger = triggers[job_index]
    context = _source_context(source_root, trigger)
    identity = {
        "schema": STALL_TRIGGER_COUNTERFACTUAL_SCHEMA,
        "schema_version": STALL_TRIGGER_COUNTERFACTUAL_VERSION,
        "mode": "single-trigger",
        "source_run_config_sha256": sha256_file(source_root / "run_config.json"),
        "source_trace_sha256": str(context["manifest"]["trace_sha256"]),
        "trigger_csv_sha256": sha256_file(trigger_path),
        "trigger": trigger,
        "candidate_pool_fingerprint": context["candidate_pool_fingerprint"],
        "trials": trials,
        "horizon": horizon,
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
    }
    runner = prepare_run_output(output_root, resume=resume, identity=identity)
    run_fingerprint = str(runner["identity_fingerprint"])
    rows: list[dict[str, Any]] = []
    for trial_index in range(trials):
        trial_branches = ordered_trial_branches(
            context["branches"],
            state_anchor_fingerprint=str(trigger["state_anchor_fingerprint"]),
            trial_index=trial_index,
        )
        for execution_order, branch in enumerate(trial_branches):
            checkpoint = (
                output_root
                / "checkpoints"
                / str(branch["branch"])
                / f"trial-{trial_index:03d}.json"
            )
            existing = _read_json(checkpoint) if resume and checkpoint.is_file() else None
            if existing is not None:
                if (
                    existing.get("complete") is not True
                    or str(existing.get("run_fingerprint")) != run_fingerprint
                ):
                    raise ValueError("counterfactual checkpoint identity mismatch")
                rows.append(existing)
                continue
            result = _branch_rollout(
                trigger=trigger,
                context=context,
                branch=branch,
                trial_index=trial_index,
                horizon=horizon,
            )
            result["branch_execution_order"] = execution_order
            result["run_fingerprint"] = run_fingerprint
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            _write_json(checkpoint, result)
            rows.append(result)
    comparisons = compare_trial_branches(rows)
    atomic_write_csv(output_root / "branch_trials.csv", rows)
    atomic_write_csv(output_root / "paired_comparisons.csv", comparisons)
    report = {
        "schema": STALL_TRIGGER_COUNTERFACTUAL_SCHEMA,
        "schema_version": STALL_TRIGGER_COUNTERFACTUAL_VERSION,
        "complete": True,
        "run_fingerprint": run_fingerprint,
        "job_index": job_index,
        "trigger": trigger,
        "trial_count": trials,
        "horizon": horizon,
        "branch_count": len(context["branches"]),
        "rollout_count": len(rows),
        "comparison_count": len(comparisons),
        "initial_selection_seconds": context["initial_selection_seconds"],
        "controller_actions_changed_in_source": False,
        "counterfactual_actions_executed": True,
        "deployment_promoted": False,
    }
    _write_json(output_root / "job_report.json", report)
    return report


def _run_logged(command: list[str], *, cwd: Path, log_prefix: Path) -> None:
    completed = subprocess.run(
        command, cwd=cwd, check=False, text=True, capture_output=True
    )
    log_prefix.parent.mkdir(parents=True, exist_ok=True)
    log_prefix.with_suffix(".stdout.log").write_text(
        completed.stdout, encoding="utf-8"
    )
    log_prefix.with_suffix(".stderr.log").write_text(
        completed.stderr, encoding="utf-8"
    )
    if completed.returncode:
        raise RuntimeError(
            f"counterfactual subprocess failed ({completed.returncode})"
        )


def _validated_job_report(root: Path, expected: dict[str, Any]) -> dict[str, Any]:
    report = _read_json(root / "job_report.json")
    if (
        report.get("complete") is not True
        or dict(report.get("trigger") or {}) != expected
        or report.get("deployment_promoted") is not False
    ):
        raise ValueError("counterfactual job report is invalid")
    return report


def _run_batch_job(
    *,
    project_root: Path,
    source_root: Path,
    trigger_path: Path,
    jobs_root: Path,
    index: int,
    trigger: dict[str, Any],
    trials: int,
    horizon: int,
    resume: bool,
) -> dict[str, Any]:
    root = jobs_root / _job_id(index, trigger)
    if resume and (root / "job_report.json").is_file():
        return _validated_job_report(root, trigger)
    command = [
        sys.executable,
        str(project_root / "scripts" / "run_stall_trigger_counterfactual.py"),
        "--source-v2",
        str(source_root),
        "--trigger-csv",
        str(trigger_path),
        "--output",
        str(root),
        "--job-index",
        str(index),
        "--trials",
        str(trials),
        "--horizon",
        str(horizon),
    ]
    if root.is_dir() and any(root.iterdir()):
        command.append("--resume")
    _run_logged(command, cwd=project_root, log_prefix=root / "runner")
    return _validated_job_report(root, trigger)


def run_stall_trigger_counterfactual(
    source_v2: str | Path,
    trigger_csv: str | Path,
    output: str | Path,
    *,
    trials: int = 2,
    horizon: int = 3,
    workers: int = 4,
    resume: bool = False,
) -> dict[str, Any]:
    if trials <= 0 or horizon <= 0 or workers <= 0:
        raise ValueError("counterfactual trials, horizon, and workers must be positive")
    project_root = Path(__file__).resolve().parents[1]
    source_root = Path(source_v2).resolve()
    trigger_path = Path(trigger_csv).resolve()
    output_root = Path(output).resolve()
    triggers = load_trigger_plan(trigger_path)
    identity = {
        "schema": STALL_TRIGGER_COUNTERFACTUAL_SCHEMA,
        "schema_version": STALL_TRIGGER_COUNTERFACTUAL_VERSION,
        "mode": "batch",
        "source_run_config_sha256": sha256_file(source_root / "run_config.json"),
        "source_manifest_sha256": sha256_file(
            source_root / "realized_dynamic_manifest.jsonl"
        ),
        "trigger_csv_sha256": sha256_file(trigger_path),
        "trigger_count": len(triggers),
        "trials": trials,
        "horizon": horizon,
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
        "script_sha256": sha256_file(
            project_root / "scripts" / "run_stall_trigger_counterfactual.py"
        ),
    }
    prepare_run_output(output_root, resume=resume, identity=identity)
    jobs_root = output_root / "jobs"
    jobs_root.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    _write_json(
        output_root / "status.json",
        {
            "schema": STALL_TRIGGER_COUNTERFACTUAL_SCHEMA,
            "status": "running",
            "total_jobs": len(triggers),
            "completed_jobs": 0,
            "error_jobs": 0,
            "workers": workers,
        },
    )
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _run_batch_job,
                project_root=project_root,
                source_root=source_root,
                trigger_path=trigger_path,
                jobs_root=jobs_root,
                index=index,
                trigger=trigger,
                trials=trials,
                horizon=horizon,
                resume=resume,
            ): (index, trigger)
            for index, trigger in enumerate(triggers)
        }
        for future in as_completed(futures):
            index, trigger = futures[future]
            try:
                reports.append(future.result())
            except Exception as error:
                errors.append(
                    {
                        "job_id": _job_id(index, trigger),
                        "task_id": trigger["task_id"],
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
            _write_json(
                output_root / "status.json",
                {
                    "schema": STALL_TRIGGER_COUNTERFACTUAL_SCHEMA,
                    "status": (
                        "running"
                        if len(reports) + len(errors) < len(triggers)
                        else "finalizing"
                    ),
                    "total_jobs": len(triggers),
                    "completed_jobs": len(reports),
                    "error_jobs": len(errors),
                    "workers": workers,
                },
            )
    all_trials: list[dict[str, Any]] = []
    all_comparisons: list[dict[str, Any]] = []
    for index, trigger in enumerate(triggers):
        root = jobs_root / _job_id(index, trigger)
        if not (root / "job_report.json").is_file():
            continue
        with (root / "branch_trials.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            all_trials.extend(dict(row) for row in csv.DictReader(handle))
        with (root / "paired_comparisons.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            all_comparisons.extend(dict(row) for row in csv.DictReader(handle))
    comparison_summary: list[dict[str, Any]] = []
    for policy in ("first_suggestion", "oracle_rescue"):
        rows = [row for row in all_comparisons if str(row["policy"]) == policy]
        comparison_summary.append(
            {
                "policy": policy,
                "comparison_count": len(rows),
                "mean_conflict_advantage": _mean(
                    float(row["rescue_conflict_advantage"]) for row in rows
                ),
                "mean_time_delta_seconds": _mean(
                    float(row["rescue_time_delta_seconds"]) for row in rows
                ),
                "faster_count": sum(
                    float(row["rescue_time_delta_seconds"]) < 0.0 for row in rows
                ),
                "lower_final_conflict_count": sum(
                    float(row["rescue_conflict_advantage"]) > 0.0 for row in rows
                ),
                "dominates_count": sum(
                    str(row["rescue_dominates"]).lower() == "true" for row in rows
                ),
            }
        )
    complete = not errors and len(reports) == len(triggers)
    report = {
        "schema": STALL_TRIGGER_COUNTERFACTUAL_SCHEMA,
        "schema_version": STALL_TRIGGER_COUNTERFACTUAL_VERSION,
        "complete": complete,
        "evidence_level": "same-state paired three-repair counterfactual",
        "trigger_count": len(triggers),
        "completed_job_count": len(reports),
        "error_count": len(errors),
        "errors": errors,
        "trial_count_per_branch": trials,
        "horizon": horizon,
        "rollout_count": len(all_trials),
        "paired_comparison_count": len(all_comparisons),
        "comparison_summary": comparison_summary,
        "controller_actions_changed_in_source": False,
        "counterfactual_actions_executed": True,
        "training_started": False,
        "deployment_promoted": False,
        "decision": (
            "diagnostic_complete_no_promotion" if complete else "incomplete_investigate"
        ),
    }
    atomic_write_csv(output_root / "all_branch_trials.csv", all_trials)
    atomic_write_csv(output_root / "all_paired_comparisons.csv", all_comparisons)
    atomic_write_csv(output_root / "comparison_summary.csv", comparison_summary)
    _write_json(output_root / "counterfactual_report.json", report)
    lines = [
        "# Stall-trigger three-repair counterfactual",
        "",
        f"- Triggers: `{len(triggers)}`; completed: `{len(reports)}`; errors: `{len(errors)}`.",
        f"- Trials per branch: `{trials}`; horizon: `{horizon}` repairs.",
        "- Evidence is same-state diagnostic only; deployment promoted: `false`.",
        "",
        "| Policy | Comparisons | Mean conflict advantage | Mean time delta (s) | Faster | Dominates |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in comparison_summary:
        lines.append(
            f"| {row['policy']} | {row['comparison_count']} | "
            f"{float(row['mean_conflict_advantage'] or 0.0):.3f} | "
            f"{float(row['mean_time_delta_seconds'] or 0.0):.6f} | "
            f"{row['faster_count']} | {row['dominates_count']} |"
        )
    lines.extend(
        [
            "",
            "The first repair differs by branch; subsequent repairs always use the frozen v2 controller. Prefix replay/reset time is excluded. Initial v2 selection time is shared equally by every branch.",
            "",
        ]
    )
    (output_root / "counterfactual_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    _write_json(
        output_root / "status.json",
        {
            "schema": STALL_TRIGGER_COUNTERFACTUAL_SCHEMA,
            "status": "complete" if complete else "error",
            "total_jobs": len(triggers),
            "completed_jobs": len(reports),
            "error_jobs": len(errors),
            "workers": workers,
        },
    )
    return report


__all__ = [
    "STALL_TRIGGER_COUNTERFACTUAL_SCHEMA",
    "compare_trial_branches",
    "load_trigger_plan",
    "ordered_trial_branches",
    "paired_continuation_seed",
    "run_counterfactual_job",
    "run_stall_trigger_counterfactual",
]
