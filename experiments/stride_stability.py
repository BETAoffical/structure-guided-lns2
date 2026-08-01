from __future__ import annotations

import concurrent.futures
import os
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

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
    _balanced_result_blind_selection,
    _paired_action,
    _replay_job,
    _state_artifact_valid,
    _validate_native_repair,
    load_stride_selection,
)
from experiments.stride_lns import (
    STRIDE_TRIAL_SCHEMA,
    aggregate_stride_candidate,
    assign_structure_scores,
    post_structure_metrics,
    stride_dominates,
)
from experiments.trace_replay import replay_prefix
from lns2_selector.runtime.fingerprints import repair_structure_fingerprint
from lns2_selector.runtime.repair_outcomes import classify_repair_outcome


STRIDE_STABILITY_SELECTION_SCHEMA = "lns2.stride.stability_selection.v1"
STRIDE_STABILITY_COLLECTION_SCHEMA = "lns2.stride.stability_collection.v1"
STRIDE_STABILITY_REPORT_SCHEMA = "lns2.stride.stability_report.v1"
EXTRA_TRIAL_INDICES = (4, 5, 6, 7)


def stride_extended_pp_seed(state_repair_fingerprint: str, trial_index: int) -> int:
    if trial_index < 0 or trial_index > 7:
        raise ValueError("STRIDE extended trial index must be between 0 and 7")
    return int(
        _fingerprint(
            {
                "namespace": "stride-lns-paired-pp-v1",
                "repair_state": str(state_repair_fingerprint),
                "trial_index": int(trial_index),
            }
        )[:16],
        16,
    ) % (2**31)


def select_stride_stability_states(
    *, selection_path: Path, output: Path, count_per_policy: int = 24
) -> dict[str, Any]:
    selected = load_stride_selection(selection_path)
    cohort, policy_reports = _balanced_result_blind_selection(
        selected, target_per_policy=count_per_policy, max_per_episode=1
    )
    cohort.sort(key=lambda row: (str(row["source_policy"]), str(row["state_id"])))
    total = count_per_policy * 2
    gates = {
        "twenty_percent": len(cohort) == total and total / len(selected) == 0.20,
        "policy_balance": all(
            report["selected_state_count"] == count_per_policy
            for report in policy_reports.values()
        ),
        "one_per_episode": all(
            report["max_states_in_episode"] <= 1 for report in policy_reports.values()
        ),
    }
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output / "stability_selection.jsonl", cohort)
    report = {
        "schema": STRIDE_STABILITY_SELECTION_SCHEMA,
        "source_selection": str(selection_path.resolve()),
        "source_state_count": len(selected),
        "selected_state_count": len(cohort),
        "count_per_policy": count_per_policy,
        "policies": policy_reports,
        "gates": gates,
        "passed": all(gates.values()),
    }
    _write_json(output / "stability_selection_report.json", report)
    return report


def _source_state_files(collection: Path) -> dict[str, Path]:
    run = _read_json(collection / "run_config.json")
    fingerprint = str(run["run_fingerprint"])
    result: dict[str, Path] = {}
    for path in sorted((collection / "states").glob("*.json")):
        payload = _read_json(path)
        state_id = str(payload.get("state_id", ""))
        if not _state_artifact_valid(
            payload, run_fingerprint=fingerprint, state_id=state_id
        ):
            raise ValueError(f"invalid source STRIDE state artifact: {path}")
        result[state_id] = path
    return result


def _extra_artifact_valid(payload: dict[str, Any], *, identity: str, state_id: str) -> bool:
    if (
        payload.get("schema") != STRIDE_STABILITY_COLLECTION_SCHEMA
        or payload.get("identity") != identity
        or payload.get("state_id") != state_id
        or payload.get("complete") is not True
    ):
        return False
    candidates = payload.get("candidate_ids")
    trials = payload.get("trials")
    if not isinstance(candidates, list) or not isinstance(trials, list):
        return False
    expected = {
        (str(candidate_id), trial_index)
        for candidate_id in candidates
        for trial_index in EXTRA_TRIAL_INDICES
    }
    observed = {
        (str(row.get("candidate_id")), int(row.get("trial_index", -1)))
        for row in trials
        if isinstance(row, dict)
    }
    return observed == expected and len(trials) == len(expected)


def _collect_extra_state(job: dict[str, Any]) -> dict[str, Any]:
    decision = dict(job["decision"])
    source_payload = _read_json(Path(str(job["source_file"])))
    output_path = Path(str(job["output_file"]))
    identity = str(job["identity"])
    state_id = str(decision["state_id"])
    if output_path.is_file():
        existing = _read_json(output_path)
        if _extra_artifact_valid(existing, identity=identity, state_id=state_id):
            return {
                "state_id": state_id,
                "status": "resumed",
                "output_file": str(output_path),
                "trial_count": len(existing["trials"]),
            }
        raise ValueError(f"invalid completed STRIDE stability artifact: {output_path}")
    replay = _replay_job(decision)
    _, initial = replay_prefix(replay, decision["prefix_actions"])
    before_fingerprint = state_fingerprint(initial)
    if before_fingerprint != str(decision["before_fingerprint"]):
        raise RuntimeError("STRIDE stability replay fingerprint mismatch")
    before_repair_fingerprint = repair_structure_fingerprint(initial)
    if before_repair_fingerprint != str(source_payload["before_repair_fingerprint"]):
        raise RuntimeError("STRIDE stability repair fingerprint mismatch")
    before_conflicts = int(initial["num_of_colliding_pairs"])
    candidates = list(source_payload["candidates"])
    source_features: dict[str, dict[str, float]] = {}
    for row in source_payload["trials"]:
        source_features.setdefault(str(row["candidate_id"]), dict(row["features"]))
    trials: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        agents = list(map(int, candidate["agents"]))
        for trial_index in EXTRA_TRIAL_INDICES:
            environment, branch = replay_prefix(replay, decision["prefix_actions"])
            if state_fingerprint(branch) != before_fingerprint:
                raise RuntimeError("STRIDE stability paired replay changed")
            seed = stride_extended_pp_seed(before_repair_fingerprint, trial_index)
            result = _plain(environment.step(_paired_action(agents, seed)))
            after, metrics = _validate_native_repair(
                result, expected_agents=agents, expected_seed=seed
            )
            conflicts_after = int(after["num_of_colliding_pairs"])
            after_repair_fingerprint = repair_structure_fingerprint(after)
            trials.append(
                {
                    "schema": STRIDE_TRIAL_SCHEMA,
                    "feature_schema_id": source_payload["trials"][0]["feature_schema_id"],
                    "state_id": state_id,
                    "candidate_id": candidate_id,
                    "map_id": str(decision["map_id"]),
                    "split": str(decision["split"]),
                    "source_policy": str(decision["source_policy"]),
                    "decision_stage": str(decision["decision_stage"]),
                    "agent_count": int(decision["agent_count"]),
                    "before_conflicts": before_conflicts,
                    "before_fingerprint": before_fingerprint,
                    "before_repair_fingerprint": before_repair_fingerprint,
                    "features": source_features[candidate_id],
                    "trial_index": trial_index,
                    "pp_seed": seed,
                    "feasible": bool(after.get("feasible")),
                    "replan_success": bool(metrics["replan_success"]),
                    "repair_outcome": classify_repair_outcome(
                        before_fingerprint=before_repair_fingerprint,
                        after_fingerprint=after_repair_fingerprint,
                        replan_success=bool(metrics["replan_success"]),
                        conflicts_before=before_conflicts,
                        conflicts_after=conflicts_after,
                        feasible=bool(after.get("feasible")),
                    ),
                    "conflicts_after": conflicts_after,
                    "after_fingerprint": state_fingerprint(after),
                    "after_repair_fingerprint": after_repair_fingerprint,
                    "post_structure": post_structure_metrics(after),
                    "native_step_seconds": float(metrics["native_step_seconds"]),
                    "pp_replan_seconds": float(metrics.get("pp_replan_seconds", 0.0)),
                }
            )
    payload = {
        "schema": STRIDE_STABILITY_COLLECTION_SCHEMA,
        "identity": identity,
        "complete": True,
        "state_id": state_id,
        "decision": decision,
        "candidate_ids": [str(candidate["candidate_id"]) for candidate in candidates],
        "trials": trials,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(output_path.name + ".partial")
    _write_json(partial, payload)
    os.replace(partial, output_path)
    return {
        "state_id": state_id,
        "status": "ok",
        "output_file": str(output_path),
        "trial_count": len(trials),
    }


def collect_stride_stability_trials(
    *, selection_path: Path, collection: Path, output: Path, workers: int = 4
) -> dict[str, Any]:
    if workers <= 0:
        raise ValueError("STRIDE stability workers must be positive")
    selected = load_stride_selection(selection_path)
    source_files = _source_state_files(collection)
    missing = sorted(str(row["state_id"]) for row in selected if str(row["state_id"]) not in source_files)
    if missing:
        raise ValueError(f"STRIDE stability source states are missing: {missing}")
    identity_payload = {
        "schema": STRIDE_STABILITY_COLLECTION_SCHEMA,
        "selection": str(selection_path.resolve()),
        "selection_rows": selected,
        "source_run_fingerprint": _read_json(collection / "run_config.json")["run_fingerprint"],
        "extra_trial_indices": list(EXTRA_TRIAL_INDICES),
    }
    identity = _fingerprint(identity_payload)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "run_config.json", {**identity_payload, "identity": identity})
    jobs = [
        {
            "decision": row,
            "source_file": str(source_files[str(row["state_id"])]),
            "output_file": str(
                output
                / "states"
                / f"{_fingerprint({'state_id': row['state_id']})[:20]}.json"
            ),
            "identity": identity,
        }
        for row in selected
    ]
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_collect_extra_state, job): job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            job = futures[future]
            try:
                results.append(future.result())
            except Exception as error:
                errors.append(
                    {
                        "state_id": str(job["decision"]["state_id"]),
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
            _write_json(
                output / "collection_status.json",
                {
                    "schema": STRIDE_STABILITY_COLLECTION_SCHEMA,
                    "requested_state_count": len(jobs),
                    "completed_state_count": len(results),
                    "error_state_count": len(errors),
                    "errors": errors,
                    "status": "running",
                },
            )
    all_trials: list[dict[str, Any]] = []
    for result in sorted(results, key=lambda row: str(row["state_id"])):
        payload = _read_json(Path(str(result["output_file"])))
        if not _extra_artifact_valid(payload, identity=identity, state_id=str(result["state_id"])):
            errors.append({"state_id": str(result["state_id"]), "error": "invalid artifact"})
        else:
            all_trials.extend(payload["trials"])
    complete = len(results) == len(jobs) and not errors
    if complete:
        _write_jsonl(output / "extra_trials.jsonl", all_trials)
    report = {
        "schema": STRIDE_STABILITY_COLLECTION_SCHEMA,
        "identity": identity,
        "requested_state_count": len(jobs),
        "completed_state_count": len(results),
        "error_state_count": len(errors),
        "errors": errors,
        "new_state_count": sum(row["status"] == "ok" for row in results),
        "resumed_state_count": sum(row["status"] == "resumed" for row in results),
        "trial_count": len(all_trials),
        "complete": complete,
    }
    _write_json(output / "collection_report.json", report)
    _write_json(output / "collection_status.json", {**report, "status": "complete" if complete else "error"})
    return report


def _rank_candidates(aggregates: list[dict[str, Any]]) -> list[str]:
    wins: Counter[str] = Counter()
    for left, right in combinations(aggregates, 2):
        before = int(left["before_conflicts"])
        if stride_dominates(left, right, before_conflicts=before):
            wins[str(left["candidate_id"])] += 1
        elif stride_dominates(right, left, before_conflicts=before):
            wins[str(right["candidate_id"])] += 1
    ordered = sorted(
        aggregates,
        key=lambda row: (
            -wins[str(row["candidate_id"])],
            -float(row["feasible_rate"]),
            -float(row["progress_rate"]),
            -float(row["robust_reduction"]),
            float(row["structural_score"]),
            str(row["candidate_id"]),
        ),
    )
    return [str(row["candidate_id"]) for row in ordered]


def analyze_stride_stability(
    *, base_trials: Path, extra_trials: Path, output: Path
) -> dict[str, Any]:
    extra = _read_jsonl(extra_trials)
    state_ids = {str(row["state_id"]) for row in extra}
    base = [row for row in _read_jsonl(base_trials) if str(row["state_id"]) in state_ids]
    grouped: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: {"first": [], "second": []}
    )
    for row in base:
        grouped[(str(row["state_id"]), str(row["candidate_id"]))]["first"].append(row)
    for row in extra:
        grouped[(str(row["state_id"]), str(row["candidate_id"]))]["second"].append(row)
    by_state: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: {"first": [], "second": []}
    )
    for (state_id, candidate_id), halves in grouped.items():
        if len(halves["first"]) != 4 or len(halves["second"]) != 4:
            raise ValueError(f"STRIDE stability candidate lacks four outcomes per half: {state_id}/{candidate_id}")
        before = int(halves["first"][0]["before_conflicts"])
        for half in ("first", "second"):
            aggregate = aggregate_stride_candidate(before_conflicts=before, outcomes=halves[half])
            aggregate.update(
                {
                    "state_id": state_id,
                    "candidate_id": candidate_id,
                    "before_conflicts": before,
                    "map_id": str(halves[half][0]["map_id"]),
                    "source_policy": str(halves[half][0]["source_policy"]),
                }
            )
            by_state[state_id][half].append(aggregate)
    state_reports: list[dict[str, Any]] = []
    total_union = 0
    total_agree = 0
    top3_scores: list[float] = []
    for state_id in sorted(by_state):
        halves = by_state[state_id]
        assign_structure_scores(halves["first"])
        assign_structure_scores(halves["second"])
        by_candidate = {
            half: {str(row["candidate_id"]): row for row in halves[half]}
            for half in ("first", "second")
        }
        directions: dict[str, dict[tuple[str, str], str]] = {"first": {}, "second": {}}
        candidate_ids = sorted(by_candidate["first"])
        for half in ("first", "second"):
            for left_id, right_id in combinations(candidate_ids, 2):
                left = by_candidate[half][left_id]
                right = by_candidate[half][right_id]
                before = int(left["before_conflicts"])
                if stride_dominates(left, right, before_conflicts=before):
                    directions[half][(left_id, right_id)] = left_id
                elif stride_dominates(right, left, before_conflicts=before):
                    directions[half][(left_id, right_id)] = right_id
        union = set(directions["first"]) | set(directions["second"])
        agree = sum(
            directions["first"].get(pair) == directions["second"].get(pair)
            and pair in directions["first"]
            and pair in directions["second"]
            for pair in union
        )
        first_top3 = set(_rank_candidates(halves["first"])[:3])
        second_top3 = set(_rank_candidates(halves["second"])[:3])
        top3 = len(first_top3 & second_top3) / 3.0
        total_union += len(union)
        total_agree += agree
        top3_scores.append(top3)
        state_reports.append(
            {
                "state_id": state_id,
                "map_id": halves["first"][0]["map_id"],
                "source_policy": halves["first"][0]["source_policy"],
                "candidate_count": len(candidate_ids),
                "decisive_pair_union": len(union),
                "agreed_pair_count": agree,
                "pairwise_consistency": agree / len(union) if union else 1.0,
                "top3_overlap": top3,
            }
        )
    pairwise = total_agree / total_union if total_union else 1.0
    top3_mean = sum(top3_scores) / len(top3_scores)
    gates = {
        "pairwise_consistency_at_least_70_percent": pairwise >= 0.70,
        "mean_top3_overlap_at_least_80_percent": top3_mean >= 0.80,
        "state_coverage": len(state_reports) == 48,
    }
    report = {
        "schema": STRIDE_STABILITY_REPORT_SCHEMA,
        "state_count": len(state_reports),
        "candidate_count": len(grouped),
        "outcome_count": len(base) + len(extra),
        "pairwise_decisive_union_count": total_union,
        "pairwise_agreement_count": total_agree,
        "pairwise_consistency": pairwise,
        "mean_top3_overlap": top3_mean,
        "gates": gates,
        "passed": all(gates.values()),
        "states": state_reports,
    }
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "stability_report.json", report)
    return report
