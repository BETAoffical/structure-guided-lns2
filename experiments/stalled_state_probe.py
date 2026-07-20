from __future__ import annotations

import csv
import hashlib
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any, Iterable

from experiments.closed_loop_confirmation import (
    feature_range_diagnostic,
    generate_online_candidates,
    score_online_candidates,
    validate_closed_loop_trace,
)
from experiments.compact_controller_model import load_controller_bundle
from experiments.online_feature_engine import OnlineFeatureEngine
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
from research.engineering.balanced.route_counterfactual import _decision_rows, _replay_prefix
from experiments.run_output_guard import prepare_run_output
from experiments.stall_guard import repair_structure_fingerprint


STALLED_STATE_PROBE_SCHEMA = "lns2.stalled_state_probe.v1"
STALLED_STATE_PROBE_VERSION = 1


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = sorted({str(key) for row in rows for key in row})
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(partial, path)


def _median(values: Iterable[float]) -> float | None:
    numbers = list(map(float, values))
    return statistics.median(numbers) if numbers else None


def find_terminal_stall(decisions: list[dict[str, Any]], minimum: int = 3) -> dict[str, Any]:
    """Return the first decision in the terminal unchanged-state failed-replan run."""

    if minimum <= 0:
        raise ValueError("minimum terminal stall length must be positive")
    start = len(decisions)
    while start > 0:
        row = decisions[start - 1]
        metrics = dict(row.get("actual_metrics") or {})
        failed = not bool(metrics.get("replan_success"))
        unchanged = int(row.get("before_conflicts", -1)) == int(
            metrics.get("conflicts_after", -2)
        )
        if not unchanged or not failed:
            break
        start -= 1
    length = len(decisions) - start
    if length < minimum:
        raise ValueError(
            f"source trace has no terminal unchanged-state failed-replan run of at least {minimum} decisions"
        )
    target = dict(decisions[start])
    return {
        "decision": target,
        "start_decision_index": int(target["decision_index"]),
        "length": length,
        "before_fingerprint": str(target["before_fingerprint"]),
    }


def ranked_candidate_indices(
    candidates: list[dict[str, Any]], scores: list[float]
) -> list[int]:
    if len(candidates) != len(scores) or not candidates:
        raise ValueError("candidate ranking inputs are invalid")
    return sorted(
        range(len(candidates)),
        key=lambda index: (
            -round(float(scores[index]), 12),
            str(candidates[index]["candidate_id"]),
        ),
    )


def choose_probe_branches(
    candidates: list[dict[str, Any]], scores: list[float]
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Choose and deduplicate the fixed diagnostic alternatives."""

    order = ranked_candidate_indices(candidates, scores)
    requested: list[tuple[str, int | None]] = [
        ("rank1", order[0]),
        ("rank2", order[1] if len(order) > 1 else order[0]),
    ]
    for cap in (8, 4):
        eligible = [index for index in order if int(candidates[index]["actual_size"]) <= cap]
        if not eligible:
            raise ValueError(f"candidate pool has no neighborhood with actual_size <= {cap}")
        requested.append((f"size_le_{cap}", eligible[0]))
    requested.append(("official_adaptive", None))

    unique: list[dict[str, Any]] = []
    by_key: dict[str, dict[str, Any]] = {}
    alias_to_key: dict[str, str] = {}
    for alias, index in requested:
        if index is None:
            key = "official_adaptive"
            branch = {
                "branch_key": key,
                "aliases": [],
                "mode": "official",
                "candidate": None,
                "rank": None,
                "score": None,
            }
        else:
            candidate = dict(candidates[index])
            key = str(candidate["candidate_id"])
            branch = {
                "branch_key": key,
                "aliases": [],
                "mode": "explicit_neighborhood",
                "candidate": candidate,
                "rank": order.index(index) + 1,
                "score": float(scores[index]),
            }
        if key not in by_key:
            by_key[key] = branch
            unique.append(branch)
        by_key[key]["aliases"].append(alias)
        alias_to_key[alias] = key
    return unique, alias_to_key


def paired_probe_seed(before_fingerprint: str, trial_index: int) -> int:
    if trial_index < 0:
        raise ValueError("trial index must be non-negative")
    return int(
        _fingerprint(
            {
                "namespace": "stalled-state-paired-repair-v1",
                "before_fingerprint": str(before_fingerprint),
                "trial_index": int(trial_index),
            }
        )[:16],
        16,
    ) % (2**31)


def summarize_probe_trials(
    trials: list[dict[str, Any]], alias_to_key: dict[str, str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_key: dict[str, list[dict[str, Any]]] = {}
    for row in trials:
        by_key.setdefault(str(row["branch_key"]), []).append(row)
    summaries: list[dict[str, Any]] = []
    by_alias: dict[str, dict[str, Any]] = {}
    for alias, key in alias_to_key.items():
        rows = by_key.get(key, [])
        if not rows:
            raise ValueError(f"probe branch is missing trials: {alias}")
        summary = {
            "branch": alias,
            "branch_key": key,
            "trial_count": len(rows),
            "replan_success_count": sum(bool(row["replan_success"]) for row in rows),
            "replan_success_rate": sum(bool(row["replan_success"]) for row in rows)
            / len(rows),
            "positive_conflict_reduction_count": sum(
                int(row["conflict_delta"]) > 0 for row in rows
            ),
            "mean_conflict_delta": statistics.fmean(
                float(row["conflict_delta"]) for row in rows
            ),
            "median_conflict_delta": _median(
                float(row["conflict_delta"]) for row in rows
            ),
            "median_conflicts_after": _median(
                float(row["conflicts_after"]) for row in rows
            ),
            "median_pp_replan_seconds": _median(
                float(row["pp_replan_seconds"]) for row in rows
            ),
            "median_total_decision_seconds": _median(
                float(row["total_decision_seconds"]) for row in rows
            ),
        }
        summaries.append(summary)
        by_alias[alias] = summary
    rank1 = by_alias["rank1"]
    supported: list[str] = []
    for alias in ("size_le_8", "size_le_4", "official_adaptive"):
        row = by_alias[alias]
        success_gain = float(row["replan_success_rate"]) - float(
            rank1["replan_success_rate"]
        )
        positive_escape = (
            int(rank1["positive_conflict_reduction_count"]) == 0
            and int(row["positive_conflict_reduction_count"]) >= 2
        )
        if success_gain >= 0.25 - 1e-12 or positive_escape:
            supported.append(alias)
    gate = {
        "passed": bool(supported),
        "supported_alternatives": supported,
        "success_rate_gain_threshold": 0.25,
        "positive_reduction_threshold_when_rank1_zero": 2,
    }
    return summaries, gate


def _model_candidates(
    environment: Any,
    state: dict[str, Any],
    *,
    task_id: str,
    solver_seed: int,
    decision_index: int,
    configuration: dict[str, Any],
    model: Any,
    model_ranges: dict[str, tuple[float, float]],
) -> tuple[list[dict[str, Any]], list[float], dict[str, Any]]:
    started = time.perf_counter()
    before_fingerprint = state_fingerprint(state)
    optimized = str(configuration.get("controller_runtime", "reference")) == "optimized"
    candidates, proposal = generate_online_candidates(
        environment,
        state,
        task_id=task_id,
        solver_seed=solver_seed,
        decision_index=decision_index,
        proposal_config=dict(configuration["proposal"]),
        state_hash=before_fingerprint,
        verify_full_state=True,
        proposal_backend="optimized" if optimized else "reference",
        shadow_validation=False,
    )
    engine = OnlineFeatureEngine(
        state,
        backend=str(configuration.get("feature_backend", "auto")),
        shadow_validation=False,
        required_features={"realized_dynamic": model.base_feature_names},
        dense_output=optimized,
    )
    rows, feature_metrics = engine.realized_rows(candidates, state_hash=before_fingerprint)
    inference_started = time.perf_counter()
    selected, scores, margin = score_online_candidates(rows, model)
    inference_seconds = time.perf_counter() - inference_started
    diagnostics = [
        feature_range_diagnostic(row, "realized_dynamic", model_ranges)
        for row in rows
    ]
    return candidates, scores, {
        "selected_index": selected,
        "score_margin": margin,
        "selection_seconds": time.perf_counter() - started,
        "candidate_generation_seconds": float(
            proposal.get("candidate_generation_seconds", proposal.get("proposal_seconds", 0.0))
        ),
        "state_check_seconds": float(proposal.get("state_check_seconds", 0.0)),
        "state_analysis_seconds": float(
            engine.last_prepare_metrics.get("state_analysis_seconds", 0.0)
        ),
        "realized_feature_seconds": float(
            feature_metrics.get("realized_feature_seconds", 0.0)
        ),
        "inference_seconds": inference_seconds,
        "candidate_count": len(candidates),
        "selected_feature_out_of_range_fraction": float(
            diagnostics[selected]["outside_fraction"]
        ),
    }


def _checkpoint_valid(
    path: Path, *, run_fingerprint: str, branch_key: str, trial_index: int
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        row = _read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if (
        str(row.get("schema")) != STALLED_STATE_PROBE_SCHEMA
        or str(row.get("run_fingerprint")) != run_fingerprint
        or str(row.get("branch_key")) != branch_key
        or int(row.get("trial_index", -1)) != trial_index
        or not bool(row.get("replay_fingerprint_match"))
    ):
        return None
    return row


def _run_trial(
    *,
    job: dict[str, Any],
    decision: dict[str, Any],
    branch: dict[str, Any],
    trial_index: int,
    selection_seconds: float,
    run_fingerprint: str,
) -> dict[str, Any]:
    environment, before = _replay_prefix(job, decision["prefix_actions"])
    before_fingerprint = state_fingerprint(before)
    if before_fingerprint != str(decision["before_fingerprint"]):
        raise RuntimeError("stalled-state replay fingerprint mismatch")
    random_seed = paired_probe_seed(before_fingerprint, trial_index)
    if str(branch["mode"]) == "official":
        action = {"mode": "official", "random_seed": random_seed}
        model_selection_seconds = 0.0
    else:
        candidate = dict(branch["candidate"])
        action = {
            "mode": "explicit_neighborhood",
            "agents": list(map(int, candidate["agents"])),
            "random_seed": random_seed,
        }
        model_selection_seconds = selection_seconds
    started = time.perf_counter()
    result = _plain(environment.step(action))
    repair_wall_seconds = time.perf_counter() - started
    after = dict(result["observation"])
    metrics = dict(result["metrics"])
    conflicts_before = int(before["num_of_colliding_pairs"])
    conflicts_after = int(after["num_of_colliding_pairs"])
    native_selection_seconds = float(
        metrics.get("native_neighborhood_generation_seconds", 0.0)
    )
    return {
        "schema": STALLED_STATE_PROBE_SCHEMA,
        "schema_version": STALLED_STATE_PROBE_VERSION,
        "run_fingerprint": run_fingerprint,
        "branch_key": str(branch["branch_key"]),
        "branch_aliases": ",".join(map(str, branch["aliases"])),
        "branch_mode": str(branch["mode"]),
        "candidate_id": (
            str(dict(branch["candidate"])["candidate_id"])
            if branch["candidate"] is not None
            else None
        ),
        "candidate_rank": branch["rank"],
        "candidate_score": branch["score"],
        "candidate_size": (
            int(dict(branch["candidate"])["actual_size"])
            if branch["candidate"] is not None
            else len(list(map(int, metrics.get("neighborhood", []))))
        ),
        "trial_index": trial_index,
        "random_seed": random_seed,
        "before_fingerprint": before_fingerprint,
        "before_repair_fingerprint": repair_structure_fingerprint(before),
        "after_fingerprint": state_fingerprint(after),
        "after_repair_fingerprint": repair_structure_fingerprint(after),
        "replay_fingerprint_match": True,
        "replan_success": bool(metrics.get("replan_success")),
        "conflicts_before": conflicts_before,
        "conflicts_after": conflicts_after,
        "conflict_delta": conflicts_before - conflicts_after,
        "sum_of_costs_delta": int(after["sum_of_costs"]) - int(before["sum_of_costs"]),
        "expanded": int(_low_level_delta(before, after).get("expanded", 0)),
        "generated": int(_low_level_delta(before, after).get("generated", 0)),
        "reopened": int(_low_level_delta(before, after).get("reopened", 0)),
        "model_selection_seconds": model_selection_seconds,
        "native_neighborhood_generation_seconds": native_selection_seconds,
        "pp_replan_seconds": float(metrics.get("pp_replan_seconds", 0.0)),
        "repair_wall_seconds": repair_wall_seconds,
        "total_decision_seconds": model_selection_seconds + repair_wall_seconds,
    }


def _markdown(report: dict[str, Any], summaries: list[dict[str, Any]]) -> str:
    gate = dict(report["promotion_gate"])
    lines = [
        "# Stalled-state repair probe",
        "",
        f"- Task: `{report['task_id']}`; solver seed: `{report['solver_seed']}`.",
        f"- Target decision: `{report['decision_index']}`; terminal stagnant run: `{report['terminal_stall_length']}` decisions.",
        f"- Before fingerprint: `{report['before_fingerprint']}`.",
        f"- Trials per unique branch: `{report['trials_per_branch']}`.",
        f"- Guard evidence gate: `{'passed' if gate['passed'] else 'failed'}`.",
        "",
        "| Branch | PP success | Positive reductions | Median conflict delta | Median PP seconds |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {branch} | {replan_success_count}/{trial_count} | "
            "{positive_conflict_reduction_count}/{trial_count} | {median_conflict_delta:.3f} | "
            "{median_pp_replan_seconds:.6f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "The probe executes exactly one repair per branch. It does not use Horizon-4 continuation and does not replace complete-episode evaluation.",
            "",
            "Supported alternatives: "
            + (", ".join(gate["supported_alternatives"]) or "none"),
            "",
        ]
    )
    return "\n".join(lines)


def run_stalled_state_probe(
    source: str | Path,
    output: str | Path,
    *,
    task_id: str,
    solver_seed: int,
    trials: int = 8,
    auto_terminal_stall: bool = True,
    decision_index: int | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    if trials <= 0:
        raise ValueError("trials must be positive")
    if auto_terminal_stall == (decision_index is not None):
        raise ValueError("choose exactly one of auto terminal stall or decision index")
    collection_root = Path(source).resolve()
    output_root = Path(output).resolve()
    source_run = _read_json(collection_root / "run_config.json")
    if str(source_run.get("controller")) != "v2-full":
        raise ValueError("stalled-state source must use v2-full")
    configuration = dict(source_run["configuration"])
    manifests = [
        row
        for row in _read_jsonl(collection_root / "realized_dynamic_manifest.jsonl")
        if str(row.get("task_id")) == task_id
        and int(row.get("solver_seed", -1)) == int(solver_seed)
        and str(row.get("status")) in {"ok", "resumed"}
    ]
    if len(manifests) != 1:
        raise ValueError("source collection does not contain exactly one requested episode")
    manifest = dict(manifests[0])
    trace_path = collection_root / str(manifest["trace_file"])
    validate_closed_loop_trace(
        trace_path,
        str(source_run["run_fingerprint"]),
        expected_episode_id=str(manifest["episode_id"]),
        expected_policy="realized_dynamic",
        expected_solver_seed=int(solver_seed),
        metric_iteration_budget=configuration.get("metric_iteration_budget"),
        collection_root=collection_root,
    )
    decisions, _events = _decision_rows(collection_root, manifest)
    if auto_terminal_stall:
        stall = find_terminal_stall(decisions)
        decision = dict(stall["decision"])
        terminal_stall_length = int(stall["length"])
    else:
        matches = [
            row for row in decisions if int(row["decision_index"]) == int(decision_index)
        ]
        if len(matches) != 1:
            raise ValueError("requested decision index is absent from the source trace")
        decision = dict(matches[0])
        terminal_stall_length = 0

    dataset_root = Path(str(source_run["dataset"])).resolve()
    rows = {
        str(row["task_id"]): row
        for row in _load_dataset_rows(dataset_root, [str(configuration["split"])])
    }
    if task_id not in rows:
        raise ValueError("requested task is missing from the source dataset")
    job = {
        "dataset_root": str(dataset_root),
        "row": rows[task_id],
        "environment": dict(configuration["environment"]),
        "solver_seed": int(solver_seed),
    }
    implementation_sha256 = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    identity = {
        "schema": STALLED_STATE_PROBE_SCHEMA,
        "source_collection": str(collection_root),
        "source_run_fingerprint": str(source_run["run_fingerprint"]),
        "source_trace_sha256": str(manifest["trace_sha256"]),
        "task_id": task_id,
        "solver_seed": int(solver_seed),
        "decision_index": int(decision["decision_index"]),
        "trials": int(trials),
        "implementation_sha256": implementation_sha256,
    }
    runner = prepare_run_output(output_root, resume=resume, identity=identity)
    run_fingerprint = str(runner["identity_fingerprint"])

    controller_path = Path(str(configuration["controller_bundle"])).resolve()
    bundle = load_controller_bundle(controller_path)
    model = bundle.main_models["realized_dynamic"]
    model_ranges = bundle.main_ranges["realized_dynamic"]
    environment, before = _replay_prefix(job, decision["prefix_actions"])
    before_fingerprint = state_fingerprint(before)
    if before_fingerprint != str(decision["before_fingerprint"]):
        raise RuntimeError("target replay fingerprint does not match the source trace")
    candidates, scores, selection = _model_candidates(
        environment,
        before,
        task_id=task_id,
        solver_seed=int(solver_seed),
        decision_index=int(decision["decision_index"]),
        configuration=configuration,
        model=model,
        model_ranges=model_ranges,
    )
    branches, alias_to_key = choose_probe_branches(candidates, scores)

    reproduction_environment, reproduction_before = _replay_prefix(
        job, decision["prefix_actions"]
    )
    if state_fingerprint(reproduction_before) != before_fingerprint:
        raise RuntimeError("source reproduction before fingerprint mismatch")
    reproduced = _plain(reproduction_environment.step(dict(decision["actual_action"])))
    reproduction_after = dict(reproduced["observation"])
    reproduction = {
        "before_fingerprint_match": True,
        "after_fingerprint_match": state_fingerprint(reproduction_after)
        == str(decision["after_fingerprint"]),
        "replan_success_match": bool(dict(reproduced["metrics"]).get("replan_success"))
        == bool(dict(decision["actual_metrics"]).get("replan_success")),
        "conflicts_after_match": int(reproduction_after["num_of_colliding_pairs"])
        == int(dict(decision["actual_metrics"])["conflicts_after"]),
    }
    if not all(reproduction.values()):
        raise RuntimeError("source decision could not be reproduced exactly")
    _write_json(output_root / "source_reproduction.json", reproduction)

    completed: list[dict[str, Any]] = []
    for branch in branches:
        for trial_index in range(trials):
            checkpoint = (
                output_root
                / "checkpoints"
                / str(branch["branch_key"])
                / f"trial-{trial_index:03d}.json"
            )
            existing = (
                _checkpoint_valid(
                    checkpoint,
                    run_fingerprint=run_fingerprint,
                    branch_key=str(branch["branch_key"]),
                    trial_index=trial_index,
                )
                if resume
                else None
            )
            if existing is not None:
                completed.append(existing)
                continue
            result = _run_trial(
                job=job,
                decision=decision,
                branch=branch,
                trial_index=trial_index,
                selection_seconds=float(selection["selection_seconds"]),
                run_fingerprint=run_fingerprint,
            )
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            _write_json(checkpoint, result)
            completed.append(result)

    summaries, gate = summarize_probe_trials(completed, alias_to_key)
    report = {
        "schema": STALLED_STATE_PROBE_SCHEMA,
        "schema_version": STALLED_STATE_PROBE_VERSION,
        "run_fingerprint": run_fingerprint,
        "source_collection": str(collection_root),
        "source_episode_id": str(manifest["episode_id"]),
        "task_id": task_id,
        "solver_seed": int(solver_seed),
        "decision_index": int(decision["decision_index"]),
        "terminal_stall_length": terminal_stall_length,
        "before_fingerprint": before_fingerprint,
        "trials_per_branch": int(trials),
        "unique_branch_count": len(branches),
        "candidate_count": len(candidates),
        "model_selection": selection,
        "source_reproduction": reproduction,
        "promotion_gate": gate,
    }
    _write_csv(output_root / "stalled_state_trials.csv", completed)
    _write_csv(output_root / "stalled_state_summary.csv", summaries)
    _write_json(output_root / "stalled_state_probe_report.json", report)
    markdown = _markdown(report, summaries)
    partial = output_root / "stalled_state_probe_report.md.partial"
    partial.write_text(markdown, encoding="utf-8")
    os.replace(partial, output_root / "stalled_state_probe_report.md")
    return report


__all__ = [
    "STALLED_STATE_PROBE_SCHEMA",
    "choose_probe_branches",
    "find_terminal_stall",
    "paired_probe_seed",
    "ranked_candidate_indices",
    "run_stalled_state_probe",
    "summarize_probe_trials",
]
