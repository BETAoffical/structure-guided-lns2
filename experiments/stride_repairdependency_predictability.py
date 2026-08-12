from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

from experiments._common import producer_identity, ratio, registered_input, sha256_file
from experiments.closed_loop_trace_storage import read_state_blob
from experiments.repair_collection import (
    _read_json,
    _read_jsonl,
    _write_json,
    _write_jsonl,
    state_fingerprint,
)
from experiments.state_analysis import analyze_static_grid
from experiments.stride_closurepool_longtail import _mean, _pair_set
from experiments.stride_closurepool_temporal import (
    build_corridor_layout,
    segment_visits,
    temporal_dependency_edges,
)
from experiments.stride_repairability_causal_audit import build_causal_cohort


CONFIG_SCHEMA = "lns2.stride.repairdependency_predictability_registration.v1"
MANIFEST_SCHEMA = "lns2.stride.repairdependency_input_state.v1"
ROW_SCHEMA = "lns2.stride.repairdependency_predictability_row.v1"
REPORT_SCHEMA = "lns2.stride.repairdependency_predictability_report.v1"
EXECUTION_SCHEMA = "lns2.stride.repairdependency_predictability_execution.v1"
EXPERIMENT_ID = "stride-repairdependency-predictability-v1"
PREDICTORS = (
    "direct_conflict_boundary",
    "spatial_low_degree_frontier",
    "temporal_corridor_frontier",
    "repair_dependency_frontier",
    "full_temporal_boundary_reference",
)
PRIMARY_PREDICTOR = "repair_dependency_frontier"
PRODUCER_FILES = (
    "experiments/stride_repairdependency_predictability.py",
    "scripts/run_stride_repairdependency_predictability.py",
    "experiments/stride_closurepool_longtail.py",
    "experiments/stride_closurepool_temporal.py",
    "experiments/state_analysis.py",
)


def _manifest_state_row(path: Path) -> dict[str, Any]:
    state = _read_json(path)
    fingerprint = str(state["state_fingerprint"])
    if (
        state.get("complete") is not True
        or path.stem != fingerprint
        or len(state.get("trials", ())) != 80
    ):
        raise ValueError(f"invalid causal state artifact: {path.name}")
    return {
        "schema": MANIFEST_SCHEMA,
        "state_fingerprint": fingerprint,
        "file_name": path.name,
        "sha256": sha256_file(path),
        "trial_row_count": 80,
        "run_fingerprint": str(state["run_fingerprint"]),
    }


def freeze_input_state_manifest(
    collection: str | Path, output: str | Path, *, workers: int = 16
) -> dict[str, Any]:
    collection = Path(collection).resolve()
    output = Path(output).resolve()
    status = _read_json(collection / "collection_status.json")
    report = _read_json(collection / "repairability_causal_report.json")
    if (
        status.get("status") != "complete"
        or int(status.get("completed_state_count", -1)) != 45
        or report.get("integrity_passed") is not True
        or int(report.get("trial_row_count", -1)) != 3600
    ):
        raise ValueError("Repairability causal collection is not complete")
    files = sorted((collection / "states").glob("*.json"))
    if len(files) != 45 or list((collection / "states").glob("*.partial")):
        raise ValueError("Repairability causal state files are incomplete")
    if int(workers) != 16:
        raise ValueError("RepairDependency manifest freeze requires 16 workers")
    with ThreadPoolExecutor(max_workers=int(workers)) as executor:
        rows = list(executor.map(_manifest_state_row, files))
    _write_jsonl(output, rows)
    return {
        "state_count": len(rows),
        "trial_row_count": sum(int(row["trial_row_count"]) for row in rows),
        "manifest_sha256": sha256_file(output),
    }


def _contained(root: Path, value: str, *, label: str) -> Path:
    path = (root / value).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} leaves repository root") from error
    return path


def load_registration(
    config_path: str | Path,
) -> tuple[Path, Path, dict[str, Any], dict[str, Path], Path]:
    config_path = Path(config_path).resolve()
    root = config_path.parents[1]
    config = _read_json(config_path)
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("experiment_id") != EXPERIMENT_ID
    ):
        raise ValueError("RepairDependency predictability registration changed")
    inputs = {
        name: registered_input(root, row, label=f"repair dependency {name}")
        for name, row in dict(config["inputs"]).items()
    }
    collection = _contained(
        root, str(config["causal_collection"]["path"]), label="causal collection"
    )
    if tuple(config["predictors"]["ids"]) != PREDICTORS:
        raise ValueError("RepairDependency predictor set changed")
    if str(config["predictors"]["primary"]) != PRIMARY_PREDICTOR:
        raise ValueError("RepairDependency primary predictor changed")
    if int(config["execution"]["worker_count"]) != 16:
        raise ValueError("RepairDependency worker count changed")
    return config_path, root, config, inputs, collection


def _pareto_frontier(
    rows: dict[int, dict[str, int]], dimensions: tuple[str, ...]
) -> set[int]:
    eligible = {
        int(agent): metrics
        for agent, metrics in rows.items()
        if any(int(metrics[name]) > 0 for name in dimensions)
    }
    frontier: set[int] = set()
    for agent, metrics in eligible.items():
        dominated = False
        for other, other_metrics in eligible.items():
            if agent == other:
                continue
            no_worse = all(
                int(other_metrics[name]) >= int(metrics[name]) for name in dimensions
            )
            strictly_better = any(
                int(other_metrics[name]) > int(metrics[name]) for name in dimensions
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.add(agent)
    return frontier


def dependency_predictors(
    state: dict[str, Any], selected_agents: Iterable[int]
) -> dict[str, Any]:
    selected = set(map(int, selected_agents))
    known = {int(row["id"]) for row in state["agents"]}
    if not selected or not selected < known:
        raise ValueError("selected neighborhood is empty or global")
    paths = {
        int(row["id"]): tuple(map(int, row["path"])) for row in state["agents"]
    }
    conflicts = _pair_set(state)
    static = analyze_static_grid(state)
    layout = build_corridor_layout(static)
    visits = segment_visits(state, layout)
    temporal = temporal_dependency_edges(visits)
    agent_segments = {
        agent: {int(row["segment_id"]) for row in agent_visits}
        for agent, agent_visits in visits.items()
    }
    low_degree_cells = set(map(int, layout["cell_to_segment"]))
    low_paths = {
        agent: set(path) & low_degree_cells for agent, path in paths.items()
    }
    metrics: dict[int, dict[str, int]] = {}
    for outside in sorted(known - selected):
        direct_partners = {
            inside
            for inside in selected
            if tuple(sorted((inside, outside))) in conflicts
        }
        temporal_rows = [
            temporal[tuple(sorted((inside, outside)))]
            for inside in selected
            if tuple(sorted((inside, outside))) in temporal
        ]
        spatial_partners = {
            inside
            for inside in selected
            if agent_segments.get(inside, set()) & agent_segments.get(outside, set())
        }
        shared_segments = set().union(
            *(
                agent_segments.get(inside, set()) & agent_segments.get(outside, set())
                for inside in selected
            )
        )
        shared_low_cells = set().union(
            *(
                low_paths.get(inside, set()) & low_paths.get(outside, set())
                for inside in selected
            )
        )
        metrics[outside] = {
            "direct_conflict_count": len(direct_partners),
            "spatial_selected_partner_count": len(spatial_partners),
            "shared_low_degree_segment_count": len(shared_segments),
            "shared_low_degree_cell_count": len(shared_low_cells),
            "temporal_selected_partner_count": len(temporal_rows),
            "temporal_overlap_count": sum(
                int(row["overlap_count"]) for row in temporal_rows
            ),
            "temporal_overlap_duration": sum(
                int(row["overlap_duration"]) for row in temporal_rows
            ),
            "opposing_overlap_count": sum(
                int(row["opposing_overlap_count"]) for row in temporal_rows
            ),
        }
    direct = {
        agent
        for agent, row in metrics.items()
        if int(row["direct_conflict_count"]) > 0
    }
    spatial = _pareto_frontier(
        metrics,
        (
            "spatial_selected_partner_count",
            "shared_low_degree_segment_count",
            "shared_low_degree_cell_count",
        ),
    )
    temporal_frontier = _pareto_frontier(
        metrics,
        (
            "temporal_selected_partner_count",
            "opposing_overlap_count",
            "temporal_overlap_duration",
        ),
    )
    combined = _pareto_frontier(
        metrics,
        (
            "direct_conflict_count",
            "spatial_selected_partner_count",
            "shared_low_degree_cell_count",
            "temporal_selected_partner_count",
            "opposing_overlap_count",
            "temporal_overlap_duration",
        ),
    )
    full_temporal = {
        agent
        for agent, row in metrics.items()
        if int(row["temporal_selected_partner_count"]) > 0
    }
    predictor_sets = {
        "direct_conflict_boundary": direct,
        "spatial_low_degree_frontier": spatial,
        "temporal_corridor_frontier": temporal_frontier,
        "repair_dependency_frontier": direct | combined,
        "full_temporal_boundary_reference": full_temporal,
    }
    return {
        "agent_count": len(known),
        "selected_agent_count": len(selected),
        "predictor_sets": {
            name: sorted(map(int, predictor_sets[name])) for name in PREDICTORS
        },
        "outside_metrics": {str(agent): row for agent, row in sorted(metrics.items())},
    }


def _context_job(job: dict[str, Any]) -> dict[str, Any]:
    state_record = dict(job["state_record"])
    blob = Path(str(state_record["state_blob"]))
    if sha256_file(blob) != str(state_record["state_blob_sha256"]):
        raise ValueError("RepairDependency source state blob changed")
    state = read_state_blob(blob)
    state["context"] = dict(state_record["state_context"])
    if state_fingerprint(state) != str(state_record["state_fingerprint"]):
        raise ValueError("RepairDependency source state fingerprint changed")
    selected = list(map(int, state_record["selected_candidate"]["agents"]))
    return {
        "state_fingerprint": str(state_record["state_fingerprint"]),
        "map_id": str(state_record["map_id"]),
        "selected_agents": selected,
        "context": dependency_predictors(state, selected),
        "worker_pid": os.getpid(),
    }


def _trial_truth(state_artifact: dict[str, Any], trial_index: int) -> list[int]:
    selected = set(map(int, state_artifact["selected_candidate"]["agents"]))
    rows = {
        (int(row["trial_index"]), str(row["arm"])): row
        for row in state_artifact["trials"]
    }
    tail = rows[(int(trial_index), "blocker_augmented_tail")]
    if tail.get("applicable") is not True:
        return []
    return sorted(set(map(int, tail["agents"])) - selected)


def _evaluation_job(job: dict[str, Any]) -> dict[str, Any]:
    truth = set(map(int, job["truth_blockers"]))
    predicted = set(map(int, job["predicted_agents"]))
    selected_count = int(job["selected_agent_count"])
    agent_count = int(job["agent_count"])
    outside_count = agent_count - selected_count
    intersection = truth & predicted
    applicable = bool(truth)
    observed_recall = ratio(len(intersection), len(truth)) if applicable else None
    random_expected = ratio(len(predicted), outside_count) if applicable else None
    return {
        "schema": ROW_SCHEMA,
        "state_fingerprint": str(job["state_fingerprint"]),
        "map_id": str(job["map_id"]),
        "trial_index": int(job["trial_index"]),
        "seed_half": "first" if int(job["trial_index"]) < 8 else "second",
        "predictor_id": str(job["predictor_id"]),
        "truth_applicable": applicable,
        "truth_blocker_count": len(truth),
        "predicted_agent_count": len(predicted),
        "intersection_count": len(intersection),
        "observed_blocker_recall": observed_recall,
        "any_observed_blocker_hit": bool(intersection) if applicable else None,
        "matched_random_expected_recall": random_expected,
        "recall_lift_over_random": (
            float(observed_recall) - float(random_expected) if applicable else None
        ),
        "expansion_ratio": ratio(len(predicted), selected_count),
        "total_neighborhood_fraction": ratio(selected_count + len(predicted), agent_count),
        "runtime_fields_stored": False,
        "future_trajectory_stored": False,
        "worker_pid": os.getpid(),
    }


def _scientific_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "worker_pid"}


def _summary(rows: list[dict[str, Any]], predictor: str) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if str(row["predictor_id"]) == predictor and bool(row["truth_applicable"])
    ]
    return {
        "applicable_trial_count": len(selected),
        "mean_observed_blocker_recall": _mean(
            row["observed_blocker_recall"] for row in selected
        ),
        "any_observed_blocker_hit_rate": _mean(
            row["any_observed_blocker_hit"] for row in selected
        ),
        "mean_matched_random_expected_recall": _mean(
            row["matched_random_expected_recall"] for row in selected
        ),
        "mean_recall_lift_over_random": _mean(
            row["recall_lift_over_random"] for row in selected
        ),
        "mean_predicted_agent_count": _mean(
            row["predicted_agent_count"] for row in selected
        ),
        "mean_expansion_ratio": _mean(row["expansion_ratio"] for row in selected),
        "maximum_total_neighborhood_fraction": max(
            (float(row["total_neighborhood_fraction"]) for row in selected),
            default=0.0,
        ),
    }


def _group_summary(
    rows: list[dict[str, Any]], predictor: str, key: str
) -> dict[str, dict[str, Any]]:
    values = sorted({str(row[key]) for row in rows})
    return {
        value: _summary([row for row in rows if str(row[key]) == value], predictor)
        for value in values
    }


def analyze_predictability(
    config_path: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    config_path, root, config, inputs, collection = load_registration(config_path)
    output_dir = Path(output_dir).resolve()
    manifest = _read_jsonl(inputs["causal_state_manifest"])
    if len(manifest) != int(config["cohort"]["required_state_count"]):
        raise ValueError("RepairDependency input state count changed")
    manifest_by_state = {str(row["state_fingerprint"]): row for row in manifest}
    if len(manifest_by_state) != len(manifest):
        raise ValueError("RepairDependency input manifest has duplicate states")
    _metadata, cohort = build_causal_cohort(inputs["causal_registration"])
    cohort_by_state = {str(row["state_fingerprint"]): row for row in cohort}
    if set(cohort_by_state) != set(manifest_by_state):
        raise ValueError("RepairDependency causal cohort changed")
    state_artifacts: dict[str, dict[str, Any]] = {}
    for state_key, manifest_row in manifest_by_state.items():
        path = collection / "states" / str(manifest_row["file_name"])
        if sha256_file(path) != str(manifest_row["sha256"]):
            raise ValueError("RepairDependency causal state artifact changed")
        state_artifacts[state_key] = _read_json(path)
    workers = int(config["execution"]["worker_count"])
    context_jobs = [
        {"state_record": cohort_by_state[state_key]}
        for state_key in sorted(cohort_by_state)
    ]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        contexts = list(executor.map(_context_job, context_jobs, chunksize=1))
    context_by_state = {str(row["state_fingerprint"]): row for row in contexts}
    evaluation_jobs: list[dict[str, Any]] = []
    for state_key in sorted(context_by_state):
        context_row = context_by_state[state_key]
        context = dict(context_row["context"])
        artifact = state_artifacts[state_key]
        for trial_index in map(int, config["cohort"]["trial_indices"]):
            truth = _trial_truth(artifact, trial_index)
            for predictor in PREDICTORS:
                evaluation_jobs.append(
                    {
                        "state_fingerprint": state_key,
                        "map_id": context_row["map_id"],
                        "trial_index": trial_index,
                        "predictor_id": predictor,
                        "truth_blockers": truth,
                        "predicted_agents": context["predictor_sets"][predictor],
                        "selected_agent_count": context["selected_agent_count"],
                        "agent_count": context["agent_count"],
                    }
                )
    with ProcessPoolExecutor(max_workers=workers) as executor:
        raw_rows = list(executor.map(_evaluation_job, evaluation_jobs, chunksize=1))
    rows = [_scientific_row(row) for row in raw_rows]
    rows.sort(
        key=lambda row: (
            str(row["state_fingerprint"]),
            int(row["trial_index"]),
            str(row["predictor_id"]),
        )
    )
    expected_rows = (
        int(config["cohort"]["required_state_count"])
        * len(config["cohort"]["trial_indices"])
        * len(PREDICTORS)
    )
    integrity = {
        "state_count": len(contexts) == int(config["cohort"]["required_state_count"]),
        "row_count": len(rows) == expected_rows,
        "map_count": len({str(row["map_id"]) for row in rows})
        == int(config["cohort"]["required_map_count"]),
        "trial_coverage": all(
            {
                int(row["trial_index"])
                for row in rows
                if str(row["state_fingerprint"]) == state_key
            }
            == set(map(int, config["cohort"]["trial_indices"]))
            for state_key in context_by_state
        ),
        "no_runtime_or_future_labels": all(
            row["runtime_fields_stored"] is False
            and row["future_trajectory_stored"] is False
            for row in rows
        ),
    }
    summaries = {predictor: _summary(rows, predictor) for predictor in PREDICTORS}
    by_map = {
        predictor: _group_summary(rows, predictor, "map_id")
        for predictor in PREDICTORS
    }
    by_half = {
        predictor: _group_summary(rows, predictor, "seed_half")
        for predictor in PREDICTORS
    }
    primary = summaries[PRIMARY_PREDICTOR]
    direct = summaries["direct_conflict_boundary"]
    gates_config = dict(config["readiness_gates"])
    primary_map = by_map[PRIMARY_PREDICTOR]
    direct_map = by_map["direct_conflict_boundary"]
    primary_half = by_half[PRIMARY_PREDICTOR]
    direct_half = by_half["direct_conflict_boundary"]
    gates = {
        "integrity": all(integrity.values()),
        "overall_recall": float(primary["mean_observed_blocker_recall"])
        >= float(gates_config["minimum_overall_recall"]),
        "per_map_recall": all(
            float(row["mean_observed_blocker_recall"])
            >= float(gates_config["minimum_per_map_recall"])
            for row in primary_map.values()
        ),
        "seed_half_recall": all(
            float(row["mean_observed_blocker_recall"])
            >= float(gates_config["minimum_seed_half_recall"])
            for row in primary_half.values()
        ),
        "overall_direct_advantage": float(primary["mean_observed_blocker_recall"])
        - float(direct["mean_observed_blocker_recall"])
        >= float(gates_config["minimum_direct_boundary_recall_advantage"]),
        "per_map_direct_advantage": all(
            float(primary_map[key]["mean_observed_blocker_recall"])
            > float(direct_map[key]["mean_observed_blocker_recall"])
            for key in primary_map
        ),
        "seed_half_direct_advantage": all(
            float(primary_half[key]["mean_observed_blocker_recall"])
            > float(direct_half[key]["mean_observed_blocker_recall"])
            for key in primary_half
        ),
        "random_recall_lift": float(primary["mean_recall_lift_over_random"])
        >= float(gates_config["minimum_overall_random_recall_lift"])
        and all(
            float(row["mean_recall_lift_over_random"]) > 0.0
            for row in primary_map.values()
        ),
        "mean_expansion": float(primary["mean_expansion_ratio"])
        <= float(gates_config["maximum_mean_expansion_ratio"]),
        "not_near_global": float(primary["maximum_total_neighborhood_fraction"])
        < float(gates_config["maximum_total_neighborhood_fraction"]),
    }
    context_pids = {int(row["worker_pid"]) for row in contexts}
    evaluation_pids = [int(row["worker_pid"]) for row in raw_rows]
    execution = {
        "schema": EXECUTION_SCHEMA,
        "worker_limit": workers,
        "task_granularity": "state_x_trial_index_x_predictor",
        "context_task_count": len(context_jobs),
        "evaluation_task_count": len(evaluation_jobs),
        "context_worker_process_count": len(context_pids),
        "evaluation_worker_process_count": len(set(evaluation_pids)),
        "final_32_task_worker_process_count": len(set(evaluation_pids[-32:])),
        "single_coordinator_output_writes": True,
        "scientific_rows_contain_worker_identity": False,
    }
    gates["sixteen_worker_execution"] = (
        execution["context_worker_process_count"] == workers
        and execution["evaluation_worker_process_count"] == workers
    )
    gates["tail_parallelism"] = execution["final_32_task_worker_process_count"] > 1
    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "integrity": integrity,
        "integrity_passed": all(integrity.values()),
        "state_count": len(contexts),
        "trial_count": len(contexts) * len(config["cohort"]["trial_indices"]),
        "row_count": len(rows),
        "predictor_summaries": summaries,
        "by_map": by_map,
        "by_seed_half": by_half,
        "readiness_gates": gates,
        "predictability_passed": all(gates.values()),
        "execution": execution,
        "claim_boundary": dict(config["claim_boundary"]),
        "next_step": (
            "preregister paired PP replay of the frozen outcome-blind frontier"
            if all(gates.values())
            else "stop RepairDependency pool construction; pre-action blocker signal is insufficient"
        ),
        "producer": producer_identity(root, source_files=PRODUCER_FILES),
        "input_sha256": {
            name: sha256_file(path) for name, path in sorted(inputs.items())
        },
        "config_sha256": sha256_file(config_path),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "predictability_rows.jsonl", rows)
    report["rows_sha256"] = sha256_file(output_dir / "predictability_rows.jsonl")
    _write_json(output_dir / "execution_report.json", execution)
    _write_json(output_dir / "predictability_report.json", report)
    return report
