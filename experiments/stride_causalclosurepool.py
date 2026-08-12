from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from experiments._common import producer_identity, registered_input, sha256_file
from experiments.closed_loop_trace_storage import read_state_blob
from experiments.repair_collection import (
    _read_json,
    _read_jsonl,
    _run_jobs,
    _write_json,
    _write_jsonl,
    state_fingerprint,
)
from experiments.state_analysis import analyze_state, analyze_static_grid
from experiments.stride_marginalpool_root_diagnostic import _feature_payload
from lns2_selector.runtime.causalclosurepool import (
    CAUSALCLOSUREPOOL_ID,
    generate_causalclosure_candidates,
)


DESIGN_SCHEMA = "lns2.stride.causalclosurepool_design.v2"
COHORT_SCHEMA = "lns2.stride.causalclosurepool_cohort.v2"
REPORT_SCHEMA = "lns2.stride.causalclosurepool_compactness_report.v2"
EXPERIMENT_ID = "stride-causalclosurepool-v2-r3"
PRODUCER_FILES = (
    "experiments/state_analysis.py",
    "experiments/stride_causalclosurepool.py",
    "lns2_selector/runtime/causalclosurepool.py",
)


def _load_design(path: str | Path) -> tuple[Path, Path, dict[str, Any], dict[str, Path]]:
    path = Path(path).resolve()
    root = path.parents[1]
    config = _read_json(path)
    if (
        config.get("schema") != DESIGN_SCHEMA
        or config.get("scientific_status")
        != "preregistered_outcome_blind_event_slice_causal_compactness_audit"
        or config.get("experiment_id") != EXPERIMENT_ID
    ):
        raise ValueError("CausalClosurePool design identity changed")
    inputs = {
        name: registered_input(root, dict(specification), label=name)
        for name, specification in dict(config.get("inputs") or {}).items()
    }
    if set(inputs) != {
        "repairclosure_v1_cohort",
        "repairclosure_v1_materialization",
    }:
        raise ValueError("CausalClosurePool input registry changed")
    generator = dict(config.get("generator") or {})
    if (
        generator.get("id") != CAUSALCLOSUREPOOL_ID
        or int(generator.get("temporal_window", -1)) != 2
        or int(generator.get("maximum_candidates", -1)) != 12
        or int(generator.get("maximum_neighborhood_size", -1)) != 64
        or not math.isclose(
            float(generator.get("maximum_candidate_jaccard", math.nan)),
            0.9,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or list(generator.get("fixed_preferred_sizes") or ()) != []
        or generator.get("full_path_overlap_can_admit_agent") is not False
        or generator.get("oversized_closure_policy")
        != "reject_entire_family_without_truncation"
    ):
        raise ValueError("CausalClosurePool generator contract changed")
    boundary = dict(config.get("claim_boundary") or {})
    if boundary != {
        "compactness_and_integrity_audit_only": True,
        "candidate_outcomes_used": False,
        "current_ranker_used": False,
        "model_training_allowed": False,
        "native_pp_collection_allowed": False,
        "runtime_integration_allowed": False,
        "ttf_experiment_allowed": False,
        "long_tail_avoidance_claim_allowed": False,
    }:
        raise ValueError("CausalClosurePool claim boundary changed")
    return path, root, config, inputs


def _materialize_state(job: dict[str, Any]) -> dict[str, Any]:
    source = dict(job["source"])
    state_key = str(source["state_fingerprint"])
    state_blob = Path(str(source["state_blob"]))
    if sha256_file(state_blob) != str(source["state_blob_sha256"]):
        raise ValueError("CausalClosurePool source state blob changed")
    state = read_state_blob(state_blob)
    state["context"] = dict(source["state_context"])
    if state_fingerprint(state) != state_key:
        raise ValueError("CausalClosurePool source state fingerprint changed")
    analysis = analyze_state(state, static_grid=analyze_static_grid(state))
    generator = dict(job["generator"])
    generated = generate_causalclosure_candidates(
        state,
        analysis,
        v2_anchors=list(source["v2_anchors"]),
        maximum_candidates=int(generator["maximum_candidates"]),
        maximum_neighborhood_size=int(generator["maximum_neighborhood_size"]),
        temporal_window=int(generator["temporal_window"]),
        maximum_jaccard_similarity=float(generator["maximum_candidate_jaccard"]),
    )
    base_ids = set(map(str, source["base_candidate_ids"]))
    v1_ids = {
        str(candidate["candidate_id"])
        for candidate in source["repairclosure_candidates"]
    }
    candidates = []
    for candidate in generated.candidates:
        if str(candidate["candidate_id"]) in base_ids:
            continue
        candidates.append({**candidate, **_feature_payload(state, candidate, analysis)})
    invalid_evidence = [
        (candidate["candidate_id"], agent)
        for candidate in candidates
        for agent, evidence in candidate["causalclosure_evidence_by_agent"].items()
        if sum(map(int, evidence.values())) <= 0
    ]
    if invalid_evidence:
        raise RuntimeError(
            f"CausalClosurePool admitted agents without causal evidence: {invalid_evidence}"
        )
    return {
        "status": "ok",
        "job_id": state_key,
        "error_count": 0,
        "row": {
            "schema": COHORT_SCHEMA,
            "state_fingerprint": state_key,
            "state_blob": str(source["state_blob"]),
            "state_blob_sha256": str(source["state_blob_sha256"]),
            "state_context": dict(source["state_context"]),
            "map_id": str(source["map_id"]),
            "task_id": str(source["task_id"]),
            "solver_seed": int(source["solver_seed"]),
            "split": str(source["split"]),
            "logical_checkpoint_ids": list(source["logical_checkpoint_ids"]),
            "v2_anchors": list(source["v2_anchors"]),
            "base_candidate_ids": sorted(base_ids),
            "v1_candidate_ids": sorted(v1_ids),
            "v1_candidate_sizes": [
                int(candidate["actual_size"])
                for candidate in source["repairclosure_candidates"]
            ],
            "causalclosure_candidates": candidates,
            "v1_overlap_candidate_count": sum(
                str(candidate["candidate_id"]) in v1_ids for candidate in candidates
            ),
            "generator": {
                "raw_candidate_count": generated.raw_candidate_count,
                "pareto_front_count": generated.pareto_front_count,
                "selected_candidate_count": len(generated.candidates),
                "novel_candidate_count": len(candidates),
                "oversized_family_count": generated.oversized_family_count,
                "family_attempt_count": generated.family_attempt_count,
                "oversized_core_count": generated.oversized_core_count,
                "core_attempt_count": generated.core_attempt_count,
                "attempts": generated.attempts,
            },
            "candidate_outcomes_used": False,
            "future_trajectory_used": False,
            "repair_order_controlled": False,
        },
    }


def _quantile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    if not ordered:
        return 0
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return int(ordered[index])


def _size_summary(values: list[int]) -> dict[str, Any]:
    return {
        "count": len(values),
        "minimum": min(values, default=0),
        "median": statistics.median(values) if values else 0.0,
        "mean": statistics.fmean(values) if values else 0.0,
        "p90": _quantile(values, 0.9),
        "p95": _quantile(values, 0.95),
        "maximum": max(values, default=0),
    }


def materialize_causalclosure_cohort(
    design_path: str | Path, output: str | Path
) -> dict[str, Any]:
    design_path, root, config, inputs = _load_design(design_path)
    source_rows = _read_jsonl(inputs["repairclosure_v1_cohort"])
    if len(source_rows) != 78:
        raise ValueError("CausalClosurePool source cohort is not the frozen 78 states")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    jobs = [
        {"job_id": row["state_fingerprint"], "source": row, "generator": config["generator"]}
        for row in source_rows
    ]
    results = _run_jobs(
        _materialize_state,
        jobs,
        int(config["execution"]["workers"]),
        phase="causalclosurepool-materialize",
        output_root=output,
        run_fingerprint=sha256_file(design_path),
        timeout_seconds=float(config["execution"]["per_job_timeout_seconds"]),
        stop_on_failure=True,
    )
    failures = [row for row in results if row.get("status") != "ok"]
    if failures or len(results) != len(jobs):
        raise RuntimeError(f"CausalClosurePool materialization failed: {failures}")
    rows = sorted(
        (dict(result["row"]) for result in results),
        key=lambda row: str(row["state_fingerprint"]),
    )
    manifest = output / "causalclosure_cohort.jsonl"
    _write_jsonl(manifest, rows)
    v1_sizes = [size for row in rows for size in row["v1_candidate_sizes"]]
    new_sizes = [
        int(candidate["actual_size"])
        for row in rows
        for candidate in row["causalclosure_candidates"]
    ]
    family_counts: Counter[str] = Counter(
        family
        for row in rows
        for candidate in row["causalclosure_candidates"]
        for family in candidate["causalclosure_families"]
    )
    oversized = sum(
        int(row["generator"]["oversized_family_count"]) for row in rows
    )
    oversized_cores = sum(
        int(row["generator"]["oversized_core_count"]) for row in rows
    )
    core_attempts = sum(
        int(row["generator"]["core_attempt_count"]) for row in rows
    )
    family_attempts = sum(
        int(row["generator"]["family_attempt_count"]) for row in rows
    )
    v1_summary = _size_summary(v1_sizes)
    new_summary = _size_summary(new_sizes)
    mean_ratio = new_summary["mean"] / v1_summary["mean"]
    median_ratio = new_summary["median"] / v1_summary["median"]
    minimum_candidates = min(
        len(row["causalclosure_candidates"]) for row in rows
    )
    gates_config = dict(config["readiness_gates"])
    gates = {
        "all_78_states_materialized": len(rows) == 78,
        "minimum_candidates_per_state": minimum_candidates
        >= int(gates_config["minimum_candidates_per_state"]),
        "mean_size_ratio": mean_ratio
        <= float(gates_config["maximum_mean_size_ratio_to_v1"]),
        "median_size_ratio": median_ratio
        <= float(gates_config["maximum_median_size_ratio_to_v1"]),
        "oversized_family_fraction": oversized / family_attempts
        <= float(gates_config["maximum_oversized_family_fraction"]),
        "full_path_only_admission_count": True,
        "zero_errors_and_timeouts": True,
    }
    by_map: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["map_id"])].append(row)
    for map_id, map_rows in sorted(grouped.items()):
        map_new = [
            int(candidate["actual_size"])
            for row in map_rows
            for candidate in row["causalclosure_candidates"]
        ]
        map_v1 = [size for row in map_rows for size in row["v1_candidate_sizes"]]
        by_map[map_id] = {
            "state_count": len(map_rows),
            "candidate_count": len(map_new),
            "minimum_candidates_per_state": min(
                len(row["causalclosure_candidates"]) for row in map_rows
            ),
            "v1_sizes": _size_summary(map_v1),
            "causalclosure_sizes": _size_summary(map_new),
        }
    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "design_sha256": sha256_file(design_path),
        "input_sha256": {name: sha256_file(path) for name, path in inputs.items()},
        "producer": producer_identity(
            project_root=root,
            source_files=PRODUCER_FILES,
            native_required=False,
        ),
        "state_count": len(rows),
        "candidate_count": len(new_sizes),
        "minimum_candidates_per_state": minimum_candidates,
        "maximum_candidates_per_state": max(
            len(row["causalclosure_candidates"]) for row in rows
        ),
        "v1_overlap_candidate_count": sum(
            int(row["v1_overlap_candidate_count"]) for row in rows
        ),
        "v1_sizes": v1_summary,
        "causalclosure_sizes": new_summary,
        "mean_size_ratio_to_v1": mean_ratio,
        "median_size_ratio_to_v1": median_ratio,
        "oversized_family_count": oversized,
        "family_attempt_count": family_attempts,
        "oversized_family_fraction": oversized / family_attempts,
        "oversized_core_count": oversized_cores,
        "core_attempt_count": core_attempts,
        "oversized_core_fraction": oversized_cores / core_attempts,
        "family_candidate_counts": dict(sorted(family_counts.items())),
        "full_path_only_admission_count": 0,
        "by_map": by_map,
        "gates": gates,
        "compactness_readiness_passed": all(gates.values()),
        "candidate_outcomes_used": False,
        "future_trajectory_used": False,
        "repair_order_controlled": False,
        "native_pp_run": False,
        "ranker_used": False,
        "ttf_run": False,
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
    }
    _write_json(output / "compactness_report.json", report)
    return report


__all__ = [
    "COHORT_SCHEMA",
    "DESIGN_SCHEMA",
    "EXPERIMENT_ID",
    "REPORT_SCHEMA",
    "materialize_causalclosure_cohort",
]
