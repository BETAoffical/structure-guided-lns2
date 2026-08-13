from __future__ import annotations

import math
import os
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
from lns2_selector.runtime.causaltopopool import (
    CAUSALTOPOPOOL_ID,
    generate_causaltopopool_candidates,
)


DESIGN_SCHEMA = "lns2.stride.causaltopopool_design.v1"
COHORT_SCHEMA = "lns2.stride.causaltopopool_cohort.v1"
REPORT_SCHEMA = "lns2.stride.causaltopopool_compactness_report.v1"
EXPERIMENT_ID = "stride-causaltopopool-v1"
PRODUCER_FILES = (
    "experiments/state_analysis.py",
    "experiments/stride_causaltopopool.py",
    "lns2_selector/runtime/causaltopopool.py",
)


def _load_design(
    path: str | Path,
) -> tuple[Path, Path, dict[str, Any], dict[str, Path]]:
    path = Path(path).resolve()
    root = path.parents[1]
    config = _read_json(path)
    if (
        config.get("schema") != DESIGN_SCHEMA
        or config.get("scientific_status")
        != "preregistered_outcome_blind_hybrid_compactness_audit"
        or config.get("experiment_id") != EXPERIMENT_ID
    ):
        raise ValueError("CausalTopoPool design identity changed")
    inputs = {
        name: registered_input(root, dict(specification), label=name)
        for name, specification in dict(config.get("inputs") or {}).items()
    }
    if set(inputs) != {"causalclosure_cohort", "causalclosure_compactness"}:
        raise ValueError("CausalTopoPool input registry changed")
    generator = dict(config.get("generator") or {})
    if (
        generator.get("id") != CAUSALTOPOPOOL_ID
        or int(generator.get("maximum_topology_candidates", -1)) != 12
        or int(generator.get("maximum_neighborhood_size", -1)) != 64
        or not math.isclose(
            float(generator.get("maximum_topology_jaccard", math.nan)),
            0.9,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or list(generator.get("fixed_preferred_sizes") or ()) != []
        or generator.get("oversized_closure_policy")
        != "reject_exact_closure_without_truncation"
        or generator.get("history_conditioned_candidates") is not False
    ):
        raise ValueError("CausalTopoPool generator contract changed")
    execution = dict(config.get("execution") or {})
    if (
        int(execution.get("workers", -1)) != 16
        or execution.get("job_granularity") != "state"
        or int(execution.get("per_job_timeout_seconds", -1)) != 300
        or execution.get("stop_on_first_error_or_timeout") is not True
    ):
        raise ValueError("CausalTopoPool execution contract changed")
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
        raise ValueError("CausalTopoPool claim boundary changed")
    return path, root, config, inputs


def _runtime_path(value: str) -> Path:
    path = Path(value)
    if path.exists() or os.name != "nt" or not value.startswith("/mnt/"):
        return path
    parts = value.split("/")
    if len(parts) < 4 or len(parts[2]) != 1:
        return path
    return Path(f"{parts[2].upper()}:/" + "/".join(parts[3:]))


def _materialize_state(job: dict[str, Any]) -> dict[str, Any]:
    source = dict(job["source"])
    state_key = str(source["state_fingerprint"])
    state_blob = _runtime_path(str(source["state_blob"]))
    if sha256_file(state_blob) != str(source["state_blob_sha256"]):
        raise ValueError("CausalTopoPool source state blob changed")
    state = read_state_blob(state_blob)
    state["context"] = dict(source["state_context"])
    if state_fingerprint(state) != state_key:
        raise ValueError("CausalTopoPool source state fingerprint changed")
    analysis = analyze_state(state, static_grid=analyze_static_grid(state))
    generator = dict(job["generator"])
    causal = [dict(candidate) for candidate in source["causalclosure_candidates"]]
    existing = [
        *causal,
        *({"candidate_id": identity} for identity in source["base_candidate_ids"]),
    ]
    generated = generate_causaltopopool_candidates(
        state,
        analysis,
        existing_candidates=existing,
        maximum_candidates=int(generator["maximum_topology_candidates"]),
        maximum_neighborhood_size=int(generator["maximum_neighborhood_size"]),
        maximum_jaccard_similarity=float(generator["maximum_topology_jaccard"]),
    )
    topology = [
        {**candidate, **_feature_payload(state, candidate, analysis)}
        for candidate in generated.candidates
    ]
    existing_ids = {
        *map(str, source["base_candidate_ids"]),
        *(str(candidate["candidate_id"]) for candidate in causal),
    }
    if any(str(candidate["candidate_id"]) in existing_ids for candidate in topology):
        raise RuntimeError("CausalTopoPool retained an exact existing duplicate")
    if any(candidate.get("causaltopo_truncated") is not False for candidate in topology):
        raise RuntimeError("CausalTopoPool emitted a truncated closure")
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
            "source_run_config": str(source["source_run_config"]),
            "source_run_config_sha256": str(source["source_run_config_sha256"]),
            "logical_checkpoint_ids": list(source["logical_checkpoint_ids"]),
            "v2_anchors": list(source["v2_anchors"]),
            "base_candidate_ids": list(source["base_candidate_ids"]),
            "causalclosure_candidates": causal,
            "causaltopology_candidates": topology,
            "generator": {
                "closure_attempt_count": generated.closure_attempt_count,
                "oversized_closure_count": generated.oversized_closure_count,
                "undersized_closure_count": generated.undersized_closure_count,
                "exact_existing_duplicate_count": (
                    generated.exact_existing_duplicate_count
                ),
                "raw_candidate_count": generated.raw_candidate_count,
                "pareto_front_count": generated.pareto_front_count,
                "selected_candidate_count": len(topology),
                "attempts": generated.attempts,
            },
            "candidate_outcomes_used": False,
            "future_trajectory_used": False,
            "history_context_used": False,
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


def materialize_causaltopopool_cohort(
    design_path: str | Path, output: str | Path
) -> dict[str, Any]:
    design_path, root, config, inputs = _load_design(design_path)
    source_rows = _read_jsonl(inputs["causalclosure_cohort"])
    if len(source_rows) != 78:
        raise ValueError("CausalTopoPool source cohort is not the frozen 78 states")
    if sum(len(row["causalclosure_candidates"]) for row in source_rows) != 930:
        raise ValueError("CausalTopoPool source CausalClosure pool changed")
    if sum(len(row["base_candidate_ids"]) for row in source_rows) != 1367:
        raise ValueError("CausalTopoPool source V2 base pool changed")

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    jobs = [
        {
            "job_id": row["state_fingerprint"],
            "source": row,
            "generator": config["generator"],
        }
        for row in source_rows
    ]
    results = _run_jobs(
        _materialize_state,
        jobs,
        int(config["execution"]["workers"]),
        phase="causaltopopool-materialize",
        output_root=output,
        run_fingerprint=sha256_file(design_path),
        timeout_seconds=float(config["execution"]["per_job_timeout_seconds"]),
        stop_on_failure=True,
    )
    failures = [row for row in results if row.get("status") != "ok"]
    if failures or len(results) != len(jobs):
        raise RuntimeError(f"CausalTopoPool materialization failed: {failures}")
    rows = sorted(
        (dict(result["row"]) for result in results),
        key=lambda row: str(row["state_fingerprint"]),
    )
    manifest = output / "causaltopopool_cohort.jsonl"
    _write_jsonl(manifest, rows)

    causal_sizes = [
        int(candidate["actual_size"])
        for row in rows
        for candidate in row["causalclosure_candidates"]
    ]
    topology_sizes = [
        int(candidate["actual_size"])
        for row in rows
        for candidate in row["causaltopology_candidates"]
    ]
    family_counts: Counter[str] = Counter(
        family
        for row in rows
        for candidate in row["causaltopology_candidates"]
        for family in candidate["structpool_family_groups"]
    )
    level_counts: Counter[str] = Counter(
        level
        for row in rows
        for candidate in row["causaltopology_candidates"]
        for level in candidate["causaltopo_closure_levels"]
    )
    closure_attempts = sum(int(row["generator"]["closure_attempt_count"]) for row in rows)
    oversized = sum(int(row["generator"]["oversized_closure_count"]) for row in rows)
    undersized = sum(int(row["generator"]["undersized_closure_count"]) for row in rows)
    exact_existing = sum(
        int(row["generator"]["exact_existing_duplicate_count"]) for row in rows
    )
    topology_counts = [len(row["causaltopology_candidates"]) for row in rows]
    truncated = sum(
        candidate.get("causaltopo_truncated") is not False
        for row in rows
        for candidate in row["causaltopology_candidates"]
    )
    source_compactness = _read_json(inputs["causalclosure_compactness"])
    source_manifest_hash_ok = (
        str(source_compactness.get("manifest_sha256"))
        == sha256_file(inputs["causalclosure_cohort"])
    )
    configured_gates = dict(config["readiness_gates"])
    gates = {
        "all_78_states_materialized": len(rows) == 78,
        "v2_base_pool_preserved": sum(len(row["base_candidate_ids"]) for row in rows)
        == int(configured_gates["expected_base_candidate_count"]),
        "causal_pool_preserved": sum(
            len(row["causalclosure_candidates"]) for row in rows
        )
        == int(configured_gates["expected_causal_candidate_count"]),
        "minimum_topology_candidates_per_state": min(topology_counts, default=0)
        >= int(configured_gates["minimum_topology_candidates_per_state"]),
        "maximum_topology_candidates_per_state": max(topology_counts, default=0)
        <= int(configured_gates["maximum_topology_candidates_per_state"]),
        "no_truncated_closures": truncated == 0,
        "maximum_neighborhood_size_respected": max(topology_sizes, default=0)
        <= int(config["generator"]["maximum_neighborhood_size"]),
        "no_exact_existing_duplicates": all(
            not (
                {str(candidate["candidate_id"]) for candidate in row["causaltopology_candidates"]}
                & (
                    set(map(str, row["base_candidate_ids"]))
                    | {
                        str(candidate["candidate_id"])
                        for candidate in row["causalclosure_candidates"]
                    }
                )
            )
            for row in rows
        ),
        "source_manifest_hash_consistent": source_manifest_hash_ok,
        "zero_errors_and_timeouts": True,
    }

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["map_id"])].append(row)
    by_map = {}
    for map_id, map_rows in sorted(grouped.items()):
        map_topology = [
            int(candidate["actual_size"])
            for row in map_rows
            for candidate in row["causaltopology_candidates"]
        ]
        by_map[map_id] = {
            "state_count": len(map_rows),
            "causal_candidate_count": sum(
                len(row["causalclosure_candidates"]) for row in map_rows
            ),
            "topology_candidate_count": len(map_topology),
            "minimum_topology_candidates_per_state": min(
                len(row["causaltopology_candidates"]) for row in map_rows
            ),
            "maximum_topology_candidates_per_state": max(
                len(row["causaltopology_candidates"]) for row in map_rows
            ),
            "topology_sizes": _size_summary(map_topology),
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
        "base_candidate_count": sum(len(row["base_candidate_ids"]) for row in rows),
        "causal_candidate_count": len(causal_sizes),
        "topology_candidate_count": len(topology_sizes),
        "mean_topology_candidates_per_state": statistics.fmean(topology_counts),
        "minimum_topology_candidates_per_state": min(topology_counts),
        "maximum_topology_candidates_per_state": max(topology_counts),
        "causal_sizes": _size_summary(causal_sizes),
        "topology_sizes": _size_summary(topology_sizes),
        "closure_attempt_count": closure_attempts,
        "oversized_exact_closure_rejection_count": oversized,
        "undersized_closure_rejection_count": undersized,
        "exact_existing_duplicate_rejection_count": exact_existing,
        "truncated_closure_count": truncated,
        "topology_family_counts": dict(sorted(family_counts.items())),
        "topology_closure_level_counts": dict(sorted(level_counts.items())),
        "by_map": by_map,
        "gates": gates,
        "compactness_readiness_passed": all(gates.values()),
        "candidate_outcomes_used": False,
        "history_context_used": False,
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
    "materialize_causaltopopool_cohort",
]
