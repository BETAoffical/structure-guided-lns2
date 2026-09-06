from __future__ import annotations

import collections
import statistics
from pathlib import Path
from typing import Any, Mapping

from experiments._common import producer_identity, registered_input, sha256_file
from experiments.closed_loop_trace_storage import read_state_blob
from experiments.repair_collection import (
    _fingerprint,
    _read_json,
    _read_jsonl,
    _run_jobs,
    _write_json,
    _write_jsonl,
    state_fingerprint,
)
from experiments.state_analysis import analyze_state, analyze_static_grid
from experiments.stride_closurepool_longtail import reconstruct_trace
from lns2_selector.runtime.frontierdependencypool import (
    FRONTIERDEPENDENCYPOOL_ID,
    generate_frontierdependency_candidates,
    select_frontierdependency_candidate,
)
from lns2_selector.runtime.temporal_state import history_before_decision


CONFIG_SCHEMA = "lns2.stride.frontierdependencypool_registration.v1"
COHORT_SCHEMA = "lns2.stride.frontierdependencypool_cohort.v1"
REPORT_SCHEMA = "lns2.stride.frontierdependencypool_materialization_report.v1"
STATUS_SCHEMA = "lns2.stride.frontierdependencypool_status.v1"
EXPERIMENT_ID = "stride-frontierdependencypool-v1"
PRE_REGISTRATION_COMMIT = "1c80b144a0290f778e37e2d2fa78ec1bddbe07bb"
PRODUCER_FILES = (
    "experiments/stride_frontierdependencypool.py",
    "experiments/stride_closurepool_longtail.py",
    "lns2_selector/runtime/fingerprints.py",
    "lns2_selector/runtime/frontierdependencypool.py",
    "lns2_selector/runtime/temporal_state.py",
    "lns2_selector/runtime/topology_candidates.py",
)


def load_registration(
    path: str | Path,
) -> tuple[Path, Path, dict[str, Any], dict[str, Path]]:
    path = Path(path).resolve()
    root = path.parents[1]
    config = _read_json(path)
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("scientific_status")
        != "preregistered_platform_entry_zero_solver_frontier_composition_audit"
        or config.get("experiment_id") != EXPERIMENT_ID
        or config.get("pre_registration_commit") != PRE_REGISTRATION_COMMIT
    ):
        raise ValueError("FrontierDependencyPool registration identity changed")
    inputs = {
        name: registered_input(root, dict(specification), label=name)
        for name, specification in dict(config["inputs"]).items()
    }
    if set(inputs) != {
        "root_checkpoints",
        "platform_entry_witnesses",
        "causal_report",
        "causal_status",
    }:
        raise ValueError("FrontierDependencyPool input registry changed")
    generator = dict(config["generator"])
    if generator != {
        "id": FRONTIERDEPENDENCYPOOL_ID,
        "base": "frozen_controller_selected_candidate",
        "motif": "one current selected-to-unselected conflict frontier pivot",
        "semantic_variants": ["compact-augment", "same-size-exchange"],
        "maximum_added_agents_per_candidate": 8,
        "maximum_neighborhood_size": 40,
        "maximum_candidates_per_state": 6,
        "maximum_candidate_jaccard": 0.9,
        "fixed_preferred_sizes": [],
        "repair_order_controlled": False,
        "future_outcomes_used": False,
        "runtime_pp_probe_used": False,
    }:
        raise ValueError("FrontierDependencyPool generator contract changed")
    execution = dict(config["execution"])
    if execution != {
        "workers": 16,
        "per_state_timeout_seconds": 300,
        "stop_on_first_error_or_timeout": True,
        "new_solver_runs": 0,
    }:
        raise ValueError("FrontierDependencyPool execution contract changed")
    return path, root, config, inputs


def _source_trace(
    root: Path, checkpoint: Mapping[str, Any]
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    source = (
        root
        / "build"
        / "stride-tailswitch-v1"
        / "states"
        / _fingerprint({"state_id": str(checkpoint["state_id"])})[:20]
        / str(checkpoint["treatment_policy"])
    )
    manifests = _read_jsonl(source / "realized_dynamic_manifest.jsonl")
    if len(manifests) != 1 or manifests[0].get("status") != "ok":
        raise ValueError("FrontierDependencyPool source trace manifest is invalid")
    manifest = dict(manifests[0])
    trace = reconstruct_trace(source, manifest)
    return source, manifest, trace


def _selected_candidate(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    selected = [
        dict(row)
        for row in checkpoint["candidate_pool"]
        if str(row["candidate_id"]) == str(checkpoint["selected_candidate_id"])
    ]
    if len(selected) != 1:
        raise ValueError("FrontierDependencyPool selected core identity changed")
    row = selected[0]
    if tuple(map(int, row["agents"])) != tuple(map(int, checkpoint["selected_agents"])):
        raise ValueError("FrontierDependencyPool selected core agents changed")
    return row


def _worker(job: dict[str, Any]) -> dict[str, Any]:
    root = Path(str(job["root"]))
    checkpoint = dict(job["checkpoint"])
    witness = dict(job["witness"])
    state_path = Path(str(job["state_path"]))
    if sha256_file(state_path) != str(checkpoint["state_blob_sha256"]):
        raise ValueError("FrontierDependencyPool state blob hash changed")
    state = read_state_blob(state_path)
    state["context"] = dict(checkpoint["state_context"])
    if state_fingerprint(state) != str(checkpoint["state_fingerprint"]):
        raise ValueError("FrontierDependencyPool state fingerprint changed")
    _source, manifest, trace = _source_trace(root, checkpoint)
    if str(manifest["trace_sha256"]) != str(checkpoint["trace_sha256"]):
        raise ValueError("FrontierDependencyPool source trace identity changed")
    target = int(checkpoint["decision_index"])
    if target >= len(trace["states"]):
        raise ValueError("FrontierDependencyPool source trace ended before checkpoint")
    source_state = dict(trace["states"][target])
    source_state["context"] = dict(checkpoint["state_context"])
    if state_fingerprint(source_state) != str(checkpoint["state_fingerprint"]):
        raise ValueError("FrontierDependencyPool source trace state changed")
    history = history_before_decision(
        trace["states"], trace["transitions"], decision_index=target
    )
    analysis = analyze_state(state, static_grid=analyze_static_grid(state))
    base = _selected_candidate(checkpoint)
    limits = dict(job["limits"])
    generated = generate_frontierdependency_candidates(
        state,
        analysis,
        base_candidate=base,
        history=history,
        maximum_candidates=int(limits["maximum_candidates_per_state"]),
        maximum_added_agents=int(limits["maximum_added_agents_per_candidate"]),
        maximum_neighborhood_size=int(limits["maximum_neighborhood_size"]),
        maximum_jaccard_similarity=float(limits["maximum_candidate_jaccard"]),
    )
    selections = {}
    for variant in limits["semantic_variants"]:
        selected, diagnostic = select_frontierdependency_candidate(
            base, generated.candidates, variant=str(variant)
        )
        selections[str(variant)] = {
            **diagnostic,
            "selected_candidate_id": str(selected["candidate_id"]),
            "selected_agents": list(map(int, selected["agents"])),
            "fell_back_to_base": str(selected["candidate_id"])
            == str(base["candidate_id"]),
        }
    diagnostic_target = sorted(
        set(map(int, witness["escape"]["added_agents"]))
        & set(map(int, witness["causal_context"]["external_blocker_union"]))
    )
    target_set = set(diagnostic_target)
    return {
        "status": "ok",
        "job_id": str(checkpoint["case_id"]),
        "error_count": 0,
        "row": {
            "schema": COHORT_SCHEMA,
            "case_id": str(checkpoint["case_id"]),
            "map_id": str(checkpoint["map_id"]),
            "task_id": str(checkpoint["task_id"]),
            "solver_seed": int(checkpoint["solver_seed"]),
            "decision_index": target,
            "state_fingerprint": str(checkpoint["state_fingerprint"]),
            "state_blob": str(state_path),
            "state_blob_sha256": str(checkpoint["state_blob_sha256"]),
            "source_trace_sha256": str(manifest["trace_sha256"]),
            "history_context": history.payload(),
            "history_context_sha256": history.sha256,
            "base_candidate": base,
            "frontierdependency_candidates": generated.candidates,
            "deterministic_selections": selections,
            "generator": {
                "raw_motif_count": generated.raw_motif_count,
                "raw_variant_count": generated.raw_variant_count,
                "selected_candidate_count": len(generated.candidates),
                "exact_duplicate_count": generated.exact_duplicate_count,
                "jaccard_rejection_count": generated.jaccard_rejection_count,
                "attempts": generated.attempts,
            },
            "posthoc_witness_diagnostic": {
                "used_for_generation": False,
                "required_escape_blockers": diagnostic_target,
                "covered_by_one_candidate": bool(target_set)
                and any(
                    target_set <= set(map(int, row["agents"]))
                    for row in generated.candidates
                ),
                "covered_by_candidate_union": bool(target_set)
                and target_set
                <= set().union(
                    *(set(map(int, row["agents"])) for row in generated.candidates)
                ),
                "best_candidate_coverage_count": max(
                    (
                        len(target_set & set(map(int, row["agents"])))
                        for row in generated.candidates
                    ),
                    default=0,
                ),
            },
            "future_outcome_used_for_generation": False,
            "pp_probe_used_for_generation": False,
            "repair_order_controlled": False,
        },
    }


def _render_markdown(report: Mapping[str, Any]) -> str:
    gate = dict(report["static_readiness_gate"])
    sizes = dict(report["candidate_sizes"])
    lines = [
        "# STRIDE FrontierDependencyPool v1 zero-solver report",
        "",
        "This stage uses pre-action state and history only. It runs no PP, continuation, training, or TTF.",
        "",
        "## Candidate pool",
        "",
        f"- States: `{report['state_count']}`",
        f"- States with at least one candidate: `{report['states_with_candidate']}`",
        f"- Candidates: `{report['candidate_count']}`",
        f"- Candidate size mean/range: `{sizes['mean']:.3f}` / `{sizes['minimum']}..{sizes['maximum']}`",
        f"- Compact augment candidates: `{report['variant_counts'].get('compact-augment', 0)}`",
        f"- Same-size exchange candidates: `{report['variant_counts'].get('same-size-exchange', 0)}`",
        "",
        "## Static readiness",
        "",
        f"- State coverage: `{gate['state_coverage_fraction']:.4f}` (required `{gate['minimum_state_coverage_fraction']:.4f}`)",
        f"- Both variants on every map: `{gate['both_variants_on_each_map']}`",
        f"- All size/evidence constraints: `{gate['all_candidate_constraints_passed']}`",
        f"- Passed: `{report['static_readiness_passed']}`",
        "",
        "A pass permits only a separately preregistered native-order bounded forced-continuation test of platform entry.",
        "",
    ]
    return "\n".join(lines)


def materialize_frontierdependencypool(
    config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    config_path, root, config, inputs = load_registration(config_path)
    causal_report = _read_json(inputs["causal_report"])
    causal_status = _read_json(inputs["causal_status"])
    if causal_report.get("integrity_passed") is not True or causal_status.get("status") != "complete":
        raise ValueError("FrontierDependencyPool causal source integrity changed")
    checkpoints = [
        dict(row)
        for row in _read_jsonl(inputs["root_checkpoints"])
        if row.get("checkpoint_kind") == config["cohort"]["checkpoint_kind"]
    ]
    witnesses = {
        str(row["case_id"]): dict(row)
        for row in _read_jsonl(inputs["platform_entry_witnesses"])
    }
    if len(checkpoints) != int(config["cohort"]["state_count"]) or set(
        str(row["case_id"]) for row in checkpoints
    ) != set(witnesses):
        raise ValueError("FrontierDependencyPool cohort identity changed")
    checkpoint_root = inputs["root_checkpoints"].parent
    jobs = [
        {
            "job_id": str(row["case_id"]),
            "root": str(root),
            "checkpoint": row,
            "witness": witnesses[str(row["case_id"])],
            "state_path": str(checkpoint_root / str(row["state_blob"])),
            "limits": config["generator"],
        }
        for row in sorted(checkpoints, key=lambda item: str(item["case_id"]))
    ]
    producer = producer_identity(
        project_root=root,
        source_files=PRODUCER_FILES,
        native_required=False,
        package_names=("numpy",),
    )
    run_fingerprint = _fingerprint(
        {"config_sha256": sha256_file(config_path), "producer": producer}
    )
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    results = _run_jobs(
        _worker,
        jobs,
        int(config["execution"]["workers"]),
        phase="frontierdependencypool-materialize",
        output_root=output,
        run_fingerprint=run_fingerprint,
        timeout_seconds=float(config["execution"]["per_state_timeout_seconds"]),
        stop_on_failure=True,
    )
    failures = [row for row in results if row.get("status") != "ok"]
    if failures or len(results) != len(jobs):
        raise RuntimeError(f"FrontierDependencyPool materialization failed: {failures}")
    cohort = sorted(
        (dict(row["row"]) for row in results), key=lambda row: str(row["case_id"])
    )
    candidates = [
        candidate
        for row in cohort
        for candidate in row["frontierdependency_candidates"]
    ]
    variants = collections.Counter(
        str(row["frontierdependency_variant"]) for row in candidates
    )
    maps = sorted({str(row["map_id"]) for row in cohort})
    by_map = {
        map_id: {
            "state_count": sum(row["map_id"] == map_id for row in cohort),
            "candidate_count": sum(
                len(row["frontierdependency_candidates"])
                for row in cohort
                if row["map_id"] == map_id
            ),
            "variant_counts": dict(
                collections.Counter(
                    candidate["frontierdependency_variant"]
                    for row in cohort
                    if row["map_id"] == map_id
                    for candidate in row["frontierdependency_candidates"]
                )
            ),
        }
        for map_id in maps
    }
    thresholds = dict(config["static_readiness_gate"])
    states_with_candidate = sum(
        bool(row["frontierdependency_candidates"]) for row in cohort
    )
    state_fraction = states_with_candidate / len(cohort)
    all_constraints = all(
        len(candidate["added_blockers"])
        <= int(config["generator"]["maximum_added_agents_per_candidate"])
        and int(candidate["actual_size"])
        <= int(config["generator"]["maximum_neighborhood_size"])
        and int(candidate["converted_boundary_edge_count"])
        >= int(thresholds["minimum_converted_boundary_edges"])
        and set(map(str, candidate["added_blockers"]))
        == set(candidate["blocker_evidence"])
        and all(
            bool(evidence["current_boundary_edges"])
            for evidence in candidate["blocker_evidence"].values()
        )
        and candidate["future_outcome_used"] is False
        and candidate["pp_probe_used"] is False
        and (
            candidate["frontierdependency_variant"] != "same-size-exchange"
            or (
                candidate["same_size_as_core"] is True
                and len(candidate["added_blockers"])
                == len(candidate["removed_agents"])
            )
        )
        and (
            candidate["frontierdependency_variant"] != "compact-augment"
            or (
                not candidate["removed_agents"]
                and int(candidate["actual_size"])
                - len(candidate["core_agents"])
                <= int(thresholds["compact_augment_maximum_growth"])
            )
        )
        for candidate in candidates
    )
    both_variants_on_each_map = len(maps) == int(config["cohort"]["map_count"]) and all(
        set(row["variant_counts"]) == {"compact-augment", "same-size-exchange"}
        for row in by_map.values()
    )
    gate = {
        "state_coverage_fraction": state_fraction,
        "minimum_state_coverage_fraction": float(
            thresholds["minimum_states_with_candidate_fraction"]
        ),
        "state_coverage_passed": state_fraction
        >= float(thresholds["minimum_states_with_candidate_fraction"]),
        "both_variants_on_each_map": both_variants_on_each_map,
        "all_candidate_constraints_passed": all_constraints,
        "future_outcome_not_used": all(
            row["future_outcome_used_for_generation"] is False for row in cohort
        ),
        "pp_probe_not_used": all(
            row["pp_probe_used_for_generation"] is False for row in cohort
        ),
    }
    gate_passed = all(
        value
        for key, value in gate.items()
        if key
        not in {"state_coverage_fraction", "minimum_state_coverage_fraction"}
    )
    target_rows = [
        row["posthoc_witness_diagnostic"]
        for row in cohort
        if row["posthoc_witness_diagnostic"]["required_escape_blockers"]
    ]
    sizes = [int(row["actual_size"]) for row in candidates]
    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "integrity_passed": len(cohort) == int(config["cohort"]["state_count"]),
        "new_solver_run_count": 0,
        "model_training_performed": False,
        "state_count": len(cohort),
        "states_with_candidate": states_with_candidate,
        "candidate_count": len(candidates),
        "variant_counts": dict(variants),
        "candidate_sizes": {
            "minimum": min(sizes),
            "maximum": max(sizes),
            "mean": statistics.fmean(sizes),
        },
        "by_map": by_map,
        "static_readiness_gate": gate,
        "static_readiness_passed": gate_passed,
        "posthoc_witness_diagnostic_not_a_gate": {
            "eligible_case_count": len(target_rows),
            "covered_by_one_candidate_count": sum(
                row["covered_by_one_candidate"] for row in target_rows
            ),
            "covered_by_candidate_union_count": sum(
                row["covered_by_candidate_union"] for row in target_rows
            ),
            "best_candidate_coverage_total": sum(
                row["best_candidate_coverage_count"] for row in target_rows
            ),
        },
        "causal_source_context": {
            "root_cause_counts": dict(causal_report["root_cause_counts"]),
            "mean_baseline_replan_success_rate": float(
                causal_report["mean_baseline_replan_success_rate"]
            ),
            "mean_baseline_external_blocker_count": float(
                causal_report["mean_baseline_external_blocker_count"]
            ),
        },
        "worker_count": int(config["execution"]["workers"]),
        "run_fingerprint": run_fingerprint,
        "producer": producer,
        "config_sha256": sha256_file(config_path),
        "input_sha256": {
            name: sha256_file(path) for name, path in sorted(inputs.items())
        },
        "claim_boundary": dict(config["claim_boundary"]),
        "next_step": (
            config["next_stage_if_passed"]["action"]
            if gate_passed
            else "stop_before_forced_continuation"
        ),
    }
    cohort_path = output / "frontierdependency_cohort.jsonl"
    report_path = output / "materialization_report.json"
    _write_jsonl(cohort_path, cohort)
    report["cohort_sha256"] = sha256_file(cohort_path)
    _write_json(report_path, report)
    (output / "materialization_report.md").write_text(
        _render_markdown(report), encoding="utf-8"
    )
    _write_json(
        output / "collection_status.json",
        {
            "schema": STATUS_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "status": "complete" if gate_passed else "stopped_gate_failed",
            "complete": True,
            "completed_jobs": len(cohort),
            "error_jobs": 0,
            "timeout_jobs": 0,
            "run_fingerprint": run_fingerprint,
            "report_sha256": sha256_file(report_path),
        },
    )
    return report


__all__ = [
    "CONFIG_SCHEMA",
    "EXPERIMENT_ID",
    "materialize_frontierdependencypool",
]
