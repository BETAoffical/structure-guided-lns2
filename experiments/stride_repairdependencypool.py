from __future__ import annotations

import collections
import statistics
from pathlib import Path
from typing import Any

from experiments._common import producer_identity, registered_input, sha256_file
from experiments.closed_loop_trace_storage import read_state_blob
from experiments.compact_controller_model import load_controller_bundle
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
from experiments.stride_plateau_escape_witness import canonical_artifact_manifest
from lns2_selector.runtime.online_selection import (
    online_candidate_rows,
    score_online_candidates,
)
from lns2_selector.runtime.repairdependencypool import (
    REPAIRDEPENDENCYPOOL_ID,
    generate_repairdependency_candidates,
    select_repairdependency_candidate,
)


CONFIG_SCHEMA = "lns2.stride.repairdependencypool_registration.v1"
COHORT_SCHEMA = "lns2.stride.repairdependencypool_cohort.v1"
WITNESS_SCHEMA = "lns2.stride.platform_entry_witness.v1"
REPORT_SCHEMA = "lns2.stride.repairdependencypool_materialization_report.v1"
STATUS_SCHEMA = "lns2.stride.repairdependencypool_status.v1"
EXPERIMENT_ID = "stride-repairdependencypool-v1"
PRE_REGISTRATION_COMMIT = "63b758de2fbaf8d26f635f67fee5b438f944b56a"
PRODUCER_FILES = (
    "experiments/stride_repairdependencypool.py",
    "lns2_selector/runtime/repairdependencypool.py",
    "lns2_selector/runtime/causaltopopool.py",
    "lns2_selector/runtime/topology_candidates.py",
)


def _is_structural_candidate(candidate: dict[str, Any]) -> bool:
    return any(
        str(family).startswith("structpool-")
        for family in candidate.get("selection_families") or ()
    )


def _escape_classification(witness: dict[str, Any]) -> str:
    exit_row = dict(witness["exit"])
    if exit_row["outcome"] == "right_censored":
        return "right_censored"
    if exit_row["outcome"] != "strict_reduction":
        raise ValueError(f"non-strict observed escape: {witness['case_id']}")
    if exit_row["set_change"]:
        return "set_change_escape"
    if not exit_row["same_agent_set"]:
        raise ValueError(f"escape set semantics changed: {witness['case_id']}")
    return "same_set_order_escape"


def _causal_classification(root_cause: str) -> str:
    mapping = {
        "set_defect": "set",
        "order_defect": "order",
        "set_and_order_joint": "joint",
        "residual_pp_instability": "residual_pp",
    }
    try:
        return mapping[str(root_cause)]
    except KeyError as error:
        raise ValueError(f"unknown causal root class: {root_cause}") from error


def _pp_trial_summary(trial: dict[str, Any]) -> dict[str, Any]:
    diagnostic = dict(trial.get("pp_diagnostic") or {})
    agent_rows = []
    for source in diagnostic.get("agents") or ():
        row = dict(source)
        agent_rows.append(
            {
                "agent_id": int(row["agent_id"]),
                "order_index": int(row["order_index"]),
                "inserted_into_path_table": bool(row["inserted_into_path_table"]),
                "path_changed": bool(row["path_changed"]),
                "new_conflict_pairs": [
                    list(map(int, edge)) for edge in row.get("new_conflict_pairs") or ()
                ],
                "internal_blocker_agents": list(
                    map(int, row.get("internal_blocker_agents") or ())
                ),
                "external_blocker_agents": list(
                    map(int, row.get("external_blocker_agents") or ())
                ),
            }
        )
    return {
        "trial_index": int(trial["trial_index"]),
        "pp_seed": int(trial["pp_seed"]),
        "agents": list(map(int, trial["agents"])),
        "repair_order": list(map(int, trial["repair_order"])),
        "replan_success": bool(trial["replan_success"]),
        "repair_outcome": str(trial["repair_outcome"]),
        "no_progress": bool(trial["no_progress"]),
        "conflicts_before": int(trial["before_conflicts"]),
        "conflicts_after": (
            int(trial["conflicts_after"])
            if trial.get("conflicts_after") is not None
            else None
        ),
        "rolled_back": bool(diagnostic.get("rolled_back")),
        "failure_reason": str(diagnostic.get("failure_reason") or "none"),
        "failed_agent": (
            int(diagnostic["failed_agent"])
            if diagnostic.get("failed_agent") is not None
            else None
        ),
        "failed_order_index": int(diagnostic.get("failed_order_index", -1)),
        "before_repair_fingerprint": str(trial["before_repair_fingerprint"]),
        "after_repair_fingerprint": str(trial["after_repair_fingerprint"]),
        "agent_diagnostics": agent_rows,
    }


def _expanded_witness(
    witness: dict[str, Any],
    causal_state: dict[str, Any],
    *,
    first_structural_checkpoint: dict[str, Any],
    platform_checkpoint: dict[str, Any],
) -> dict[str, Any]:
    baseline = sorted(
        (
            dict(row)
            for row in causal_state.get("trials") or ()
            if row.get("arm") == "selected_native_order" and row.get("applicable")
        ),
        key=lambda row: int(row["trial_index"]),
    )
    if [int(row["trial_index"]) for row in baseline] != list(range(16)):
        raise ValueError(f"causal trial coverage changed: {witness['case_id']}")
    root = str(witness["root_cause"])
    return {
        "schema": WITNESS_SCHEMA,
        "case_id": str(witness["case_id"]),
        "state_id": str(witness["state_id"]),
        "map_id": str(witness["map_id"]),
        "task_id": str(witness["task_id"]),
        "solver_seed": int(witness["solver_seed"]),
        "challenger": str(witness["challenger"]),
        "treatment_policy": str(witness["treatment_policy"]),
        "historical_outcome_classification": str(witness["classification"]),
        "escape_classification": _escape_classification(witness),
        "causal_intervention_classification": _causal_classification(root),
        "causal_root_cause": root,
        "first_structural_action": {
            **dict(witness["first_structural"]),
            "candidate_id": str(first_structural_checkpoint["selected_candidate_id"]),
            "agents": list(map(int, first_structural_checkpoint["selected_agents"])),
            "selection_families": list(
                map(str, first_structural_checkpoint["selected_families"])
            ),
            "pre_state_fingerprints": dict(
                first_structural_checkpoint["state_fingerprints"]
            ),
        },
        "platform": dict(witness["plateau"]),
        "platform_state_fingerprint": str(witness["causal_state_fingerprint"]),
        "platform_trigger": {
            "decision_index": int(platform_checkpoint["decision_index"]),
            "candidate_id": str(platform_checkpoint["selected_candidate_id"]),
            "agents": list(map(int, platform_checkpoint["selected_agents"])),
            "selection_families": list(map(str, platform_checkpoint["selected_families"])),
            "pre_state_fingerprints": dict(platform_checkpoint["state_fingerprints"]),
            "before_repair_fingerprint": str(
                causal_state["before_repair_fingerprint"]
            ),
            "before_conflicts": int(causal_state["before_conflicts"]),
        },
        "persistent_conflict_edges": list(witness["plateau"]["persistent_core_edges"]),
        "persistent_conflict_agents": sorted(
            {
                int(agent)
                for edge in witness["plateau"]["persistent_core_edges"]
                for agent in edge
            }
        ),
        "repeated_candidate_id": str(witness["plateau"]["dominant_candidate_id"]),
        "repeated_agent_set": list(witness["plateau"]["dominant_candidate_agents"]),
        "escape": dict(witness["exit"]),
        "causal_native_order_pp_trials_at_platform_checkpoint": [
            _pp_trial_summary(row) for row in baseline
        ],
        "causal_context": dict(witness["causal_context"]),
        "episode": dict(witness["episode"]),
        "source_trace_sha256": str(witness["trace_sha256"]),
        "mechanism_evidence_only": True,
        "online_feature_use_allowed": False,
    }


def _checkpoint_with_fingerprints(
    checkpoint: dict[str, Any], checkpoint_directory: Path
) -> dict[str, Any]:
    state_blob = checkpoint_directory / str(checkpoint["state_blob"])
    if sha256_file(state_blob) != str(checkpoint["state_blob_sha256"]):
        raise ValueError(f"checkpoint blob changed: {checkpoint['case_id']}")
    state = read_state_blob(state_blob)
    state["context"] = dict(checkpoint["state_context"])
    if state_fingerprint(state) != str(checkpoint["state_fingerprint"]):
        raise ValueError(f"checkpoint state changed: {checkpoint['case_id']}")
    result = dict(checkpoint)
    result["state_fingerprints"] = {
        "state": str(checkpoint["state_fingerprint"]),
        "path": _fingerprint(
            [
                {
                    "agent_id": int(row["id"]),
                    "path": list(map(int, row["path"])),
                }
                for row in sorted(state["agents"], key=lambda item: int(item["id"]))
            ]
        ),
        "conflict_edges": _fingerprint(
            sorted(
                tuple(sorted(map(int, edge)))
                for edge in state.get("conflict_edges") or ()
            )
        ),
        "state_blob_sha256": str(checkpoint["state_blob_sha256"]),
    }
    return result


def _materialize_state(job: dict[str, Any]) -> dict[str, Any]:
    checkpoint = dict(job["checkpoint"])
    state_blob = Path(str(job["state_blob"]))
    if sha256_file(state_blob) != str(checkpoint["state_blob_sha256"]):
        raise ValueError("RepairDependencyPool source state blob changed")
    state = read_state_blob(state_blob)
    state["context"] = dict(checkpoint["state_context"])
    state_key = str(checkpoint["state_fingerprint"])
    if state_fingerprint(state) != state_key:
        raise ValueError("RepairDependencyPool source state fingerprint changed")
    analysis = analyze_state(state, static_grid=analyze_static_grid(state))
    v2_candidates = [
        dict(candidate)
        for candidate in checkpoint["candidate_pool"]
        if not _is_structural_candidate(candidate)
    ]
    if not v2_candidates or any(row.get("score") is None for row in v2_candidates):
        raise ValueError("RepairDependencyPool checkpoint has no scored V2 pool")
    anchor = max(
        v2_candidates,
        key=lambda row: (float(row["score"]), str(row["candidate_id"])),
    )
    limits = dict(job["limits"])
    generated = generate_repairdependency_candidates(
        state,
        analysis,
        existing_candidates=v2_candidates,
        maximum_candidates=int(limits["maximum_candidates_per_state"]),
        maximum_added_agents=int(limits["maximum_added_agents_per_core"]),
        maximum_neighborhood_size=int(limits["maximum_neighborhood_size"]),
        maximum_jaccard_similarity=float(limits["maximum_candidate_jaccard"]),
    )
    combined = [*v2_candidates, *generated.candidates]
    model = load_controller_bundle(Path(str(job["v2_bundle_root"]))).main_models[
        "realized_dynamic"
    ]
    feature_rows = online_candidate_rows(state, combined)
    _selected_index, scores, _margin = score_online_candidates(feature_rows, model)
    denominator = max(1, len(combined) - 1)
    scores_by_id = {
        str(candidate["candidate_id"]): float(score) / denominator
        for candidate, score in zip(combined, scores)
    }
    scored_new = [
        {
            **candidate,
            "frozen_v2_quality_score": scores_by_id[str(candidate["candidate_id"])],
        }
        for candidate in generated.candidates
    ]
    scored_anchor = {
        **anchor,
        "frozen_v2_quality_score": scores_by_id[str(anchor["candidate_id"])],
        "selection_source": "frozen_v2_anchor",
        "rejection_reason": None,
    }
    selected, selection = select_repairdependency_candidate(scored_anchor, scored_new)
    return {
        "status": "ok",
        "job_id": str(checkpoint["case_id"]),
        "error_count": 0,
        "row": {
            "schema": COHORT_SCHEMA,
            "case_id": str(checkpoint["case_id"]),
            "state_fingerprint": state_key,
            "state_blob": str(state_blob),
            "state_blob_sha256": str(checkpoint["state_blob_sha256"]),
            "state_context": dict(checkpoint["state_context"]),
            "decision_index": int(checkpoint["decision_index"]),
            "map_id": str(checkpoint["map_id"]),
            "task_id": str(checkpoint["task_id"]),
            "solver_seed": int(checkpoint["solver_seed"]),
            "v2_candidates": v2_candidates,
            "v2_anchor": scored_anchor,
            "repairdependency_candidates": scored_new,
            "deterministic_selection": {
                **selection,
                "selected_candidate_id": str(selected["candidate_id"]),
                "selected_agents": list(map(int, selected["agents"])),
            },
            "generator": {
                "core_count": generated.core_count,
                "raw_variant_count": generated.raw_variant_count,
                "selected_candidate_count": len(scored_new),
                "exact_existing_duplicate_count": (
                    generated.exact_existing_duplicate_count
                ),
                "exact_new_duplicate_count": generated.exact_new_duplicate_count,
                "jaccard_rejection_count": generated.jaccard_rejection_count,
                "oversized_rejection_count": generated.oversized_rejection_count,
                "attempts": generated.attempts,
            },
            "candidate_outcomes_used": False,
            "future_trajectory_used": False,
            "pp_failure_diagnostics_used_for_generation": False,
            "repair_order_controlled": False,
        },
    }


def load_registration(
    config_path: str | Path,
) -> tuple[Path, Path, dict[str, Any], dict[str, Path], Path]:
    path = Path(config_path).resolve()
    root = path.parents[1]
    config = _read_json(path)
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("scientific_status")
        != "preregistered_platform_entry_zero_solver_witness_reconstruction"
        or config.get("experiment_id") != EXPERIMENT_ID
        or config.get("pre_registration_commit") != PRE_REGISTRATION_COMMIT
    ):
        raise ValueError("RepairDependencyPool registration identity changed")
    inputs = {
        name: registered_input(root, dict(specification), label=name)
        for name, specification in dict(config["inputs"]).items()
    }
    if set(inputs) != {
        "plateau_witnesses",
        "plateau_report",
        "root_checkpoints",
        "causal_report",
        "causal_status",
        "frozen_v2_manifest",
    }:
        raise ValueError("RepairDependencyPool input registry changed")
    limits = dict(config["generator"])
    if limits != {
        "id": REPAIRDEPENDENCYPOOL_ID,
        "core_families": [
            "topology_boundary",
            "spatiotemporal_hotspot",
            "path_overlap",
        ],
        "semantic_variants_per_core": ["direct-boundary", "temporal-corridor"],
        "maximum_added_agents_per_core": 8,
        "maximum_neighborhood_size": 32,
        "maximum_candidates_per_state": 6,
        "maximum_candidate_jaccard": 0.9,
        "fixed_preferred_sizes": [],
        "oversized_policy": "reject_without_truncation",
        "future_outcomes_used": False,
        "pp_failure_diagnostics_used_online": False,
    }:
        raise ValueError("RepairDependencyPool generator contract changed")
    execution = dict(config["execution"])
    if execution != {
        "workers": 16,
        "per_state_timeout_seconds": 300,
        "stop_on_first_error_or_timeout": True,
        "new_solver_runs": 0,
    }:
        raise ValueError("RepairDependencyPool execution contract changed")
    boundary = dict(config["claim_boundary"])
    if boundary != {
        "existing_evidence_only": True,
        "new_solver_runs_allowed": False,
        "model_training_allowed": False,
        "runtime_integration_allowed": False,
        "ttf_experiment_allowed": False,
        "longtail_prevention_claim_allowed": False,
        "outcome_enriched_evidence_used_for_generation": False,
        "no_result_based_state_selection": True,
    }:
        raise ValueError("RepairDependencyPool claim boundary changed")
    causal_directory = inputs["causal_report"].parent / "states"
    return path, root, config, inputs, causal_directory


def _render_markdown(report: dict[str, Any]) -> str:
    gate = dict(report["witness_reconstruction_gate"])
    counts = dict(report["escape_classification_counts"])
    lines = [
        "# STRIDE RepairDependencyPool v1 zero-solver report",
        "",
        "This stage reconstructs witnesses and materializes pre-action candidates only; it runs no PP or continuation.",
        "",
        "## Witnesses",
        "",
        f"- Cases: `{report['witness_count']}`",
        f"- Same-set/order escape: `{counts.get('same_set_order_escape', 0)}`",
        f"- Set-change escape: `{counts.get('set_change_escape', 0)}`",
        f"- Right-censored: `{counts.get('right_censored', 0)}`",
        "",
        "## Candidate reconstruction gate",
        "",
        f"- External-blocker-backed set-change cases: `{gate['eligible_case_count']}`",
        f"- Fully covered cases: `{gate['covered_case_count']}`",
        f"- Required: `{gate['minimum_covered_case_count']}`",
        f"- Three-map legal coverage: `{gate['all_maps_have_legal_candidates']}`",
        f"- Passed: `{report['witness_reconstruction_passed']}`",
        "",
        "## Boundary",
        "",
        "A pass only authorizes preregistration of bounded forced-continuation. It is not a TTF or long-tail prevention result.",
        "",
    ]
    return "\n".join(lines)


def _coverage_failure_diagnostic(
    target: list[int], cohort_row: dict[str, Any]
) -> dict[str, Any]:
    target_set = set(map(int, target))
    candidates = list(cohort_row["repairdependency_candidates"])
    best_candidate_coverage = max(
        (
            len(target_set & set(map(int, candidate["agents"])))
            for candidate in candidates
        ),
        default=0,
    )
    attempts = [
        dict(row)
        for row in cohort_row["generator"]["attempts"]
        if row.get("core_agents") is not None
    ]
    feasible_attempts = [row for row in attempts if row.get("decision") == "generated"]

    def attempt_agents(row: dict[str, Any]) -> set[int]:
        return set(map(int, row.get("core_agents") or ())) | set(
            map(int, row.get("proposed_added_blockers") or ())
        )

    best_feasible_coverage = max(
        (len(target_set & attempt_agents(row)) for row in feasible_attempts),
        default=0,
    )
    any_core = {
        agent
        for row in attempts
        for agent in map(int, row.get("core_agents") or ())
    }
    any_visible_blocker = {
        agent
        for row in attempts
        for agent in map(int, row.get("proposed_added_blockers") or ())
    }
    full_in_rejected_oversized = any(
        row.get("rejection_reason") == "variant_oversized_without_truncation"
        and target_set <= attempt_agents(row)
        for row in attempts
    )
    return {
        "required_count": len(target_set),
        "best_final_candidate_coverage_count": best_candidate_coverage,
        "best_feasible_raw_variant_coverage_count": best_feasible_coverage,
        "required_agents_in_any_natural_core": sorted(target_set & any_core),
        "required_agents_visible_as_supported_blockers": sorted(
            target_set & any_visible_blocker
        ),
        "required_agents_not_visible_in_any_core_or_blocker": sorted(
            target_set - any_core - any_visible_blocker
        ),
        "full_coverage_only_in_rejected_oversized_variant": (
            full_in_rejected_oversized
        ),
    }


def materialize_repairdependencypool(
    config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    config_path, root, config, inputs, causal_directory = load_registration(
        config_path
    )
    plateau_report = _read_json(inputs["plateau_report"])
    causal_report = _read_json(inputs["causal_report"])
    causal_status = _read_json(inputs["causal_status"])
    if not (
        plateau_report.get("integrity_passed") is True
        and causal_report.get("integrity_passed") is True
        and causal_status.get("status") == "complete"
    ):
        raise ValueError("RepairDependencyPool source integrity changed")
    witnesses = _read_jsonl(inputs["plateau_witnesses"])
    all_checkpoints = [dict(row) for row in _read_jsonl(inputs["root_checkpoints"])]
    checkpoints = [
        row for row in all_checkpoints if row.get("checkpoint_kind") == "first_repeat_stall"
    ]
    first_structural_checkpoints = [
        row
        for row in all_checkpoints
        if row.get("checkpoint_kind") == "first_structural_selection"
    ]
    if (
        len(witnesses) != 45
        or len(checkpoints) != 45
        or len(first_structural_checkpoints) != 45
    ):
        raise ValueError("RepairDependencyPool frozen cohort changed")
    checkpoint_by_case = {str(row["case_id"]): row for row in checkpoints}
    first_by_case = {
        str(row["case_id"]): row for row in first_structural_checkpoints
    }
    if set(checkpoint_by_case) != {str(row["case_id"]) for row in witnesses}:
        raise ValueError("RepairDependencyPool case identity mismatch")
    artifact_digest, artifact_rows = canonical_artifact_manifest(causal_directory)
    artifact_spec = dict(config["causal_state_artifacts"])
    if (
        len(artifact_rows) != int(artifact_spec["file_count"])
        or artifact_digest != str(artifact_spec["canonical_manifest_sha256"])
    ):
        raise ValueError("RepairDependencyPool causal artifacts changed")
    causal_states = {
        path.stem: _read_json(path) for path in sorted(causal_directory.glob("*.json"))
    }
    root_checkpoint_directory = inputs["root_checkpoints"].parent
    fingerprinted_platform = {
        case_id: _checkpoint_with_fingerprints(row, root_checkpoint_directory)
        for case_id, row in checkpoint_by_case.items()
    }
    fingerprinted_first = {
        case_id: _checkpoint_with_fingerprints(row, root_checkpoint_directory)
        for case_id, row in first_by_case.items()
    }
    expanded = []
    for witness in sorted(witnesses, key=lambda row: str(row["case_id"])):
        state_key = str(witness["causal_state_fingerprint"])
        state = causal_states[state_key]
        if state.get("complete") is not True or state.get("state_fingerprint") != state_key:
            raise ValueError(f"causal state changed: {state_key}")
        case_id = str(witness["case_id"])
        expanded.append(
            _expanded_witness(
                witness,
                state,
                first_structural_checkpoint=fingerprinted_first[case_id],
                platform_checkpoint=fingerprinted_platform[case_id],
            )
        )

    jobs = []
    for case_id, checkpoint in sorted(checkpoint_by_case.items()):
        state_blob = root_checkpoint_directory / str(checkpoint["state_blob"])
        jobs.append(
            {
                "job_id": case_id,
                "checkpoint": checkpoint,
                "state_blob": str(state_blob),
                "limits": config["generator"],
                "v2_bundle_root": str(inputs["frozen_v2_manifest"].parent),
            }
        )
    producer = producer_identity(
        project_root=root,
        source_files=PRODUCER_FILES,
        native_required=False,
        package_names=("numpy",),
    )
    run_fingerprint = _fingerprint(
        {
            "config_sha256": sha256_file(config_path),
            "producer": producer,
        }
    )
    results = _run_jobs(
        _materialize_state,
        jobs,
        int(config["execution"]["workers"]),
        phase="repairdependencypool-materialize",
        output_root=Path(output).resolve(),
        run_fingerprint=run_fingerprint,
        timeout_seconds=float(config["execution"]["per_state_timeout_seconds"]),
        stop_on_failure=True,
    )
    failures = [row for row in results if row.get("status") != "ok"]
    if failures or len(results) != len(jobs):
        raise RuntimeError(f"RepairDependencyPool materialization failed: {failures}")
    cohort = sorted(
        (dict(row["row"]) for row in results), key=lambda row: str(row["case_id"])
    )
    cohort_by_case = {str(row["case_id"]): row for row in cohort}

    reconstruction_rows = []
    for witness in expanded:
        if witness["escape_classification"] != "set_change_escape":
            continue
        target = sorted(
            set(map(int, witness["escape"]["added_agents"]))
            & set(map(int, witness["causal_context"]["external_blocker_union"]))
        )
        if not target:
            continue
        candidates = cohort_by_case[witness["case_id"]][
            "repairdependency_candidates"
        ]
        covering = [
            str(candidate["candidate_id"])
            for candidate in candidates
            if set(target) <= set(map(int, candidate["agents"]))
        ]
        diagnostic = _coverage_failure_diagnostic(target, cohort_by_case[witness["case_id"]])
        reconstruction_rows.append(
            {
                "case_id": witness["case_id"],
                "map_id": witness["map_id"],
                "required_escape_blockers": target,
                "covering_candidate_ids": sorted(covering),
                "covered": bool(covering),
                "coverage_failure_diagnostic": diagnostic,
            }
        )
    thresholds = dict(config["witness_reconstruction_gate"])
    maps = sorted({str(row["map_id"]) for row in cohort})
    by_map = {
        map_id: {
            "state_count": sum(row["map_id"] == map_id for row in cohort),
            "legal_candidate_count": sum(
                len(row["repairdependency_candidates"])
                for row in cohort
                if row["map_id"] == map_id
            ),
        }
        for map_id in maps
    }
    all_candidates = [
        candidate
        for row in cohort
        for candidate in row["repairdependency_candidates"]
    ]
    gate = {
        "eligible_case_count": len(reconstruction_rows),
        "covered_case_count": sum(row["covered"] for row in reconstruction_rows),
        "minimum_covered_case_count": int(
            thresholds["minimum_covered_external_blocker_cases"]
        ),
        "eligible_case_count_is_six": len(reconstruction_rows)
        == int(thresholds["expected_external_blocker_case_count"]),
        "coverage_requirement_met": sum(row["covered"] for row in reconstruction_rows)
        >= int(thresholds["minimum_covered_external_blocker_cases"]),
        "all_maps_have_legal_candidates": len(maps)
        == int(thresholds["expected_map_count"])
        and all(row["legal_candidate_count"] > 0 for row in by_map.values()),
        "maximum_added_agents_respected": all(
            len(row["added_blockers"])
            <= int(config["generator"]["maximum_added_agents_per_core"])
            for row in all_candidates
        ),
        "maximum_neighborhood_size_respected": all(
            int(row["actual_size"])
            <= int(config["generator"]["maximum_neighborhood_size"])
            for row in all_candidates
        ),
        "candidate_budget_respected": all(
            len(row["repairdependency_candidates"])
            <= int(config["generator"]["maximum_candidates_per_state"])
            for row in cohort
        ),
        "all_added_agents_have_evidence": all(
            set(map(str, row["added_blockers"])) == set(row["blocker_evidence"])
            for row in all_candidates
        ),
        "no_near_global_truncation": all(
            row["truncated_to_near_global"] is False for row in all_candidates
        ),
    }
    gate_passed = all(
        value
        for key, value in gate.items()
        if key not in {
            "eligible_case_count",
            "covered_case_count",
            "minimum_covered_case_count",
        }
    )
    escape_counts = collections.Counter(row["escape_classification"] for row in expanded)
    causal_counts = collections.Counter(
        row["causal_intervention_classification"] for row in expanded
    )
    candidate_sizes = [int(row["actual_size"]) for row in all_candidates]
    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "integrity_passed": len(expanded) == 45 and len(cohort) == 45,
        "new_solver_run_count": 0,
        "model_training_performed": False,
        "worker_count": int(config["execution"]["workers"]),
        "witness_count": len(expanded),
        "escape_classification_counts": dict(sorted(escape_counts.items())),
        "causal_intervention_classification_counts": dict(sorted(causal_counts.items())),
        "state_count": len(cohort),
        "v2_candidate_count": sum(len(row["v2_candidates"]) for row in cohort),
        "repairdependency_candidate_count": len(all_candidates),
        "states_with_repairdependency_candidate": sum(
            bool(row["repairdependency_candidates"]) for row in cohort
        ),
        "deterministic_repairdependency_selection_count": sum(
            row["deterministic_selection"]["selection_source"]
            == "repairdependency_unique_pareto"
            for row in cohort
        ),
        "candidate_size": {
            "minimum": min(candidate_sizes, default=0),
            "mean": statistics.fmean(candidate_sizes) if candidate_sizes else 0.0,
            "maximum": max(candidate_sizes, default=0),
        },
        "witness_reconstruction_rows": reconstruction_rows,
        "witness_reconstruction_gate": gate,
        "witness_reconstruction_passed": gate_passed,
        "by_map": by_map,
        "next_step": (
            "preregister_bounded_forced_continuation"
            if gate_passed
            else "stop_before_continuation"
        ),
        "claim_boundary": dict(config["claim_boundary"]),
        "inputs": {
            "registration_sha256": sha256_file(config_path),
            "registered_file_sha256": {
                name: sha256_file(path) for name, path in sorted(inputs.items())
            },
            "causal_state_artifact_manifest_sha256": artifact_digest,
            "causal_state_artifact_sha256": artifact_rows,
            "producer": producer,
            "run_fingerprint": run_fingerprint,
        },
    }
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    witness_path = output / "platform_entry_witnesses.jsonl"
    cohort_path = output / "repairdependency_cohort.jsonl"
    report_path = output / "materialization_report.json"
    markdown_path = output / "materialization_report.md"
    _write_jsonl(witness_path, expanded)
    _write_jsonl(cohort_path, cohort)
    report["witness_sha256"] = sha256_file(witness_path)
    report["cohort_sha256"] = sha256_file(cohort_path)
    _write_json(report_path, report)
    markdown_path.write_text(_render_markdown(report), encoding="utf-8")
    _write_json(
        output / "status.json",
        {
            "schema": STATUS_SCHEMA,
            "complete": True,
            "integrity_passed": report["integrity_passed"],
            "witness_reconstruction_passed": gate_passed,
            "state_count": len(cohort),
            "report_sha256": sha256_file(report_path),
            "witness_sha256": report["witness_sha256"],
            "cohort_sha256": report["cohort_sha256"],
            "markdown_sha256": sha256_file(markdown_path),
        },
    )
    return report


__all__ = [
    "CONFIG_SCHEMA",
    "COHORT_SCHEMA",
    "EXPERIMENT_ID",
    "REPORT_SCHEMA",
    "STATUS_SCHEMA",
    "WITNESS_SCHEMA",
    "load_registration",
    "materialize_repairdependencypool",
]
