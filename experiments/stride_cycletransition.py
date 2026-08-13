from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from experiments._common import producer_identity, registered_input, sha256_file
from experiments.closed_loop_trace_storage import read_state_blob
from experiments.feature_schema_v2 import PROFILE_FEATURE_NAMES
from experiments.online_feature_engine import OnlineFeatureEngine
from experiments.repair_collection import (
    _fingerprint,
    _plain,
    _read_json,
    _read_jsonl,
    _run_jobs,
    _write_json,
    _write_jsonl,
    state_fingerprint,
)
from experiments.stride_collection import _paired_action, _validate_native_repair
from experiments.stride_marginalpool_action_replay import (
    _replay_job as _marginalpool_replay_job,
    build_frozen_cohort,
)
from experiments.stride_repairability_collection import (
    repairability_pp_seed,
    repairability_restore_seed,
)
from experiments.trace_replay import restore_repair_state
from lns2_selector.runtime.fingerprints import repair_structure_fingerprint
from lns2_selector.runtime.online_selection import (
    generate_online_candidates,
    score_online_candidates,
    slotpool_runtime_augmentation,
)
from lns2_selector.runtime.repair_outcomes import classify_repair_outcome
from lns2_selector.training import load_frozen_policy_bundle


CONFIG_SCHEMA = "lns2.stride.cycletransition_pilot_registration.v1"
SCIENTIFIC_STATUS = (
    "preregistered_pre_action_quality_constrained_cycle_transition_readiness"
)
EXPERIMENT_ID = "stride-cycletransition-pilot-v1"
PREPARATION_SCHEMA = "lns2.stride.cycletransition_preparation.v1"
PREPARATION_REPORT_SCHEMA = "lns2.stride.cycletransition_preparation_report.v1"
TRIAL_SCHEMA = "lns2.stride.cycletransition_trial.v1"
COLLECTION_STATUS_SCHEMA = "lns2.stride.cycletransition_collection_status.v1"
REPORT_SCHEMA = "lns2.stride.cycletransition_readiness_report.v1"
PROFILE = "realized_dynamic"
TRIAL_INDICES = tuple(range(8))
FIRST_HALF = tuple(range(4))
SECOND_HALF = tuple(range(4, 8))
STRUCTURAL_SIZES = (8, 16, 24, 32)
STRUCTURAL_FAMILIES = (
    "bottleneck_crossing",
    "conflict_component",
    "topology_boundary_articulation",
    "topology_boundary_low_degree",
    "spatiotemporal_hotspot",
    "path_overlap",
)
PRODUCER_FILES = (
    "CMakeLists.txt",
    "experiments/_common.py",
    "experiments/closed_loop_trace_storage.py",
    "experiments/feature_schema_v2.py",
    "experiments/neighborhood_features.py",
    "experiments/online_feature_engine.py",
    "experiments/repair_collection.py",
    "experiments/state_analysis.py",
    "experiments/stride_collection.py",
    "experiments/stride_cycletransition.py",
    "experiments/stride_marginalpool_action_replay.py",
    "experiments/stride_marginalpool_root_diagnostic.py",
    "experiments/stride_repairability_collection.py",
    "experiments/trace_replay.py",
    "lns2_selector/runtime/fingerprints.py",
    "lns2_selector/runtime/online_selection.py",
    "lns2_selector/runtime/repair_outcomes.py",
    "lns2_selector/runtime/topology_candidates.py",
    "lns2_selector/training/policy_bundle.py",
    "scripts/run_stride_cycletransition.py",
    "src/python_bindings.cpp",
    "third_party/mapf_lns2/inc/RepairPolicy.h",
    "third_party/mapf_lns2/src/InitLNS.cpp",
)


def _mean(values: Iterable[float | int | bool]) -> float:
    rows = [float(value) for value in values]
    return statistics.fmean(rows) if rows else 0.0


def _candidate_jaccard(left: Iterable[int], right: Iterable[int]) -> float:
    first = set(map(int, left))
    second = set(map(int, right))
    return len(first & second) / max(1, len(first | second))


def _feature_digest(features: dict[str, float]) -> str:
    return hashlib.sha256(
        json.dumps(
            features,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _selection_digest(map_id: str, state_key: str) -> str:
    return _fingerprint(
        {
            "namespace": EXPERIMENT_ID,
            "map_id": str(map_id),
            "state_fingerprint": str(state_key),
        }
    )


def validate_registration(config: dict[str, Any]) -> None:
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("scientific_status") != SCIENTIFIC_STATUS
        or config.get("experiment_id") != EXPERIMENT_ID
        or config.get("pre_registration_head")
        != "7c1e792d56a840a406285f0a58d44e57302dd88d"
    ):
        raise ValueError("CycleTransition registration identity changed")
    cohort = dict(config.get("cohort") or {})
    if (
        cohort.get("source_checkpoint_kind") != "first_structural_selection"
        or int(cohort.get("source_logical_checkpoint_count", -1)) != 45
        or int(cohort.get("states_per_map", -1)) != 6
        or int(cohort.get("state_count", -1)) != 18
        or list(map(str, cohort.get("maps") or ()))
        != ["maze-128-128-1", "maze-128-128-2", "maze-32-32-4"]
        or cohort.get("classification_and_all_repair_outcomes_forbidden_during_selection")
        is not True
        or cohort.get("no_result_based_exclusion") is not True
    ):
        raise ValueError("CycleTransition cohort contract changed")
    candidates = dict(config.get("candidate_contract") or {})
    if (
        list(map(int, candidates.get("structural_sizes") or ()))
        != list(STRUCTURAL_SIZES)
        or list(map(str, candidates.get("structural_families") or ()))
        != list(STRUCTURAL_FAMILIES)
        or candidates.get("six_candidate_reduction_allowed") is not False
        or candidates.get("fixed_family_preferred_size_allowed") is not False
        or candidates.get("v2_anchor")
        != "score base candidates only with frozen v2-full realized_dynamic ranker"
        or int(candidates.get("maximum_structural_candidates", -1)) != 24
        or int(candidates.get("maximum_total_candidates", -1)) != 48
    ):
        raise ValueError("CycleTransition candidate contract changed")
    execution = dict(config.get("execution") or {})
    if (
        int(execution.get("prepare_workers", -1)) != 16
        or int(execution.get("collection_workers", -1)) != 16
        or int(execution.get("prepare_job_timeout_seconds", -1)) != 600
        or int(execution.get("per_job_timeout_seconds", -1)) != 300
        or tuple(map(int, execution.get("trial_indices") or ())) != TRIAL_INDICES
        or execution.get("job_unit")
        != "one state, one candidate, and one paired PP trial"
        or execution.get("stop_on_first_error_or_timeout") is not True
        or execution.get("native_repair_semantics_unchanged") is not True
        or execution.get("replan_algorithm") != "PP"
        or execution.get("use_sipp") is not True
        or execution.get("raw_ttf_timing_run") is not False
    ):
        raise ValueError("CycleTransition execution contract changed")
    observations = dict(config.get("transition_observations") or {})
    if (
        int(observations.get("forced_action_count", -1)) != 1
        or int(observations.get("continuation_repair_actions", -1)) != 0
        or int(observations.get("shadow_next_selection_count", -1)) != 1
        or observations.get("future_repair_trajectory_stored") is not False
        or observations.get("ttf_or_runtime_label_stored") is not False
    ):
        raise ValueError("CycleTransition observation contract changed")
    quality = dict(config.get("quality_constraint") or {})
    if (
        float(quality.get("minimum_seed_mean_delta", math.nan)) != 0.0
        or float(quality.get("maximum_no_progress_rate_delta", math.nan)) != 0.0
        or float(quality.get("minimum_each_fixed_four_seed_half_delta", math.nan))
        != -0.02
        or quality.get(
            "quality_is_a_hard_admissibility_constraint_not_a_weighted_risk_tradeoff"
        )
        is not True
    ):
        raise ValueError("CycleTransition quality constraint changed")
    gates = dict(config.get("label_readiness_gates") or {})
    if gates != {
        "minimum_half_split_soft_cycle_class_agreement": 0.8,
        "minimum_half_split_soft_cycle_rate_rank_correlation": 0.6,
        "minimum_each_map_half_split_soft_cycle_class_agreement": 0.7,
        "minimum_each_map_half_split_soft_cycle_rate_rank_correlation": 0.5,
        "minimum_state_fraction_with_quality_admissible_structural_cycle_escape": 0.6,
        "minimum_each_map_state_fraction_with_quality_admissible_structural_cycle_escape": 0.4,
        "minimum_soft_cycle_rate_reduction_for_escape": 0.25,
        "maximum_exact_noop_rate_delta_for_escape": 0.0,
        "minimum_maps_with_both_positive_and_negative_soft_cycle_actions": 3,
        "maximum_single_size_share_among_escape_actions": 0.75,
        "required_error_count": 0,
        "required_timeout_count": 0,
    }:
        raise ValueError("CycleTransition label-readiness gates changed")
    decision = dict(config.get("decision") or {})
    boundary = dict(config.get("claim_boundary") or {})
    if (
        decision.get("risk_only_selector_allowed") is not False
        or decision.get("online_tentative_pp_probe_allowed") is not False
        or boundary.get("model_training_allowed") is not False
        or boundary.get("runtime_integration_allowed") is not False
        or boundary.get("ttf_experiment_allowed") is not False
        or boundary.get("long_tail_avoidance_claim_allowed") is not False
    ):
        raise ValueError("CycleTransition claim boundary changed")


def load_cycletransition_registration(
    config_path: str | Path,
) -> tuple[Path, Path, dict[str, Any], dict[str, Path]]:
    path = Path(config_path).resolve()
    root = path.parents[1]
    config = _read_json(path)
    validate_registration(config)
    inputs = {
        name: registered_input(root, dict(specification), label=name)
        for name, specification in dict(config["inputs"]).items()
    }
    return path, root, config, inputs


def _select_pilot_occurrences(
    logical_rows: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    cohort = dict(config["cohort"])
    maps = list(map(str, cohort["maps"]))
    per_map = int(cohort["states_per_map"])
    by_map: defaultdict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for source in logical_rows:
        row = dict(source)
        if str(row.get("checkpoint_kind")) != str(cohort["source_checkpoint_kind"]):
            continue
        map_id = str(row["map_id"])
        if map_id not in maps:
            raise ValueError(f"unexpected first-structural map: {map_id}")
        state_key = str(row["state_fingerprint"])
        previous = by_map[map_id].get(state_key)
        if previous is None or str(row["logical_checkpoint_id"]) < str(
            previous["logical_checkpoint_id"]
        ):
            by_map[map_id][state_key] = row
    selected: list[dict[str, Any]] = []
    for map_id in maps:
        available = list(by_map[map_id].values())
        if len(available) < per_map:
            raise ValueError(
                f"CycleTransition map has too few unique states: {map_id}={len(available)}"
            )
        ordered = sorted(
            available,
            key=lambda row: (
                _selection_digest(map_id, str(row["state_fingerprint"])),
                str(row["logical_checkpoint_id"]),
            ),
        )
        selected.extend(ordered[:per_map])
    selected.sort(key=lambda row: (str(row["map_id"]), str(row["state_fingerprint"])))
    if len(selected) != int(cohort["state_count"]):
        raise RuntimeError("CycleTransition selected state count changed")
    return selected


def _full_proposal(state_record: dict[str, Any]) -> dict[str, Any]:
    run_path = Path(str(state_record["source_run_config"]))
    if sha256_file(run_path) != str(state_record["source_run_config_sha256"]):
        raise ValueError("CycleTransition source run configuration changed")
    run = _read_json(run_path)
    proposal = dict(run["configuration"]["proposal"])
    proposal.update(
        {
            "heuristics": ["target", "collision", "random"],
            "neighborhood_sizes": [4, 8, 16],
            "candidates_per_family": 2,
        }
    )
    if not dict(proposal.get("structpool") or {}):
        raise ValueError("CycleTransition source lacks StructPool configuration")
    # The frozen SlotPool runtime exposes the complete family-by-size grid and
    # marks reduction as pending.  Reuse that registered generator identity;
    # the audit consumes the raw rows and deliberately never invokes the
    # six-candidate SlotPool reducer.
    proposal["structpool"] = slotpool_runtime_augmentation()
    return proposal


def _load_v2_model(runtime_path: Path) -> Any:
    runtime = _read_json(runtime_path)
    bundle = load_frozen_policy_bundle(
        runtime["frozen_models"], dict(runtime["model_registration"])
    )
    return bundle.models[PROFILE]


def _structural(candidate: dict[str, Any]) -> bool:
    return bool(candidate.get("structpool_family_groups"))


def _nominal_sizes(candidate: dict[str, Any]) -> list[int]:
    result = set()
    for family in map(str, candidate.get("selection_families") or ()):
        prefix, separator, raw = family.rpartition(":")
        if separator and prefix.startswith("structpool-") and raw.isdigit():
            result.add(int(raw))
    return sorted(result)


def _family_variants(candidate: dict[str, Any]) -> list[str]:
    mapping = {
        "bottleneck_crossing": "bottleneck_crossing",
        "conflict_component": "conflict_component",
        "topology_boundary": "topology_boundary",
        "spatiotemporal_hotspot": "spatiotemporal_hotspot",
        "path_overlap": "path_overlap",
    }
    values = []
    for group in map(str, candidate.get("structpool_family_groups") or ()):
        if group not in mapping:
            raise ValueError(f"unknown StructPool family group: {group}")
        values.append(mapping[group])
    return sorted(set(values))


def _candidate_record(
    candidate: dict[str, Any], row: dict[str, Any], score: float
) -> dict[str, Any]:
    features = {
        str(name): float(value)
        for name, value in dict(row["features"][PROFILE]).items()
    }
    if set(features) != set(PROFILE_FEATURE_NAMES[PROFILE]) or any(
        not math.isfinite(value) for value in features.values()
    ):
        raise ValueError("CycleTransition candidate feature schema changed")
    agents = sorted(map(int, candidate["agents"]))
    return {
        "candidate_id": str(candidate["candidate_id"]),
        "candidate_kind": "structural" if _structural(candidate) else "base",
        "agents": agents,
        "actual_size": len(agents),
        "selection_families": sorted(
            map(str, candidate.get("selection_families") or ())
        ),
        "structpool_family_groups": sorted(
            map(str, candidate.get("structpool_family_groups") or ())
        ),
        "structural_family_variants": _family_variants(candidate),
        "nominal_sizes": _nominal_sizes(candidate),
        "v2_mixed_pool_score": float(score),
        "feature_schema": "lns2.realized_features.v2",
        "feature_profile": PROFILE,
        "feature_count": len(features),
        "feature_sha256": _feature_digest(features),
        "features": features,
    }


def _prepare_state(job: dict[str, Any]) -> dict[str, Any]:
    state_record = dict(job["state_record"])
    logical = dict(job["logical"])
    state = read_state_blob(Path(str(state_record["state_blob"])))
    state_key = str(state_record["state_fingerprint"])
    if (
        state_fingerprint(state) != state_key
        or sha256_file(Path(str(state_record["state_blob"])))
        != str(state_record["state_blob_sha256"])
    ):
        raise ValueError("CycleTransition frozen state blob changed")
    replay = _marginalpool_replay_job(state_record)
    restore_seed = repairability_restore_seed(repair_structure_fingerprint(state))
    environment, restored = restore_repair_state(replay, state, seed=restore_seed)
    before_repair = repair_structure_fingerprint(state)
    if repair_structure_fingerprint(restored) != before_repair:
        raise RuntimeError("CycleTransition restored repair structure differs")
    proposal = _full_proposal(state_record)
    candidates, generation = generate_online_candidates(
        environment,
        state,
        task_id=str(state_record["task_id"]),
        solver_seed=int(state_record["solver_seed"]),
        decision_index=int(logical["decision_index"]),
        proposal_config=proposal,
        state_hash=state_key,
        # reset_paths intentionally resets iteration/low-level counters.  The
        # native state revision and the repair-structure check below protect
        # proposal purity without comparing those reset-only counters.
        verify_full_state=False,
        proposal_backend="optimized",
        shadow_validation=False,
    )
    if (
        generation.get("structpool_gate_passed") is not True
        or generation.get("slotpool_reduction_pending") is not True
        or int(generation.get("structpool_generated_count", -1)) <= 0
    ):
        raise RuntimeError("CycleTransition full structural grid was not generated")
    if repair_structure_fingerprint(_plain(environment.get_state())) != before_repair:
        raise RuntimeError("CycleTransition preparation changed repair structure")
    feature_engine = OnlineFeatureEngine(
        state,
        backend="native",
        required_features={PROFILE: PROFILE_FEATURE_NAMES[PROFILE]},
        dense_output=False,
    )
    rows, _metrics = feature_engine.realized_rows(candidates, state_hash=state_key)
    model = _load_v2_model(Path(str(job["runtime_path"])))
    full_index, scores, _margin = score_online_candidates(rows, model)
    base_indices = [index for index, candidate in enumerate(candidates) if not _structural(candidate)]
    structural_indices = [index for index, candidate in enumerate(candidates) if _structural(candidate)]
    if not base_indices or not structural_indices:
        raise RuntimeError("CycleTransition full pool is missing one candidate class")
    base_rows = [rows[index] for index in base_indices]
    base_local, _base_scores, _base_margin = score_online_candidates(base_rows, model)
    anchor_index = base_indices[base_local]
    records = [
        _candidate_record(candidate, row, score)
        for candidate, row, score in zip(candidates, rows, scores)
    ]
    if len(records) != len({row["candidate_id"] for row in records}):
        raise RuntimeError("CycleTransition candidate IDs are not unique")
    if len({tuple(row["agents"]) for row in records}) != len(records):
        raise RuntimeError("CycleTransition exact-set deduplication changed")
    family_sizes = {
        (family.removeprefix("structpool-").rsplit(":", 1)[0], int(family.rsplit(":", 1)[1]))
        for row in records
        for family in row["selection_families"]
        if family.startswith("structpool-") and family.rsplit(":", 1)[1].isdigit()
    }
    if {size for _family, size in family_sizes} != set(STRUCTURAL_SIZES):
        raise RuntimeError("CycleTransition four-size coverage changed")
    source_rows = {
        str(row["candidate_id"]): dict(row) for row in state_record["candidates"]
    }
    source_original_id = str(logical["selected_candidate_id"])
    if source_original_id not in source_rows:
        raise RuntimeError("CycleTransition source-selected action is missing")
    source_agents = sorted(map(int, source_rows[source_original_id]["agents"]))
    mapped_source = [row for row in records if row["agents"] == source_agents]
    if len(mapped_source) != 1 or mapped_source[0]["candidate_kind"] != "structural":
        raise RuntimeError(
            "CycleTransition source-selected structural action was not preserved"
        )
    return {
        "schema": PREPARATION_SCHEMA,
        "state_fingerprint": state_key,
        "logical_checkpoint_id": str(logical["logical_checkpoint_id"]),
        "map_id": str(logical["map_id"]),
        "task_id": str(logical["task_id"]),
        "solver_seed": int(logical["solver_seed"]),
        "decision_index": int(logical["decision_index"]),
        "selection_digest": _selection_digest(str(logical["map_id"]), state_key),
        "state_blob": str(state_record["state_blob"]),
        "state_blob_sha256": str(state_record["state_blob_sha256"]),
        "source_run_config": str(state_record["source_run_config"]),
        "source_run_config_sha256": str(state_record["source_run_config_sha256"]),
        "split": str(state_record["split"]),
        "before_conflicts": int(state["num_of_colliding_pairs"]),
        "before_repair_fingerprint": before_repair,
        "restore_seed": restore_seed,
        "candidate_count": len(records),
        "base_candidate_count": len(base_indices),
        "structural_candidate_count": len(structural_indices),
        "generated_structural_candidate_count": int(generation["structpool_generated_count"]),
        "v2_anchor_candidate_id": str(records[anchor_index]["candidate_id"]),
        "full_pool_selected_candidate_id": str(records[full_index]["candidate_id"]),
        "source_selected_original_candidate_id": source_original_id,
        "source_selected_candidate_id": str(mapped_source[0]["candidate_id"]),
        "source_selected_agents": source_agents,
        "candidate_pool_sha256": _fingerprint(records),
        "candidates": records,
        "outcome_fields_read": False,
        "six_candidate_reduction_applied": False,
    }


def _failure(job: dict[str, Any], status: str, error: str) -> dict[str, Any]:
    logical = dict(job.get("logical") or {})
    candidate = dict(job.get("candidate") or {})
    return {
        "status": status,
        "error": error,
        "state_fingerprint": str(
            logical.get("state_fingerprint", job.get("state_fingerprint", ""))
        ),
        "candidate_id": str(candidate.get("candidate_id", "")),
        "trial_index": int(job.get("trial_index", -1)),
    }


def prepare_cycletransition(
    *, config_path: str | Path, output: str | Path, workers: int | None = None
) -> dict[str, Any]:
    config_path, root, config, inputs = load_cycletransition_registration(config_path)
    metadata, state_rows, logical_rows = build_frozen_cohort(
        inputs["marginalpool_registration"]
    )
    if (
        _read_json(inputs["root_status"]).get("integrity_passed") is not True
        or _read_json(inputs["root_report"]).get("integrity_passed") is not True
        or len(_read_jsonl(inputs["root_checkpoints"])) != 90
    ):
        raise ValueError("CycleTransition root diagnostic integrity changed")
    selected = _select_pilot_occurrences(logical_rows, config)
    state_by_key = {str(row["state_fingerprint"]): row for row in state_rows}
    jobs = [
        {
            "state_record": state_by_key[str(logical["state_fingerprint"])],
            "logical": logical,
            "runtime_path": str(inputs["v2_runtime_registration"]),
            "job_id": f"prepare:{logical['state_fingerprint'][:16]}",
        }
        for logical in selected
    ]
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    worker_count = int(workers or config["execution"]["prepare_workers"])
    if worker_count != int(config["execution"]["prepare_workers"]):
        raise ValueError("CycleTransition preparation requires 16 workers")
    results = _run_jobs(
        _prepare_state,
        jobs,
        worker_count,
        phase="cycletransition-prepare",
        output_root=output,
        run_fingerprint=_fingerprint(
            {"config": sha256_file(config_path), "cohort": metadata["cohort_fingerprint"]}
        ),
        timeout_seconds=float(config["execution"]["prepare_job_timeout_seconds"]),
        failure_result=_failure,
        stop_on_failure=True,
    )
    failures = [row for row in results if row.get("status") in {"error", "timeout"}]
    prepared = [row for row in results if row.get("schema") == PREPARATION_SCHEMA]
    prepared.sort(key=lambda row: (str(row["map_id"]), str(row["state_fingerprint"])))
    passed = not failures and len(prepared) == int(config["cohort"]["state_count"])
    if passed:
        _write_jsonl(output / "cohort_manifest.jsonl", prepared)
    report = {
        "schema": PREPARATION_REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "passed": passed,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "input_sha256": {name: sha256_file(path) for name, path in sorted(inputs.items())},
        "state_count": len(prepared),
        "candidate_count": sum(int(row["candidate_count"]) for row in prepared),
        "base_candidate_count": sum(int(row["base_candidate_count"]) for row in prepared),
        "structural_candidate_count": sum(
            int(row["structural_candidate_count"]) for row in prepared
        ),
        "map_state_counts": dict(sorted(Counter(row["map_id"] for row in prepared).items())),
        "error_count": sum(row.get("status") == "error" for row in failures),
        "timeout_count": sum(row.get("status") == "timeout" for row in failures),
        "failures": failures,
        "producer": producer_identity(
            project_root=root,
            source_files=PRODUCER_FILES,
            native_required=True,
            package_names=("numpy",),
        ),
        "outcome_fields_read": False,
        "manifest_sha256": (
            sha256_file(output / "cohort_manifest.jsonl") if passed else None
        ),
    }
    _write_json(output / "preparation_report.json", report)
    return report


def _edge_set(state: dict[str, Any]) -> set[tuple[int, int]]:
    edges = {
        tuple(sorted((int(row[0]), int(row[1]))))
        for row in state.get("conflict_edges", [])
    }
    if len(edges) != int(state["num_of_colliding_pairs"]):
        raise ValueError("CycleTransition conflict edge count changed")
    return edges


def _edge_metrics(
    before: set[tuple[int, int]], after: set[tuple[int, int]]
) -> dict[str, float | bool]:
    shared = before & after
    union = before | after
    return {
        "conflict_edge_exact_recurrence": before == after,
        "unresolved_edge_fraction": len(shared) / max(1, len(before)),
        "new_edge_fraction": len(after - before) / max(1, len(after)),
        "conflict_edge_jaccard": len(shared) / max(1, len(union)),
    }


def _trial_path(output: Path, state_key: str, candidate_id: str, index: int) -> Path:
    return output / "trials" / state_key[:16] / candidate_id / f"trial-{index:02d}.json"


def _trial_valid(
    payload: dict[str, Any], *, manifest: dict[str, Any], candidate: dict[str, Any], index: int
) -> bool:
    required_boolean = (
        "repair_exact_noop",
        "conflict_edge_exact_recurrence",
        "next_full_pool_selected_exact_agent_set",
        "exact_policy_cycle",
        "soft_policy_cycle",
        "replan_success",
        "feasible",
    )
    required_float = (
        "normalized_conflict_reduction",
        "unresolved_edge_fraction",
        "new_edge_fraction",
        "conflict_edge_jaccard",
        "next_full_pool_selected_agent_jaccard",
        "next_v2_anchor_agent_jaccard",
    )
    return (
        payload.get("schema") == TRIAL_SCHEMA
        and payload.get("state_fingerprint") == manifest["state_fingerprint"]
        and payload.get("candidate_id") == candidate["candidate_id"]
        and int(payload.get("trial_index", -1)) == index
        and int(payload.get("pp_seed", -1))
        == repairability_pp_seed(str(manifest["before_repair_fingerprint"]), index)
        and all(isinstance(payload.get(field), bool) for field in required_boolean)
        and all(
            isinstance(payload.get(field), (int, float))
            and math.isfinite(float(payload[field]))
            for field in required_float
        )
        and payload.get("native_action_validated") is True
        and payload.get("runtime_or_ttf_stored") is False
        and payload.get("future_repair_trajectory_stored") is False
    )


def _next_selection(
    *,
    environment: Any,
    after: dict[str, Any],
    manifest: dict[str, Any],
    state_record: dict[str, Any],
    proposal: dict[str, Any],
    model: Any,
) -> dict[str, Any]:
    after_key = state_fingerprint(after)
    candidates, _generation = generate_online_candidates(
        environment,
        after,
        task_id=str(manifest["task_id"]),
        solver_seed=int(manifest["solver_seed"]),
        decision_index=int(manifest["decision_index"]) + 1,
        proposal_config=proposal,
        state_hash=after_key,
        verify_full_state=True,
        proposal_backend="optimized",
        shadow_validation=False,
    )
    engine = OnlineFeatureEngine(
        after,
        backend="native",
        required_features={PROFILE: PROFILE_FEATURE_NAMES[PROFILE]},
        dense_output=False,
    )
    rows, _metrics = engine.realized_rows(candidates, state_hash=after_key)
    full_index, _scores, _margin = score_online_candidates(rows, model)
    base_indices = [index for index, candidate in enumerate(candidates) if not _structural(candidate)]
    if not base_indices:
        raise RuntimeError("CycleTransition next state has no V2 base candidate")
    base_local, _base_scores, _base_margin = score_online_candidates(
        [rows[index] for index in base_indices], model
    )
    anchor_index = base_indices[base_local]
    return {
        "full_candidate_id": str(candidates[full_index]["candidate_id"]),
        "full_agents": sorted(map(int, candidates[full_index]["agents"])),
        "anchor_candidate_id": str(candidates[anchor_index]["candidate_id"]),
        "anchor_agents": sorted(map(int, candidates[anchor_index]["agents"])),
        "candidate_count": len(candidates),
        "structural_candidate_count": sum(_structural(row) for row in candidates),
    }


def _collect_trial(job: dict[str, Any]) -> dict[str, Any]:
    output = Path(str(job["output"]))
    manifest = dict(job["manifest"])
    candidate = dict(job["candidate"])
    trial_index = int(job["trial_index"])
    if trial_index not in TRIAL_INDICES:
        raise ValueError("CycleTransition trial index changed")
    state_record = {
        key: manifest[key]
        for key in (
            "state_fingerprint",
            "state_blob",
            "state_blob_sha256",
            "source_run_config",
            "source_run_config_sha256",
            "task_id",
            "solver_seed",
            "split",
        )
    }
    state = read_state_blob(Path(str(manifest["state_blob"])))
    state_key = str(manifest["state_fingerprint"])
    if state_fingerprint(state) != state_key:
        raise ValueError("CycleTransition collection state changed")
    replay = _marginalpool_replay_job(state_record)
    proposal = _full_proposal(state_record)
    model = _load_v2_model(Path(str(job["runtime_path"])))
    before_repair = repair_structure_fingerprint(state)
    if before_repair != str(manifest["before_repair_fingerprint"]):
        raise RuntimeError("CycleTransition before-repair fingerprint changed")
    before_edges = _edge_set(state)
    before_conflicts = int(manifest["before_conflicts"])
    path = _trial_path(output, state_key, str(candidate["candidate_id"]), trial_index)
    if path.is_file():
        if not bool(job.get("resume", True)):
            raise FileExistsError(f"CycleTransition trial already exists: {path}")
        existing = _read_json(path)
        if not _trial_valid(
            existing, manifest=manifest, candidate=candidate, index=trial_index
        ):
            raise ValueError(f"invalid CycleTransition checkpoint: {path}")
        return {
            "status": "resumed",
            "state_fingerprint": state_key,
            "candidate_id": str(candidate["candidate_id"]),
            "trial_index": trial_index,
            "trial_count": 1,
        }
    branch_environment, branch_state = restore_repair_state(
        replay, state, seed=int(manifest["restore_seed"])
    )
    if repair_structure_fingerprint(branch_state) != before_repair:
        raise RuntimeError("CycleTransition paired restore changed")
    pp_seed = repairability_pp_seed(before_repair, trial_index)
    result = _plain(
        branch_environment.step(_paired_action(candidate["agents"], pp_seed))
    )
    after, repair_metrics = _validate_native_repair(
        result,
        expected_agents=list(map(int, candidate["agents"])),
        expected_seed=pp_seed,
    )
    conflicts_after = int(after["num_of_colliding_pairs"])
    after_repair = repair_structure_fingerprint(after)
    edge_metrics = _edge_metrics(before_edges, _edge_set(after))
    terminal = bool(after.get("done")) or conflicts_after == 0
    if terminal:
        next_selection = {
            "full_candidate_id": None,
            "full_agents": [],
            "anchor_candidate_id": None,
            "anchor_agents": [],
            "candidate_count": 0,
            "structural_candidate_count": 0,
        }
    else:
        next_selection = _next_selection(
            environment=branch_environment,
            after=after,
            manifest=manifest,
            state_record=state_record,
            proposal=proposal,
            model=model,
        )
    forced_agents = list(map(int, candidate["agents"]))
    selected_jaccard = _candidate_jaccard(forced_agents, next_selection["full_agents"])
    anchor_jaccard = _candidate_jaccard(forced_agents, next_selection["anchor_agents"])
    selected_exact = bool(next_selection["full_agents"]) and set(forced_agents) == set(
        next_selection["full_agents"]
    )
    repair_noop = after_repair == before_repair
    no_strict_decrease = conflicts_after >= before_conflicts
    trial = {
        "schema": TRIAL_SCHEMA,
        "state_fingerprint": state_key,
        "map_id": str(manifest["map_id"]),
        "candidate_id": str(candidate["candidate_id"]),
        "candidate_kind": str(candidate["candidate_kind"]),
        "actual_size": int(candidate["actual_size"]),
        "nominal_sizes": list(map(int, candidate["nominal_sizes"])),
        "trial_index": trial_index,
        "pp_seed": pp_seed,
        "before_conflicts": before_conflicts,
        "conflicts_after": conflicts_after,
        "normalized_conflict_reduction": (before_conflicts - conflicts_after)
        / max(1, before_conflicts),
        "replan_success": bool(repair_metrics["replan_success"]),
        "feasible": bool(after.get("feasible")),
        "repair_outcome": classify_repair_outcome(
            before_fingerprint=before_repair,
            after_fingerprint=after_repair,
            replan_success=bool(repair_metrics["replan_success"]),
            conflicts_before=before_conflicts,
            conflicts_after=conflicts_after,
            feasible=bool(after.get("feasible")),
        ),
        "after_repair_fingerprint": after_repair,
        "repair_exact_noop": repair_noop,
        **edge_metrics,
        "next_full_pool_selected_candidate_id": next_selection["full_candidate_id"],
        "next_full_pool_selected_agent_jaccard": selected_jaccard,
        "next_full_pool_selected_exact_agent_set": selected_exact,
        "next_v2_anchor_candidate_id": next_selection["anchor_candidate_id"],
        "next_v2_anchor_agent_jaccard": anchor_jaccard,
        "next_candidate_count": int(next_selection["candidate_count"]),
        "next_structural_candidate_count": int(
            next_selection["structural_candidate_count"]
        ),
        "exact_policy_cycle": repair_noop and selected_exact,
        "soft_policy_cycle": (
            no_strict_decrease
            and float(edge_metrics["unresolved_edge_fraction"]) >= 0.8
            and selected_jaccard >= 0.8
        ),
        "native_action_validated": True,
        "runtime_or_ttf_stored": False,
        "future_repair_trajectory_stored": False,
    }
    if not _trial_valid(
        trial, manifest=manifest, candidate=candidate, index=trial_index
    ):
        raise RuntimeError("CycleTransition trial artifact failed validation")
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, trial)
    return {
        "status": "ok",
        "state_fingerprint": state_key,
        "candidate_id": str(candidate["candidate_id"]),
        "trial_index": trial_index,
        "trial_count": 1,
    }


def collect_cycletransition(
    *,
    config_path: str | Path,
    output: str | Path,
    workers: int | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    config_path, _root, config, inputs = load_cycletransition_registration(config_path)
    output = Path(output).resolve()
    preparation = _read_json(output / "preparation_report.json")
    if (
        preparation.get("passed") is not True
        or preparation.get("config_sha256") != sha256_file(config_path)
        or preparation.get("manifest_sha256")
        != sha256_file(output / "cohort_manifest.jsonl")
    ):
        raise ValueError("CycleTransition preparation is not frozen and complete")
    manifests = _read_jsonl(output / "cohort_manifest.jsonl")
    jobs = [
        {
            "output": str(output),
            "manifest": manifest,
            "candidate": candidate,
            "trial_index": trial_index,
            "runtime_path": str(inputs["v2_runtime_registration"]),
            "resume": bool(resume),
            "job_id": (
                f"{manifest['state_fingerprint'][:12]}:"
                f"{candidate['candidate_id']}:{trial_index:02d}"
            ),
        }
        for manifest in manifests
        for candidate in manifest["candidates"]
        for trial_index in TRIAL_INDICES
    ]
    worker_count = int(workers or config["execution"]["collection_workers"])
    if worker_count != int(config["execution"]["collection_workers"]):
        raise ValueError("CycleTransition collection requires 16 workers")
    run_fingerprint = _fingerprint(
        {
            "config": sha256_file(config_path),
            "manifest": preparation["manifest_sha256"],
            "producer": preparation["producer"],
        }
    )
    status_path = output / "collection_status.json"
    _write_json(
        status_path,
        {
            "schema": COLLECTION_STATUS_SCHEMA,
            "run_fingerprint": run_fingerprint,
            "status": "running",
            "total_jobs": len(jobs),
            "total_trials": len(jobs),
            "workers": worker_count,
            "error_jobs": 0,
            "timeout_jobs": 0,
        },
    )
    results = _run_jobs(
        _collect_trial,
        jobs,
        worker_count,
        phase="cycletransition-collect",
        output_root=output,
        run_fingerprint=run_fingerprint,
        timeout_seconds=float(config["execution"]["per_job_timeout_seconds"]),
        failure_result=_failure,
        stop_on_failure=True,
    )
    failures = [row for row in results if row.get("status") in {"error", "timeout"}]
    status = {
        "schema": COLLECTION_STATUS_SCHEMA,
        "run_fingerprint": run_fingerprint,
        "status": "failed" if failures else "complete",
        "total_jobs": len(jobs),
        "completed_jobs": len(results) - len(failures),
        "total_trials": len(jobs),
        "completed_trials": sum(int(row.get("trial_count", 0)) for row in results),
        "workers": worker_count,
        "error_jobs": sum(row.get("status") == "error" for row in failures),
        "timeout_jobs": sum(row.get("status") == "timeout" for row in failures),
        "failures": failures,
    }
    _write_json(status_path, status)
    return status


def _ranks(values: list[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=lambda index: (values[index], index))
    result = [0.0] * len(values)
    position = 0
    while position < len(ordered):
        end = position + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[position]]:
            end += 1
        rank = 0.5 * (position + end - 1)
        for offset in range(position, end):
            result[ordered[offset]] = rank
        position = end
    return result


def _spearman(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("CycleTransition rank vectors do not align")
    first = _ranks(left)
    second = _ranks(right)
    first_mean = _mean(first)
    second_mean = _mean(second)
    covariance = sum(
        (a - first_mean) * (b - second_mean) for a, b in zip(first, second)
    )
    first_scale = math.sqrt(sum((value - first_mean) ** 2 for value in first))
    second_scale = math.sqrt(sum((value - second_mean) ** 2 for value in second))
    if first_scale <= 1e-15 or second_scale <= 1e-15:
        return 1.0 if first == second else 0.0
    return covariance / (first_scale * second_scale)


def _aggregate_candidate(
    candidate: dict[str, Any], trials: list[dict[str, Any]]
) -> dict[str, Any]:
    ordered = sorted(trials, key=lambda row: int(row["trial_index"]))
    if [int(row["trial_index"]) for row in ordered] != list(TRIAL_INDICES):
        raise ValueError("CycleTransition candidate lacks eight paired trials")
    quality = [float(row["normalized_conflict_reduction"]) for row in ordered]
    noop = [float(row["repair_exact_noop"]) for row in ordered]
    soft_cycle = [float(row["soft_policy_cycle"]) for row in ordered]
    first_noop = _mean(noop[:4])
    second_noop = _mean(noop[4:])
    first_soft_cycle = _mean(soft_cycle[:4])
    second_soft_cycle = _mean(soft_cycle[4:])
    return {
        "candidate_id": str(candidate["candidate_id"]),
        "candidate_kind": str(candidate["candidate_kind"]),
        "actual_size": int(candidate["actual_size"]),
        "nominal_sizes": list(map(int, candidate["nominal_sizes"])),
        "seed_mean": _mean(quality),
        "first_half_mean": _mean(quality[:4]),
        "second_half_mean": _mean(quality[4:]),
        "no_progress_rate": _mean(
            int(row["conflicts_after"]) >= int(row["before_conflicts"])
            for row in ordered
        ),
        "exact_noop_rate": _mean(noop),
        "first_half_exact_noop_rate": first_noop,
        "second_half_exact_noop_rate": second_noop,
        "half_exact_noop_class_agreement": (first_noop >= 0.5) == (second_noop >= 0.5),
        "exact_policy_cycle_rate": _mean(row["exact_policy_cycle"] for row in ordered),
        "soft_policy_cycle_rate": _mean(soft_cycle),
        "first_half_soft_policy_cycle_rate": first_soft_cycle,
        "second_half_soft_policy_cycle_rate": second_soft_cycle,
        "half_soft_policy_cycle_class_agreement": (
            first_soft_cycle >= 0.5
        ) == (second_soft_cycle >= 0.5),
        "unresolved_edge_fraction_mean": _mean(
            row["unresolved_edge_fraction"] for row in ordered
        ),
        "new_edge_fraction_mean": _mean(row["new_edge_fraction"] for row in ordered),
        "next_selection_jaccard_mean": _mean(
            row["next_full_pool_selected_agent_jaccard"] for row in ordered
        ),
    }


def _quality_admissible(
    candidate: dict[str, Any], anchor: dict[str, Any], quality: dict[str, Any]
) -> bool:
    return (
        float(candidate["seed_mean"]) - float(anchor["seed_mean"])
        >= float(quality["minimum_seed_mean_delta"]) - 1e-15
        and float(candidate["no_progress_rate"]) - float(anchor["no_progress_rate"])
        <= float(quality["maximum_no_progress_rate_delta"]) + 1e-15
        and float(candidate["first_half_mean"]) - float(anchor["first_half_mean"])
        >= float(quality["minimum_each_fixed_four_seed_half_delta"]) - 1e-15
        and float(candidate["second_half_mean"]) - float(anchor["second_half_mean"])
        >= float(quality["minimum_each_fixed_four_seed_half_delta"]) - 1e-15
    )


def _map_metrics(states: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [candidate for state in states for candidate in state["candidates"]]
    positive = sum(float(row["soft_policy_cycle_rate"]) >= 0.5 for row in candidates)
    negative = sum(float(row["soft_policy_cycle_rate"]) < 0.5 for row in candidates)
    return {
        "state_count": len(states),
        "candidate_count": len(candidates),
        "half_split_soft_cycle_class_agreement": _mean(
            row["half_soft_policy_cycle_class_agreement"] for row in candidates
        ),
        "half_split_soft_cycle_rate_rank_correlation": _mean(
            state["half_split_soft_cycle_rate_rank_correlation"] for state in states
        ),
        "half_split_exact_noop_class_agreement": _mean(
            row["half_exact_noop_class_agreement"] for row in candidates
        ),
        "half_split_noop_rate_rank_correlation": _mean(
            state["half_split_noop_rate_rank_correlation"] for state in states
        ),
        "quality_admissible_structural_cycle_escape_state_fraction": _mean(
            state["has_quality_admissible_structural_cycle_escape"] for state in states
        ),
        "positive_soft_cycle_candidate_count": positive,
        "negative_soft_cycle_candidate_count": negative,
        "has_both_soft_cycle_classes": positive > 0 and negative > 0,
    }


def analyze_cycletransition(
    *, config_path: str | Path, collection: str | Path, output: str | Path
) -> dict[str, Any]:
    config_path, _root, config, _inputs = load_cycletransition_registration(
        config_path
    )
    collection = Path(collection).resolve()
    output = Path(output).resolve()
    status = _read_json(collection / "collection_status.json")
    preparation = _read_json(collection / "preparation_report.json")
    manifests = _read_jsonl(collection / "cohort_manifest.jsonl")
    if status.get("status") != "complete" or preparation.get("passed") is not True:
        raise ValueError("CycleTransition collection is not complete")
    expected_jobs = sum(len(row["candidates"]) for row in manifests) * len(
        TRIAL_INDICES
    )
    if (
        len(manifests) != int(config["cohort"]["state_count"])
        or int(status.get("total_jobs", -1)) != expected_jobs
        or int(status.get("completed_jobs", -1)) != expected_jobs
        or int(status.get("error_jobs", -1)) != 0
        or int(status.get("timeout_jobs", -1)) != 0
    ):
        raise ValueError("CycleTransition collection cardinality changed")
    quality = dict(config["quality_constraint"])
    cycle_reduction = float(
        config["label_readiness_gates"][
            "minimum_soft_cycle_rate_reduction_for_escape"
        ]
    )
    maximum_noop_delta = float(
        config["label_readiness_gates"][
            "maximum_exact_noop_rate_delta_for_escape"
        ]
    )
    state_results: list[dict[str, Any]] = []
    trial_sha: dict[str, str] = {}
    for manifest in manifests:
        aggregates = []
        for candidate in manifest["candidates"]:
            trials = []
            for index in TRIAL_INDICES:
                path = _trial_path(
                    collection,
                    str(manifest["state_fingerprint"]),
                    str(candidate["candidate_id"]),
                    index,
                )
                payload = _read_json(path)
                if not _trial_valid(
                    payload, manifest=manifest, candidate=candidate, index=index
                ):
                    raise ValueError(f"invalid CycleTransition trial: {path}")
                relative = path.relative_to(collection).as_posix()
                trial_sha[relative] = sha256_file(path)
                trials.append(payload)
            aggregates.append(_aggregate_candidate(candidate, trials))
        by_id = {row["candidate_id"]: row for row in aggregates}
        anchor = by_id[str(manifest["v2_anchor_candidate_id"])]
        full_selected = by_id[str(manifest["full_pool_selected_candidate_id"])]
        source_selected = by_id[str(manifest["source_selected_candidate_id"])]
        escape = []
        for row in aggregates:
            row["quality_admissible_vs_v2_anchor"] = _quality_admissible(
                row, anchor, quality
            )
            row["exact_noop_rate_delta_vs_v2_anchor"] = (
                float(row["exact_noop_rate"]) - float(anchor["exact_noop_rate"])
            )
            row["exact_noop_rate_delta_vs_source_selected"] = (
                float(row["exact_noop_rate"])
                - float(source_selected["exact_noop_rate"])
            )
            row["soft_policy_cycle_rate_delta_vs_source_selected"] = (
                float(row["soft_policy_cycle_rate"])
                - float(source_selected["soft_policy_cycle_rate"])
            )
            row["quality_admissible_structural_cycle_escape"] = bool(
                row["candidate_kind"] == "structural"
                and row["quality_admissible_vs_v2_anchor"]
                and float(row["soft_policy_cycle_rate"])
                <= float(source_selected["soft_policy_cycle_rate"])
                - cycle_reduction
                + 1e-15
                and float(row["exact_noop_rate"])
                - float(source_selected["exact_noop_rate"])
                <= maximum_noop_delta + 1e-15
            )
            if row["quality_admissible_structural_cycle_escape"]:
                escape.append(row)
        noop_correlation = _spearman(
            [float(row["first_half_exact_noop_rate"]) for row in aggregates],
            [float(row["second_half_exact_noop_rate"]) for row in aggregates],
        )
        cycle_correlation = _spearman(
            [
                float(row["first_half_soft_policy_cycle_rate"])
                for row in aggregates
            ],
            [
                float(row["second_half_soft_policy_cycle_rate"])
                for row in aggregates
            ],
        )
        state_results.append(
            {
                "state_fingerprint": str(manifest["state_fingerprint"]),
                "map_id": str(manifest["map_id"]),
                "candidate_count": len(aggregates),
                "v2_anchor_candidate_id": str(anchor["candidate_id"]),
                "source_selected_candidate_id": str(source_selected["candidate_id"]),
                "source_selected_exact_noop_rate": float(
                    source_selected["exact_noop_rate"]
                ),
                "source_selected_soft_policy_cycle_rate": float(
                    source_selected["soft_policy_cycle_rate"]
                ),
                "full_pool_selected_candidate_id": str(full_selected["candidate_id"]),
                "full_pool_selected_exact_noop_rate": float(
                    full_selected["exact_noop_rate"]
                ),
                "full_pool_selected_soft_policy_cycle_rate": float(
                    full_selected["soft_policy_cycle_rate"]
                ),
                "half_split_noop_rate_rank_correlation": noop_correlation,
                "half_split_soft_cycle_rate_rank_correlation": cycle_correlation,
                "has_quality_admissible_structural_cycle_escape": bool(escape),
                "escape_candidate_ids": sorted(row["candidate_id"] for row in escape),
                "candidates": aggregates,
            }
        )
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for state in state_results:
        grouped[str(state["map_id"])].append(state)
    map_results = {map_id: _map_metrics(rows) for map_id, rows in sorted(grouped.items())}
    all_candidates = [row for state in state_results for row in state["candidates"]]
    escapes = [
        row
        for row in all_candidates
        if row["quality_admissible_structural_cycle_escape"]
    ]
    size_counts: Counter[str] = Counter()
    for row in escapes:
        sizes = list(map(int, row["nominal_sizes"])) or [int(row["actual_size"])]
        for size in sizes:
            size_counts[str(size)] += 1
    single_size_share = max(size_counts.values(), default=0) / max(
        1, sum(size_counts.values())
    )
    gates_config = dict(config["label_readiness_gates"])
    global_cycle_agreement = _mean(
        row["half_soft_policy_cycle_class_agreement"] for row in all_candidates
    )
    global_cycle_correlation = _mean(
        state["half_split_soft_cycle_rate_rank_correlation"]
        for state in state_results
    )
    global_noop_agreement = _mean(
        row["half_exact_noop_class_agreement"] for row in all_candidates
    )
    global_noop_correlation = _mean(
        state["half_split_noop_rate_rank_correlation"] for state in state_results
    )
    global_escape = _mean(
        state["has_quality_admissible_structural_cycle_escape"]
        for state in state_results
    )
    maps_with_both = sum(
        row["has_both_soft_cycle_classes"] for row in map_results.values()
    )
    gates = {
        "half_split_soft_cycle_class_agreement": global_cycle_agreement
        >= float(gates_config["minimum_half_split_soft_cycle_class_agreement"]),
        "half_split_soft_cycle_rate_rank_correlation": global_cycle_correlation
        >= float(gates_config["minimum_half_split_soft_cycle_rate_rank_correlation"]),
        "each_map_half_split_soft_cycle_class_agreement": all(
            row["half_split_soft_cycle_class_agreement"]
            >= float(gates_config["minimum_each_map_half_split_soft_cycle_class_agreement"])
            for row in map_results.values()
        ),
        "each_map_half_split_soft_cycle_rate_rank_correlation": all(
            row["half_split_soft_cycle_rate_rank_correlation"]
            >= float(gates_config["minimum_each_map_half_split_soft_cycle_rate_rank_correlation"])
            for row in map_results.values()
        ),
        "quality_admissible_structural_cycle_escape": global_escape
        >= float(
            gates_config[
                "minimum_state_fraction_with_quality_admissible_structural_cycle_escape"
            ]
        ),
        "each_map_quality_admissible_structural_cycle_escape": all(
            row["quality_admissible_structural_cycle_escape_state_fraction"]
            >= float(
                gates_config[
                    "minimum_each_map_state_fraction_with_quality_admissible_structural_cycle_escape"
                ]
            )
            for row in map_results.values()
        ),
        "both_soft_cycle_classes_on_required_maps": maps_with_both
        >= int(
            gates_config[
                "minimum_maps_with_both_positive_and_negative_soft_cycle_actions"
            ]
        ),
        "escape_action_size_not_collapsed": single_size_share
        <= float(gates_config["maximum_single_size_share_among_escape_actions"]),
        "zero_errors": int(status["error_jobs"]) == int(gates_config["required_error_count"]),
        "zero_timeouts": int(status["timeout_jobs"])
        == int(gates_config["required_timeout_count"]),
    }
    passed = all(gates.values())
    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "integrity_passed": True,
        "label_readiness_passed": passed,
        "state_count": len(state_results),
        "candidate_count": len(all_candidates),
        "trial_count": len(trial_sha),
        "half_split_soft_cycle_class_agreement": global_cycle_agreement,
        "half_split_soft_cycle_rate_rank_correlation": global_cycle_correlation,
        "half_split_exact_noop_class_agreement": global_noop_agreement,
        "half_split_noop_rate_rank_correlation": global_noop_correlation,
        "quality_admissible_structural_cycle_escape_state_fraction": global_escape,
        "escape_candidate_count": len(escapes),
        "escape_nominal_size_counts": dict(sorted(size_counts.items())),
        "maximum_escape_single_size_share": single_size_share,
        "maps_with_both_soft_cycle_classes": maps_with_both,
        "map_results": map_results,
        "gates": gates,
        "decision": (
            config["decision"]["if_all_gates_pass"]
            if passed
            else config["decision"]["if_any_gate_fails"]
        ),
        "claim_boundary": dict(config["claim_boundary"]),
        "state_results": state_results,
        "sha256": {
            "config": sha256_file(config_path),
            "preparation_report": sha256_file(collection / "preparation_report.json"),
            "cohort_manifest": sha256_file(collection / "cohort_manifest.jsonl"),
            "collection_status": sha256_file(collection / "collection_status.json"),
            "trial_manifest": _fingerprint(trial_sha),
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "cycletransition_report.json", report)
    return report


__all__ = [
    "analyze_cycletransition",
    "collect_cycletransition",
    "load_cycletransition_registration",
    "prepare_cycletransition",
    "validate_registration",
]
