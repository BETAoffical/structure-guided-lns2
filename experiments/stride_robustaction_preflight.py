from __future__ import annotations

import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from experiments._common import producer_identity, sha256_file
from experiments.feature_schema_v2 import PROFILE_FEATURE_NAMES
from experiments.online_feature_engine import OnlineFeatureEngine
from experiments.repair_collection import (
    _fingerprint,
    _plain,
    _read_json,
    _run_jobs,
    _write_json,
    _write_jsonl,
    state_fingerprint,
)
from experiments.state_analysis import analyze_state, summarize_initial_state_complexity
from experiments.stride_collection import FULL_POOL_PROPOSAL, _replay_job, load_stride_selection
from experiments.stride_lns import FROZEN_FEATURE_DIMENSION, FROZEN_FEATURE_SCHEMA_ID
from experiments.stride_repairability_collection import (
    _source_target_state,
    repairability_restore_seed,
)
from experiments.stride_structpool import high_stress_gate, load_structpool_design
from experiments.trace_replay import restore_repair_state
from lns2_selector.runtime.fingerprints import repair_structure_fingerprint
from lns2_selector.runtime.online_selection import generate_online_candidates
from lns2_selector.runtime.topology_candidates import (
    generate_structpool_candidates,
    generate_topology_boundary_candidates,
    merge_structpool_candidates,
)


CONFIG_SCHEMA = "lns2.stride.robustaction_structpool_label_preflight_config.v1"
RUN_SCHEMA = "lns2.stride.robustaction_structpool_label_preflight_run.v1"
STATE_SCHEMA = "lns2.stride.robustaction_structpool_label_preflight_state.v1"
REPORT_SCHEMA = "lns2.stride.robustaction_structpool_label_preflight_report.v1"
SELECTION_POLICIES = ("official_adaptive", "v2-full")
PRODUCER_FILES = (
    "CMakeLists.txt",
    "experiments/online_feature_engine.py",
    "experiments/repair_collection.py",
    "experiments/state_analysis.py",
    "experiments/stride_collection.py",
    "experiments/stride_lns.py",
    "experiments/stride_repairability_collection.py",
    "experiments/stride_robustaction_preflight.py",
    "experiments/stride_structpool.py",
    "experiments/trace_replay.py",
    "lns2_selector/runtime/fingerprints.py",
    "lns2_selector/runtime/online_selection.py",
    "lns2_selector/runtime/topology_candidates.py",
    "src/python_bindings.cpp",
    "third_party/mapf_lns2/inc/RepairPolicy.h",
    "third_party/mapf_lns2/src/InitLNS.cpp",
)


def _registered(project_root: Path, spec: dict[str, Any]) -> Path:
    path = (project_root / str(spec["path"])).resolve()
    if not path.is_file():
        raise ValueError(f"registered label-preflight input is missing: {path}")
    observed = sha256_file(path)
    expected = str(spec["sha256"])
    if observed != expected:
        raise ValueError(
            f"registered label-preflight input changed: {path}: "
            f"expected {expected}, got {observed}"
        )
    return path


def validate_robustaction_label_preflight_config(
    config: dict[str, Any], *, project_root: Path | None = None
) -> None:
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("RobustAction label-preflight schema changed")
    if (
        config.get("scientific_status")
        != "preregistered_proposal_and_feature_preflight_before_new_candidate_repairs"
        or config.get("preflight_id")
        != "stride-robustaction-structpool-label-preflight-v1"
        or config.get("planned_model_id") != "stride-robustaction-v1"
        or config.get("candidate_pool_id") != "v2-plus-stride-structpool-v1"
        or config.get("pre_registration_git_commit")
        != "0dd04d896ae31c8b211b16e4f782d0f72517909e"
    ):
        raise ValueError("RobustAction label-preflight identity changed")

    selection = dict(config.get("selection_contract") or {})
    if selection != {
        "expected_state_count": 320,
        "expected_episode_count": 216,
        "expected_states_per_policy": 160,
        "expected_structpool_eligible_state_count": 98,
        "expected_structpool_eligible_by_policy": {
            "official_adaptive": 51,
            "v2-full": 47,
        },
        "expected_map_count": 28,
        "expected_task_count": 56,
        "maximum_states_per_episode": 2,
    }:
        raise ValueError("RobustAction label-preflight selection contract changed")

    candidate = dict(config.get("candidate_space") or {})
    if candidate != {
        "base_heuristics": ["target", "collision", "random"],
        "base_neighborhood_sizes": [4, 8, 16],
        "base_candidates_per_family": 2,
        "structpool_neighborhood_sizes": [8, 16, 24, 32],
        "maximum_added_candidates": 6,
        "maximum_jaccard_similarity": 0.8,
        "maximum_total_candidates": 24,
        "proposal_backend": "optimized",
        "proposal_repetitions": 2,
        "inactive_state_policy": (
            "exact_frozen_v2_full_pool_without_topology_analysis"
        ),
    }:
        raise ValueError("RobustAction label-preflight candidate space changed")

    features = dict(config.get("feature_contract") or {})
    if features != {
        "schema_id": "lns2.realized_features.v2",
        "profile": "realized_dynamic",
        "dimension": 124,
        "backend": "native",
        "dense_output": False,
    }:
        raise ValueError("RobustAction label-preflight feature contract changed")

    gates = dict(config.get("gates") or {})
    if (
        int(gates.get("minimum_states_with_added_candidates", -1)) != 80
        or int(gates.get("minimum_states_with_added_candidates_per_policy", -1))
        != 40
        or float(
            gates.get("minimum_fraction_of_active_states_with_three_added_candidates", -1)
        )
        != 0.9
        or int(gates.get("minimum_aggregate_family_group_count", -1)) != 5
        or int(gates.get("minimum_aggregate_neighborhood_size_count", -1)) != 4
        or dict(gates.get("minimum_maps_with_added_candidates_by_topology_group") or {})
        != {
            "dao_high_topology": 3,
            "dao_mid_topology": 3,
            "dao_low_topology_control": 2,
        }
        or not all(
            gates.get(name) is True
            for name in (
                "require_all_320_states",
                "require_exact_high_stress_gate_identity",
                "require_deterministic_proposals",
                "require_exact_base_preservation",
                "require_incumbent_boundary_preservation",
                "require_candidate_cap",
                "require_native_explicit_action_legality",
                "require_state_fingerprint_preservation",
                "require_complete_124_feature_rows",
                "require_zero_errors",
            )
        )
    ):
        raise ValueError("RobustAction label-preflight gates changed")

    execution = dict(config.get("execution") or {})
    if (
        int(execution.get("recommended_workers", -1)) != 2
        or float(execution.get("per_state_timeout_seconds", -1)) != 600.0
        or execution.get("resume_required_for_existing_output") is not True
        or execution.get("preserve_complete_failed_product") is not True
    ):
        raise ValueError("RobustAction label-preflight execution contract changed")

    boundary = dict(config.get("outcome_boundary") or {})
    if any(
        boundary.get(name) is not False
        for name in (
            "candidate_repair_step_allowed",
            "controller_action_allowed",
            "candidate_repair_outcomes_read",
            "controller_outcomes_read",
            "ttf_read",
            "future_trajectory_used_for_candidate_selection",
            "formal_speed_claim",
        )
    ):
        raise ValueError("RobustAction label-preflight outcome boundary changed")

    if project_root is None:
        return
    for spec in dict(config["inputs"]).values():
        _registered(project_root.resolve(), dict(spec))


def _candidate_signature(rows: list[dict[str, Any]]) -> str:
    return _fingerprint(
        [
            {
                "candidate_id": str(row["candidate_id"]),
                "agents": list(map(int, row["agents"])),
                "selection_families": sorted(
                    map(str, row.get("selection_families") or ())
                ),
                "structpool_family_groups": sorted(
                    map(str, row.get("structpool_family_groups") or ())
                ),
            }
            for row in rows
        ]
    )


def _state_artifact_valid(
    payload: dict[str, Any], *, run_fingerprint: str, decision: dict[str, Any]
) -> bool:
    if (
        payload.get("schema") != STATE_SCHEMA
        or payload.get("run_fingerprint") != run_fingerprint
        or payload.get("state_id") != decision.get("state_id")
        or payload.get("decision") != decision
        or payload.get("complete") is not True
        or payload.get("candidate_repair_trials_executed") is not False
        or payload.get("controller_actions_executed") is not False
    ):
        return False
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return False
    ids = [str(row.get("candidate_id", "")) for row in candidates]
    if not all(ids) or len(ids) != len(set(ids)):
        return False
    if payload.get("candidate_signature") != _candidate_signature(candidates):
        return False
    required = set(PROFILE_FEATURE_NAMES["realized_dynamic"])
    for candidate in candidates:
        features = candidate.get("features")
        if not isinstance(features, dict) or set(features) != required:
            return False
        if len(features) != FROZEN_FEATURE_DIMENSION:
            return False
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in features.values()
        ):
            return False
    forbidden = {
        "after_conflicts",
        "candidate_trials",
        "conflicts_after",
        "native_step_seconds",
        "repair_outcome",
        "trials",
        "ttf",
    }
    return not (forbidden & set(payload))


def _jaccard(left: list[int], right: list[int]) -> float:
    left_set = set(map(int, left))
    right_set = set(map(int, right))
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 0.0


def _preflight_state(job: dict[str, Any]) -> dict[str, Any]:
    decision = dict(job["decision"])
    output_path = Path(str(job["output_path"]))
    run_fingerprint = str(job["run_fingerprint"])
    if bool(job["resume"]) and output_path.is_file():
        existing = _read_json(output_path)
        if _state_artifact_valid(
            existing, run_fingerprint=run_fingerprint, decision=decision
        ):
            return {
                "state_id": str(decision["state_id"]),
                "state_file": str(output_path),
                "status": "resumed",
                "state_count": 1,
                "outcome_count": 0,
                "error_count": 0,
            }
        raise ValueError(f"invalid completed label-preflight state: {output_path}")

    candidate_space = dict(job["candidate_space"])
    feature_contract = dict(job["feature_contract"])
    design = dict(job["structpool_design"])
    replay = _replay_job(decision)
    state, source_manifest, source_trace_path = _source_target_state(decision)
    before_fingerprint = state_fingerprint(state)
    if before_fingerprint != str(decision["before_fingerprint"]):
        raise RuntimeError("label-preflight source-state fingerprint changed")
    before_repair = repair_structure_fingerprint(state)
    restore_seed = repairability_restore_seed(before_repair)
    environment, restored = restore_repair_state(replay, state, seed=restore_seed)
    if repair_structure_fingerprint(restored) != before_repair:
        raise RuntimeError("label-preflight restore changed repair structure")

    summary = summarize_initial_state_complexity(state)
    gate_passed = high_stress_gate(summary, design)
    selection_proxy = (
        int(decision["before_conflicts"]) >= 16
        and int(decision["agent_count"]) >= 96
    )
    if gate_passed != selection_proxy:
        raise RuntimeError("selection proxy and exact StructPool gate differ")

    proposal = {
        **dict(replay["proposal"]),
        **FULL_POOL_PROPOSAL,
        "heuristics": list(candidate_space["base_heuristics"]),
        "neighborhood_sizes": list(candidate_space["base_neighborhood_sizes"]),
        "candidates_per_family": int(candidate_space["base_candidates_per_family"]),
    }
    proposal.pop("topology_boundary", None)
    base, generation = generate_online_candidates(
        environment,
        state,
        task_id=str(decision["task_id"]),
        solver_seed=int(decision["solver_seed"]),
        decision_index=int(decision["decision_index"]),
        proposal_config=proposal,
        state_hash=before_fingerprint,
        verify_full_state=False,
        proposal_backend=str(candidate_space["proposal_backend"]),
        shadow_validation=False,
    )

    additions: list[dict[str, Any]] = []
    analysis = None
    deterministic = True
    incumbent_sets: set[tuple[int, ...]] = set()
    if gate_passed:
        analysis = analyze_state(state)
        repetitions = [
            generate_structpool_candidates(
                state,
                analysis,
                neighborhood_sizes=candidate_space["structpool_neighborhood_sizes"],
                maximum_added_candidates=int(
                    candidate_space["maximum_added_candidates"]
                ),
                maximum_jaccard_similarity=float(
                    candidate_space["maximum_jaccard_similarity"]
                ),
            )
            for _ in range(int(candidate_space["proposal_repetitions"]))
        ]
        additions = repetitions[0]
        deterministic = all(
            _candidate_signature(rows) == _candidate_signature(additions)
            for rows in repetitions[1:]
        )
        incumbent = generate_topology_boundary_candidates(
            state, analysis, neighborhood_size=16, core_budget=4
        )
        incumbent_sets = {tuple(map(int, row["agents"])) for row in incumbent}

    merged = merge_structpool_candidates(base, additions)
    base_sets = {tuple(map(int, row["agents"])) for row in base}
    addition_sets = {tuple(map(int, row["agents"])) for row in additions}
    unique_additions = [
        row for row in additions if tuple(map(int, row["agents"])) not in base_sets
    ]
    unique_addition_sets = {
        tuple(map(int, row["agents"])) for row in unique_additions
    }
    novel_additions = [
        row
        for row in unique_additions
        if tuple(map(int, row["agents"])) not in incumbent_sets
    ]
    similarities = [
        _jaccard(left["agents"], right["agents"])
        for index, left in enumerate(novel_additions)
        for right in novel_additions[index + 1 :]
    ]
    active_agents = {
        int(agent)
        for event in analysis.events
        for agent in (event.left, event.right)
    } if analysis is not None and additions else set()

    feature_engine = OnlineFeatureEngine(
        state,
        backend=str(feature_contract["backend"]),
        required_features={
            str(feature_contract["profile"]): PROFILE_FEATURE_NAMES[
                str(feature_contract["profile"])
            ]
        },
        dense_output=bool(feature_contract["dense_output"]),
    )
    feature_rows, feature_metrics = feature_engine.realized_rows(
        merged, state_hash=before_fingerprint
    )
    profile = str(feature_contract["profile"])
    features_by_id = {
        str(row["candidate_id"]): dict(row["features"][profile])
        for row in feature_rows
    }
    if set(features_by_id) != {str(row["candidate_id"]) for row in merged}:
        raise RuntimeError("label-preflight candidate features are incomplete")

    candidates = []
    for candidate in merged:
        candidate_id = str(candidate["candidate_id"])
        agents = list(map(int, candidate["agents"]))
        candidate_set = tuple(agents)
        candidates.append(
            {
                "candidate_id": candidate_id,
                "candidate_kind": (
                    "structpool" if candidate_set in unique_addition_sets else "base"
                ),
                "agents": agents,
                "actual_size": len(agents),
                "selection_families": sorted(
                    map(str, candidate.get("selection_families") or ())
                ),
                "structpool_family_groups": sorted(
                    map(str, candidate.get("structpool_family_groups") or ())
                ),
                "features": features_by_id[candidate_id],
            }
        )
    state_preserved = (
        state_fingerprint(state) == before_fingerprint
        and repair_structure_fingerprint(_plain(environment.get_state()))
        == before_repair
    )
    payload = {
        "schema": STATE_SCHEMA,
        "run_fingerprint": run_fingerprint,
        "complete": True,
        "state_id": str(decision["state_id"]),
        "decision": decision,
        "before_fingerprint": before_fingerprint,
        "before_repair_fingerprint": before_repair,
        "before_conflicts": int(decision["before_conflicts"]),
        "source_trace_file": str(source_manifest["trace_file"]),
        "source_trace_path": str(source_trace_path),
        "restore_seed": restore_seed,
        "high_stress_gate_passed": gate_passed,
        "selection_proxy_eligible": selection_proxy,
        "topology_analysis_executed": gate_passed,
        "base_generation": generation,
        "base_candidate_count": len(base),
        "added_candidate_count": len(unique_additions),
        "total_candidate_count": len(merged),
        "added_family_groups": sorted(
            {
                group
                for row in unique_additions
                for group in row.get("structpool_family_groups") or ()
            }
        ),
        "added_sizes": sorted(
            {int(row.get("actual_size", len(row["agents"]))) for row in unique_additions}
        ),
        "deterministic": deterministic,
        "base_preserved": merged[: len(base)] == base,
        "incumbent_boundary_preserved": (
            incumbent_sets <= addition_sets if gate_passed else True
        ),
        "candidate_cap_preserved": (
            len(additions) <= int(candidate_space["maximum_added_candidates"])
            and len(merged) <= int(candidate_space["maximum_total_candidates"])
        ),
        "native_explicit_action_legal": (
            bool(active_agents)
            and all(
                set(map(int, row["agents"])) & active_agents
                for row in unique_additions
            )
            if unique_additions
            else True
        ),
        "maximum_novel_jaccard_similarity": max(similarities, default=0.0),
        "state_fingerprint_preserved": state_preserved,
        "feature_schema_id": FROZEN_FEATURE_SCHEMA_ID,
        "feature_dimension": FROZEN_FEATURE_DIMENSION,
        "feature_metrics": feature_metrics,
        "candidate_signature": _candidate_signature(candidates),
        "feature_signature": _fingerprint(
            [
                {
                    "candidate_id": row["candidate_id"],
                    "features": row["features"],
                }
                for row in candidates
            ]
        ),
        "candidates": candidates,
        "candidate_repair_trials_executed": False,
        "controller_actions_executed": False,
        "candidate_repair_outcomes_read": False,
        "ttf_read": False,
    }
    if not _state_artifact_valid(
        payload, run_fingerprint=run_fingerprint, decision=decision
    ):
        raise RuntimeError("label-preflight state artifact is invalid")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(output_path.name + ".partial")
    _write_json(partial, payload)
    os.replace(partial, output_path)
    return {
        "state_id": str(decision["state_id"]),
        "state_file": str(output_path),
        "status": "ok",
        "state_count": 1,
        "outcome_count": 0,
        "error_count": 0,
    }


def _summary_row(payload: dict[str, Any]) -> dict[str, Any]:
    decision = dict(payload["decision"])
    return {
        "schema": STATE_SCHEMA,
        "state_id": str(payload["state_id"]),
        "map_id": str(decision["map_id"]),
        "task_id": str(decision["task_id"]),
        "source_policy": str(decision["source_policy"]),
        "layout_mode": str(decision.get("layout_mode", "unknown")),
        "before_conflicts": int(payload["before_conflicts"]),
        "high_stress_gate_passed": bool(payload["high_stress_gate_passed"]),
        "selection_proxy_eligible": bool(payload["selection_proxy_eligible"]),
        "topology_analysis_executed": bool(payload["topology_analysis_executed"]),
        "base_candidate_count": int(payload["base_candidate_count"]),
        "added_candidate_count": int(payload["added_candidate_count"]),
        "total_candidate_count": int(payload["total_candidate_count"]),
        "added_family_groups": list(payload["added_family_groups"]),
        "added_sizes": list(payload["added_sizes"]),
        "deterministic": bool(payload["deterministic"]),
        "base_preserved": bool(payload["base_preserved"]),
        "incumbent_boundary_preserved": bool(
            payload["incumbent_boundary_preserved"]
        ),
        "candidate_cap_preserved": bool(payload["candidate_cap_preserved"]),
        "native_explicit_action_legal": bool(
            payload["native_explicit_action_legal"]
        ),
        "maximum_novel_jaccard_similarity": float(
            payload["maximum_novel_jaccard_similarity"]
        ),
        "state_fingerprint_preserved": bool(
            payload["state_fingerprint_preserved"]
        ),
        "feature_schema_id": str(payload["feature_schema_id"]),
        "feature_dimension": int(payload["feature_dimension"]),
        "candidate_signature": str(payload["candidate_signature"]),
        "feature_signature": str(payload["feature_signature"]),
    }


def analyze_robustaction_preflight_rows(
    config: dict[str, Any], rows: list[dict[str, Any]], *, error_count: int
) -> dict[str, Any]:
    selection = dict(config["selection_contract"])
    gates_config = dict(config["gates"])
    active = [row for row in rows if bool(row["high_stress_gate_passed"])]
    with_additions = [row for row in rows if int(row["added_candidate_count"]) > 0]
    added_by_policy = Counter(str(row["source_policy"]) for row in with_additions)
    active_by_policy = Counter(str(row["source_policy"]) for row in active)
    maps_by_layout: dict[str, set[str]] = defaultdict(set)
    for row in with_additions:
        maps_by_layout[str(row["layout_mode"])].add(str(row["map_id"]))
    family_groups = {
        str(group) for row in rows for group in row["added_family_groups"]
    }
    sizes = {int(size) for row in rows for size in row["added_sizes"]}
    fraction_three = (
        sum(int(row["added_candidate_count"]) >= 3 for row in active) / len(active)
        if active
        else 0.0
    )
    minimum_maps = dict(
        gates_config["minimum_maps_with_added_candidates_by_topology_group"]
    )
    gates = {
        "all_320_states": len(rows) == int(selection["expected_state_count"]),
        "zero_errors": error_count == 0,
        "exact_high_stress_gate_identity": (
            len(active) == int(selection["expected_structpool_eligible_state_count"])
            and all(
                active_by_policy[policy] == int(expected)
                for policy, expected in dict(
                    selection["expected_structpool_eligible_by_policy"]
                ).items()
            )
            and all(
                bool(row["high_stress_gate_passed"])
                == bool(row["selection_proxy_eligible"])
                for row in rows
            )
        ),
        "minimum_states_with_added_candidates": len(with_additions)
        >= int(gates_config["minimum_states_with_added_candidates"]),
        "minimum_states_with_added_candidates_per_policy": all(
            added_by_policy[policy]
            >= int(gates_config["minimum_states_with_added_candidates_per_policy"])
            for policy in SELECTION_POLICIES
        ),
        "minimum_fraction_of_active_states_with_three_added_candidates": fraction_three
        >= float(
            gates_config[
                "minimum_fraction_of_active_states_with_three_added_candidates"
            ]
        ),
        "minimum_aggregate_family_group_count": len(family_groups)
        >= int(gates_config["minimum_aggregate_family_group_count"]),
        "minimum_aggregate_neighborhood_size_count": len(sizes)
        >= int(gates_config["minimum_aggregate_neighborhood_size_count"]),
        "minimum_maps_with_added_candidates_by_topology_group": all(
            len(maps_by_layout[group]) >= int(minimum)
            for group, minimum in minimum_maps.items()
        ),
        "deterministic_proposals": all(bool(row["deterministic"]) for row in rows),
        "exact_base_preservation": all(bool(row["base_preserved"]) for row in rows),
        "incumbent_boundary_preservation": all(
            bool(row["incumbent_boundary_preserved"]) for row in active
        ),
        "candidate_cap": all(bool(row["candidate_cap_preserved"]) for row in rows),
        "native_explicit_action_legality": all(
            bool(row["native_explicit_action_legal"]) for row in rows
        ),
        "state_fingerprint_preservation": all(
            bool(row["state_fingerprint_preserved"]) for row in rows
        ),
        "complete_124_feature_rows": all(
            row["feature_schema_id"] == FROZEN_FEATURE_SCHEMA_ID
            and int(row["feature_dimension"]) == FROZEN_FEATURE_DIMENSION
            for row in rows
        ),
        "inactive_exact_v2_fallback": all(
            int(row["added_candidate_count"]) == 0
            and not bool(row["topology_analysis_executed"])
            for row in rows
            if not bool(row["high_stress_gate_passed"])
        ),
        "jaccard_filter": all(
            float(row["maximum_novel_jaccard_similarity"])
            <= float(config["candidate_space"]["maximum_jaccard_similarity"])
            for row in rows
        ),
    }
    return {
        "active_state_count": len(active),
        "active_state_count_by_policy": dict(sorted(active_by_policy.items())),
        "states_with_added_candidates": len(with_additions),
        "states_with_added_candidates_by_policy": dict(
            sorted(added_by_policy.items())
        ),
        "fraction_of_active_states_with_three_added_candidates": fraction_three,
        "maps_with_added_candidates_by_topology_group": {
            group: sorted(maps) for group, maps in sorted(maps_by_layout.items())
        },
        "aggregate_family_groups": sorted(family_groups),
        "aggregate_neighborhood_sizes": sorted(sizes),
        "candidate_count_distribution": dict(
            sorted(Counter(int(row["total_candidate_count"]) for row in rows).items())
        ),
        "added_candidate_count_distribution": dict(
            sorted(Counter(int(row["added_candidate_count"]) for row in rows).items())
        ),
        "gates": gates,
        "passed": all(gates.values()),
    }


def run_robustaction_label_preflight(
    *,
    config_path: str | Path,
    output: str | Path,
    workers: int = 2,
    resume: bool = False,
) -> dict[str, Any]:
    if workers <= 0:
        raise ValueError("label-preflight workers must be positive")
    config_path = Path(config_path).resolve()
    project_root = config_path.parents[1]
    output = Path(output).resolve()
    config = _read_json(config_path)
    validate_robustaction_label_preflight_config(
        config, project_root=project_root
    )
    inputs = {
        name: _registered(project_root, dict(spec))
        for name, spec in dict(config["inputs"]).items()
    }
    selection_report = _read_json(inputs["selection_report"])
    coverage_report = _read_json(inputs["proposal_coverage_report"])
    headroom_report = _read_json(inputs["headroom_report"])
    if not all(
        report.get("passed") is True
        for report in (selection_report, coverage_report, headroom_report)
    ):
        raise ValueError("label-preflight requires passed predecessor reports")
    selected = load_stride_selection(inputs["state_selection"])
    contract = dict(config["selection_contract"])
    if (
        len(selected) != int(contract["expected_state_count"])
        or len({str(row["episode_id"]) for row in selected})
        != int(contract["expected_episode_count"])
        or Counter(str(row["source_policy"]) for row in selected)
        != Counter(
            {
                policy: int(contract["expected_states_per_policy"])
                for policy in SELECTION_POLICIES
            }
        )
    ):
        raise ValueError("label-preflight selection dimensions changed")
    structpool_design = load_structpool_design(inputs["structpool_design"])
    producer = producer_identity(
        project_root=project_root,
        source_files=PRODUCER_FILES,
        native_required=True,
        package_names=("numpy",),
    )
    source_run_hashes = {
        str(Path(str(row["source_root"])).resolve()): sha256_file(
            Path(str(row["source_root"])).resolve() / "run_config.json"
        )
        for row in selected
    }
    identity = {
        "schema": RUN_SCHEMA,
        "config_sha256": sha256_file(config_path),
        "input_sha256": {
            name: sha256_file(path) for name, path in sorted(inputs.items())
        },
        "selection_sha256": sha256_file(inputs["state_selection"]),
        "selected_state_ids": [str(row["state_id"]) for row in selected],
        "candidate_space": dict(config["candidate_space"]),
        "feature_contract": dict(config["feature_contract"]),
        "source_run_config_sha256": dict(sorted(source_run_hashes.items())),
        "producer": producer,
    }
    run_fingerprint = _fingerprint(identity)
    run_path = output / "run_config.json"
    if run_path.is_file():
        if _read_json(run_path).get("run_fingerprint") != run_fingerprint:
            raise ValueError("label-preflight output belongs to another run")
        if not resume:
            raise ValueError("label-preflight output exists; pass --resume")
    output.mkdir(parents=True, exist_ok=True)
    _write_json(run_path, {**identity, "run_fingerprint": run_fingerprint})
    jobs = []
    for decision in selected:
        key = _fingerprint(
            {
                "state_id": decision["state_id"],
                "before": decision["before_fingerprint"],
            }
        )[:20]
        jobs.append(
            {
                "job_id": str(decision["state_id"]),
                "state_id": str(decision["state_id"]),
                "row": decision,
                "solver_seed": int(decision["solver_seed"]),
                "decision": decision,
                "output_path": str(output / "states" / f"{key}.json"),
                "run_fingerprint": run_fingerprint,
                "candidate_space": dict(config["candidate_space"]),
                "feature_contract": dict(config["feature_contract"]),
                "structpool_design": structpool_design,
                "resume": bool(resume),
            }
        )
    status_path = output / "preflight_status.json"
    observed: list[dict[str, Any]] = []

    def update_status(result: dict[str, Any]) -> None:
        observed.append(result)
        failures = [
            row for row in observed if row.get("status") in {"error", "timeout"}
        ]
        _write_json(
            status_path,
            {
                "schema": REPORT_SCHEMA,
                "run_fingerprint": run_fingerprint,
                "requested_state_count": len(jobs),
                "completed_state_count": len(observed) - len(failures),
                "error_state_count": len(failures),
                "timeout_state_count": sum(
                    row.get("status") == "timeout" for row in observed
                ),
                "status": "running",
                "errors": [
                    {
                        "state_id": str(row.get("state_id", row.get("job_id"))),
                        "error": str(row.get("error")),
                    }
                    for row in failures
                ],
            },
        )

    _write_json(
        status_path,
        {
            "schema": REPORT_SCHEMA,
            "run_fingerprint": run_fingerprint,
            "requested_state_count": len(jobs),
            "completed_state_count": 0,
            "error_state_count": 0,
            "timeout_state_count": 0,
            "status": "running",
            "errors": [],
        },
    )
    try:
        observed = _run_jobs(
            _preflight_state,
            jobs,
            workers,
            phase="stride-robustaction-label-preflight",
            output_root=output,
            run_fingerprint=run_fingerprint,
            timeout_seconds=float(config["execution"]["per_state_timeout_seconds"]),
            on_result=update_status,
        )
    except BaseException as error:
        current = _read_json(status_path)
        _write_json(
            status_path,
            {
                **current,
                "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "error",
                "runner_error": f"{type(error).__name__}: {error}",
            },
        )
        raise

    successes = [
        row for row in observed if row.get("status") in {"ok", "resumed"}
    ]
    errors = [
        {
            "state_id": str(row.get("state_id", row.get("job_id"))),
            "error": str(row.get("error")),
            "status": str(row.get("status")),
        }
        for row in observed
        if row.get("status") in {"error", "timeout"}
    ]
    selected_by_id = {str(row["state_id"]): row for row in selected}
    summaries: list[dict[str, Any]] = []
    for result in sorted(successes, key=lambda row: str(row["state_id"])):
        payload = _read_json(Path(str(result["state_file"])))
        decision = selected_by_id[str(result["state_id"])]
        if not _state_artifact_valid(
            payload, run_fingerprint=run_fingerprint, decision=decision
        ):
            errors.append(
                {
                    "state_id": str(result["state_id"]),
                    "error": "invalid state artifact",
                    "status": "error",
                }
            )
            continue
        summaries.append(_summary_row(payload))
    analysis = analyze_robustaction_preflight_rows(
        config, summaries, error_count=len(errors)
    )
    rows_path = output / "preflight_states.jsonl"
    _write_jsonl(rows_path, summaries)
    report = {
        "schema": REPORT_SCHEMA,
        "scientific_status": "proposal_and_feature_preflight_no_repairs",
        "preflight_id": str(config["preflight_id"]),
        "data_line_id": str(config["data_line_id"]),
        "planned_model_id": str(config["planned_model_id"]),
        "candidate_pool_id": str(config["candidate_pool_id"]),
        "run_fingerprint": run_fingerprint,
        "requested_state_count": len(jobs),
        "completed_state_count": len(summaries),
        "new_state_count": sum(row.get("status") == "ok" for row in successes),
        "resumed_state_count": sum(
            row.get("status") == "resumed" for row in successes
        ),
        "error_state_count": len(errors),
        "errors": errors,
        **analysis,
        "candidate_repair_trials_executed": False,
        "controller_actions_executed": False,
        "candidate_repair_outcomes_read": False,
        "controller_outcomes_read": False,
        "ttf_read": False,
        "formal_speed_claim": False,
        "config_sha256": sha256_file(config_path),
        "state_rows_sha256": sha256_file(rows_path),
        "next_decision": config[
            "next_decision_on_pass" if analysis["passed"] else "next_decision_on_failure"
        ],
    }
    _write_json(output / "preflight_report.json", report)
    _write_json(
        status_path,
        {
            "schema": REPORT_SCHEMA,
            "run_fingerprint": run_fingerprint,
            "requested_state_count": len(jobs),
            "completed_state_count": len(summaries),
            "error_state_count": len(errors),
            "timeout_state_count": sum(
                row.get("status") == "timeout" for row in observed
            ),
            "status": "complete" if analysis["passed"] else "failed",
            "passed": bool(analysis["passed"]),
            "errors": errors,
        },
    )
    return report


__all__ = [
    "CONFIG_SCHEMA",
    "REPORT_SCHEMA",
    "analyze_robustaction_preflight_rows",
    "run_robustaction_label_preflight",
    "validate_robustaction_label_preflight_config",
]
