from __future__ import annotations

import collections
import copy
import math
from pathlib import Path
from typing import Any

from experiments._common import sha256_file
from experiments.repair_collection import (
    _read_json,
    _read_jsonl,
    _write_json,
    _write_jsonl,
    state_fingerprint,
)
from experiments.state_analysis import analyze_state, summarize_initial_state_complexity
from experiments.stride_collection import FULL_POOL_PROPOSAL, _replay_job
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


CONFIG_SCHEMA = "lns2.stride.structpool_coverage_config.v1"
ROW_SCHEMA = "lns2.stride.structpool_coverage_state.v1"
REPORT_SCHEMA = "lns2.stride.structpool_coverage_report.v1"


def _mean(values: list[float]) -> float:
    return math.fsum(values) / len(values) if values else 0.0


def _registered_path(project_root: Path, spec: dict[str, Any]) -> Path:
    path = (project_root / str(spec["path"])).resolve()
    if not path.is_file() or sha256_file(path) != str(spec["sha256"]):
        raise ValueError(f"StructPool registered input changed: {spec['path']}")
    return path


def validate_structpool_coverage_config(config: dict[str, Any]) -> None:
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unexpected StructPool coverage config")
    if (
        config.get("scientific_status") != "proposal_only_train_development_audit"
        or config.get("diagnostic_id") != "stride-structpool-coverage-v1"
        or config.get("candidate_generator_id") != "stride-structpool-v1"
        or bool(config.get("formal_speed_claim"))
        or bool(config.get("default_replacement_allowed"))
        or bool(config.get("runtime_export_allowed"))
        or bool(config.get("formal_ood_allowed"))
        or bool(config.get("candidate_repair_trials_allowed"))
        or bool(config.get("controller_actions_allowed"))
        or bool(config.get("controller_outcomes_allowed"))
        or bool(config.get("selection_outcome_fields_allowed"))
    ):
        raise ValueError("StructPool coverage must remain proposal-only")
    if set(config.get("inputs") or {}) != {
        "design",
        "state_selection",
        "source_run_config",
        "official_adaptive_manifest",
        "v2_full_manifest",
    }:
        raise ValueError("StructPool coverage input registry changed")
    if dict(config.get("selection") or {}) != {
        "research_split": "train",
        "minimum_agent_count": 96,
        "minimum_before_conflicts": 16,
        "rule": "deterministic_round_robin_by_map_and_source_policy_then_state_id",
        "expected_selected_state_count": 48,
        "minimum_distinct_map_count": 12,
        "required_source_policies": ["official_adaptive", "v2-full"],
    }:
        raise ValueError("StructPool outcome-blind state selection changed")
    if (
        int(config.get("proposal_repetitions", -1)) != 2
        or config.get("proposal_backend") != "optimized"
        or dict(config.get("candidate_space") or {})
        != {
            "neighborhood_sizes": [8, 16, 24, 32],
            "maximum_added_candidates": 6,
            "maximum_jaccard_similarity": 0.8,
        }
    ):
        raise ValueError("StructPool proposal contract changed")
    if dict(config.get("gates") or {}) != {
        "minimum_eligible_state_count": 48,
        "minimum_fraction_with_three_added_candidates": 0.9,
        "minimum_aggregate_novel_family_coverage": 5,
        "minimum_aggregate_neighborhood_size_coverage": 4,
        "require_deterministic_replay": True,
        "require_exact_base_preservation": True,
        "require_exact_incumbent_boundary_preservation": True,
        "require_candidate_cap": True,
        "require_native_explicit_action_legality": True,
        "require_state_fingerprint_preservation": True,
    }:
        raise ValueError("StructPool proposal gates changed")


def select_structpool_states(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    """Select train states using only fields present before candidate repair."""

    selection = dict(config["selection"])
    eligible = [
        copy.deepcopy(row)
        for row in rows
        if str(row.get("research_split")) == selection["research_split"]
        and int(row.get("agent_count", -1)) >= int(selection["minimum_agent_count"])
        and int(row.get("before_conflicts", -1))
        >= int(selection["minimum_before_conflicts"])
    ]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in eligible:
        groups[(str(row["map_id"]), str(row["source_policy"]))].append(row)
    for values in groups.values():
        values.sort(key=lambda row: str(row["state_id"]))
    selected: list[dict[str, Any]] = []
    keys = sorted(groups)
    while len(selected) < int(selection["expected_selected_state_count"]):
        progressed = False
        for key in keys:
            if groups[key] and len(selected) < int(
                selection["expected_selected_state_count"]
            ):
                selected.append(groups[key].pop(0))
                progressed = True
        if not progressed:
            break
    if len(selected) != int(selection["expected_selected_state_count"]):
        raise ValueError("StructPool source has insufficient outcome-blind states")
    if len({str(row["state_id"]) for row in selected}) != len(selected):
        raise ValueError("StructPool state selection repeats a state")
    if len({str(row["map_id"]) for row in selected}) < int(
        selection["minimum_distinct_map_count"]
    ):
        raise ValueError("StructPool state selection has insufficient map coverage")
    if {str(row["source_policy"]) for row in selected} != set(
        selection["required_source_policies"]
    ):
        raise ValueError("StructPool state selection lacks a source policy")
    return selected


def _signature(rows: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    return [
        (
            str(row["candidate_id"]),
            tuple(map(int, row["agents"])),
            tuple(map(str, row["selection_families"])),
            tuple(map(str, row.get("structpool_family_groups", []))),
        )
        for row in rows
    ]


def analyze_structpool_coverage_rows(
    config: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    gates_config = dict(config["gates"])
    expected = int(config["selection"]["expected_selected_state_count"])
    family_groups = {
        group for row in rows for group in row["added_family_groups"]
    }
    sizes = {int(size) for row in rows for size in row["added_sizes"]}
    at_least_three = [int(row["added_candidate_count"]) >= 3 for row in rows]
    gates = {
        "minimum_eligible_state_count": len(rows)
        >= int(gates_config["minimum_eligible_state_count"]),
        "expected_selected_state_count": len(rows) == expected,
        "all_states_gate_eligible": all(bool(row["high_stress_gate_passed"]) for row in rows),
        "minimum_fraction_with_three_added_candidates": (
            sum(at_least_three) / len(at_least_three) if at_least_three else 0.0
        )
        >= float(gates_config["minimum_fraction_with_three_added_candidates"]),
        "minimum_aggregate_novel_family_coverage": len(family_groups)
        >= int(gates_config["minimum_aggregate_novel_family_coverage"]),
        "minimum_aggregate_neighborhood_size_coverage": len(sizes)
        >= int(gates_config["minimum_aggregate_neighborhood_size_coverage"]),
        "deterministic_replay": all(bool(row["deterministic"]) for row in rows),
        "exact_base_preservation": all(bool(row["base_preserved"]) for row in rows),
        "exact_incumbent_boundary_preservation": all(
            bool(row["incumbent_boundary_preserved"]) for row in rows
        ),
        "candidate_cap": all(bool(row["candidate_cap_preserved"]) for row in rows),
        "native_explicit_action_legality": all(
            bool(row["native_explicit_action_legal"]) for row in rows
        ),
        "state_fingerprint_preservation": all(
            bool(row["state_fingerprint_preserved"]) for row in rows
        ),
        "jaccard_filter": all(
            float(row["maximum_novel_jaccard_similarity"])
            <= float(config["candidate_space"]["maximum_jaccard_similarity"])
            for row in rows
        ),
    }
    passed = all(gates.values())
    return {
        "schema": REPORT_SCHEMA,
        "scientific_status": "proposal_only_train_development_audit",
        "candidate_generator_id": "stride-structpool-v1",
        "state_count": len(rows),
        "map_count": len({str(row["map_id"]) for row in rows}),
        "source_policies": sorted({str(row["source_policy"]) for row in rows}),
        "added_candidate_count": sum(int(row["added_candidate_count"]) for row in rows),
        "mean_added_candidate_count": _mean(
            [float(row["added_candidate_count"]) for row in rows]
        ),
        "fraction_with_three_added_candidates": (
            sum(at_least_three) / len(at_least_three) if at_least_three else 0.0
        ),
        "aggregate_family_groups": sorted(family_groups),
        "aggregate_neighborhood_sizes": sorted(sizes),
        "candidate_repair_trials_executed": False,
        "controller_actions_executed": False,
        "controller_outcomes_read": False,
        "selection_outcome_fields_read": False,
        "formal_speed_claim": False,
        "gates": gates,
        "passed": passed,
        "next_decision": config[
            "next_decision_on_pass" if passed else "next_decision_on_failure"
        ],
    }


def collect_structpool_coverage(
    config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    project_root = config_path.parents[1]
    config = _read_json(config_path)
    validate_structpool_coverage_config(config)
    inputs = {
        name: _registered_path(project_root, dict(spec))
        for name, spec in dict(config["inputs"]).items()
    }
    design = load_structpool_design(inputs["design"])
    selected = select_structpool_states(
        _read_jsonl(inputs["state_selection"]), config
    )
    candidate_space = dict(config["candidate_space"])
    state_rows: list[dict[str, Any]] = []
    for decision in selected:
        replay = _replay_job(decision)
        state, _manifest, _trace = _source_target_state(decision)
        before_fingerprint = state_fingerprint(state)
        if before_fingerprint != str(decision["before_fingerprint"]):
            raise RuntimeError("StructPool source state fingerprint changed")
        repair_fingerprint = repair_structure_fingerprint(state)
        environment, restored = restore_repair_state(
            replay, state, seed=repairability_restore_seed(repair_fingerprint)
        )
        if repair_structure_fingerprint(restored) != repair_fingerprint:
            raise RuntimeError("StructPool state restore changed repair structure")
        summary = summarize_initial_state_complexity(state)
        gate_passed = high_stress_gate(summary, design)
        if not gate_passed:
            raise RuntimeError("outcome-blind StructPool selection violated high-stress gate")

        proposal = {**dict(replay["proposal"]), **FULL_POOL_PROPOSAL}
        proposal.pop("topology_boundary", None)
        base, _generation = generate_online_candidates(
            environment,
            state,
            task_id=str(decision["task_id"]),
            solver_seed=int(decision["solver_seed"]),
            decision_index=int(decision["decision_index"]),
            proposal_config=proposal,
            state_hash=before_fingerprint,
            verify_full_state=False,
            proposal_backend=str(config["proposal_backend"]),
            shadow_validation=False,
        )
        analysis = analyze_state(state)
        repetitions = [
            generate_structpool_candidates(
                state,
                analysis,
                neighborhood_sizes=candidate_space["neighborhood_sizes"],
                maximum_added_candidates=int(
                    candidate_space["maximum_added_candidates"]
                ),
                maximum_jaccard_similarity=float(
                    candidate_space["maximum_jaccard_similarity"]
                ),
            )
            for _ in range(int(config["proposal_repetitions"]))
        ]
        additions = repetitions[0]
        merged = merge_structpool_candidates(base, additions)
        incumbent = generate_topology_boundary_candidates(
            state, analysis, neighborhood_size=16, core_budget=4
        )
        incumbent_sets = {tuple(map(int, row["agents"])) for row in incumbent}
        addition_sets = {tuple(map(int, row["agents"])) for row in additions}
        base_sets = {tuple(map(int, row["agents"])) for row in base}
        unique_added = [
            row for row in additions if tuple(map(int, row["agents"])) not in base_sets
        ]
        novel = [
            row
            for row in additions
            if tuple(map(int, row["agents"])) not in incumbent_sets
        ]
        similarities = []
        for index, left in enumerate(novel):
            left_set = set(map(int, left["agents"]))
            for right in novel[index + 1 :]:
                right_set = set(map(int, right["agents"]))
                similarities.append(
                    len(left_set & right_set) / len(left_set | right_set)
                )
        active_agents = {
            int(agent)
            for event in analysis.events
            for agent in (event.left, event.right)
        }
        after_fingerprint = state_fingerprint(state)
        restored_after = repair_structure_fingerprint(environment.get_state())
        state_rows.append(
            {
                "schema": ROW_SCHEMA,
                "state_id": str(decision["state_id"]),
                "map_id": str(decision["map_id"]),
                "task_id": str(decision["task_id"]),
                "source_policy": str(decision["source_policy"]),
                "solver_seed": int(decision["solver_seed"]),
                "agent_count": int(summary["agent_count"]),
                "before_conflicts": int(summary["conflict_pair_count"]),
                "high_stress_gate_passed": gate_passed,
                "base_candidate_count": len(base),
                "added_candidate_count": len(unique_added),
                "total_candidate_count": len(merged),
                "added_family_groups": sorted(
                    {
                        group
                        for row in additions
                        for group in row["structpool_family_groups"]
                    }
                ),
                "added_sizes": sorted(
                    {int(row["actual_size"]) for row in additions}
                ),
                "deterministic": all(
                    _signature(rows) == _signature(additions)
                    for rows in repetitions[1:]
                ),
                "base_preserved": merged[: len(base)] == base,
                "incumbent_boundary_preserved": incumbent_sets <= addition_sets,
                "candidate_cap_preserved": len(additions)
                <= int(candidate_space["maximum_added_candidates"])
                and len(merged) <= len(base) + int(
                    candidate_space["maximum_added_candidates"]
                ),
                "native_explicit_action_legal": bool(active_agents)
                and all(set(map(int, row["agents"])) & active_agents for row in additions),
                "maximum_novel_jaccard_similarity": max(similarities, default=0.0),
                "state_fingerprint_preserved": after_fingerprint
                == before_fingerprint
                and restored_after == repair_fingerprint,
            }
        )

    state_rows.sort(key=lambda row: str(row["state_id"]))
    report = analyze_structpool_coverage_rows(config, state_rows)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows_path = output / "structpool_coverage_states.jsonl"
    _write_jsonl(rows_path, state_rows)
    report["inputs"] = {
        "config_sha256": sha256_file(config_path),
        **{
            f"{name}_sha256": sha256_file(path)
            for name, path in sorted(inputs.items())
        },
    }
    report["artifacts"] = {"state_rows_sha256": sha256_file(rows_path)}
    _write_json(output / "structpool_coverage_report.json", report)
    return report


__all__ = [
    "analyze_structpool_coverage_rows",
    "collect_structpool_coverage",
    "select_structpool_states",
    "validate_structpool_coverage_config",
]
