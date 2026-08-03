from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from experiments.stride_topology_anchor_coverage import _base_config
from experiments.stride_topology_coverage import (
    _collect_topology_coverage,
    _mean,
    analyze_topology_coverage_rows,
    validate_topology_coverage_config,
)
from lns2_selector.runtime.topology_candidates import (
    generate_topology_boundary_candidates,
    merge_topology_anchor_candidates,
    topology_candidate_audit,
)


CONFIG_SCHEMA = "lns2.stride.topology_boundary_coverage_config.v1"
REPORT_SCHEMA = "lns2.stride.topology_boundary_coverage_report.v1"


def validate_topology_boundary_coverage_config(config: dict[str, Any]) -> None:
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unexpected topology-boundary coverage config")
    validate_topology_coverage_config(_base_config(config))
    if (
        config.get("diagnostic_id") != "stride-topoboundary-coverage-v1"
        or config.get("candidate_generator_id") != "stride-topoboundary-v1"
        or config.get("predecessor_id") != "stride-topoanchor-quality-pilot-v1"
    ):
        raise ValueError("topology-boundary diagnostic identity changed")
    if dict(config.get("augmentation") or {}) != {
        "strategy": "one_endpoint_topology_core_then_global_boundary_fill",
        "kinds": ["articulation", "low_degree"],
        "neighborhood_size": 16,
        "topology_core_budget": 4,
        "maximum_added_candidates_per_state": 2,
        "maximum_total_candidates_per_state": 20,
        "selection_objective": "new_incident_minus_internal_closure_then_component_diversity",
    }:
        raise ValueError("topology-boundary augmentation protocol changed")
    if dict(config.get("boundary_gates") or {}) != {
        "minimum_mean_incident_articulation_coverage": 0.70,
        "minimum_mean_incident_low_degree_coverage": 0.70,
        "minimum_mean_boundary_articulation_ratio": 0.65,
        "minimum_mean_boundary_low_degree_ratio": 0.65,
        "minimum_ultra_mean_boundary_articulation_ratio": 0.65,
        "minimum_ultra_mean_boundary_low_degree_ratio": 0.65,
        "minimum_mean_global_event_incident_delta_vs_base_frontier": -0.02,
        "minimum_mean_global_boundary_ratio_delta_vs_base_frontier": 0.0,
        "minimum_mean_component_reach_delta_vs_base_frontier": -0.05,
    }:
        raise ValueError("topology-boundary gates changed")
    if set(config.get("inputs") or {}) != {
        "preflight_report", "dataset_manifest", "qualification_manifest",
        "qualification_report", "source_config", "runtime_config",
        "anchor_quality_report", "anchor_failure_report",
    }:
        raise ValueError("topology-boundary input registry changed")


def _is_family(row: dict[str, Any], prefix: str) -> bool:
    return any(str(family).startswith(prefix) for family in row["selection_families"])


def _best(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        rows,
        key=lambda row: (
            float(row["proposal_audit"]["global_event_incident_coverage"]),
            float(row["proposal_audit"]["global_event_boundary_ratio"]),
            float(row["proposal_audit"]["conflict_component_reach"]),
            str(row["candidate_id"]),
        ),
    )


def analyze_topology_boundary_coverage_rows(
    config: dict[str, Any], state_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    base_report = analyze_topology_coverage_rows(_base_config(config), state_rows)
    by_state: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        by_state[str(row["state_id"])].append(row)
    details = []
    art_incident: list[float] = []
    low_incident: list[float] = []
    art_boundary: list[float] = []
    low_boundary: list[float] = []
    ultra_art_boundary: list[float] = []
    ultra_low_boundary: list[float] = []
    global_incident_deltas: list[float] = []
    global_boundary_deltas: list[float] = []
    component_deltas: list[float] = []
    all_boundary_rows = []
    for state in sorted(state_rows, key=lambda row: str(row["state_id"])):
        state_id = str(state["state_id"])
        rows = by_state[state_id]
        base = [row for row in rows if any(_is_family(row, prefix) for prefix in ("target:", "collision:", "random:"))]
        boundary = [row for row in rows if _is_family(row, "topology-boundary-")]
        if not base:
            raise ValueError(f"topology-boundary state lacks a pool partition: {state_id}")
        if any("proposal_audit" not in row for row in rows):
            raise ValueError(f"topology-boundary state lacks proposal audit: {state_id}")
        if not boundary:
            if bool(state["articulation_relevant"]) or bool(state["low_degree_relevant"]):
                raise ValueError(f"topology-boundary relevant state lacks a candidate: {state_id}")
            details.append({
                "state_id": state_id,
                "layout_family": state["layout_family"],
                "candidate_count": state["candidate_count"],
                "added_candidate_count": state["added_candidate_count"],
                "status": "no_relevant_topology_events",
                "topology_kinds": {},
            })
            continue
        all_boundary_rows.extend(boundary)
        best_base = _best(base)
        best_boundary = _best(boundary)
        base_audit = best_base["proposal_audit"]
        boundary_audit = best_boundary["proposal_audit"]
        global_incident_deltas.append(float(boundary_audit["global_event_incident_coverage"]) - float(base_audit["global_event_incident_coverage"]))
        global_boundary_deltas.append(float(boundary_audit["global_event_boundary_ratio"]) - float(base_audit["global_event_boundary_ratio"]))
        component_deltas.append(float(boundary_audit["conflict_component_reach"]) - float(base_audit["conflict_component_reach"]))
        kinds = {}
        for kind in ("articulation", "low_degree"):
            candidates = [row for row in boundary if _is_family(row, f"topology-boundary-{kind}:")]
            if candidates:
                chosen = _best(candidates)
                features = chosen["features"]
                incident = float(features[f"topology.realized.incident_{kind}_event_coverage"])
                ratio = float(features[f"topology.realized.boundary_{kind}_event_ratio"])
                kinds[kind] = {"candidate_id": chosen["candidate_id"], "incident_coverage": incident, "boundary_ratio": ratio}
                if bool(state[f"{kind}_relevant"]):
                    (art_incident if kind == "articulation" else low_incident).append(incident)
                    (art_boundary if kind == "articulation" else low_boundary).append(ratio)
                    if state["layout_family"] == "dao_ultra_bottleneck":
                        (ultra_art_boundary if kind == "articulation" else ultra_low_boundary).append(ratio)
        details.append({
            "state_id": state_id,
            "layout_family": state["layout_family"],
            "candidate_count": state["candidate_count"],
            "added_candidate_count": state["added_candidate_count"],
            "best_base_candidate_id": best_base["candidate_id"],
            "best_boundary_candidate_id": best_boundary["candidate_id"],
            "global_event_incident_delta_vs_base_frontier": global_incident_deltas[-1],
            "global_boundary_ratio_delta_vs_base_frontier": global_boundary_deltas[-1],
            "component_reach_delta_vs_base_frontier": component_deltas[-1],
            "topology_kinds": kinds,
        })
    summary = {
        "mean_incident_articulation_coverage": _mean(art_incident),
        "mean_incident_low_degree_coverage": _mean(low_incident),
        "mean_boundary_articulation_ratio": _mean(art_boundary),
        "mean_boundary_low_degree_ratio": _mean(low_boundary),
        "ultra_mean_boundary_articulation_ratio": _mean(ultra_art_boundary),
        "ultra_mean_boundary_low_degree_ratio": _mean(ultra_low_boundary),
        "mean_global_event_incident_delta_vs_base_frontier": _mean(global_incident_deltas),
        "mean_global_boundary_ratio_delta_vs_base_frontier": _mean(global_boundary_deltas),
        "mean_component_reach_delta_vs_base_frontier": _mean(component_deltas),
    }
    thresholds = config["boundary_gates"]
    gates = {
        key: bool(value)
        for key, value in base_report["gates"].items()
        if key not in {
            "minimum_mean_max_incident_articulation_coverage",
            "minimum_articulation_state_fraction_at_half_coverage",
            "minimum_mean_max_incident_low_degree_coverage",
            "minimum_low_degree_state_fraction_at_half_coverage",
        }
    }
    gates.update({
        "maximum_candidate_count_per_state": max(int(row["candidate_count"]) for row in state_rows) <= 20,
        "maximum_added_candidates_per_state": max(int(row["added_candidate_count"]) for row in state_rows) <= 2,
        "all_boundary_candidates_size_16": bool(all_boundary_rows) and all(int(row["actual_size"]) == 16 for row in all_boundary_rows),
        **{
            gate: summary[name] >= float(thresholds[gate])
            for gate, name in {
                "minimum_mean_incident_articulation_coverage": "mean_incident_articulation_coverage",
                "minimum_mean_incident_low_degree_coverage": "mean_incident_low_degree_coverage",
                "minimum_mean_boundary_articulation_ratio": "mean_boundary_articulation_ratio",
                "minimum_mean_boundary_low_degree_ratio": "mean_boundary_low_degree_ratio",
                "minimum_ultra_mean_boundary_articulation_ratio": "ultra_mean_boundary_articulation_ratio",
                "minimum_ultra_mean_boundary_low_degree_ratio": "ultra_mean_boundary_low_degree_ratio",
                "minimum_mean_global_event_incident_delta_vs_base_frontier": "mean_global_event_incident_delta_vs_base_frontier",
                "minimum_mean_global_boundary_ratio_delta_vs_base_frontier": "mean_global_boundary_ratio_delta_vs_base_frontier",
                "minimum_mean_component_reach_delta_vs_base_frontier": "mean_component_reach_delta_vs_base_frontier",
            }.items()
        },
    })
    passed = all(gates.values())
    return {
        "schema": REPORT_SCHEMA,
        "scientific_status": "proposal_only_post_failure_design_diagnostic",
        "candidate_repair_trials_executed": False,
        "controller_actions_executed": False,
        "formal_speed_claim": False,
        "state_count": len(state_rows),
        "candidate_count": sum(int(row["candidate_count"]) for row in state_rows),
        "added_candidate_count": sum(int(row["added_candidate_count"]) for row in state_rows),
        "summary": summary,
        "states": details,
        "gates": gates,
        "passed": passed,
        "next_decision": config["next_decision_on_pass" if passed else "next_decision_on_failure"],
    }


def _augment(state: dict[str, Any], analysis: Any, candidates: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    audited_base = [{**row, "proposal_audit": topology_candidate_audit(analysis, row["agents"])} for row in candidates]
    boundary = generate_topology_boundary_candidates(
        state, analysis,
        neighborhood_size=int(config["augmentation"]["neighborhood_size"]),
        core_budget=int(config["augmentation"]["topology_core_budget"]),
    )
    return merge_topology_anchor_candidates(audited_base, boundary)


def collect_topology_boundary_coverage(config_path: str | Path, output: str | Path) -> dict[str, Any]:
    return _collect_topology_coverage(
        config_path, output,
        validator=validate_topology_boundary_coverage_config,
        analyzer=analyze_topology_boundary_coverage_rows,
        augmenter=_augment,
        analyzer_uses_candidate_rows=True,
    )


__all__ = [
    "analyze_topology_boundary_coverage_rows",
    "collect_topology_boundary_coverage",
    "validate_topology_boundary_coverage_config",
]
