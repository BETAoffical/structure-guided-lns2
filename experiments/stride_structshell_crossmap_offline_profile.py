from __future__ import annotations

import collections
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

from experiments._common import sha256_file, write_json
from experiments.repair_collection import state_fingerprint
from experiments.state_analysis import StaticGridAnalysis, analyze_state, analyze_static_grid
from experiments.stride_closurepool_longtail import reconstruct_trace
from lns2_selector.runtime.topology_candidates import (
    generate_structpool_candidate_subset,
)


REPORT_SCHEMA = "lns2.stride.structshell_crossmap_offline_profile_report.v1"
EXPERIMENT_ID = "stride-structshell-crossmap-offline-profile-v1"
SOURCE_EXPERIMENT_ID = "stride-structshell-crossmap-quick-screen-v1"
SOURCE_REPORT_SCHEMA = "lns2.stride.structshell_crossmap_quick_screen_report.v1"
REPORT_FILENAME = "offline_profile_report.json"
SOURCE_CONTROLLER = "v2_only"
MAP_IDS = (
    "den020d",
    "maze-32-32-4",
    "random-32-32-20-high-load",
    "room-64-64-16",
)
CHECKPOINT_NAMES = ("initial", "middle", "last_pre_action")
CHECKPOINT_ROLES = {
    "initial": "shared_initial_outcome_blind_profile",
    "middle": "v2_outcome_conditioned_unlabelled_retrospective_profile",
    "last_pre_action": "v2_outcome_conditioned_unlabelled_retrospective_profile",
}
PROFILE_IMPLEMENTATION_FILES = (
    "experiments/stride_structshell_crossmap_offline_profile.py",
    "experiments/state_analysis.py",
    "lns2_selector/runtime/topology_candidates.py",
    "experiments/stride_closurepool_longtail.py",
)
FAMILY_NAMES = {
    "component16": "structpool-conflict-component:16",
    "hotspot16": "structpool-spatiotemporal-hotspot:16",
}
FAMILY_SIZES = {
    "conflict_component": (16,),
    "spatiotemporal_hotspot": (16,),
}


def _loaded_native_modules() -> list[str]:
    return sorted(
        name
        for name in sys.modules
        if name == "lns2_env" or name.startswith("lns2_selector.solver")
    )


def _require_source_only_process() -> None:
    loaded = _loaded_native_modules()
    if loaded:
        raise RuntimeError(
            f"offline profile process loaded forbidden native modules: {loaded}"
        )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"JSONL row must be an object: {path}")
    return rows


def checkpoint_indices(decision_count: int) -> tuple[int, int, int]:
    """Return three deterministic pre-action checkpoints for one source trace."""

    count = int(decision_count)
    if count < 3:
        raise ValueError("offline profile requires at least three source decisions")
    indices = (0, count // 2, count - 1)
    if len(set(indices)) != 3:
        raise RuntimeError("offline profile checkpoint selection collapsed")
    return indices


def _concentration(values: Iterable[int]) -> dict[str, float | int]:
    counts = [int(value) for value in values if int(value) > 0]
    total = sum(counts)
    return {
        "support_count": len(counts),
        "peak_share": max(counts, default=0) / total if total else 0.0,
        "hhi": sum(value * value for value in counts) / (total * total) if total else 0.0,
    }


def _candidate_summary(
    candidate: Mapping[str, Any] | None, *, family: str
) -> dict[str, Any] | None:
    if candidate is None:
        return None
    return {
        "candidate_id": str(candidate["candidate_id"]),
        "agents": list(map(int, candidate["agents"])),
        "actual_size": int(candidate["actual_size"]),
        "proposal_audit": {
            str(key): float(value)
            for key, value in dict(candidate["proposal_audit"]).items()
        },
        "structpool_score": float(candidate["structpool_score"]),
        "support_count": int(
            dict(candidate["structpool_support_count_by_family"])[family]
        ),
        "support_ratio": float(
            dict(candidate["structpool_support_ratio_by_family"])[family]
        ),
    }


def profile_state(
    state: dict[str, Any],
    *,
    map_id: str,
    checkpoint: str,
    decision_index: int,
    static_grid: StaticGridAnalysis,
) -> dict[str, Any]:
    """Describe one recorded pre-action state without inspecting its outcome."""

    analysis = analyze_state(state, static_grid=static_grid)
    reported_pairs = {
        tuple(sorted(map(int, edge))) for edge in state.get("conflict_edges", ())
    }
    if reported_pairs != analysis.pair_set or int(
        state.get("num_of_colliding_pairs", -1)
    ) != len(analysis.pair_set):
        raise ValueError("offline profile conflict reconstruction changed")

    agent_count = len(state["agents"])
    active_agents = {agent for pair in analysis.pair_set for agent in pair}
    component_sizes = sorted(
        (len(members) for members in analysis.component_members.values()), reverse=True
    )
    time_counts = collections.Counter(event.time for event in analysis.events)
    cell_counts = collections.Counter(
        cell for event in analysis.events for cell in event.cells
    )
    low_degree = {
        cell for cell in analysis.free_cells if int(analysis.degrees.get(cell, 0)) <= 2
    }
    event_count = len(analysis.events)
    articulation_events = sum(
        any(cell in analysis.articulation for cell in event.cells)
        for event in analysis.events
    )
    low_degree_events = sum(
        any(cell in low_degree for cell in event.cells) for event in analysis.events
    )

    candidates = generate_structpool_candidate_subset(
        state,
        analysis,
        family_sizes=FAMILY_SIZES,
    )
    by_family: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        for family in map(str, candidate["selection_families"]):
            if family in FAMILY_NAMES.values():
                by_family[family] = candidate
    family_candidates = {
        short: by_family.get(family) for short, family in FAMILY_NAMES.items()
    }
    component_agents = set(
        map(int, (family_candidates["component16"] or {}).get("agents", ()))
    )
    hotspot_agents = set(
        map(int, (family_candidates["hotspot16"] or {}).get("agents", ()))
    )
    union = component_agents | hotspot_agents

    denominator = agent_count * (agent_count - 1)
    return {
        "map_id": str(map_id),
        "checkpoint": str(checkpoint),
        "profile_role": CHECKPOINT_ROLES[str(checkpoint)],
        "counterfactual_family_label_allowed": False,
        "decision_index": int(decision_index),
        "state_fingerprint": state_fingerprint(state),
        "profile": {
            "agent_count": agent_count,
            "conflict_pair_count": len(analysis.pair_set),
            "conflict_pair_density": (
                2.0 * len(analysis.pair_set) / denominator if denominator else 0.0
            ),
            "conflict_event_count": event_count,
            "events_per_pair": event_count / max(1, len(analysis.pair_set)),
            "active_conflict_agent_count": len(active_agents),
            "active_conflict_agent_ratio": len(active_agents) / max(1, agent_count),
            "conflict_component_count": len(component_sizes),
            "largest_conflict_component_size": max(component_sizes, default=0),
            "largest_conflict_component_ratio": max(component_sizes, default=0)
            / max(1, agent_count),
            "component_size_concentration": _concentration(component_sizes),
            "event_time_concentration": _concentration(time_counts.values()),
            "event_cell_incidence_concentration": _concentration(
                cell_counts.values()
            ),
            "path_visit_concentration": _concentration(
                analysis.visit_heat.values()
            ),
            "path_agent_concentration": _concentration(
                analysis.agent_heat.values()
            ),
            "articulation_event_ratio": articulation_events / max(1, event_count),
            "low_degree_event_ratio": low_degree_events / max(1, event_count),
            "articulation_cell_count": len(analysis.articulation),
            "low_degree_free_cell_ratio": len(low_degree)
            / max(1, len(analysis.free_cells)),
        },
        "fixed16_candidates": {
            short: _candidate_summary(candidate, family=FAMILY_NAMES[short])
            for short, candidate in family_candidates.items()
        },
        "candidate_comparison": {
            "both_present": bool(component_agents and hotspot_agents),
            "exact_same_agent_set": bool(component_agents)
            and component_agents == hotspot_agents,
            "jaccard": len(component_agents & hotspot_agents) / len(union)
            if union
            else None,
        },
    }


def _source_outcome(row: Mapping[str, Any]) -> dict[str, Any]:
    component = dict(row.get("component16") or {})
    hotspot = dict(row.get("hotspot16") or {})
    component_ttf = float(component["restricted_ttf"])
    hotspot_ttf = float(hotspot["restricted_ttf"])
    if component_ttf < hotspot_ttf:
        winner = "component16"
    elif hotspot_ttf < component_ttf:
        winner = "hotspot16"
    else:
        winner = "tie"
    return {
        "component16": {
            "success": bool(component["success"]),
            "restricted_ttf": component_ttf,
        },
        "hotspot16": {
            "success": bool(hotspot["success"]),
            "restricted_ttf": hotspot_ttf,
        },
        "component_minus_hotspot_restricted_ttf_seconds": component_ttf
        - hotspot_ttf,
        "point_winner": winner,
        "label_status": "retrospective_map_level_diagnostic_only",
    }


def source_count_evidence(maps: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Gate only an additional independent collection, never Router training."""

    informative_winners = collections.Counter(
        str(row["source_outcome"]["point_winner"])
        for row in maps
        if bool(row["informative_for_family_crossover_hypothesis"])
    )
    component_maps = int(informative_winners["component16"])
    hotspot_maps = int(informative_winners["hotspot16"])
    source_count_gate_met = component_maps >= 2 and hotspot_maps >= 2
    return {
        "informative_map_count": component_maps + hotspot_maps,
        "component16_informative_point_win_map_count": component_maps,
        "hotspot16_informative_point_win_map_count": hotspot_maps,
        "minimum_independent_maps_required_per_family": 2,
        "source_count_gate_met": source_count_gate_met,
        "additional_collection_allowed": source_count_gate_met,
        "router_training_allowed": False,
        "decision": (
            "additional_independent_collection_allowed"
            if source_count_gate_met
            else "stop_additional_collection_insufficient_independent_map_sources"
        ),
        "scope": (
            "initial profiles support a feature-separability hypothesis only; "
            "middle and last profiles are V2-outcome-conditioned, unlabelled, and "
            "must not be treated as counterfactual family labels"
        ),
    }


def analyze(source: str | Path, output: str | Path) -> dict[str, Any]:
    """Audit fixed16 profile differences using only frozen source artifacts."""

    _require_source_only_process()
    source_root = Path(source).resolve()
    output_root = Path(output).resolve()
    try:
        output_root.relative_to(source_root)
    except ValueError:
        pass
    else:
        raise ValueError("offline profile output must not modify its source tree")

    source_report_path = source_root / "quick_screen_report.json"
    source_report = _read_json(source_report_path)
    if (
        source_report.get("schema") != SOURCE_REPORT_SCHEMA
        or source_report.get("experiment_id") != SOURCE_EXPERIMENT_ID
        or source_report.get("integrity_passed") is not True
        or int(source_report.get("map_count", -1)) != len(MAP_IDS)
        or set(map(str, dict(source_report.get("per_map") or {}))) != set(MAP_IDS)
    ):
        raise ValueError("cross-map quick-screen source identity changed")

    profiles: list[dict[str, Any]] = []
    maps: list[dict[str, Any]] = []
    trace_hashes: dict[str, str] = {}
    manifest_hashes: dict[str, str] = {}
    for map_id in MAP_IDS:
        collection = source_root / "maps" / map_id / SOURCE_CONTROLLER
        manifest_path = collection / "realized_dynamic_manifest.jsonl"
        manifests = _read_jsonl(manifest_path)
        if len(manifests) != 1:
            raise ValueError(f"{map_id} must have exactly one V2 source episode")
        manifest = manifests[0]
        summary = dict(manifest.get("summary") or {})
        if (
            manifest.get("status") not in {"ok", "resumed"}
            or str(manifest.get("policy")) != "realized_dynamic"
            or str(summary.get("controller_mode")) != "v2-full"
            or int(summary.get("repair_iterations", -1)) < 3
        ):
            raise ValueError(f"{map_id} V2 source episode changed")

        reconstructed = reconstruct_trace(collection, manifest)
        transitions = list(reconstructed["transitions"])
        states = list(reconstructed["states"])
        if len(states) != len(transitions) + 1:
            raise ValueError(f"{map_id} reconstructed trace length changed")
        pre_action_states = states[:-1]
        indices = checkpoint_indices(len(pre_action_states))
        static_grid = analyze_static_grid(pre_action_states[0])
        map_profiles = []
        for checkpoint, index in zip(CHECKPOINT_NAMES, indices):
            transition = transitions[index]
            if int(transition.get("decision_index", -1)) != index:
                raise ValueError(f"{map_id} decision indices changed")
            state = pre_action_states[index]
            fingerprint = state_fingerprint(state)
            if fingerprint != str(transition.get("before_fingerprint")):
                raise ValueError(f"{map_id} pre-action fingerprint changed")
            row = profile_state(
                state,
                map_id=map_id,
                checkpoint=checkpoint,
                decision_index=index,
                static_grid=static_grid,
            )
            map_profiles.append(row)
            profiles.append(row)

        initial_fingerprint = str(summary.get("initial_fingerprint") or "")
        if map_profiles[0]["state_fingerprint"] != initial_fingerprint:
            raise ValueError(f"{map_id} initial state identity changed")
        outcome = _source_outcome(dict(source_report["per_map"])[map_id])
        initial_same = bool(
            map_profiles[0]["candidate_comparison"]["exact_same_agent_set"]
        )
        both_failed = not outcome["component16"]["success"] and not outcome[
            "hotspot16"
        ]["success"]
        informative = (
            not initial_same
            and not both_failed
            and outcome["point_winner"] in FAMILY_NAMES
        )
        maps.append(
            {
                "map_id": map_id,
                "source_task_id": str(manifest.get("task_id")),
                "source_solver_seed": int(manifest.get("solver_seed", -1)),
                "source_decision_count": len(transitions),
                "checkpoint_indices": list(indices),
                "source_outcome": outcome,
                "initial_candidates_exact_same": initial_same,
                "informative_for_family_crossover_hypothesis": informative,
            }
        )
        trace_hashes[map_id] = str(manifest["trace_sha256"])
        manifest_hashes[map_id] = sha256_file(manifest_path)

    source_count = source_count_evidence(maps)
    project_root = Path(__file__).resolve().parents[1]
    implementation_sha256 = {
        relative: sha256_file(project_root / relative)
        for relative in PROFILE_IMPLEMENTATION_FILES
    }
    source_report_sha256 = sha256_file(source_report_path)
    _require_source_only_process()
    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "scientific_status": "source_only_retrospective_mechanism_diagnostic",
        "contract": {
            "source_controller": SOURCE_CONTROLLER,
            "map_count": len(MAP_IDS),
            "checkpoints_per_map": len(CHECKPOINT_NAMES),
            "checkpoint_names": list(CHECKPOINT_NAMES),
            "checkpoint_roles": dict(CHECKPOINT_ROLES),
            "profile_state_count": len(profiles),
            "fixed_nominal_size": 16,
            "families": list(FAMILY_NAMES),
            "native_solver_imported_or_invoked": False,
            "pp_imported_or_invoked": False,
            "controller_invoked": False,
            "new_reset_or_episode_invoked": False,
            "model_training_invoked": False,
            "auc_computed": False,
            "counterfactual_family_labels_created": False,
            "loaded_native_modules": [],
        },
        "source_integrity": {
            "source_experiment_id": SOURCE_EXPERIMENT_ID,
            "source_report_sha256": source_report_sha256,
            "v2_manifest_sha256_by_map": manifest_hashes,
            "v2_trace_sha256_by_map": trace_hashes,
            "all_v2_profile_trace_and_state_fingerprints_verified": True,
            "component_hotspot_outcomes": {
                "source": "quick_screen_report.json",
                "bound_by_source_report_sha256": source_report_sha256,
                "challenger_traces_read_or_reverified": False,
            },
            "profile_implementation_sha256": implementation_sha256,
        },
        "maps": maps,
        "profiles": profiles,
        "source_count_evidence": source_count,
        "router_training_allowed": False,
        "default_replacement_allowed": False,
        "formal_speed_claim": False,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / REPORT_FILENAME, report)
    return report


__all__ = [
    "CHECKPOINT_NAMES",
    "EXPERIMENT_ID",
    "FAMILY_NAMES",
    "MAP_IDS",
    "REPORT_FILENAME",
    "REPORT_SCHEMA",
    "analyze",
    "checkpoint_indices",
    "profile_state",
    "source_count_evidence",
]
