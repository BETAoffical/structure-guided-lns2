from __future__ import annotations

import collections
import itertools
from pathlib import Path
from typing import Any

from experiments._common import (
    producer_identity,
    ratio,
    registered_input,
    sha256_file,
    write_json,
    write_jsonl,
)
from experiments.repair_collection import _read_json, _read_jsonl, state_fingerprint
from experiments.state_analysis import StaticGridAnalysis, analyze_static_grid
from experiments.stride_closurepool_longtail import (
    _action_agents,
    _manifest_index,
    _mean,
    _pair_conflict_events,
    load_closurepool_longtail_config,
    reconstruct_trace,
)


CONFIG_SCHEMA = "lns2.stride.closurepool_temporal_forensics_config.v1"
REPORT_SCHEMA = "lns2.stride.closurepool_temporal_forensics_report.v1"
COMPARISON_SCHEMA = "lns2.stride.closurepool_temporal_comparison.v1"
PAIR_SCHEMA = "lns2.stride.closurepool_temporal_persistent_pair.v1"
EXPERIMENT_ID = "stride-closurepool-temporal-forensics-v1"

def _grid_neighbors(cell: int, static: StaticGridAnalysis) -> list[int]:
    row, col = divmod(cell, static.cols)
    result = []
    for row_delta, col_delta in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        next_row, next_col = row + row_delta, col + col_delta
        next_cell = next_row * static.cols + next_col
        if (
            0 <= next_row < static.rows
            and 0 <= next_col < static.cols
            and next_cell in static.free_cells
        ):
            result.append(next_cell)
    return result


def build_corridor_layout(static: StaticGridAnalysis) -> dict[str, Any]:
    low_degree = {
        cell for cell in static.free_cells if int(static.degrees[cell]) <= 2
    }
    components: list[set[int]] = []
    visited: set[int] = set()
    for start in sorted(low_degree):
        if start in visited:
            continue
        component = {start}
        stack = [start]
        visited.add(start)
        while stack:
            current = stack.pop()
            for neighbor in _grid_neighbors(current, static):
                if neighbor in low_degree and neighbor not in visited:
                    visited.add(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    components.sort(key=lambda cells: (min(cells), len(cells)))
    cell_to_segment: dict[int, int] = {}
    segments = []
    for identifier, cells in enumerate(components):
        for cell in cells:
            cell_to_segment[cell] = identifier
        endpoints = sorted(
            cell
            for cell in cells
            if sum(neighbor in cells for neighbor in _grid_neighbors(cell, static)) <= 1
        )
        adjacent_junctions = sorted(
            {
                neighbor
                for cell in cells
                for neighbor in _grid_neighbors(cell, static)
                if neighbor not in cells and int(static.degrees[neighbor]) >= 3
            }
        )
        segments.append(
            {
                "segment_id": identifier,
                "cell_count": len(cells),
                "minimum_cell": min(cells),
                "endpoint_cells": endpoints,
                "adjacent_junction_cells": adjacent_junctions,
            }
        )
    return {
        "rows": static.rows,
        "cols": static.cols,
        "low_degree_cell_count": len(low_degree),
        "cell_to_segment": cell_to_segment,
        "segments": segments,
    }


def segment_visits(
    state: dict[str, Any], layout: dict[str, Any]
) -> dict[int, list[dict[str, Any]]]:
    cell_to_segment = dict(layout["cell_to_segment"])
    result: dict[int, list[dict[str, Any]]] = {}
    for agent in state["agents"]:
        identifier = int(agent["id"])
        path = list(map(int, agent["path"]))
        visits: list[dict[str, Any]] = []
        start = 0
        while start < len(path):
            segment = cell_to_segment.get(path[start])
            if segment is None:
                start += 1
                continue
            end = start
            while (
                end + 1 < len(path)
                and cell_to_segment.get(path[end + 1]) == segment
            ):
                end += 1
            directed_edges = {
                (path[index - 1], path[index])
                for index in range(start + 1, end + 1)
                if path[index - 1] != path[index]
            }
            visits.append(
                {
                    "agent_id": identifier,
                    "segment_id": int(segment),
                    "start_time": start,
                    "end_time": end,
                    "duration": end - start + 1,
                    "wait_steps": sum(
                        path[index - 1] == path[index]
                        for index in range(start + 1, end + 1)
                    ),
                    "directed_edges": directed_edges,
                }
            )
            start = end + 1
        result[identifier] = visits
    return result


def temporal_dependency_edges(
    visits: dict[int, list[dict[str, Any]]]
) -> dict[tuple[int, int], dict[str, Any]]:
    by_segment: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for agent_visits in visits.values():
        for visit in agent_visits:
            by_segment[int(visit["segment_id"])].append(visit)
    edges: dict[tuple[int, int], dict[str, Any]] = {}
    for segment, members in sorted(by_segment.items()):
        for left, right in itertools.combinations(members, 2):
            left_agent = int(left["agent_id"])
            right_agent = int(right["agent_id"])
            if left_agent == right_agent:
                continue
            overlap_start = max(int(left["start_time"]), int(right["start_time"]))
            overlap_end = min(int(left["end_time"]), int(right["end_time"]))
            if overlap_start > overlap_end:
                continue
            pair = tuple(sorted((left_agent, right_agent)))
            row = edges.setdefault(
                pair,
                {
                    "segments": set(),
                    "overlap_count": 0,
                    "overlap_duration": 0,
                    "opposing_overlap_count": 0,
                },
            )
            reversed_right = {(target, source) for source, target in right["directed_edges"]}
            opposing = bool(set(left["directed_edges"]) & reversed_right)
            row["segments"].add(segment)
            row["overlap_count"] += 1
            row["overlap_duration"] += overlap_end - overlap_start + 1
            row["opposing_overlap_count"] += int(opposing)
    return edges


def _partition_edges(
    edges: dict[tuple[int, int], dict[str, Any]], selected: set[int]
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
    pairs = set(edges)
    internal = {pair for pair in pairs if pair[0] in selected and pair[1] in selected}
    boundary = {
        pair for pair in pairs if (pair[0] in selected) != (pair[1] in selected)
    }
    return internal, boundary, pairs - internal - boundary


def _components(pairs: set[tuple[int, int]]) -> list[set[int]]:
    adjacency: dict[int, set[int]] = collections.defaultdict(set)
    for left, right in pairs:
        adjacency[left].add(right)
        adjacency[right].add(left)
    result = []
    visited: set[int] = set()
    for start in sorted(adjacency):
        if start in visited:
            continue
        component = {start}
        stack = [start]
        visited.add(start)
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        result.append(component)
    return result


def temporal_action_diagnostics(
    before: dict[str, Any],
    after: dict[str, Any],
    selected: set[int],
    *,
    layout: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if layout is None:
        layout = build_corridor_layout(analyze_static_grid(before))
    known = {int(agent["id"]) for agent in before["agents"]}
    if not selected or not selected <= known:
        raise ValueError("selected neighborhood is empty or illegal")
    before_visits = segment_visits(before, layout)
    after_visits = segment_visits(after, layout)
    before_edges = temporal_dependency_edges(before_visits)
    after_edges = temporal_dependency_edges(after_visits)
    internal, boundary, external = _partition_edges(before_edges, selected)
    after_internal, after_boundary, after_external = _partition_edges(
        after_edges, selected
    )
    incident = internal | boundary
    opposing = {
        pair
        for pair, row in before_edges.items()
        if int(row["opposing_overlap_count"]) > 0
    }
    after_opposing = {
        pair
        for pair, row in after_edges.items()
        if int(row["opposing_overlap_count"]) > 0
    }
    opposing_internal = opposing & internal
    opposing_boundary = opposing & boundary
    components = _components(set(before_edges))
    touched = [component for component in components if component & selected]
    coverages = [ratio(len(component & selected), len(component)) for component in touched]
    boundary_outside = {
        right if left in selected else left for left, right in boundary
    }
    selected_visits = [
        visit for agent in selected for visit in before_visits.get(agent, ())
    ]
    selected_segments = {int(visit["segment_id"]) for visit in selected_visits}
    segment_lengths = {
        int(row["segment_id"]): int(row["cell_count"])
        for row in layout["segments"]
    }
    retained = set(before_edges) & set(after_edges)
    new = set(after_edges) - set(before_edges)
    removed = set(before_edges) - set(after_edges)
    return {
        "selected_agent_count": len(selected),
        "corridor_segment_count": len(layout["segments"]),
        "selected_corridor_segment_count": len(selected_segments),
        "selected_corridor_visit_count": len(selected_visits),
        "mean_selected_segment_length": _mean(
            segment_lengths[int(visit["segment_id"])] for visit in selected_visits
        ),
        "mean_selected_visit_duration": _mean(
            int(visit["duration"]) for visit in selected_visits
        ),
        "selected_visit_wait_ratio": ratio(
            sum(int(visit["wait_steps"]) for visit in selected_visits),
            sum(max(0, int(visit["duration"]) - 1) for visit in selected_visits),
        ),
        "dependency_edge_count": len(before_edges),
        "internal_dependency_edge_count": len(internal),
        "boundary_dependency_edge_count": len(boundary),
        "external_dependency_edge_count": len(external),
        "incident_dependency_edge_coverage": ratio(len(incident), len(before_edges)),
        "boundary_dependency_edge_ratio": ratio(len(boundary), len(incident)),
        "boundary_dependency_outside_agent_count": len(boundary_outside),
        "opposing_dependency_edge_count": len(opposing),
        "internal_opposing_dependency_edge_count": len(opposing_internal),
        "boundary_opposing_dependency_edge_count": len(opposing_boundary),
        "boundary_opposing_dependency_ratio": ratio(
            len(opposing_boundary), len(opposing_internal | opposing_boundary)
        ),
        "touched_dependency_component_count": len(touched),
        "fully_covered_dependency_component_count": sum(
            coverage == 1.0 for coverage in coverages
        ),
        "minimum_touched_dependency_component_coverage": min(
            coverages, default=0.0
        ),
        "mean_touched_dependency_component_coverage": _mean(coverages),
        "retained_dependency_edge_count": len(retained),
        "removed_dependency_edge_count": len(removed),
        "new_dependency_edge_count": len(new),
        "dependency_edge_retention_ratio": ratio(len(retained), len(before_edges)),
        "new_dependency_edge_fraction": ratio(len(new), len(after_edges)),
        "after_internal_dependency_edge_count": len(after_internal),
        "after_boundary_dependency_edge_count": len(after_boundary),
        "after_external_dependency_edge_count": len(after_external),
        "after_boundary_dependency_edge_ratio": ratio(
            len(after_boundary), len(after_internal | after_boundary)
        ),
        "retained_opposing_dependency_edge_count": len(opposing & after_opposing),
        "new_opposing_dependency_edge_count": len(after_opposing - opposing),
        "removed_opposing_dependency_edge_count": len(opposing - after_opposing),
    }


FEATURES = (
    "boundary_dependency_edge_ratio",
    "boundary_dependency_outside_agent_count",
    "boundary_opposing_dependency_ratio",
    "internal_opposing_dependency_edge_count",
    "minimum_touched_dependency_component_coverage",
    "mean_touched_dependency_component_coverage",
    "selected_visit_wait_ratio",
    "dependency_edge_retention_ratio",
    "new_dependency_edge_fraction",
    "after_boundary_dependency_edge_ratio",
    "new_opposing_dependency_edge_count",
)


def _deltas(challenger: dict[str, Any], v2: dict[str, Any]) -> dict[str, float]:
    return {name: float(challenger[name]) - float(v2[name]) for name in FEATURES}


def load_temporal_config(
    path: str | Path,
) -> tuple[Path, Path, dict[str, Any], dict[str, Path]]:
    path = Path(path).resolve()
    root = path.parent.parent.resolve()
    config = _read_json(path)
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("scientific_status")
        != "preregistered_retrospective_temporal_dependency_diagnostic"
        or config.get("experiment_id") != EXPERIMENT_ID
        or config.get("pre_registration_parent_commit") != "8b2fafc"
    ):
        raise ValueError("ClosurePool temporal diagnostic identity changed")
    if dict(config.get("corridor_definition") or {}) != {
        "cell_rule": "free_grid_degree_at_most_2",
        "segment_rule": "connected_component_in_low_degree_induced_subgraph",
        "visit_rule": "maximal_contiguous_path_run_in_one_segment",
        "temporal_overlap_rule": "closed_visit_intervals_intersect",
        "opposing_rule": "visits_traverse_at_least_one_shared_grid_edge_in_reverse_directions",
        "time_window_parameter": None,
        "outcome_dependent_parameter": False,
    }:
        raise ValueError("ClosurePool temporal corridor definition changed")
    expected_inputs = {
        "longtail_config",
        "longtail_report",
        "comparison_diagnostics",
        "persistent_pair_diagnostics",
    }
    if set(config.get("inputs") or {}) != expected_inputs:
        raise ValueError("ClosurePool temporal input registry changed")
    inputs = {
        name: registered_input(root, dict(specification), label=f"Temporal {name}")
        for name, specification in dict(config["inputs"]).items()
    }
    return path, root, config, inputs


def _trace_sources(
    longtail_config: Path,
) -> tuple[dict[str, Path], dict[str, dict[tuple[str, int], dict[str, Any]]]]:
    _path, _root, _config, inputs, roots = load_closurepool_longtail_config(
        longtail_config
    )
    manifests = {
        "historical_v2": _manifest_index(inputs["v2_manifest"]),
        "historical_slotpool": _manifest_index(inputs["slotpool_manifest"]),
        "historical_structpool": _manifest_index(inputs["structpool_manifest"]),
        "known_v2": _manifest_index(roots["known_v2"] / "realized_dynamic_manifest.jsonl"),
        "known_slotpool": _manifest_index(
            roots["known_slotpool"] / "realized_dynamic_manifest.jsonl"
        ),
        "known_structpool": _manifest_index(
            roots["known_structpool"] / "realized_dynamic_manifest.jsonl"
        ),
    }
    return roots, manifests


def _source_names(row: dict[str, Any]) -> tuple[str, str]:
    known = row["evidence_role"] == "known_regression_only"
    controller = str(row["challenger_controller"])
    suffix = "slotpool" if controller == "v2-plus-slotpool" else "structpool"
    return ("known_v2", f"known_{suffix}") if known else (
        "historical_v2",
        f"historical_{suffix}",
    )


def _location_context(
    event: dict[str, Any], static: StaticGridAnalysis, layout: dict[str, Any]
) -> dict[str, Any]:
    cell_to_segment = dict(layout["cell_to_segment"])
    cells = list(map(int, event["cells"]))
    segment_context = sorted(
        {
            segment
            for cell in cells
            for segment in (
                [cell_to_segment[cell]]
                if cell in cell_to_segment
                else [
                    cell_to_segment[neighbor]
                    for neighbor in _grid_neighbors(cell, static)
                    if neighbor in cell_to_segment
                ]
            )
        }
    )
    return {
        "time": int(event["time"]),
        "kind": str(event["kind"]),
        "cells": cells,
        "coordinates": [list(divmod(cell, static.cols)) for cell in cells],
        "cell_degrees": [int(static.degrees[cell]) for cell in cells],
        "articulation_flags": [cell in static.articulation for cell in cells],
        "corridor_segment_context": segment_context,
    }


def _contrast(rows: list[dict[str, Any]], controller: str | None) -> dict[str, Any]:
    eligible = [
        row
        for row in rows
        if row["evidence_role"] == "historical_mechanism_diagnostic"
        and (controller is None or row["challenger_controller"] == controller)
    ]
    adverse = [
        row for row in eligible if row["outcome_category"] in {"adverse", "severe_tail"}
    ]
    controls = [
        row for row in eligible if row["outcome_category"] in {"beneficial", "tied"}
    ]
    return {
        "challenger_controller": controller or "all",
        "adverse_or_severe_count": len(adverse),
        "beneficial_or_tied_count": len(controls),
        "adverse_task_count": len({row["task_id"] for row in adverse}),
        "control_task_count": len({row["task_id"] for row in controls}),
        "adverse_minus_control_mean_challenger_minus_v2": {
            name: _mean(row["challenger_minus_v2"][name] for row in adverse)
            - _mean(row["challenger_minus_v2"][name] for row in controls)
            for name in FEATURES
        },
        "individual_historical_rows": [
            {
                "comparison_id": row["comparison_id"],
                "outcome_category": row["outcome_category"],
                "repair_iteration_delta": row["repair_iteration_delta"],
                "challenger_minus_v2": row["challenger_minus_v2"],
            }
            for row in eligible
        ],
    }


def analyze_closurepool_temporal(
    config_path: str | Path,
    output: str | Path,
    *,
    producer: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path, root, config, inputs = load_temporal_config(config_path)
    output = Path(output).resolve()
    parent_report = _read_json(inputs["longtail_report"])
    if parent_report.get("integrity_passed") is not True:
        raise ValueError("parent long-tail report integrity failed")
    all_comparisons = _read_jsonl(inputs["comparison_diagnostics"])
    comparisons = [
        row
        for row in all_comparisons
        if (
            row["evidence_role"] == "historical_mechanism_diagnostic"
            and row["group_id"] == config["cohort"]["historical_group_id"]
        )
        or row["evidence_role"] == "known_regression_only"
    ]
    historical = [
        row for row in comparisons if row["evidence_role"] == "historical_mechanism_diagnostic"
    ]
    known = [row for row in comparisons if row["evidence_role"] == "known_regression_only"]
    cohort = dict(config["cohort"])
    if (
        len(historical) != int(cohort["historical_comparison_count"])
        or len({(row["task_id"], row["solver_seed"]) for row in historical})
        != int(cohort["historical_paired_key_count"])
        or len({row["task_id"] for row in historical})
        != int(cohort["historical_task_count"])
        or sum(row["outcome_category"] in {"adverse", "severe_tail"} for row in historical)
        != int(cohort["historical_adverse_or_severe_count"])
        or sum(row["outcome_category"] in {"beneficial", "tied"} for row in historical)
        != int(cohort["historical_beneficial_or_tied_count"])
        or len(known) != int(cohort["known_regression_comparison_count"])
    ):
        raise ValueError("ClosurePool temporal cohort changed")

    roots, manifests = _trace_sources(inputs["longtail_config"])
    trace_cache: dict[tuple[str, str, int], dict[str, Any]] = {}

    def trace(source: str, task_id: str, solver_seed: int) -> dict[str, Any]:
        cache_key = (source, task_id, solver_seed)
        if cache_key not in trace_cache:
            manifest = manifests[source].get((task_id, solver_seed))
            if manifest is None:
                raise ValueError(f"missing temporal trace source: {cache_key}")
            trace_cache[cache_key] = reconstruct_trace(roots[source], manifest)
        return trace_cache[cache_key]

    result_rows = []
    context_cache: dict[str, tuple[StaticGridAnalysis, dict[str, Any], dict, dict]] = {}
    for source_row in comparisons:
        task_id = str(source_row["task_id"])
        solver_seed = int(source_row["solver_seed"])
        v2_source, challenger_source = _source_names(source_row)
        v2_trace = trace(v2_source, task_id, solver_seed)
        challenger_trace = trace(challenger_source, task_id, solver_seed)
        if (
            v2_trace["trace_sha256"] != source_row["source_trace_sha256"]["v2"]
            or challenger_trace["trace_sha256"]
            != source_row["source_trace_sha256"]["challenger"]
        ):
            raise ValueError("temporal source trace identity changed")
        index = int(source_row["divergence_index"])
        v2_before = v2_trace["states"][index]
        challenger_before = challenger_trace["states"][index]
        if (
            state_fingerprint(v2_before) != source_row["divergence_state_fingerprint"]
            or state_fingerprint(challenger_before) != source_row["divergence_state_fingerprint"]
        ):
            raise ValueError("temporal divergence state changed")
        known_agents = {int(agent["id"]) for agent in v2_before["agents"]}
        v2_selected = _action_agents(v2_trace["transitions"][index], known_agents)
        challenger_selected = _action_agents(
            challenger_trace["transitions"][index], known_agents
        )
        if sorted(v2_selected) != source_row["v2_selected_agents"] or sorted(
            challenger_selected
        ) != source_row["challenger_selected_agents"]:
            raise ValueError("temporal realized action changed")
        static = analyze_static_grid(v2_before)
        layout = build_corridor_layout(static)
        v2_metrics = temporal_action_diagnostics(
            v2_before,
            v2_trace["states"][index + 1],
            v2_selected,
            layout=layout,
        )
        challenger_metrics = temporal_action_diagnostics(
            challenger_before,
            challenger_trace["states"][index + 1],
            challenger_selected,
            layout=layout,
        )
        row = {
            "schema": COMPARISON_SCHEMA,
            "comparison_id": source_row["comparison_id"],
            "evidence_role": source_row["evidence_role"],
            "challenger_controller": source_row["challenger_controller"],
            "group_id": source_row["group_id"],
            "task_id": task_id,
            "solver_seed": solver_seed,
            "outcome_category": source_row["outcome_category"],
            "repair_iteration_delta": int(source_row["repair_iteration_delta"]),
            "divergence_index": index,
            "divergence_state_fingerprint": source_row["divergence_state_fingerprint"],
            "v2_action": v2_metrics,
            "challenger_action": challenger_metrics,
            "challenger_minus_v2": _deltas(challenger_metrics, v2_metrics),
            "corridor_layout": {
                "low_degree_cell_count": layout["low_degree_cell_count"],
                "segment_count": len(layout["segments"]),
                "maximum_segment_cell_count": max(
                    (row["cell_count"] for row in layout["segments"]), default=0
                ),
            },
            "source_trace_sha256": source_row["source_trace_sha256"],
        }
        result_rows.append(row)
        context_cache[str(source_row["comparison_id"])] = (
            static,
            layout,
            v2_trace,
            challenger_trace,
        )

    persistent_source = _read_jsonl(inputs["persistent_pair_diagnostics"])
    eligible_ids = {str(row["comparison_id"]) for row in comparisons}
    pair_rows = []
    for source_pair in persistent_source:
        comparison_id = str(source_pair["comparison_id"])
        if comparison_id not in eligible_ids or int(source_pair["state_presence_count"]) < 3:
            continue
        static, layout, v2_trace, challenger_trace = context_cache[comparison_id]
        comparison = next(row for row in comparisons if row["comparison_id"] == comparison_id)
        chosen_trace = v2_trace if source_pair["trajectory_role"] == "v2" else challenger_trace
        absolute_index = int(comparison["divergence_index"]) + int(
            source_pair["last_relative_state_index"]
        )
        state = chosen_trace["states"][absolute_index]
        pair = (int(source_pair["left_agent"]), int(source_pair["right_agent"]))
        events = _pair_conflict_events(state, pair)
        pair_rows.append(
            {
                "schema": PAIR_SCHEMA,
                "comparison_id": comparison_id,
                "evidence_role": source_pair["evidence_role"],
                "challenger_controller": source_pair["controller"],
                "trajectory_role": source_pair["trajectory_role"],
                "outcome_category": source_pair["outcome_category"],
                "task_id": source_pair["task_id"],
                "solver_seed": int(source_pair["solver_seed"]),
                "left_agent": pair[0],
                "right_agent": pair[1],
                "state_presence_count": int(source_pair["state_presence_count"]),
                "maximum_consecutive_state_count": int(
                    source_pair["maximum_consecutive_state_count"]
                ),
                "new_after_divergence": bool(source_pair["new_after_divergence"]),
                "both_selected_at_divergence": bool(
                    source_pair["both_selected_at_divergence"]
                ),
                "crosses_divergence_neighborhood": bool(
                    source_pair["crosses_divergence_neighborhood"]
                ),
                "last_state_index": absolute_index,
                "event_locations": [
                    _location_context(event, static, layout) for event in events
                ],
            }
        )

    if producer is None:
        producer = producer_identity(
            project_root=root,
            source_files=(
                "experiments/stride_closurepool_temporal.py",
                "experiments/stride_closurepool_longtail.py",
                "experiments/state_analysis.py",
            ),
            native_required=False,
        )
    comparison_path = output / "temporal_comparison_diagnostics.jsonl"
    pair_path = output / "temporal_persistent_pair_locations.jsonl"
    write_jsonl(comparison_path, result_rows)
    write_jsonl(pair_path, pair_rows)
    checks = {
        "historical_maze_comparison_count": len(historical)
        == int(cohort["historical_comparison_count"]),
        "historical_maze_key_count": len(
            {(row["task_id"], row["solver_seed"]) for row in historical}
        )
        == int(cohort["historical_paired_key_count"]),
        "known_regression_comparison_count": len(known)
        == int(cohort["known_regression_comparison_count"]),
        "all_divergence_states_and_actions_verified": len(result_rows)
        == len(comparisons),
        "known_regression_excluded_from_contrasts": True,
        "no_time_window_or_outcome_parameter": config["corridor_definition"][
            "time_window_parameter"
        ]
        is None
        and config["corridor_definition"]["outcome_dependent_parameter"] is False,
        "no_candidate_rule_model_or_solver_run": True,
    }
    report = {
        "schema": REPORT_SCHEMA,
        "scientific_status": "completed_retrospective_temporal_dependency_diagnostic_no_rule_freeze",
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": sha256_file(path),
        "producer_identity": producer,
        "inputs": {
            name: {"path": str(value), "sha256": sha256_file(value)}
            for name, value in inputs.items()
        },
        "historical_comparison_count": len(historical),
        "known_regression_comparison_count": len(known),
        "persistent_pair_location_count": len(pair_rows),
        "contrasts": [
            _contrast(result_rows, None),
            _contrast(result_rows, "v2-plus-slotpool"),
            _contrast(result_rows, "v2-plus-structpool"),
        ],
        "checks": checks,
        "integrity_passed": all(checks.values()),
        "claim_boundary": config["claim_boundary"],
        "decision": "human_cross_case_review_required_no_rule_freeze",
        "artifacts": {
            "temporal_comparison_diagnostics": comparison_path.name,
            "temporal_comparison_diagnostics_sha256": sha256_file(comparison_path),
            "temporal_persistent_pair_locations": pair_path.name,
            "temporal_persistent_pair_locations_sha256": sha256_file(pair_path),
        },
    }
    write_json(output / "closurepool_temporal_forensics_report.json", report)
    return report
