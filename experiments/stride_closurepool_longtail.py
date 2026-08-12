from __future__ import annotations

import collections
import math
import statistics
from pathlib import Path
from typing import Any, Iterable

from experiments._common import (
    _native_filesystem_path,
    producer_identity,
    ratio,
    registered_input,
    sha256_file,
    write_json,
    write_jsonl,
)
from experiments.closed_loop_trace_storage import (
    EPISODE_SCHEMA_V2,
    apply_extras_delta,
    apply_state_delta,
    read_state_blob,
    read_trace_events,
    resolve_state_blob,
)
from experiments.repair_collection import _read_json, _read_jsonl, state_fingerprint
from experiments.state_analysis import (
    StaticGridAnalysis,
    analyze_state,
    analyze_static_grid,
)


CONFIG_SCHEMA = "lns2.stride.closurepool_longtail_forensics_config.v1"
REPORT_SCHEMA = "lns2.stride.closurepool_longtail_forensics_report.v1"
COMPARISON_SCHEMA = "lns2.stride.closurepool_longtail_comparison.v1"
PAIR_SCHEMA = "lns2.stride.closurepool_longtail_pair.v1"
AGENT_SCHEMA = "lns2.stride.closurepool_longtail_agent.v1"
EXPERIMENT_ID = "stride-closurepool-longtail-forensics-v1"
CHALLENGERS = ("v2-plus-structpool", "v2-plus-slotpool")


def _mean(values: Iterable[float | int | bool]) -> float:
    numbers = [float(value) for value in values]
    return statistics.fmean(numbers) if numbers else 0.0


def _maximum_consecutive(indices: Iterable[int]) -> int:
    ordered = sorted(set(map(int, indices)))
    maximum = 0
    current = 0
    previous: int | None = None
    for index in ordered:
        current = current + 1 if previous is not None and index == previous + 1 else 1
        maximum = max(maximum, current)
        previous = index
    return maximum


def _contained_directory(root: Path, value: Any, *, field: str) -> Path:
    text = str(value or "")
    relative = Path(text)
    if not text or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{field} must be a contained relative directory")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{field} escapes the project root") from error
    if not _native_filesystem_path(resolved).is_dir():
        raise FileNotFoundError(f"{field} does not exist: {value}")
    return resolved


def load_closurepool_longtail_config(
    path: str | Path,
) -> tuple[Path, Path, dict[str, Any], dict[str, Path], dict[str, Path]]:
    path = Path(path).resolve()
    root = path.parent.parent.resolve()
    config = _read_json(path)
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("scientific_status")
        != "retrospective_mechanism_diagnostic_before_closurepool_design"
        or config.get("experiment_id") != EXPERIMENT_ID
        or config.get("pre_registration_parent_commit")
        != "70761ac41106b91e1eeff32bcc6f442e1b09c1e5"
    ):
        raise ValueError("ClosurePool long-tail diagnostic identity changed")
    if dict(config.get("cohort") or {}) != {
        "historical_comparison_count": 58,
        "historical_paired_key_count": 29,
        "known_regression_comparison_count": 2,
        "known_regression_key_count": 1,
        "known_regression_used_for_rule_selection": False,
        "all_historical_comparisons_retained": True,
    }:
        raise ValueError("ClosurePool long-tail diagnostic cohort changed")
    if dict(config.get("tail_definition") or {}) != {
        "role": "descriptive_outcome_grouping_only",
        "adverse_iteration_delta_minimum": 1,
        "severe_iteration_delta_minimum": 10,
        "severe_iteration_ratio_minimum": 2.0,
        "persistent_pair_minimum_state_count": 3,
        "threshold_selection_or_model_label_use_allowed": False,
    }:
        raise ValueError("ClosurePool long-tail descriptive grouping changed")
    if dict(config.get("claim_boundary") or {}) != {
        "read_only_existing_trace_analysis": True,
        "candidate_rule_freeze_allowed": False,
        "model_training_allowed": False,
        "runtime_integration_allowed": False,
        "formal_ttf_claim": False,
        "known_regression_is_generalization_evidence": False,
        "fresh_solver_outcomes_collected": False,
        "human_cross_case_review_required": True,
    }:
        raise ValueError("ClosurePool long-tail claim boundary changed")
    expected_inputs = {
        "known_regression_report",
        "historical_divergence_report",
        "v2_manifest",
        "slotpool_manifest",
        "structpool_manifest",
    }
    if set(config.get("inputs") or {}) != expected_inputs:
        raise ValueError("ClosurePool long-tail input registry changed")
    inputs = {
        name: registered_input(root, dict(specification), label=f"ClosurePool {name}")
        for name, specification in dict(config["inputs"]).items()
    }
    expected_roots = {
        "known_v2",
        "known_slotpool",
        "known_structpool",
        "historical_v2",
        "historical_slotpool",
        "historical_structpool",
    }
    if set(config.get("collection_roots") or {}) != expected_roots:
        raise ValueError("ClosurePool long-tail collection roots changed")
    roots = {
        name: _contained_directory(root, value, field=f"ClosurePool {name}")
        for name, value in dict(config["collection_roots"]).items()
    }
    return path, root, config, inputs, roots


def _manifest_index(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for row in _read_jsonl(path):
        key = (str(row["task_id"]), int(row["solver_seed"]))
        if key in result:
            raise ValueError(f"duplicate manifest key: {key}")
        if row.get("status") != "ok" or row.get("error") is not None:
            raise ValueError(f"non-ok source manifest row: {key}")
        result[key] = row
    return result


def _pair_set(state: dict[str, Any]) -> set[tuple[int, int]]:
    result = {
        tuple(sorted((int(edge[0]), int(edge[1]))))
        for edge in state.get("conflict_edges", ())
    }
    if len(result) != len(state.get("conflict_edges", ())):
        raise ValueError("state conflict edges contain duplicates")
    if int(state.get("num_of_colliding_pairs", -1)) != len(result):
        raise ValueError("state conflict count disagrees with conflict edges")
    return result


def _initial_state(
    collection_root: Path, trace_path: Path, event: dict[str, Any]
) -> dict[str, Any]:
    if str(event.get("schema")) != EPISODE_SCHEMA_V2:
        state = event.get("state")
        if not isinstance(state, dict):
            raise ValueError("trace initial event has no state")
        return dict(state)
    reference = str(event.get("state_blob") or "")
    blob = resolve_state_blob(trace_path, reference, collection_root)
    state = read_state_blob(_native_filesystem_path(blob))
    extras = event.get("state_extras")
    if not isinstance(extras, dict):
        raise ValueError("trace initial event has invalid state extras")
    state.update(extras)
    return state


def reconstruct_trace(
    collection_root: Path, manifest: dict[str, Any]
) -> dict[str, Any]:
    ordinary_trace = (collection_root / str(manifest["trace_file"])).resolve()
    try:
        ordinary_trace.relative_to(collection_root.resolve())
    except ValueError as error:
        raise ValueError("trace path escapes collection root") from error
    trace_path = _native_filesystem_path(ordinary_trace)
    if sha256_file(trace_path) != str(manifest["trace_sha256"]):
        raise ValueError(f"trace hash changed: {ordinary_trace}")
    events = read_trace_events(trace_path)
    if not events or events[0].get("event") != "initial":
        raise ValueError("trace has no initial event")
    state = _initial_state(collection_root, ordinary_trace, events[0])
    if state_fingerprint(state) != str(events[0].get("state_fingerprint")):
        raise ValueError("trace initial state fingerprint changed")
    states = [state]
    transitions: list[dict[str, Any]] = []
    finish: dict[str, Any] | None = None
    for event in events[1:]:
        kind = str(event.get("event"))
        if kind == "finish":
            finish = event
            continue
        if kind != "transition":
            raise ValueError(f"unexpected trace event: {kind}")
        if state_fingerprint(state) != str(event.get("before_fingerprint")):
            raise ValueError("trace before fingerprint changed")
        if str(event.get("schema")) == EPISODE_SCHEMA_V2:
            after = apply_state_delta(state, dict(event["state_delta"]))
            after.update(apply_extras_delta(state, dict(event["state_extras_delta"])))
        else:
            after = dict(event["after"])
        if state_fingerprint(after) != str(event.get("after_fingerprint")):
            raise ValueError("trace after fingerprint changed")
        transitions.append(event)
        states.append(after)
        state = after
    if finish is None:
        raise ValueError("trace has no finish event")
    summary = dict(manifest.get("summary") or finish.get("summary") or {})
    if int(summary.get("repair_iterations", -1)) != len(transitions):
        raise ValueError("trace repair count changed")
    trajectory = list(map(int, summary.get("conflict_trajectory") or ()))
    observed = [int(row["num_of_colliding_pairs"]) for row in states]
    if trajectory != observed:
        raise ValueError("trace conflict trajectory changed")
    return {
        "trace_path": ordinary_trace,
        "trace_sha256": str(manifest["trace_sha256"]),
        "initial_event": events[0],
        "states": states,
        "transitions": transitions,
        "summary": summary,
    }


def _action_agents(event: dict[str, Any], known_agents: set[int]) -> set[int]:
    metrics = dict(event.get("metrics") or {})
    raw = metrics.get("neighborhood")
    if not isinstance(raw, list) or not raw:
        raw = dict(event.get("action") or {}).get("agents")
    if not isinstance(raw, list) or not raw:
        raise ValueError("transition has no realized neighborhood")
    agents = set(map(int, raw))
    if len(agents) != len(raw) or not agents <= known_agents:
        raise ValueError("transition neighborhood is duplicate or illegal")
    return agents


def _components(pair_set: set[tuple[int, int]]) -> list[set[int]]:
    adjacency: dict[int, set[int]] = collections.defaultdict(set)
    for left, right in pair_set:
        adjacency[left].add(right)
        adjacency[right].add(left)
    result: list[set[int]] = []
    visited: set[int] = set()
    for start in sorted(adjacency):
        if start in visited:
            continue
        members = {start}
        stack = [start]
        visited.add(start)
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    members.add(neighbor)
                    stack.append(neighbor)
        result.append(members)
    return result


def _partition_pairs(
    pairs: set[tuple[int, int]], selected: set[int]
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
    internal = {pair for pair in pairs if pair[0] in selected and pair[1] in selected}
    boundary = {
        pair for pair in pairs if (pair[0] in selected) != (pair[1] in selected)
    }
    external = pairs - internal - boundary
    return internal, boundary, external


def _wait_ratio(path: list[int]) -> float:
    if len(path) < 2:
        return 0.0
    return ratio(sum(left == right for left, right in zip(path, path[1:])), len(path) - 1)


def action_diagnostics(
    before: dict[str, Any],
    after: dict[str, Any],
    selected: set[int],
    *,
    static_grid: StaticGridAnalysis | None = None,
) -> dict[str, Any]:
    static_grid = static_grid or analyze_static_grid(before)
    before_analysis = analyze_state(before, static_grid=static_grid)
    after_analysis = analyze_state(after, static_grid=static_grid)
    agents = {int(agent["id"]): agent for agent in before["agents"]}
    if not selected or not selected <= set(agents):
        raise ValueError("selected neighborhood is empty or references unknown agents")
    before_pairs = set(before_analysis.pair_set)
    after_pairs = set(after_analysis.pair_set)
    internal, boundary, external = _partition_pairs(before_pairs, selected)
    after_internal, after_boundary, after_external = _partition_pairs(
        after_pairs, selected
    )
    active = {agent for pair in before_pairs for agent in pair}
    boundary_outside = {
        right if left in selected else left for left, right in boundary
    }
    components = _components(before_pairs)
    touched = [component for component in components if component & selected]
    coverages = [ratio(len(component & selected), len(component)) for component in touched]
    largest = max(components, key=lambda component: (len(component), -min(component)), default=set())

    event_internal = 0
    event_boundary = 0
    bottleneck_events = 0
    bottleneck_boundary_events = 0
    for event in before_analysis.events:
        is_internal = event.left in selected and event.right in selected
        is_boundary = (event.left in selected) != (event.right in selected)
        event_internal += int(is_internal)
        event_boundary += int(is_boundary)
        bottleneck = any(
            cell in static_grid.articulation
            or int(static_grid.degrees.get(cell, 0)) <= 2
            for cell in event.cells
        )
        bottleneck_events += int(bottleneck)
        bottleneck_boundary_events += int(bottleneck and is_boundary)

    selected_paths = [list(map(int, agents[identifier]["path"])) for identifier in sorted(selected)]
    selected_path_cells = {cell for path in selected_paths for cell in path}
    bottleneck_cells = {
        cell
        for cell in selected_path_cells
        if cell in static_grid.articulation
        or int(static_grid.degrees.get(cell, 0)) <= 2
    }
    unselected_active = active - selected
    outside_queue = {
        identifier
        for identifier in unselected_active
        if set(map(int, agents[identifier]["path"])) & bottleneck_cells
    }
    path_steps = [cell for path in selected_paths for cell in path]
    retained = before_pairs & after_pairs
    new_pairs = after_pairs - before_pairs
    removed = before_pairs - after_pairs
    after_repeated_excess = max(0, len(after_analysis.events) - len(after_pairs))
    return {
        "selected_agent_count": len(selected),
        "before_conflict_pair_count": len(before_pairs),
        "after_conflict_pair_count": len(after_pairs),
        "conflict_pair_reduction": len(before_pairs) - len(after_pairs),
        "selected_active_agent_coverage": ratio(len(selected & active), len(active)),
        "internal_conflict_pair_count": len(internal),
        "boundary_conflict_pair_count": len(boundary),
        "external_conflict_pair_count": len(external),
        "incident_conflict_pair_coverage": ratio(len(internal | boundary), len(before_pairs)),
        "boundary_conflict_pair_ratio": ratio(len(boundary), len(internal | boundary)),
        "boundary_outside_agent_count": len(boundary_outside),
        "touched_component_count": len(touched),
        "fully_covered_component_count": sum(coverage == 1.0 for coverage in coverages),
        "partially_covered_component_count": sum(coverage < 1.0 for coverage in coverages),
        "mean_touched_component_coverage": _mean(coverages),
        "minimum_touched_component_coverage": min(coverages, default=0.0),
        "largest_component_coverage": ratio(len(largest & selected), len(largest)),
        "conflict_event_count": len(before_analysis.events),
        "internal_conflict_event_count": event_internal,
        "boundary_conflict_event_count": event_boundary,
        "boundary_conflict_event_ratio": ratio(
            event_boundary, event_internal + event_boundary
        ),
        "bottleneck_conflict_event_ratio": ratio(
            bottleneck_events, len(before_analysis.events)
        ),
        "bottleneck_boundary_event_ratio": ratio(
            bottleneck_boundary_events, bottleneck_events
        ),
        "selected_unique_path_cell_count": len(selected_path_cells),
        "selected_path_low_degree_ratio": ratio(
            sum(int(static_grid.degrees.get(cell, 0)) <= 2 for cell in path_steps),
            len(path_steps),
        ),
        "selected_path_articulation_ratio": ratio(
            sum(cell in static_grid.articulation for cell in path_steps), len(path_steps)
        ),
        "selected_path_wait_ratio": _mean(_wait_ratio(path) for path in selected_paths),
        "unselected_bottleneck_queue_count": len(outside_queue),
        "unselected_bottleneck_queue_ratio": ratio(len(outside_queue), len(unselected_active)),
        "retained_conflict_pair_count": len(retained),
        "removed_conflict_pair_count": len(removed),
        "new_conflict_pair_count": len(new_pairs),
        "conflict_pair_retention_ratio": ratio(len(retained), len(before_pairs)),
        "new_conflict_pair_fraction": ratio(len(new_pairs), len(after_pairs)),
        "after_internal_conflict_pair_count": len(after_internal),
        "after_boundary_conflict_pair_count": len(after_boundary),
        "after_external_conflict_pair_count": len(after_external),
        "after_boundary_conflict_pair_ratio": ratio(
            len(after_boundary), len(after_internal | after_boundary)
        ),
        "after_conflict_event_count": len(after_analysis.events),
        "after_repeated_conflict_event_excess": after_repeated_excess,
        "after_repeated_event_excess_per_pair": ratio(
            after_repeated_excess, len(after_pairs)
        ),
    }


def _pair_conflict_events(
    state: dict[str, Any], pair: tuple[int, int]
) -> list[dict[str, Any]]:
    agents = {int(agent["id"]): agent for agent in state["agents"]}
    left_path = list(map(int, agents[pair[0]]["path"]))
    right_path = list(map(int, agents[pair[1]]["path"]))
    horizon = max(len(left_path), len(right_path))
    result: list[dict[str, Any]] = []
    for time in range(horizon):
        left = left_path[min(time, len(left_path) - 1)]
        right = right_path[min(time, len(right_path) - 1)]
        if left == right:
            result.append({"time": time, "kind": "vertex", "cells": [left]})
        if time == 0:
            continue
        left_previous = left_path[min(time - 1, len(left_path) - 1)]
        right_previous = right_path[min(time - 1, len(right_path) - 1)]
        if left_previous == right and right_previous == left and left != right:
            result.append(
                {
                    "time": time,
                    "kind": "edge",
                    "cells": sorted((left, right)),
                }
            )
    return result


def trajectory_diagnostics(
    states: list[dict[str, Any]],
    selected: set[int],
    *,
    start_index: int,
    persistent_pair_minimum: int,
    static_grid: StaticGridAnalysis | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if start_index < 0 or start_index >= len(states):
        raise ValueError("trajectory start index is out of range")
    static_grid = static_grid or analyze_static_grid(states[0])
    suffix = states[start_index:]
    pair_sets = [_pair_set(state) for state in suffix]
    conflicts = [len(pairs) for pairs in pair_sets]
    pair_indices: dict[tuple[int, int], list[int]] = collections.defaultdict(list)
    for index, pairs in enumerate(pair_sets):
        for pair in pairs:
            pair_indices[pair].append(index)
    pair_rows: list[dict[str, Any]] = []
    for pair in sorted(pair_indices):
        indices = pair_indices[pair]
        maximum_consecutive = _maximum_consecutive(indices)
        event_summary: dict[str, Any] | None = None
        if len(indices) >= persistent_pair_minimum:
            first_state = suffix[indices[0]]
            last_state = suffix[indices[-1]]
            first_events = _pair_conflict_events(first_state, pair)
            last_events = _pair_conflict_events(last_state, pair)
            cells = [cell for event in last_events for cell in event["cells"]]
            event_summary = {
                "first_state_event_count": len(first_events),
                "last_state_event_count": len(last_events),
                "last_state_vertex_event_count": sum(
                    event["kind"] == "vertex" for event in last_events
                ),
                "last_state_edge_event_count": sum(
                    event["kind"] == "edge" for event in last_events
                ),
                "last_state_bottleneck_event_fraction": ratio(
                    sum(
                        any(
                            cell in static_grid.articulation
                            or int(static_grid.degrees.get(cell, 0)) <= 2
                            for cell in event["cells"]
                        )
                        for event in last_events
                    ),
                    len(last_events),
                ),
                "last_state_distinct_conflict_cell_count": len(set(cells)),
                "last_state_first_conflict_time": (
                    min(event["time"] for event in last_events) if last_events else None
                ),
            }
        pair_rows.append(
            {
                "schema": PAIR_SCHEMA,
                "left_agent": pair[0],
                "right_agent": pair[1],
                "state_presence_count": len(indices),
                "maximum_consecutive_state_count": maximum_consecutive,
                "first_relative_state_index": min(indices),
                "last_relative_state_index": max(indices),
                "present_at_divergence": 0 in indices,
                "present_at_terminal": len(suffix) - 1 in indices,
                "new_after_divergence": 0 not in indices,
                "both_selected_at_divergence": pair[0] in selected and pair[1] in selected,
                "crosses_divergence_neighborhood": (pair[0] in selected)
                != (pair[1] in selected),
                "event_summary": event_summary,
            }
        )
    agent_ids = [int(agent["id"]) for agent in suffix[0]["agents"]]
    agent_presence: dict[int, list[int]] = collections.defaultdict(list)
    partners: dict[int, set[int]] = collections.defaultdict(set)
    for index, pairs in enumerate(pair_sets):
        active = {agent for pair in pairs for agent in pair}
        for agent in active:
            agent_presence[agent].append(index)
        for left, right in pairs:
            partners[left].add(right)
            partners[right].add(left)
    path_changes: collections.Counter[int] = collections.Counter()
    previous_paths = {
        int(agent["id"]): tuple(map(int, agent["path"])) for agent in suffix[0]["agents"]
    }
    for state in suffix[1:]:
        for agent in state["agents"]:
            identifier = int(agent["id"])
            path = tuple(map(int, agent["path"]))
            path_changes[identifier] += int(path != previous_paths[identifier])
            previous_paths[identifier] = path
    agent_rows = [
        {
            "schema": AGENT_SCHEMA,
            "agent_id": identifier,
            "selected_at_divergence": identifier in selected,
            "conflict_state_count": len(agent_presence.get(identifier, ())),
            "maximum_consecutive_conflict_state_count": _maximum_consecutive(
                agent_presence.get(identifier, ())
            ),
            "distinct_conflict_partner_count": len(partners.get(identifier, ())),
            "path_change_count": int(path_changes[identifier]),
            "conflicting_at_divergence": 0 in agent_presence.get(identifier, ()),
            "conflicting_at_terminal": len(suffix) - 1
            in agent_presence.get(identifier, ()),
        }
        for identifier in sorted(agent_ids)
        if agent_presence.get(identifier) or identifier in selected
    ]
    churn = []
    for before, after in zip(pair_sets, pair_sets[1:]):
        churn.append(
            {
                "retained": len(before & after),
                "removed": len(before - after),
                "new": len(after - before),
            }
        )
    no_progress_streak = 0
    maximum_no_progress_streak = 0
    single_pair_streak = 0
    maximum_single_pair_streak = 0
    for left, right in zip(conflicts, conflicts[1:]):
        no_progress_streak = 0 if right < left else no_progress_streak + 1
        maximum_no_progress_streak = max(
            maximum_no_progress_streak, no_progress_streak
        )
    for value in conflicts:
        single_pair_streak = single_pair_streak + 1 if value == 1 else 0
        maximum_single_pair_streak = max(
            maximum_single_pair_streak, single_pair_streak
        )
    persistent = [
        row for row in pair_rows if row["state_presence_count"] >= persistent_pair_minimum
    ]
    metrics = {
        "observed_state_count": len(suffix),
        "observed_transition_count": max(0, len(suffix) - 1),
        "initial_conflicts": conflicts[0],
        "terminal_conflicts": conflicts[-1],
        "maximum_no_progress_streak": maximum_no_progress_streak,
        "maximum_single_conflict_plateau": maximum_single_pair_streak,
        "distinct_conflict_pair_count": len(pair_rows),
        "persistent_conflict_pair_count": len(persistent),
        "new_persistent_conflict_pair_count": sum(
            bool(row["new_after_divergence"]) for row in persistent
        ),
        "maximum_pair_presence_state_count": max(
            (int(row["state_presence_count"]) for row in pair_rows), default=0
        ),
        "maximum_pair_consecutive_state_count": max(
            (int(row["maximum_consecutive_state_count"]) for row in pair_rows),
            default=0,
        ),
        "mean_pair_retention_count": _mean(row["retained"] for row in churn),
        "mean_new_pair_count": _mean(row["new"] for row in churn),
        "conflict_agent_count": sum(bool(agent_presence.get(agent)) for agent in agent_ids),
        "maximum_agent_conflict_state_count": max(
            (len(indices) for indices in agent_presence.values()), default=0
        ),
    }
    return metrics, pair_rows, agent_rows


def _outcome_category(
    challenger_iterations: int,
    v2_iterations: int,
    definition: dict[str, Any],
) -> str:
    delta = int(challenger_iterations) - int(v2_iterations)
    iteration_ratio = (
        float(challenger_iterations) / float(v2_iterations)
        if v2_iterations > 0
        else (1.0 if challenger_iterations == 0 else math.inf)
    )
    if (
        delta >= int(definition["severe_iteration_delta_minimum"])
        and iteration_ratio >= float(definition["severe_iteration_ratio_minimum"])
    ):
        return "severe_tail"
    if delta >= int(definition["adverse_iteration_delta_minimum"]):
        return "adverse"
    if delta < 0:
        return "beneficial"
    return "tied"


DIAGNOSTIC_FEATURES = (
    "boundary_conflict_pair_ratio",
    "boundary_conflict_event_ratio",
    "minimum_touched_component_coverage",
    "largest_component_coverage",
    "unselected_bottleneck_queue_ratio",
    "conflict_pair_retention_ratio",
    "new_conflict_pair_fraction",
    "after_boundary_conflict_pair_ratio",
    "after_repeated_event_excess_per_pair",
)


TRAJECTORY_FEATURES = (
    "maximum_no_progress_streak",
    "maximum_single_conflict_plateau",
    "persistent_conflict_pair_count",
    "new_persistent_conflict_pair_count",
    "maximum_pair_consecutive_state_count",
    "maximum_agent_conflict_state_count",
)


def _feature_deltas(
    challenger_action: dict[str, Any],
    v2_action: dict[str, Any],
    challenger_trajectory: dict[str, Any],
    v2_trajectory: dict[str, Any],
) -> dict[str, float]:
    result = {
        f"action.{name}": float(challenger_action[name]) - float(v2_action[name])
        for name in DIAGNOSTIC_FEATURES
    }
    result.update(
        {
            f"trajectory.{name}": float(challenger_trajectory[name])
            - float(v2_trajectory[name])
            for name in TRAJECTORY_FEATURES
        }
    )
    return result


def _decorate_rows(
    rows: list[dict[str, Any]],
    *,
    comparison_id: str,
    evidence_role: str,
    controller: str,
    trajectory_role: str,
    group_id: str,
    task_id: str,
    solver_seed: int,
    outcome_category: str,
) -> list[dict[str, Any]]:
    return [
        {
            **row,
            "comparison_id": comparison_id,
            "evidence_role": evidence_role,
            "controller": controller,
            "trajectory_role": trajectory_role,
            "group_id": group_id,
            "task_id": task_id,
            "solver_seed": solver_seed,
            "outcome_category": outcome_category,
        }
        for row in rows
    ]


def _comparison(
    *,
    comparison_id: str,
    evidence_role: str,
    group_id: str,
    task_id: str,
    solver_seed: int,
    controller: str,
    divergence_index: int,
    v2_trace: dict[str, Any],
    challenger_trace: dict[str, Any],
    definition: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    v2_states = list(v2_trace["states"])
    challenger_states = list(challenger_trace["states"])
    v2_transitions = list(v2_trace["transitions"])
    challenger_transitions = list(challenger_trace["transitions"])
    if divergence_index >= len(v2_transitions) or divergence_index >= len(
        challenger_transitions
    ):
        raise ValueError("divergence index exceeds a source trace")
    v2_before = v2_states[divergence_index]
    challenger_before = challenger_states[divergence_index]
    if state_fingerprint(v2_before) != state_fingerprint(challenger_before):
        raise ValueError("divergence states are not identical")
    v2_event = v2_transitions[divergence_index]
    challenger_event = challenger_transitions[divergence_index]
    v2_pp_seed = int(dict(v2_event.get("action") or {}).get("pp_random_seed", -1))
    challenger_pp_seed = int(
        dict(challenger_event.get("action") or {}).get("pp_random_seed", -1)
    )
    if v2_pp_seed < 0 or v2_pp_seed != challenger_pp_seed:
        raise ValueError("divergence PP seeds are not strictly paired")
    known_agents = {int(agent["id"]) for agent in v2_before["agents"]}
    v2_selected = _action_agents(v2_event, known_agents)
    challenger_selected = _action_agents(challenger_event, known_agents)
    if v2_selected == challenger_selected:
        raise ValueError("registered divergence has identical realized neighborhoods")
    static = analyze_static_grid(v2_before)
    v2_action = action_diagnostics(
        v2_before,
        v2_states[divergence_index + 1],
        v2_selected,
        static_grid=static,
    )
    challenger_action = action_diagnostics(
        challenger_before,
        challenger_states[divergence_index + 1],
        challenger_selected,
        static_grid=static,
    )
    persistent_minimum = int(definition["persistent_pair_minimum_state_count"])
    v2_trajectory, v2_pairs, v2_agents = trajectory_diagnostics(
        v2_states,
        v2_selected,
        start_index=divergence_index,
        persistent_pair_minimum=persistent_minimum,
        static_grid=static,
    )
    challenger_trajectory, challenger_pairs, challenger_agents = trajectory_diagnostics(
        challenger_states,
        challenger_selected,
        start_index=divergence_index,
        persistent_pair_minimum=persistent_minimum,
        static_grid=static,
    )
    v2_iterations = int(v2_trace["summary"]["repair_iterations"])
    challenger_iterations = int(challenger_trace["summary"]["repair_iterations"])
    outcome = _outcome_category(challenger_iterations, v2_iterations, definition)
    row = {
        "schema": COMPARISON_SCHEMA,
        "comparison_id": comparison_id,
        "evidence_role": evidence_role,
        "group_id": group_id,
        "task_id": task_id,
        "solver_seed": solver_seed,
        "challenger_controller": controller,
        "divergence_index": divergence_index,
        "initial_state_fingerprint": state_fingerprint(v2_states[0]),
        "divergence_state_fingerprint": state_fingerprint(v2_before),
        "paired_pp_seed": v2_pp_seed,
        "v2_repair_iterations": v2_iterations,
        "challenger_repair_iterations": challenger_iterations,
        "repair_iteration_delta": challenger_iterations - v2_iterations,
        "repair_iteration_ratio": (
            ratio(challenger_iterations, v2_iterations)
            if v2_iterations
            else (1.0 if challenger_iterations == 0 else None)
        ),
        "outcome_category": outcome,
        "v2_selected_agents": sorted(v2_selected),
        "challenger_selected_agents": sorted(challenger_selected),
        "selected_neighborhood_jaccard": ratio(
            len(v2_selected & challenger_selected), len(v2_selected | challenger_selected)
        ),
        "v2_action": v2_action,
        "challenger_action": challenger_action,
        "v2_trajectory": v2_trajectory,
        "challenger_trajectory": challenger_trajectory,
        "challenger_minus_v2": _feature_deltas(
            challenger_action, v2_action, challenger_trajectory, v2_trajectory
        ),
        "source_trace_sha256": {
            "v2": str(v2_trace["trace_sha256"]),
            "challenger": str(challenger_trace["trace_sha256"]),
        },
    }
    pair_rows = _decorate_rows(
        v2_pairs,
        comparison_id=comparison_id,
        evidence_role=evidence_role,
        controller=controller,
        trajectory_role="v2",
        group_id=group_id,
        task_id=task_id,
        solver_seed=solver_seed,
        outcome_category=outcome,
    ) + _decorate_rows(
        challenger_pairs,
        comparison_id=comparison_id,
        evidence_role=evidence_role,
        controller=controller,
        trajectory_role="challenger",
        group_id=group_id,
        task_id=task_id,
        solver_seed=solver_seed,
        outcome_category=outcome,
    )
    agent_rows = _decorate_rows(
        v2_agents,
        comparison_id=comparison_id,
        evidence_role=evidence_role,
        controller=controller,
        trajectory_role="v2",
        group_id=group_id,
        task_id=task_id,
        solver_seed=solver_seed,
        outcome_category=outcome,
    ) + _decorate_rows(
        challenger_agents,
        comparison_id=comparison_id,
        evidence_role=evidence_role,
        controller=controller,
        trajectory_role="challenger",
        group_id=group_id,
        task_id=task_id,
        solver_seed=solver_seed,
        outcome_category=outcome,
    )
    return row, pair_rows, agent_rows


def _group_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["evidence_role"]),
                str(row["challenger_controller"]),
                str(row["outcome_category"]),
            )
        ].append(row)
    result = []
    for (role, controller, category), members in sorted(grouped.items()):
        feature_names = sorted(members[0]["challenger_minus_v2"])
        result.append(
            {
                "evidence_role": role,
                "challenger_controller": controller,
                "outcome_category": category,
                "comparison_count": len(members),
                "map_group_count": len({str(row["group_id"]) for row in members}),
                "task_count": len({str(row["task_id"]) for row in members}),
                "mean_repair_iteration_delta": _mean(
                    row["repair_iteration_delta"] for row in members
                ),
                "mean_challenger_minus_v2": {
                    name: _mean(row["challenger_minus_v2"][name] for row in members)
                    for name in feature_names
                },
            }
        )
    return result


def _mechanism_contrasts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for controller in CHALLENGERS:
        eligible = [
            row
            for row in rows
            if row["evidence_role"] == "historical_mechanism_diagnostic"
            and row["challenger_controller"] == controller
        ]
        adverse = [
            row
            for row in eligible
            if row["outcome_category"] in {"adverse", "severe_tail"}
        ]
        controls = [
            row
            for row in eligible
            if row["outcome_category"] in {"beneficial", "tied"}
        ]
        features = sorted(eligible[0]["challenger_minus_v2"]) if eligible else []
        result.append(
            {
                "challenger_controller": controller,
                "adverse_count": len(adverse),
                "control_count": len(controls),
                "adverse_map_group_count": len(
                    {str(row["group_id"]) for row in adverse}
                ),
                "control_map_group_count": len(
                    {str(row["group_id"]) for row in controls}
                ),
                "adverse_minus_control_mean_feature_delta": {
                    name: _mean(
                        row["challenger_minus_v2"][name] for row in adverse
                    )
                    - _mean(row["challenger_minus_v2"][name] for row in controls)
                    for name in features
                },
                "interpretation_boundary": (
                    "descriptive retrospective contrast; no feature or threshold is frozen"
                ),
            }
        )
    return result


def analyze_closurepool_longtail(
    config_path: str | Path,
    output: str | Path,
    *,
    producer: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path, root, config, inputs, roots = load_closurepool_longtail_config(config_path)
    output = Path(output).resolve()
    historical_report = _read_json(inputs["historical_divergence_report"])
    known_report = _read_json(inputs["known_regression_report"])
    if int(historical_report.get("comparison_count", -1)) != 58:
        raise ValueError("historical divergence comparison count changed")
    if historical_report.get("integrity_passed") is not True:
        raise ValueError("historical divergence source integrity failed")
    if known_report.get("integrity_passed") is not True:
        raise ValueError("known regression source integrity failed")

    manifests = {
        "historical_v2": _manifest_index(inputs["v2_manifest"]),
        "historical_slotpool": _manifest_index(inputs["slotpool_manifest"]),
        "historical_structpool": _manifest_index(inputs["structpool_manifest"]),
    }
    for name in ("known_v2", "known_slotpool", "known_structpool"):
        manifests[name] = _manifest_index(
            roots[name] / "realized_dynamic_manifest.jsonl"
        )
    trace_cache: dict[tuple[str, str, int], dict[str, Any]] = {}

    def trace(source: str, key: tuple[str, int]) -> dict[str, Any]:
        cache_key = (source, key[0], key[1])
        if cache_key not in trace_cache:
            row = manifests[source].get(key)
            if row is None:
                raise ValueError(f"source manifest is missing key {source}: {key}")
            trace_cache[cache_key] = reconstruct_trace(roots[source], row)
        return trace_cache[cache_key]

    comparisons: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    agent_rows: list[dict[str, Any]] = []
    no_divergence_count = 0
    historical_keys: set[tuple[str, int]] = set()
    definition = dict(config["tail_definition"])
    for controller in CHALLENGERS:
        report_rows = list(dict(historical_report["comparisons"])[controller])
        if len(report_rows) != 29:
            raise ValueError(f"historical {controller} comparison coverage changed")
        source = (
            "historical_slotpool"
            if controller == "v2-plus-slotpool"
            else "historical_structpool"
        )
        for report_row in report_rows:
            key = (str(report_row["task_id"]), int(report_row["solver_seed"]))
            historical_keys.add(key)
            divergence = report_row.get("divergence")
            if divergence is None:
                no_divergence_count += 1
                continue
            comparison_id = (
                f"historical::{controller}::{key[0]}::seed-{key[1]:04d}"
            )
            row, pairs, agents = _comparison(
                comparison_id=comparison_id,
                evidence_role="historical_mechanism_diagnostic",
                group_id=str(report_row["group_id"]),
                task_id=key[0],
                solver_seed=key[1],
                controller=controller,
                divergence_index=int(divergence["decision_index"]),
                v2_trace=trace("historical_v2", key),
                challenger_trace=trace(source, key),
                definition=definition,
            )
            if row["source_trace_sha256"]["v2"] != str(
                report_row["v2_trace_sha256"]
            ) or row["source_trace_sha256"]["challenger"] != str(
                report_row["challenger_trace_sha256"]
            ):
                raise ValueError("historical comparison trace identity changed")
            comparisons.append(row)
            pair_rows.extend(pairs)
            agent_rows.extend(agents)

    known_controller_results = dict(known_report["controller_results"])
    for controller, source in (
        ("v2-plus-structpool", "known_structpool"),
        ("v2-plus-slotpool", "known_slotpool"),
    ):
        known_rows = manifests[source]
        if len(known_rows) != 1 or len(manifests["known_v2"]) != 1:
            raise ValueError("known regression manifest coverage changed")
        key = next(iter(known_rows))
        if key != next(iter(manifests["known_v2"])):
            raise ValueError("known regression controller keys differ")
        comparison_id = f"known-regression::{controller}::{key[0]}::seed-{key[1]:04d}"
        row, pairs, agents = _comparison(
            comparison_id=comparison_id,
            evidence_role="known_regression_only",
            group_id="known-maze",
            task_id=key[0],
            solver_seed=key[1],
            controller=controller,
            divergence_index=int(known_report["first_divergence_vs_v2"][controller]),
            v2_trace=trace("known_v2", key),
            challenger_trace=trace(source, key),
            definition=definition,
        )
        expected = known_controller_results[controller]
        v2_expected = known_controller_results["v2-full"]
        if (
            row["challenger_repair_iterations"] != int(expected["repair_iterations"])
            or row["v2_repair_iterations"] != int(v2_expected["repair_iterations"])
            or row["source_trace_sha256"]["challenger"]
            != str(expected["trace_sha256"])
            or row["source_trace_sha256"]["v2"] != str(v2_expected["trace_sha256"])
        ):
            raise ValueError("known regression controller evidence changed")
        comparisons.append(row)
        pair_rows.extend(pairs)
        agent_rows.extend(agents)

    if producer is None:
        producer = producer_identity(
            project_root=root,
            source_files=(
                "experiments/stride_closurepool_longtail.py",
                "experiments/closed_loop_trace_storage.py",
                "experiments/state_analysis.py",
            ),
            native_required=False,
        )
    comparison_path = output / "comparison_diagnostics.jsonl"
    pair_path = output / "persistent_pair_diagnostics.jsonl"
    agent_path = output / "agent_persistence_diagnostics.jsonl"
    write_jsonl(comparison_path, sorted(comparisons, key=lambda row: row["comparison_id"]))
    write_jsonl(
        pair_path,
        sorted(
            pair_rows,
            key=lambda row: (
                row["comparison_id"],
                row["trajectory_role"],
                row["left_agent"],
                row["right_agent"],
            ),
        ),
    )
    write_jsonl(
        agent_path,
        sorted(
            agent_rows,
            key=lambda row: (
                row["comparison_id"], row["trajectory_role"], row["agent_id"]
            ),
        ),
    )
    expected_divergence_count = sum(
        row.get("divergence") is not None
        for controller in CHALLENGERS
        for row in historical_report["comparisons"][controller]
    )
    checks = {
        "historical_key_count": len(historical_keys) == 29,
        "historical_comparison_count": expected_divergence_count + no_divergence_count
        == 58,
        "historical_divergence_count": sum(
            row["evidence_role"] == "historical_mechanism_diagnostic"
            for row in comparisons
        )
        == expected_divergence_count,
        "known_regression_comparison_count": sum(
            row["evidence_role"] == "known_regression_only" for row in comparisons
        )
        == 2,
        "all_divergence_pp_seeds_paired": all(
            int(row["paired_pp_seed"]) >= 0 for row in comparisons
        ),
        "all_actions_legal_and_nonempty": all(
            row["v2_selected_agents"] and row["challenger_selected_agents"]
            for row in comparisons
        ),
        "known_regression_excluded_from_mechanism_contrast": True,
        "no_candidate_rule_or_model_produced": True,
    }
    report = {
        "schema": REPORT_SCHEMA,
        "scientific_status": "completed_retrospective_mechanism_diagnostic_no_rule_freeze",
        "experiment_id": EXPERIMENT_ID,
        "producer_identity": producer,
        "integrity_passed": all(checks.values()),
        "checks": checks,
        "comparison_count_with_divergence": len(comparisons),
        "historical_no_divergence_comparison_count": no_divergence_count,
        "outcome_category_counts": dict(
            sorted(collections.Counter(row["outcome_category"] for row in comparisons).items())
        ),
        "group_summaries": _group_summary(comparisons),
        "mechanism_contrasts": _mechanism_contrasts(comparisons),
        "claim_boundary": dict(config["claim_boundary"]),
        "decision": (
            "human_cross_case_mechanism_review_required_before_closure_rule_freeze"
        ),
        "artifacts": {
            "comparison_diagnostics": comparison_path.name,
            "comparison_diagnostics_sha256": sha256_file(comparison_path),
            "persistent_pair_diagnostics": pair_path.name,
            "persistent_pair_diagnostics_sha256": sha256_file(pair_path),
            "agent_persistence_diagnostics": agent_path.name,
            "agent_persistence_diagnostics_sha256": sha256_file(agent_path),
        },
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in sorted(inputs.items())
        },
        "config_sha256": sha256_file(path),
    }
    write_json(output / "closurepool_longtail_forensics_report.json", report)
    return report


__all__ = [
    "action_diagnostics",
    "analyze_closurepool_longtail",
    "load_closurepool_longtail_config",
    "reconstruct_trace",
    "trajectory_diagnostics",
    "_outcome_category",
]
