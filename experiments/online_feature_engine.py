from __future__ import annotations

import collections
import functools
import importlib
import itertools
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from experiments.feature_schema_v2 import PROFILE_FEATURE_NAMES, canonicalize_features
from experiments.local_representation_audit import (
    ConflictEvent,
    StateAnalysis,
    StaticGridAnalysis,
    _conflict_components,
    analyze_state,
    analyze_static_grid,
)
from experiments.realized_neighborhood_ranking_audit import (
    CandidateFeatureCache,
    _aggregate,
    _mean,
    _path_wait_ratio,
    _ratio,
    candidate_feature_cache,
    explicit_neighborhood_features,
    proposal_features,
    state_dynamic_features,
)


FEATURE_BACKENDS = ("auto", "python", "reference", "native")


@functools.lru_cache(maxsize=1)
def _native_batch_function() -> Any | None:
    for module_name in ("lns2_env", "lns2_features_native"):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        function = getattr(module, "batch_online_features", None)
        if callable(function):
            return function
    project_root = Path(__file__).resolve().parents[1]
    for candidate in (
        project_root / "build" / "linux" / "project",
        project_root / "build" / "native-features-windows" / "Release",
    ):
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    for module_name in ("lns2_env", "lns2_features_native"):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        function = getattr(module, "batch_online_features", None)
        if callable(function):
            return function
    return None


@functools.lru_cache(maxsize=16)
def _cached_static_grid(
    rows: int, cols: int, obstacles: tuple[int, ...]
) -> StaticGridAnalysis:
    return analyze_static_grid(
        {"rows": rows, "cols": cols, "obstacles": list(obstacles)}
    )


def static_grid_for_state(state: dict[str, Any]) -> StaticGridAnalysis:
    return _cached_static_grid(
        int(state["rows"]),
        int(state["cols"]),
        tuple(map(int, state["obstacles"])),
    )


def _project_feature_values(
    features: dict[str, Any], profile: str, required_names: Iterable[str]
) -> dict[str, float]:
    allowed = set(PROFILE_FEATURE_NAMES[profile])
    names = tuple(map(str, required_names))
    if len(names) != len(set(names)) or not set(names) <= allowed:
        raise ValueError(f"invalid required feature projection for {profile}")
    values = {name: float(features.get(name, 0.0)) for name in names}
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError(f"non-finite feature value in {profile}")
    return values


def _native_static_payload(static_grid: StaticGridAnalysis) -> dict[str, Any]:
    cell_count = static_grid.rows * static_grid.cols
    return {
        "free_cell_count": len(static_grid.free_cells),
        "degrees": [float(static_grid.degrees.get(cell, 0.0)) for cell in range(cell_count)],
        "articulation": [int(cell in static_grid.articulation) for cell in range(cell_count)],
        "obstacle_rate_2": [
            float(static_grid.obstacle_rate_2.get(cell, 0.0)) for cell in range(cell_count)
        ],
        "obstacle_rate_4": [
            float(static_grid.obstacle_rate_4.get(cell, 0.0)) for cell in range(cell_count)
        ],
    }


def _position(path: list[int], time_index: int) -> int:
    return path[min(time_index, len(path) - 1)]


class TemporalConflictIndex:
    def __init__(self, paths: dict[int, list[int]]) -> None:
        if not paths or any(not path for path in paths.values()):
            raise ValueError("temporal conflict index requires non-empty paths")
        self.paths = {agent: list(path) for agent, path in paths.items()}
        self.horizon = max(map(len, self.paths.values()))
        self.occupancy: list[dict[int, set[int]]] = [
            collections.defaultdict(set) for _ in range(self.horizon)
        ]
        self.transitions: list[dict[tuple[int, int], set[int]]] = [
            collections.defaultdict(set) for _ in range(self.horizon)
        ]
        for agent_id, path in sorted(self.paths.items()):
            for time_index in range(self.horizon):
                current = _position(path, time_index)
                self.occupancy[time_index][current].add(agent_id)
                if time_index:
                    previous = _position(path, time_index - 1)
                    if previous != current:
                        self.transitions[time_index][(previous, current)].add(agent_id)

    def all_events(self) -> list[ConflictEvent]:
        events: list[ConflictEvent] = []
        for time_index in range(self.horizon):
            for cell, occupants in self.occupancy[time_index].items():
                for left, right in itertools.combinations(sorted(occupants), 2):
                    events.append(
                        ConflictEvent(time_index, "vertex", left, right, (cell,))
                    )
            if not time_index:
                continue
            transitions = self.transitions[time_index]
            for (previous, current), forward in sorted(transitions.items()):
                if previous >= current:
                    continue
                for left in sorted(forward):
                    for right in sorted(transitions.get((current, previous), set())):
                        first, second = sorted((left, right))
                        events.append(
                            ConflictEvent(
                                time_index,
                                "edge",
                                first,
                                second,
                                (previous, current),
                            )
                        )
        events.sort(key=lambda event: (event.time, event.kind, event.left, event.right))
        return events

    def update(
        self, paths: dict[int, list[int]], changed_agents: Iterable[int]
    ) -> set[int]:
        new_paths = {agent: list(path) for agent, path in paths.items()}
        if set(new_paths) != set(self.paths):
            raise ValueError("agent ids changed during an episode")
        requested = {int(agent) for agent in changed_agents}
        changed = {
            agent
            for agent in requested
            if agent not in self.paths or self.paths[agent] != new_paths[agent]
        }
        if not changed:
            self.paths = new_paths
            return set()

        old_paths = self.paths
        old_horizon = self.horizon
        new_horizon = max(map(len, new_paths.values()))
        if new_horizon > old_horizon:
            for time_index in range(old_horizon, new_horizon):
                occupancy: dict[int, set[int]] = collections.defaultdict(set)
                for agent_id, path in old_paths.items():
                    occupancy[_position(path, time_index)].add(agent_id)
                self.occupancy.append(occupancy)
                self.transitions.append(collections.defaultdict(set))

        common_horizon = min(old_horizon, new_horizon)
        for agent_id in changed:
            old_path = old_paths[agent_id]
            new_path = new_paths[agent_id]
            for time_index in range(common_horizon):
                old_cell = _position(old_path, time_index)
                new_cell = _position(new_path, time_index)
                if old_cell != new_cell:
                    occupants = self.occupancy[time_index][old_cell]
                    occupants.discard(agent_id)
                    if not occupants:
                        del self.occupancy[time_index][old_cell]
                    self.occupancy[time_index][new_cell].add(agent_id)
                if time_index:
                    old_previous = _position(old_path, time_index - 1)
                    new_previous = _position(new_path, time_index - 1)
                    old_move = (old_previous, old_cell)
                    new_move = (new_previous, new_cell)
                    if old_move != new_move:
                        if old_previous != old_cell:
                            agents = self.transitions[time_index][old_move]
                            agents.discard(agent_id)
                            if not agents:
                                del self.transitions[time_index][old_move]
                        if new_previous != new_cell:
                            self.transitions[time_index][new_move].add(agent_id)
            if new_horizon > old_horizon:
                for time_index in range(old_horizon, new_horizon):
                    old_cell = _position(old_path, time_index)
                    new_cell = _position(new_path, time_index)
                    if old_cell != new_cell:
                        occupants = self.occupancy[time_index][old_cell]
                        occupants.discard(agent_id)
                        if not occupants:
                            del self.occupancy[time_index][old_cell]
                        self.occupancy[time_index][new_cell].add(agent_id)
                    new_previous = _position(new_path, time_index - 1)
                    if new_previous != new_cell:
                        self.transitions[time_index][(new_previous, new_cell)].add(
                            agent_id
                        )

        if new_horizon < old_horizon:
            del self.occupancy[new_horizon:]
            del self.transitions[new_horizon:]
        self.paths = new_paths
        self.horizon = new_horizon
        return changed


@dataclass
class PathAggregates:
    length: int
    degree_sum: float
    low_degree_count: int
    articulation_count: int
    obstacle_2_sum: float
    obstacle_4_sum: float
    visit_heat_sum: float
    wait_ratio: float


@dataclass
class OptimizedCandidateCache:
    base: CandidateFeatureCache
    path_bits: dict[int, int]
    path_aggregates: dict[int, PathAggregates]
    conflict_adjacency: dict[int, set[int]]
    event_pair_counts: collections.Counter[tuple[int, int]]
    incident_event_counts: collections.Counter[int]


def _path_bitset(path: Iterable[int]) -> int:
    value = 0
    for cell in set(map(int, path)):
        value |= 1 << cell
    return value


def _optimized_cache(
    state: dict[str, Any], analysis: StateAnalysis
) -> OptimizedCandidateCache:
    base = candidate_feature_cache(state, analysis)
    adjacency = {agent_id: set() for agent_id in base.by_id}
    for left, right in analysis.pair_set:
        adjacency[left].add(right)
        adjacency[right].add(left)
    event_pairs: collections.Counter[tuple[int, int]] = collections.Counter()
    incident_events: collections.Counter[int] = collections.Counter()
    for event in analysis.events:
        pair = (event.left, event.right)
        event_pairs[pair] += 1
        incident_events[event.left] += 1
        incident_events[event.right] += 1
    aggregates = {}
    for agent_id, path in base.paths.items():
        degree_values = [float(analysis.degrees.get(cell, 0.0)) for cell in path]
        obstacle_2 = [float(analysis.obstacle_rate_2.get(cell, 0.0)) for cell in path]
        obstacle_4 = [float(analysis.obstacle_rate_4.get(cell, 0.0)) for cell in path]
        aggregates[agent_id] = PathAggregates(
            length=len(path),
            degree_sum=math.fsum(degree_values),
            low_degree_count=sum(value <= 2 for value in degree_values),
            articulation_count=sum(cell in analysis.articulation for cell in path),
            obstacle_2_sum=math.fsum(obstacle_2),
            obstacle_4_sum=math.fsum(obstacle_4),
            visit_heat_sum=math.fsum(float(analysis.visit_heat[cell]) for cell in path),
            wait_ratio=_path_wait_ratio(path),
        )
    return OptimizedCandidateCache(
        base=base,
        path_bits={
            agent_id: _path_bitset(path) for agent_id, path in base.paths.items()
        },
        path_aggregates=aggregates,
        conflict_adjacency=adjacency,
        event_pair_counts=event_pairs,
        incident_event_counts=incident_events,
    )


def optimized_explicit_neighborhood_features(
    state: dict[str, Any],
    analysis: StateAnalysis,
    neighborhood: list[int],
    cache: OptimizedCandidateCache,
) -> dict[str, float]:
    if not neighborhood or len(neighborhood) != len(set(neighborhood)):
        raise ValueError("explicit neighborhood must be non-empty and unique")
    selected = set(map(int, neighborhood))
    by_id = cache.base.by_id
    if any(agent_id not in by_id for agent_id in selected):
        raise ValueError("explicit neighborhood contains an unknown agent")

    internal_edges = 0
    boundary_edges = 0
    for agent_id in selected:
        for neighbor in cache.conflict_adjacency[agent_id]:
            if neighbor in selected:
                internal_edges += agent_id < neighbor
            else:
                boundary_edges += 1
    active_agents = cache.base.active_agents
    component_ids = {
        analysis.component_id[agent_id]
        for agent_id in selected
        if agent_id in analysis.component_id
    }
    component_coverages = [
        _ratio(
            len(selected & analysis.component_members[component_id]),
            len(analysis.component_members[component_id]),
        )
        for component_id in sorted(component_ids)
    ]
    selected_ids = sorted(selected)
    selected_agents = [by_id[agent_id] for agent_id in selected_ids]
    delays = [float(agent.get("delay", 0)) for agent in selected_agents]
    conflicts = [float(agent.get("conflict_degree", 0)) for agent in selected_agents]
    path_costs = [float(agent.get("path_cost", 0)) for agent in selected_agents]
    stretches = [
        _ratio(agent.get("path_cost", 0), max(1, agent.get("shortest_path_cost", 0)))
        for agent in selected_agents
    ]

    overlaps = []
    union_bits = 0
    for agent_id in selected_ids:
        union_bits |= cache.path_bits[agent_id]
    for left, right in itertools.combinations(selected_ids, 2):
        left_bits = cache.path_bits[left]
        right_bits = cache.path_bits[right]
        overlaps.append(
            _ratio((left_bits & right_bits).bit_count(), (left_bits | right_bits).bit_count())
        )
    union = set().union(*(cache.base.path_sets[agent] for agent in selected_ids))
    if union:
        coordinates = [divmod(cell, analysis.cols) for cell in union]
        span_rows = max(row for row, _ in coordinates) - min(row for row, _ in coordinates) + 1
        span_cols = max(col for _, col in coordinates) - min(col for _, col in coordinates) + 1
    else:
        span_rows = span_cols = 0

    path_values = [cache.path_aggregates[agent_id] for agent_id in selected_ids]
    path_entry_count = sum(value.length for value in path_values)
    internal_events = sum(
        cache.event_pair_counts[tuple(sorted((agent_id, neighbor)))]
        for agent_id in selected
        for neighbor in cache.conflict_adjacency[agent_id]
        if neighbor in selected and agent_id < neighbor
    )
    incident_events = (
        sum(cache.incident_event_counts[agent_id] for agent_id in selected)
        - internal_events
    )
    features = {
        "realized.actual_size": float(len(selected)),
        "realized.actual_size_ratio_agents": _ratio(len(selected), len(by_id)),
        "realized.conflicting_agent_ratio": _ratio(len(selected & active_agents), len(selected)),
        "realized.conflicting_agent_coverage": _ratio(len(selected & active_agents), len(active_agents)),
        "realized.component_count": float(len(component_ids)),
        "realized.component_coverage_mean": _mean(component_coverages),
        "realized.component_coverage_max": max(component_coverages, default=0.0),
        "realized.internal_conflict_edges": float(internal_edges),
        "realized.boundary_conflict_edges": float(boundary_edges),
        "realized.incident_conflict_coverage": _ratio(
            internal_edges + boundary_edges, len(analysis.pair_set)
        ),
        "realized.internal_conflict_coverage": _ratio(
            internal_edges, len(analysis.pair_set)
        ),
        "realized.internal_event_coverage": _ratio(internal_events, len(analysis.events)),
        "realized.incident_event_coverage": _ratio(incident_events, len(analysis.events)),
        "realized.path_overlap_mean": _mean(overlaps),
        "realized.path_overlap_max": max(overlaps, default=0.0),
        "realized.path_union_cell_ratio": _ratio(union_bits.bit_count(), len(analysis.free_cells)),
        "realized.path_bbox_area_ratio": _ratio(
            span_rows * span_cols, analysis.rows * analysis.cols
        ),
        "realized.path_degree_mean": _ratio(
            math.fsum(value.degree_sum for value in path_values), path_entry_count
        ),
        "realized.path_low_degree_ratio": _ratio(
            sum(value.low_degree_count for value in path_values), path_entry_count
        ),
        "realized.path_articulation_ratio": _ratio(
            sum(value.articulation_count for value in path_values), path_entry_count
        ),
        "realized.path_visit_heat_mean": _ratio(
            math.fsum(value.visit_heat_sum for value in path_values), path_entry_count
        ),
        "realized.path_obstacle_rate_r2": _ratio(
            math.fsum(value.obstacle_2_sum for value in path_values), path_entry_count
        ),
        "realized.path_obstacle_rate_r4": _ratio(
            math.fsum(value.obstacle_4_sum for value in path_values), path_entry_count
        ),
        "realized.path_wait_ratio_mean": _mean(value.wait_ratio for value in path_values),
    }
    features.update(_aggregate("realized.delay", delays))
    features.update(_aggregate("realized.conflict_degree", conflicts))
    features.update(_aggregate("realized.path_cost", path_costs))
    features.update(_aggregate("realized.path_stretch", stretches))
    return features


def online_rows_for_shadow(
    state: dict[str, Any], candidates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    analysis = analyze_state(state, static_grid=static_grid_for_state(state))
    dynamic = state_dynamic_features(state, analysis)
    cache = candidate_feature_cache(state, analysis)
    rows = []
    for candidate in candidates:
        proposal = proposal_features(
            state, analysis, candidate, feature_cache=cache
        )
        realized = explicit_neighborhood_features(
            state,
            analysis,
            list(map(int, candidate["agents"])),
            feature_cache=cache,
        )
        candidate_id = str(candidate["candidate_id"])
        rows.append(
            {
                "candidate_id": candidate_id,
                "candidate_key": candidate_id,
                "features": {
                    "proposal_dynamic": canonicalize_features(
                        dynamic | proposal, "proposal_dynamic"
                    ),
                    "realized_dynamic": canonicalize_features(
                        dynamic | proposal | realized, "realized_dynamic"
                    ),
                },
            }
        )
    return rows


def _assert_native_rows(
    reference: list[dict[str, Any]],
    native: list[dict[str, Any]],
    profile: str,
    tolerance: float = 1e-12,
) -> None:
    if len(reference) != len(native):
        raise ValueError("native and reference feature row counts differ")
    for expected, actual in zip(reference, native):
        if str(expected["candidate_id"]) != str(actual["candidate_id"]):
            raise ValueError("native and reference candidate orders differ")
        left = expected["features"][profile]
        right = actual["features"][profile]
        if not set(right) <= set(left):
            raise ValueError("native feature projection contains an unknown feature")
        maximum = max(
            (abs(float(left[name]) - float(right[name])) for name in right),
            default=0.0,
        )
        if maximum > tolerance:
            raise ValueError(
                f"native {profile} differs from reference by {maximum}"
            )


class OnlineFeatureEngine:
    def __init__(
        self,
        initial_state: dict[str, Any],
        *,
        backend: str = "auto",
        shadow_validation: bool = False,
        required_features: dict[str, Iterable[str]] | None = None,
    ) -> None:
        if backend not in FEATURE_BACKENDS:
            raise ValueError(f"unsupported feature backend: {backend}")
        native_function = _native_batch_function()
        if backend == "native" and native_function is None:
            raise ValueError("native feature backend is not available in this build")
        self.backend = (
            "native"
            if backend == "native" or (backend == "auto" and native_function is not None)
            else "python" if backend == "auto" else backend
        )
        self.native_function = native_function if self.backend == "native" else None
        native_doc = str(getattr(self.native_function, "__doc__", ""))
        self.native_accepts_required_features = "required_features" in native_doc
        self.shadow_validation = bool(shadow_validation)
        self.required_features: dict[str, tuple[str, ...]] = {}
        requested_profiles = dict(required_features or {})
        for profile, schema_names in PROFILE_FEATURE_NAMES.items():
            requested = tuple(
                map(str, requested_profiles.get(profile, schema_names))
            )
            requested_set = set(requested)
            if len(requested) != len(requested_set) or not requested_set <= set(
                schema_names
            ):
                raise ValueError(f"invalid required features for {profile}")
            self.required_features[profile] = tuple(
                name for name in schema_names if name in requested_set
            )
        self.static_grid = static_grid_for_state(initial_state)
        self.native_static = _native_static_payload(self.static_grid)
        self.state: dict[str, Any] | None = None
        self.analysis: StateAnalysis | None = None
        self.conflict_index: TemporalConflictIndex | None = None
        self.cache: OptimizedCandidateCache | None = None
        self.dynamic: dict[str, float] | None = None
        self.last_shadow_rows: dict[str, list[dict[str, Any]]] = {}
        self.last_prepare_metrics: dict[str, Any] = {}
        self.prepare(initial_state)

    def _native_payload(
        self,
        candidates: list[dict[str, Any]],
        *,
        profile: str,
        include_realized: bool,
    ) -> dict[str, Any]:
        assert self.native_function is not None and self.state is not None
        names = self.required_features[profile]
        required = {
            "dynamic": [
                name for name in names if name.startswith(("state.", "context."))
            ],
            "proposal": [name for name in names if name.startswith("proposal.")],
            "realized": [name for name in names if name.startswith("realized.")],
        }
        if self.native_accepts_required_features:
            return dict(
                self.native_function(
                    self.state,
                    candidates,
                    self.native_static,
                    include_realized,
                    required,
                )
            )
        return dict(
            self.native_function(
                self.state, candidates, self.native_static, include_realized
            )
        )

    def _analysis_from_index(
        self, state: dict[str, Any], index: TemporalConflictIndex
    ) -> StateAnalysis:
        paths = index.paths
        events = index.all_events()
        pair_set = {(event.left, event.right) for event in events}
        expected_pairs = {
            tuple(sorted((int(edge[0]), int(edge[1]))))
            for edge in state.get("conflict_edges", [])
        }
        if pair_set != expected_pairs:
            raise ValueError("reconstructed conflicts disagree with solver conflict edges")
        agent_ids = [int(agent["id"]) for agent in state["agents"]]
        component_id, component_members = _conflict_components(agent_ids, pair_set)
        visit_heat: collections.Counter[int] = collections.Counter()
        agent_heat: collections.Counter[int] = collections.Counter()
        for path in paths.values():
            visit_heat.update(path)
            agent_heat.update(set(path))
        return StateAnalysis(
            rows=self.static_grid.rows,
            cols=self.static_grid.cols,
            free_cells=self.static_grid.free_cells,
            degrees=self.static_grid.degrees,
            articulation=self.static_grid.articulation,
            obstacle_rate_2=self.static_grid.obstacle_rate_2,
            obstacle_rate_4=self.static_grid.obstacle_rate_4,
            visit_heat=visit_heat,
            agent_heat=agent_heat,
            events=events,
            pair_set=pair_set,
            component_id=component_id,
            component_members=component_members,
        )

    def _full_analysis(self, state: dict[str, Any]) -> tuple[StateAnalysis, TemporalConflictIndex]:
        paths = {
            int(agent["id"]): list(map(int, agent["path"]))
            for agent in state["agents"]
        }
        index = TemporalConflictIndex(paths)
        analysis = (
            analyze_state(state, static_grid=self.static_grid)
            if self.backend == "reference"
            else self._analysis_from_index(state, index)
        )
        return analysis, index

    def _incremental_analysis(
        self, state: dict[str, Any], changed_agents: Iterable[int]
    ) -> tuple[StateAnalysis, float, bool]:
        assert self.analysis is not None and self.conflict_index is not None
        started = time.perf_counter()
        paths = {
            int(agent["id"]): list(map(int, agent["path"]))
            for agent in state["agents"]
        }
        changed = self.conflict_index.update(paths, changed_agents)
        if not changed:
            conflict_seconds = time.perf_counter() - started
            return self.analysis, conflict_seconds, True
        analysis = self._analysis_from_index(state, self.conflict_index)
        return analysis, time.perf_counter() - started, False

    def prepare(
        self,
        state: dict[str, Any],
        *,
        changed_agents: Iterable[int] | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        if self.backend == "native":
            self.state = state
            self.last_prepare_metrics = {
                "state_analysis_seconds": time.perf_counter() - started,
                "conflict_update_seconds": 0.0,
                "feature_cache_seconds": 0.0,
                "state_feature_seconds": 0.0,
                "incremental": changed_agents is not None,
                "incremental_cache_hit": False,
                "feature_backend": self.backend,
            }
            return dict(self.last_prepare_metrics)
        conflict_seconds = 0.0
        cache_hit = False
        incremental = False
        if self.backend == "reference" or self.analysis is None or changed_agents is None:
            analysis_started = time.perf_counter()
            analysis, index = self._full_analysis(state)
            conflict_seconds = time.perf_counter() - analysis_started
            self.conflict_index = index
        else:
            incremental = True
            try:
                analysis, conflict_seconds, cache_hit = self._incremental_analysis(
                    state, changed_agents
                )
            except ValueError:
                analysis_started = time.perf_counter()
                analysis, index = self._full_analysis(state)
                conflict_seconds = time.perf_counter() - analysis_started
                self.conflict_index = index
                incremental = False
        if self.shadow_validation and self.backend != "reference":
            reference = analyze_state(state, static_grid=self.static_grid)
            if reference != analysis:
                raise ValueError("incremental state analysis differs from reference")
        cache_started = time.perf_counter()
        cache = _optimized_cache(state, analysis)
        cache_seconds = time.perf_counter() - cache_started
        dynamic_started = time.perf_counter()
        dynamic = state_dynamic_features(state, analysis)
        dynamic_seconds = time.perf_counter() - dynamic_started
        self.state = state
        self.analysis = analysis
        self.cache = cache
        self.dynamic = dynamic
        self.last_prepare_metrics = {
            "state_analysis_seconds": time.perf_counter() - started,
            "conflict_update_seconds": conflict_seconds,
            "feature_cache_seconds": cache_seconds,
            "state_feature_seconds": dynamic_seconds,
            "incremental": incremental,
            "incremental_cache_hit": cache_hit,
            "feature_backend": self.backend,
        }
        return dict(self.last_prepare_metrics)

    def proposal_rows(
        self,
        candidates: list[dict[str, Any]],
        *,
        state_hash: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        assert self.state is not None
        started = time.perf_counter()
        if self.backend == "native":
            payload = self._native_payload(
                candidates,
                profile="proposal_dynamic",
                include_realized=False,
            )
            dynamic = dict(payload["dynamic"])
            proposal_values = list(payload["proposal"])
            rows = []
            for candidate, proposal in zip(candidates, proposal_values):
                candidate_id = str(candidate["candidate_id"])
                rows.append(
                    {
                        "state_id": state_hash,
                        "candidate_id": candidate_id,
                        "candidate_key": candidate_id,
                        "features": {
                            "proposal_dynamic": _project_feature_values(
                                dynamic | dict(proposal),
                                "proposal_dynamic",
                                self.required_features["proposal_dynamic"],
                            )
                        },
                    }
                )
            if len(rows) != len(candidates):
                raise ValueError("native proposal feature batch has the wrong size")
            if self.shadow_validation:
                reference = online_rows_for_shadow(self.state, candidates)
                _assert_native_rows(reference, rows, "proposal_dynamic")
                self.last_shadow_rows["proposal_dynamic"] = reference
            return rows, {
                "proposal_feature_seconds": time.perf_counter() - started
            }
        assert self.analysis is not None and self.cache is not None and self.dynamic is not None
        rows = []
        for candidate in candidates:
            proposal = proposal_features(
                self.state,
                self.analysis,
                candidate,
                feature_cache=self.cache.base,
            )
            features = _project_feature_values(
                self.dynamic | proposal,
                "proposal_dynamic",
                self.required_features["proposal_dynamic"],
            )
            candidate_id = str(candidate["candidate_id"])
            rows.append(
                {
                    "state_id": state_hash,
                    "candidate_id": candidate_id,
                    "candidate_key": candidate_id,
                    "features": {"proposal_dynamic": features},
                }
            )
        if self.shadow_validation:
            reference = online_rows_for_shadow(self.state, candidates)
            _assert_native_rows(reference, rows, "proposal_dynamic")
            self.last_shadow_rows["proposal_dynamic"] = reference
        return rows, {"proposal_feature_seconds": time.perf_counter() - started}

    def realized_rows(
        self,
        candidates: list[dict[str, Any]],
        *,
        state_hash: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        assert self.state is not None
        started = time.perf_counter()
        if self.backend == "native":
            payload = self._native_payload(
                candidates,
                profile="realized_dynamic",
                include_realized=True,
            )
            dynamic = dict(payload["dynamic"])
            proposal_values = list(payload["proposal"])
            realized_values = list(payload["realized"])
            rows = []
            for candidate, proposal, realized in zip(
                candidates, proposal_values, realized_values
            ):
                candidate_id = str(candidate["candidate_id"])
                rows.append(
                    {
                        "state_id": state_hash,
                        "candidate_id": candidate_id,
                        "candidate_key": candidate_id,
                        "features": {
                            "realized_dynamic": _project_feature_values(
                                dynamic | dict(proposal) | dict(realized),
                                "realized_dynamic",
                                self.required_features["realized_dynamic"],
                            )
                        },
                    }
                )
            if len(rows) != len(candidates):
                raise ValueError("native realized feature batch has the wrong size")
            if self.shadow_validation:
                reference = online_rows_for_shadow(self.state, candidates)
                _assert_native_rows(reference, rows, "realized_dynamic")
                self.last_shadow_rows["realized_dynamic"] = reference
            return rows, {
                "realized_feature_seconds": time.perf_counter() - started
            }
        assert self.analysis is not None and self.cache is not None and self.dynamic is not None
        rows = []
        for candidate in candidates:
            proposal = proposal_features(
                self.state,
                self.analysis,
                candidate,
                feature_cache=self.cache.base,
            )
            realized = optimized_explicit_neighborhood_features(
                self.state,
                self.analysis,
                list(map(int, candidate["agents"])),
                self.cache,
            )
            features = _project_feature_values(
                self.dynamic | proposal | realized,
                "realized_dynamic",
                self.required_features["realized_dynamic"],
            )
            candidate_id = str(candidate["candidate_id"])
            rows.append(
                {
                    "state_id": state_hash,
                    "candidate_id": candidate_id,
                    "candidate_key": candidate_id,
                    "features": {"realized_dynamic": features},
                }
            )
        if self.shadow_validation:
            reference = online_rows_for_shadow(self.state, candidates)
            _assert_native_rows(reference, rows, "realized_dynamic")
            self.last_shadow_rows["realized_dynamic"] = reference
        return rows, {"realized_feature_seconds": time.perf_counter() - started}


__all__ = [
    "FEATURE_BACKENDS",
    "OnlineFeatureEngine",
    "TemporalConflictIndex",
    "optimized_explicit_neighborhood_features",
    "static_grid_for_state",
]
