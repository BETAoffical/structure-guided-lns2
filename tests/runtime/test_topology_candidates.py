from __future__ import annotations

import collections
import random
import unittest
from unittest.mock import patch

from experiments.state_analysis import ConflictEvent, StateAnalysis
import lns2_selector.runtime.topology_candidates as topology_candidates
from lns2_selector.runtime.topology_candidates import (
    _anchor_neighborhood,
    _boundary_neighborhood,
    _fill_neighborhood,
    generate_scalepool_candidates,
    generate_structpool_candidates,
    generate_structpool_candidate_grid,
    generate_structpool_candidate_subset,
    generate_topology_anchor_candidates,
    generate_topology_boundary_candidates,
    merge_structpool_candidates,
    merge_topology_anchor_candidates,
    scalepool_size_attempt_order,
)


def _reference_fill_neighborhood(
    selected: set[int],
    state: dict,
    event_weight: collections.Counter[int],
    size: int,
) -> list[int]:
    agent_rows = {int(agent["id"]): agent for agent in state["agents"]}
    adjacency = {agent: set() for agent in agent_rows}
    for edge in state.get("conflict_edges", []):
        left, right = map(int, edge)
        adjacency[left].add(right)
        adjacency[right].add(left)
    while len(selected) < min(size, len(agent_rows)):
        remaining = set(agent_rows) - selected
        chosen = min(
            remaining,
            key=lambda agent: (
                -len(adjacency[agent] & selected),
                -int(agent_rows[agent].get("conflict_degree", 0)),
                -int(event_weight[agent]),
                agent,
            ),
        )
        selected.add(chosen)
    return sorted(selected)


def _reference_anchor_neighborhood(
    state: dict, events: list[ConflictEvent], size: int
) -> list[int]:
    uncovered = set(range(len(events)))
    selected: set[int] = set()
    event_weight: collections.Counter[int] = collections.Counter()
    for event in events:
        event_weight[event.left] += 1
        event_weight[event.right] += 1
    pairs = sorted({(event.left, event.right) for event in events})
    while uncovered and len(selected) < size:
        options = []
        for left, right in pairs:
            addition = {left, right} - selected
            if not addition or len(selected) + len(addition) > size:
                continue
            covered = {
                index
                for index in uncovered
                if events[index].left in addition or events[index].right in addition
            }
            internal = {
                index
                for index in uncovered
                if events[index].left in (selected | addition)
                and events[index].right in (selected | addition)
            }
            options.append(
                (
                    len(covered) / len(addition),
                    len(covered),
                    len(internal),
                    sum(event_weight[agent] for agent in addition),
                    -left,
                    -right,
                    addition,
                )
            )
        if not options:
            break
        *_, addition = max(options, key=lambda value: value[:-1])
        selected.update(addition)
        uncovered = {
            index
            for index in uncovered
            if events[index].left not in selected and events[index].right not in selected
        }
    return _reference_fill_neighborhood(selected, state, event_weight, size)


def _reference_boundary_neighborhood(
    state: dict, analysis: StateAnalysis, events: list[ConflictEvent],
    *, size: int, core_budget: int,
) -> list[int]:
    agent_rows = {int(agent["id"]): agent for agent in state["agents"]}
    selected: set[int] = set()

    def choose(available: set[int], scored_events: list[ConflictEvent]) -> int:
        def score(agent: int) -> tuple[int, int, int, int, int, int]:
            newly_incident = sum(
                agent in {event.left, event.right}
                and event.left not in selected
                and event.right not in selected
                for event in scored_events
            )
            newly_internal = sum(
                agent in {event.left, event.right}
                and ((event.left in selected) != (event.right in selected))
                for event in scored_events
            )
            component = analysis.component_id.get(agent)
            component_novel = int(
                component is not None
                and all(
                    analysis.component_id.get(current) != component
                    for current in selected
                )
            )
            return (
                3 * newly_incident - newly_internal,
                newly_incident,
                -newly_internal,
                component_novel,
                int(agent_rows[agent].get("conflict_degree", 0)),
                -agent,
            )

        return max(available, key=score)

    relevant_agents = {
        agent for event in events for agent in (int(event.left), int(event.right))
    }
    while len(selected) < min(core_budget, size) and relevant_agents - selected:
        candidate = choose(relevant_agents - selected, events)
        before_coverage = sum(
            event.left in selected or event.right in selected for event in events
        )
        selected.add(candidate)
        after_coverage = sum(
            event.left in selected or event.right in selected for event in events
        )
        if after_coverage == before_coverage:
            selected.remove(candidate)
            break

    active_agents = {
        agent
        for event in analysis.events
        for agent in (int(event.left), int(event.right))
    }
    limit = min(size, len(agent_rows))
    while len(selected) < limit:
        remaining_active = active_agents - selected
        available = remaining_active if remaining_active else set(agent_rows) - selected
        selected.add(choose(available, analysis.events))
    return sorted(selected)


class TopologyCandidatesTest(unittest.TestCase):
    @staticmethod
    def _structpool_state() -> tuple[dict, StateAnalysis]:
        agent_count = 40
        agents = []
        events = []
        pair_set = set()
        for agent in range(agent_count):
            path = [
                agent % 20,
                20 + (agent % 10),
                40 + ((agent * 3) % 20),
                60 + ((agent * 7) % 20),
            ]
            agents.append(
                {
                    "id": agent,
                    "path": path,
                    "conflict_degree": 2 + (agent % 6),
                }
            )
        for index in range(32):
            left = index
            right = (index + 1) % 32
            pair = (min(left, right), max(left, right))
            pair_set.add(pair)
            events.append(
                ConflictEvent(
                    time=index % 8,
                    kind="vertex",
                    left=pair[0],
                    right=pair[1],
                    cells=(20 + (index % 10),),
                )
            )
        for index in range(8):
            left, right = index, 32 + index
            pair_set.add((left, right))
            events.append(
                ConflictEvent(
                    time=2 + (index % 3),
                    kind="vertex",
                    left=left,
                    right=right,
                    cells=(40 + index,),
                )
            )
        state = {
            "agents": agents,
            "conflict_edges": [list(pair) for pair in sorted(pair_set)],
        }
        visit_heat: collections.Counter[int] = collections.Counter()
        agent_heat: collections.Counter[int] = collections.Counter()
        for agent in agents:
            visit_heat.update(agent["path"])
            agent_heat.update(set(agent["path"]))
        component_id = {agent: 0 if agent < 32 else agent - 31 for agent in range(40)}
        component_members = {0: set(range(32))}
        component_members.update({index + 1: {index, index + 32} for index in range(8)})
        analysis = StateAnalysis(
            rows=5,
            cols=20,
            free_cells=set(range(100)),
            degrees={cell: (2 if 20 <= cell < 30 else 3) for cell in range(100)},
            articulation={40, 41, 42, 43},
            obstacle_rate_2={},
            obstacle_rate_4={},
            visit_heat=visit_heat,
            agent_heat=agent_heat,
            events=events,
            pair_set=pair_set,
            component_id=component_id,
            component_members=component_members,
        )
        return state, analysis

    def test_incident_index_optimization_preserves_reference_actions(self) -> None:
        for seed in range(12):
            generator = random.Random(seed)
            agent_count = 40
            state = {
                "agents": [
                    {
                        "id": agent,
                        "path": [agent],
                        "conflict_degree": generator.randrange(8),
                    }
                    for agent in range(agent_count)
                ]
            }
            events = []
            for index in range(200):
                left, right = generator.sample(range(agent_count), 2)
                events.append(
                    ConflictEvent(
                        index,
                        "vertex",
                        min(left, right),
                        max(left, right),
                        (generator.randrange(100),),
                    )
                )
            analysis = StateAnalysis(
                rows=10,
                cols=10,
                free_cells=set(range(100)),
                degrees={cell: 2 for cell in range(100)},
                articulation=set(),
                obstacle_rate_2={},
                obstacle_rate_4={},
                visit_heat=collections.Counter(),
                agent_heat=collections.Counter(),
                events=events,
                pair_set={(event.left, event.right) for event in events},
                component_id={agent: agent % 7 for agent in range(agent_count)},
                component_members={},
            )
            relevant = [event for index, event in enumerate(events) if index % 3]
            for size in (4, 8, 16):
                self.assertEqual(
                    _boundary_neighborhood(
                        state,
                        analysis,
                        relevant,
                        size=size,
                        core_budget=4,
                    ),
                    _reference_boundary_neighborhood(
                        state,
                        analysis,
                        relevant,
                        size=size,
                        core_budget=4,
                    ),
                )

    def test_incremental_fill_preserves_reference_actions(self) -> None:
        for seed in range(16):
            generator = random.Random(seed)
            agent_count = 48
            edges = {
                tuple(sorted(generator.sample(range(agent_count), 2)))
                for _ in range(180)
            }
            state = {
                "agents": [
                    {
                        "id": agent,
                        "path": [agent],
                        "conflict_degree": generator.randrange(10),
                    }
                    for agent in range(agent_count)
                ],
                "conflict_edges": [list(edge) for edge in sorted(edges)],
            }
            event_weight = collections.Counter(
                {agent: generator.randrange(20) for agent in range(agent_count)}
            )
            for size in (8, 16, 24, 32):
                initial = set(generator.sample(range(agent_count), 4))
                self.assertEqual(
                    _fill_neighborhood(
                        set(initial), state, event_weight, size
                    ),
                    _reference_fill_neighborhood(
                        set(initial), state, event_weight, size
                    ),
                )

    def test_incremental_anchor_preserves_reference_actions(self) -> None:
        for seed in range(12):
            generator = random.Random(seed)
            agent_count = 48
            events = []
            for index in range(160):
                left, right = sorted(generator.sample(range(agent_count), 2))
                events.append(
                    ConflictEvent(
                        time=index % 17,
                        kind="vertex",
                        left=left,
                        right=right,
                        cells=(generator.randrange(256),),
                    )
                )
            state = {
                "agents": [
                    {
                        "id": agent,
                        "path": [agent],
                        "conflict_degree": generator.randrange(10),
                    }
                    for agent in range(agent_count)
                ],
                "conflict_edges": [
                    list(pair)
                    for pair in sorted({(event.left, event.right) for event in events})
                ],
            }
            for size in (8, 16, 24, 32):
                self.assertEqual(
                    _anchor_neighborhood(state, events, size),
                    _reference_anchor_neighborhood(state, events, size),
                )

    def test_pair_set_cover_is_deterministic_and_closes_relevant_pairs(self) -> None:
        state = {
            "agents": [
                {"id": index, "path": [index], "conflict_degree": 1}
                for index in range(6)
            ],
            "conflict_edges": [[0, 1], [1, 2], [3, 4]],
        }
        analysis = StateAnalysis(
            rows=2,
            cols=3,
            free_cells=set(range(6)),
            degrees={0: 3, 1: 2, 2: 3, 3: 3, 4: 3, 5: 3},
            articulation={1},
            obstacle_rate_2={},
            obstacle_rate_4={},
            visit_heat=collections.Counter(),
            agent_heat=collections.Counter(),
            events=[
                ConflictEvent(1, "vertex", 0, 1, (1,)),
                ConflictEvent(2, "vertex", 1, 2, (1,)),
                ConflictEvent(3, "vertex", 3, 4, (3,)),
            ],
            pair_set={(0, 1), (1, 2), (3, 4)},
            component_id={},
            component_members={},
        )
        first = generate_topology_anchor_candidates(state, analysis, [4])
        second = generate_topology_anchor_candidates(state, analysis, [4])
        self.assertEqual(first, second)
        articulation = next(
            row
            for row in first
            if "topology-anchor-articulation:4" in row["selection_families"]
        )
        self.assertTrue({0, 1, 2}.issubset(set(articulation["agents"])))
        self.assertEqual(articulation["actual_size"], 4)

    def test_merge_deduplicates_equal_agent_sets(self) -> None:
        base = [
            {
                "candidate_id": "same",
                "agents": [0, 1],
                "actual_size": 2,
                "selection_families": ["target:4"],
                "selection_rank_by_family": {"target:4": 0},
                "proposal_count_by_family": {"target:4": 1},
                "proposal_seeds": [7],
                "seed_agents": [0],
            }
        ]
        anchors = [
            {
                "candidate_id": "same",
                "agents": [0, 1],
                "actual_size": 2,
                "selection_families": ["topology-anchor-articulation:4"],
                "selection_rank_by_family": {"topology-anchor-articulation:4": 0},
                "proposal_count_by_family": {"topology-anchor-articulation:4": 1},
                "proposal_seeds": [],
                "seed_agents": [],
            }
        ]
        merged = merge_topology_anchor_candidates(base, anchors)
        self.assertEqual(len(merged), 1)
        self.assertEqual(len(merged[0]["selection_families"]), 2)

    def test_boundary_candidate_is_deterministic_and_discourages_pair_closure(self) -> None:
        state = {
            "agents": [
                {"id": index, "path": [index], "conflict_degree": 2}
                for index in range(8)
            ],
            "conflict_edges": [[0, 1], [1, 2], [2, 3], [4, 5], [6, 7]],
        }
        events = [
            ConflictEvent(index, "vertex", left, right, (1,))
            for index, (left, right) in enumerate(state["conflict_edges"])
        ]
        analysis = StateAnalysis(
            rows=2,
            cols=4,
            free_cells=set(range(8)),
            degrees={index: (2 if index == 1 else 3) for index in range(8)},
            articulation={1},
            obstacle_rate_2={},
            obstacle_rate_4={},
            visit_heat=collections.Counter(),
            agent_heat=collections.Counter(),
            events=events,
            pair_set={tuple(edge) for edge in state["conflict_edges"]},
            component_id={0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 2, 7: 2},
            component_members={0: {0, 1, 2, 3}, 1: {4, 5}, 2: {6, 7}},
        )
        first = generate_topology_boundary_candidates(
            state, analysis, neighborhood_size=4, core_budget=2
        )
        second = generate_topology_boundary_candidates(
            state, analysis, neighborhood_size=4, core_budget=2
        )
        self.assertEqual(first, second)
        self.assertTrue(first)
        self.assertTrue(all(row["actual_size"] == 4 for row in first))
        self.assertTrue(
            all(row["proposal_audit"]["global_event_boundary_ratio"] >= 0.5 for row in first)
        )

    def test_boundary_candidate_skips_states_without_relevant_events(self) -> None:
        state = {
            "agents": [{"id": 0, "path": [0], "conflict_degree": 0}],
            "conflict_edges": [],
        }
        analysis = StateAnalysis(
            rows=1,
            cols=1,
            free_cells={0},
            degrees={0: 4},
            articulation=set(),
            obstacle_rate_2={},
            obstacle_rate_4={},
            visit_heat=collections.Counter(),
            agent_heat=collections.Counter(),
            events=[],
            pair_set=set(),
            component_id={},
            component_members={},
        )
        self.assertEqual(
            generate_topology_boundary_candidates(
                state, analysis, neighborhood_size=16, core_budget=4
            ),
            [],
        )

    def test_structpool_is_deterministic_capped_and_preserves_incumbent(self) -> None:
        state, analysis = self._structpool_state()
        first = generate_structpool_candidates(state, analysis)
        second = generate_structpool_candidates(state, analysis)
        self.assertEqual(first, second)
        self.assertLessEqual(len(first), 6)
        self.assertGreaterEqual(len(first), 5)

        incumbent = generate_topology_boundary_candidates(
            state, analysis, neighborhood_size=16, core_budget=4
        )
        incumbent_sets = {tuple(row["agents"]) for row in incumbent}
        structpool_sets = {tuple(row["agents"]) for row in first}
        self.assertTrue(incumbent_sets <= structpool_sets)

        family_groups = {
            group for row in first for group in row["structpool_family_groups"]
        }
        self.assertEqual(
            family_groups,
            {
                "bottleneck_crossing",
                "conflict_component",
                "topology_boundary",
                "spatiotemporal_hotspot",
                "path_overlap",
            },
        )
        self.assertEqual({row["actual_size"] for row in first}, {8, 16, 24, 32})
        active_agents = {
            agent
            for event in analysis.events
            for agent in (event.left, event.right)
        }
        self.assertTrue(
            all(set(row["agents"]) & active_agents for row in first),
            "every explicit StructPool action must touch the conflict graph",
        )

    def test_structpool_reuses_state_level_candidate_inputs(self) -> None:
        state, analysis = self._structpool_state()
        with (
            patch.object(
                topology_candidates,
                "_conflict_component_seed_data",
                wraps=topology_candidates._conflict_component_seed_data,
            ) as component,
            patch.object(
                topology_candidates,
                "_hotspot_seed_data",
                wraps=topology_candidates._hotspot_seed_data,
            ) as hotspot,
            patch.object(
                topology_candidates,
                "_path_overlap_seed_data",
                wraps=topology_candidates._path_overlap_seed_data,
            ) as overlap,
            patch.object(
                topology_candidates,
                "_boundary_neighborhood",
                wraps=topology_candidates._boundary_neighborhood,
            ) as boundary,
            patch.object(
                topology_candidates,
                "topology_candidate_audit",
                wraps=topology_candidates.topology_candidate_audit,
            ) as audit,
        ):
            rows = generate_structpool_candidates(state, analysis)
        self.assertEqual(component.call_count, 1)
        self.assertEqual(hotspot.call_count, 1)
        self.assertEqual(overlap.call_count, 1)
        self.assertEqual(boundary.call_count, 8)
        self.assertLessEqual(audit.call_count, len(rows) + 4)

    def test_structpool_grid_keeps_all_sizes_and_support_provenance(self) -> None:
        state, analysis = self._structpool_state()
        grid = generate_structpool_candidate_grid(state, analysis)
        self.assertGreater(len(grid), 6)
        self.assertEqual(grid, generate_structpool_candidate_grid(state, analysis))
        observed = {
            int(family.rsplit(":", 1)[1])
            for row in grid
            for family in row["selection_families"]
        }
        self.assertEqual(observed, {8, 16, 24, 32})
        for row in grid:
            families = set(row["selection_families"])
            self.assertEqual(
                set(row["structpool_support_count_by_family"]), families
            )
            self.assertEqual(
                set(row["structpool_support_ratio_by_family"]), families
            )
            self.assertEqual(
                set(row["structpool_nominal_size_by_family"]), families
            )
            self.assertEqual(
                row["structpool_grid_pure_family"],
                len(row["structpool_family_groups"]) == 1,
            )
            self.assertEqual(
                row["structpool_grid_duplicate_provenance_count"],
                len(row["selection_families"]),
            )

    def test_structpool_runtime_subset_skips_omitted_path_overlap_work(self) -> None:
        state, analysis = self._structpool_state()
        full = generate_structpool_candidate_grid(state, analysis)
        with patch.object(
            topology_candidates,
            "_path_overlap_seed_data",
            wraps=topology_candidates._path_overlap_seed_data,
        ) as overlap:
            subset = generate_structpool_candidate_subset(
                state,
                analysis,
                family_sizes={
                    "conflict_component": (24, 32),
                    "topology_boundary_articulation": (16, 24, 32),
                },
            )
        self.assertEqual(overlap.call_count, 0)
        self.assertTrue(subset)
        full_sets = {tuple(row["agents"]) for row in full}
        self.assertTrue({tuple(row["agents"]) for row in subset} <= full_sets)
        self.assertTrue(all(row["structpool_runtime_subset"] for row in subset))

    def test_scalepool_orders_sizes_by_support_with_smaller_tie_break(self) -> None:
        self.assertEqual(scalepool_size_attempt_order(2), (8, 16, 24, 32))
        self.assertEqual(scalepool_size_attempt_order(12), (8, 16, 24, 32))
        self.assertEqual(scalepool_size_attempt_order(20), (16, 24, 8, 32))
        self.assertEqual(scalepool_size_attempt_order(29), (32, 24, 16, 8))

    def test_scalepool_is_lazy_deterministic_and_anchor_relative(self) -> None:
        state, analysis = self._structpool_state()
        anchor = [34, 35, 36, 37, 38, 39]
        first = generate_scalepool_candidates(
            state, analysis, v2_anchor_agents=anchor
        )
        second = generate_scalepool_candidates(
            state, analysis, v2_anchor_agents=anchor
        )
        self.assertEqual(first, second)
        self.assertLessEqual(len(first.candidates), 6)
        self.assertLess(first.raw_candidate_count, 24)
        self.assertTrue(first.attempts)
        selected_or_merged = [
            row for row in first.attempts
            if row["decision"] in {"selected", "merged"}
        ]
        self.assertEqual(
            len({row["family_variant"] for row in selected_or_merged}),
            len(selected_or_merged),
        )
        for row in first.candidates:
            self.assertLessEqual(row["v2_anchor_jaccard"], 0.9)
            families = set(row["selection_families"])
            self.assertEqual(
                set(row["structpool_support_count_by_family"]), families
            )
            self.assertEqual(
                set(row["scalepool_size_attempt_order_by_family"]), families
            )
        for index, left in enumerate(first.candidates):
            for right in first.candidates[index + 1 :]:
                intersection = set(left["agents"]) & set(right["agents"])
                union = set(left["agents"]) | set(right["agents"])
                self.assertLessEqual(len(intersection) / len(union), 0.8)

    def test_structpool_novel_additions_obey_jaccard_filter(self) -> None:
        state, analysis = self._structpool_state()
        rows = generate_structpool_candidates(state, analysis)
        incumbent = {
            tuple(row["agents"])
            for row in generate_topology_boundary_candidates(
                state, analysis, neighborhood_size=16, core_budget=4
            )
        }
        novel = [row for row in rows if tuple(row["agents"]) not in incumbent]
        for index, left in enumerate(novel):
            for right in novel[index + 1 :]:
                left_agents = set(left["agents"])
                right_agents = set(right["agents"])
                similarity = len(left_agents & right_agents) / len(
                    left_agents | right_agents
                )
                self.assertLessEqual(similarity, 0.8)

    def test_structpool_merge_keeps_frozen_base_rows_exact(self) -> None:
        base = [
            {
                "candidate_id": "base-a",
                "agents": [0, 1],
                "actual_size": 2,
                "selection_families": ["target:4"],
                "selection_rank_by_family": {"target:4": 0},
                "proposal_count_by_family": {"target:4": 1},
                "proposal_seeds": [5],
                "seed_agents": [0],
            }
        ]
        duplicate = {**base[0], "selection_families": ["structpool-path-overlap:8"]}
        addition = {
            **base[0],
            "candidate_id": "added",
            "agents": [2, 3],
            "selection_families": ["structpool-conflict-component:8"],
        }
        merged = merge_structpool_candidates(base, [duplicate, addition])
        self.assertEqual(merged[0], base[0])
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[1]["agents"], [2, 3])

    def test_structpool_rejects_protocol_drift(self) -> None:
        state, analysis = self._structpool_state()
        with self.assertRaisesRegex(ValueError, "requires sizes"):
            generate_structpool_candidates(
                state, analysis, neighborhood_sizes=(8, 16)
            )
        with self.assertRaisesRegex(ValueError, "six-candidate"):
            generate_structpool_candidates(
                state, analysis, maximum_added_candidates=7
            )


if __name__ == "__main__":
    unittest.main()
