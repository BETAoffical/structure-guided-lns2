import copy
import json
from types import SimpleNamespace
import unittest

from experiments.state_analysis import ConflictEvent, analyze_state
from lns2_selector.runtime import topology_candidates as topology
from scripts import probe_sa_structural_focus as p


def event(t, a, b, cell):
    return ConflictEvent(t, "vertex", a, b, (cell,))


def two_components():
    return SimpleNamespace(events=[event(1, 10, 20, 1), event(1, 30, 40, 11)],
                           component_members={10: {10, 20}, 30: {30, 40}, 99: {99}},
                           component_id={10: 10, 20: 10, 30: 30, 40: 30, 99: 99})


def state_and_pool():
    agents = [dict(id=a, path=path, start=path[0], goal=path[-1], conflict_degree=1)
              for a, path in ((10, [0, 1, 2]), (20, [2, 1, 0]), (30, [10, 11, 12]), (40, [12, 11, 10]))]
    agents += [dict(id=100+i, path=[20+i], start=20+i, goal=20+i, conflict_degree=0) for i in range(20)]
    state = dict(rows=1, cols=50, obstacles=[0]*50, agents=agents,
                 conflict_edges=[[10, 20], [30, 40]], num_of_colliding_pairs=2)
    analysis = analyze_state(state)
    pool = {}
    for family, seed_data in zip(p.FAMILIES, (topology._conflict_component_seed_data(analysis), topology._hotspot_seed_data(analysis))):
        members = topology._ranked_seed_neighborhood(state, analysis, seed_data[0], size=16, priority=seed_data[1])
        cid = p.candidate_id(members)
        row = pool.setdefault(cid, dict(candidate_id=cid, agents=members, actual_size=16, selection_families=[]))
        row["selection_families"].append(family)
    return state, list(pool.values())


class StructuralFocusTests(unittest.TestCase):
    def test_second_active_component_and_primary_tie_rule(self):
        cores = p.two_cores(two_components())[p.FAMILIES[0]]
        self.assertEqual([c["seeds"] for c in cores], [[10, 20], [30, 40]])
        self.assertNotIn(99, [a for c in cores for a in c["seeds"]])

    def test_hotspot_repeated_support_is_not_a_second_core(self):
        a = two_components()
        a.events += [event(5, 10, 20, 2), event(9, 10, 20, 3)]
        cores = p.two_cores(a)[p.FAMILIES[1]]
        self.assertEqual([c["seeds"] for c in cores], [[10, 20], [30, 40]])
        self.assertEqual([c["key"] for c in cores], [[0, 1], [0, 11]])

    def test_only_one_active_component_has_no_second(self):
        a = two_components()
        a.events = a.events[:1]
        cores = p.two_cores(a)
        self.assertEqual([len(cores[f]) for f in p.FAMILIES], [1, 1])

    def test_event_order_does_not_change_core_choice(self):
        a = two_components()
        first = p.two_cores(a)
        a.events.reverse()
        a.component_members = dict(reversed(list(a.component_members.items())))
        self.assertEqual(first, p.two_cores(a))

    def test_no_future_or_score_fields_in_action_input(self):
        state, _ = state_and_pool()
        first = p.action_input(state)
        changed = copy.deepcopy(state)
        changed.update(future_path=[999], H32=True, runtime=9999, outcome=dict(success=True), score=999)
        for agent in changed["agents"]:
            agent.update(future_delay=999, repair_success=True)
        self.assertEqual(first, p.action_input(changed))

    def test_primary_reproduction_and_source_immutability(self):
        state, pool = state_and_pool()
        before = copy.deepcopy((state, pool))
        result = p.materialize(state, pool, pool[0]["candidate_id"], [])
        self.assertEqual((state, pool), before)
        self.assertEqual(len(result["primary"]), 2)
        self.assertEqual(len(result["attempts"]), len(pool[0]["selection_families"]))
        self.assertTrue(all(len(r["agents"]) == 16 for r in result["primary"].values()))

    def test_conflict_mismatch_and_unknown_members_rejected(self):
        state, pool = state_and_pool()
        state["num_of_colliding_pairs"] = 1
        with self.assertRaisesRegex(ValueError, "reconstruction"):
            p.materialize(state, pool, pool[0]["candidate_id"], [])
        state, pool = state_and_pool()
        pool[0]["agents"][0] = 999
        with self.assertRaisesRegex(ValueError, "agents"):
            p.materialize(state, pool, pool[0]["candidate_id"], [])

    def test_nonstructural_anchor_never_gets_a_hidden_intervention(self):
        state, pool = state_and_pool()
        anchor = dict(candidate_id=p.candidate_id([100+i for i in range(16)]),
                      agents=[100+i for i in range(16)], actual_size=16, selection_families=["random:16"])
        result = p.materialize(state, pool+[anchor], anchor["candidate_id"], [])
        self.assertEqual(result["attempts"], [])
        self.assertFalse(result["structural_anchor"])

    def test_failed_primary_reproduction_stops(self):
        state, pool = state_and_pool()
        pool[0]["agents"][-1] = 119
        pool[0]["candidate_id"] = p.candidate_id(pool[0]["agents"])
        with self.assertRaisesRegex(ValueError, "not reproduced"):
            p.materialize(state, pool, pool[0]["candidate_id"], [])

    def test_two_cores_may_fill_to_same_members_without_searching_a_third(self):
        state, pool = state_and_pool()
        result = p.materialize(state, pool, pool[0]["candidate_id"], [])
        self.assertTrue(all(a["status"] == "same_after_fill" for a in result["attempts"]))
        self.assertEqual(p.summarize([result | dict(map_id="m")])["roots_with_different_members"], 0)

    def test_large_disjoint_cores_produce_one_deduplicated_new_set(self):
        # A synthetic already-conflicted state tests generation, not MAPF reset validity.
        agents = [dict(id=10*i, path=[1 if i < 20 else 40], conflict_degree=19) for i in range(40)]
        edges = [[a["id"], b["id"]] for i, a in enumerate(agents) for b in agents[i+1:] if a["path"] == b["path"]]
        state = dict(rows=1, cols=50, obstacles=[0]*50, agents=agents,
                     conflict_edges=edges, num_of_colliding_pairs=len(edges))
        anchor, alternate = [10*i for i in range(16)], [10*i for i in range(20, 36)]
        pool = [dict(candidate_id=p.candidate_id(anchor), agents=anchor, actual_size=16, selection_families=list(p.FAMILIES))]
        result = p.materialize(state, pool, pool[0]["candidate_id"], [])
        self.assertEqual(json.loads(json.dumps(result)), result)
        self.assertEqual([a["agents"] for a in result["attempts"]], [alternate, alternate])
        summary = p.summarize([result | dict(map_id="m")])
        self.assertEqual(summary["unique_root_alternates"], 1)
        self.assertEqual(summary["new_to_pool_attempts"], 2)
        self.assertEqual(summary["labeled_with_anchor_attempts"], 0)
        alt = dict(candidate_id=p.candidate_id(alternate), agents=alternate, actual_size=16, selection_families=["random:16"])
        existing = p.materialize(state, pool+[alt], pool[0]["candidate_id"], [[pool[0]["candidate_id"], alt["candidate_id"]]])
        summary = p.summarize([existing | dict(map_id="m")])
        self.assertEqual(summary["already_in_pool_attempts"], 2)
        self.assertEqual(summary["labeled_with_anchor_attempts"], 2)


if __name__ == "__main__":
    unittest.main()
