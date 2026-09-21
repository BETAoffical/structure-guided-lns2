import tempfile
from pathlib import Path
import unittest

from scripts import audit_sa_member_contract as a


def candidate(cid, offset, family="target:16", size=16):
    return dict(candidate_id=cid, agents=list(range(offset, offset + size)),
                actual_size=size, selection_families=[family], score=999)


def fixture():
    pool = [candidate("a", 0), candidate("b", 20), candidate("c", 40, "random:16"), candidate("d", 60, "random:16")]
    entry = dict(state_id="s", map_id="m", decision=16, anchor_id="a", agent_ids=list(range(100)))
    root = dict(old_selected_id="a", source=dict(id="s", map_id="m", decision=16),
                control_event=dict(pool=pool, decision=16, selected_index=0, action=dict(agents=pool[0]["agents"])))
    def group(kind, ids):
        return dict(group_id=kind, kind=kind, state_id="s", map_id="m", family="random:16",
                    trial_keys=[kind+str(t) for t in range(8)],
                    candidates=[dict(candidate_id=c["candidate_id"], agents=c["agents"], values=[False]*8)
                                for c in pool if c["candidate_id"] in ids])
    return entry, root, [group("old_grid", ["a", "b", "c", "d"]), group("matched_pair", ["c", "d"])]


class MemberContractTests(unittest.TestCase):
    def test_matched_pair_is_not_automatically_an_anchor_comparison(self):
        row = a.scope_row(*fixture())
        self.assertTrue(row["eligible"])
        self.assertEqual(row["same_signature_alternatives"], ["b"])
        self.assertFalse(row["matched_pair_contains_anchor"])
        self.assertEqual(row["indexed_labeled_alternatives"], ["b"])
        self.assertEqual([p["kind"] for p in row["indexed_pairs"]], ["old_grid"])

    def test_scope_does_not_use_scores_outcomes_or_future_fields(self):
        data = fixture()
        first = a.scope_row(*data)
        for c in data[1]["control_event"]["pool"]:
            c.update(score=-999, future_success=True, generated_nodes=100000)
        for g in data[2]:
            for c in g["candidates"]:
                c["values"] = [True]*8
        self.assertEqual(first, a.scope_row(*data))

    def test_exact_signature_not_overlap_and_actual_not_requested_size(self):
        for mutation in ("overlap", "size"):
            entry, root, groups = fixture()
            c = root["control_event"]["pool"][1]
            if mutation == "overlap":
                c["selection_families"].append("collision:16")
            else:
                c["agents"] = c["agents"][:-1]
                c["actual_size"] = 15
                groups[0]["candidates"][1]["agents"] = c["agents"]
            row = a.scope_row(entry, root, groups)
            self.assertFalse(row["eligible"])
            self.assertEqual(row["indexed_labeled_alternatives"], [])

    def test_structural_singleton_is_not_an_alternative(self):
        entry, root, groups = fixture()
        root["control_event"]["pool"][0]["selection_families"] = ["structpool-conflict-component:16", "structpool-spatiotemporal-hotspot:16"]
        row = a.scope_row(entry, root, groups)
        self.assertTrue(row["structural_anchor"])
        self.assertFalse(row["eligible"])

    def test_no_cross_group_pseudo_pair(self):
        entry, root, groups = fixture()
        e = candidate("e", 80, "collision:16")
        root["control_event"]["pool"].append(e)
        groups[0]["candidates"][1] = dict(candidate_id="e", agents=e["agents"], values=[False]*8)
        # b exists online, but there is no group with both a and b outcomes.
        row = a.scope_row(entry, root, groups)
        self.assertTrue(row["eligible"])
        self.assertEqual(row["indexed_labeled_alternatives"], [])
        self.assertEqual(row["unlabeled_alternatives"], ["b"])

    def test_pool_and_group_order_do_not_change_scope(self):
        entry, root, groups = fixture()
        first = a.scope_row(entry, root, groups)
        root["control_event"]["pool"].reverse()
        root["control_event"]["selected_index"] = 3
        groups.reverse()
        for g in groups:
            g["candidates"].reverse()
        self.assertEqual(first, a.scope_row(entry, root, groups))

    def test_unknown_members_incomplete_labels_and_duplicate_ids_rejected(self):
        mutations = ("unknown", "censored", "duplicate", "size", "action", "identity", "trials")
        for mode in mutations:
            entry, root, groups = fixture()
            if mode == "unknown": root["control_event"]["pool"][0]["agents"][0] = 200
            if mode == "censored": groups[0]["candidates"][0]["values"][0] = None
            if mode == "duplicate": root["control_event"]["pool"][1]["candidate_id"] = "a"
            if mode == "size": root["control_event"]["pool"][0]["actual_size"] = 15
            if mode == "action": root["control_event"]["action"]["agents"] = [90]
            if mode == "identity": groups[0]["map_id"] = "other"
            if mode == "trials": groups[0]["trial_keys"] = ["same"]*8
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                a.scope_row(entry, root, groups)

    def test_denominators_include_no_alternative_states(self):
        entry, root, groups = fixture()
        first = a.scope_row(entry, root, groups)
        root["control_event"]["pool"][0]["selection_families"] = ["structpool-conflict-component:16"]
        second = a.scope_row(entry, root, groups)
        result = a.summarize([first, second])
        self.assertEqual(result["roots"], 2)
        self.assertEqual(result["eligible_roots"], 1)
        self.assertEqual(result["no_alternative_roots"], 1)

    def test_sha_drift_and_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "input.json"
            path.write_text('{"a": 1}', encoding="utf-8")
            digest = a.sha256_file(path)
            inputs = {}
            self.assertEqual(a.load_pinned(root, "input.json", digest, inputs), dict(a=1))
            path.write_text('{"a": 2}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "drift"):
                a.load_pinned(root, "input.json", digest, inputs)
            with self.assertRaises(ValueError):
                a.load_pinned(root, "../outside.json", digest, inputs)


if __name__ == "__main__":
    unittest.main()
