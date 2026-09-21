import copy
import unittest

from scripts import audit_sa_tail_membership as a


def fixture():
    agents = [dict(id=20, path=[4, 5, 4]), dict(id=30, path=[9, 5, 1]),
              dict(id=2, path=[10, 6]), dict(id=7, path=[8, 9])]
    state = dict(agents=agents, rows=3, cols=4, conflict_edges=[[20, 30]], num_of_colliding_pairs=1)
    pool = [dict(candidate_id="selected", agents=[2, 20, 30], score=2, selection_families=["random:3"]),
            dict(candidate_id="other", agents=[7, 20, 30], score=1, selection_families=["random:3"])]
    event = dict(decision=1, action=dict(agents=[2, 20, 30]),
                 metrics=dict(neighborhood=[30, 2, 20], replan_success=True), pool=pool, selected_index=0)
    return state, event


class TailMembershipTest(unittest.TestCase):
    def test_contacts_distinguish_filler_outsider_and_end_wait(self):
        state, event = fixture()
        row = a.audit_decision(state, state, event)
        self.assertEqual(row["near_fillers"], [2])
        self.assertEqual(row["parked_near_fillers"], [2])
        self.assertEqual(row["near_outsiders"], [7])
        self.assertEqual(row["active_members"], [20, 30])
        self.assertEqual(len(row["same_size_full_coverage_alternatives"]), 1)

    def test_no_row_wrap(self):
        self.assertEqual(a.closed_cross(3, 3, 4), {2, 3, 7})

    def test_reject_unknown_or_modified_action(self):
        state, event = fixture()
        event["metrics"]["neighborhood"] = [2, 7, 20]
        with self.assertRaises(ValueError):
            a.audit_decision(state, state, event)

    def test_reject_conflict_mismatch(self):
        state, event = fixture()
        state["conflict_edges"] = []
        with self.assertRaises(ValueError):
            a.audit_decision(state, state, event)

    def test_reject_outsider_path_change(self):
        state, event = fixture()
        after = copy.deepcopy(state)
        after["agents"][-1]["path"] = [8, 4, 5, 9]
        with self.assertRaises(ValueError):
            a.audit_decision(state, after, event)

    def test_low_id_fill_with_noncontinuous_ids(self):
        state, event = fixture()
        for i in range(100, 114):
            state["agents"].append(dict(id=i, path=[i]))
        state.update(rows=40, cols=4)
        members = sorted({20, 30, 2, 7} | set(range(100, 112)))
        event["action"]["agents"] = members
        event["metrics"]["neighborhood"] = members
        event["pool"][0]["agents"] = members
        row = a.audit_decision(state, state, event)
        self.assertTrue(row["exact_low_id_fill"])
        self.assertEqual(row["filler_count"], 14)

    def test_input_order_invariance_and_no_mutation(self):
        state, event = fixture()
        before = copy.deepcopy((state, event))
        row = a.audit_decision(state, state, event)
        self.assertEqual((state, event), before)
        state["agents"].reverse()
        self.assertEqual(row, a.audit_decision(state, state, event))

    def test_changed_filler_is_not_called_useless(self):
        state, event = fixture()
        after = copy.deepcopy(state)
        after["agents"][2]["path"] = [10, 10, 6]
        row = a.audit_decision(state, after, event)
        self.assertEqual(row["changed_fillers"], [2])
        self.assertNotIn("useless_fillers", row)

    def test_summary_uses_branch_and_decision_denominators(self):
        state, event = fixture()
        row = a.audit_decision(state, state, event)
        branch = dict(job_id="j", state_id="s", map_id="m", arm="anchor", trial=0,
                      stop="horizon", final_conflicts=1, decisions=[row, row])
        report = a.summarize([branch])
        self.assertEqual(report["groups"]["anchor/all"]["branches"], 1)
        self.assertEqual(report["groups"]["anchor/all"]["tail_decisions"], 2)
        self.assertEqual(report["low_residual_branches"][0]["unique_path_states"], 1)
        self.assertFalse(report["training_allowed"])

    def test_terminal_wait_exposure_is_not_a_direct_conflict(self):
        state, event = fixture()
        state["agents"][0]["path"] = [4, 4, 5, 4]
        state["agents"][1]["path"] = [9, 5, 5, 1]
        row = a.audit_decision(state, state, event)
        self.assertEqual(row["parked_near_fillers"], [2])
        self.assertEqual(row["pairs"], [[20, 30]])

    def test_existing_evidence_matches_saved_summary(self):
        path = a.ROOT / "artifacts/sa-tail-membership-audit-v1/evidence.json"
        if not path.exists() or not (a.OUTPUT / "report.json").exists():
            self.skipTest("local sealed membership audit not available")
        evidence = a.read_json(path)
        self.assertEqual(a.sha256_file(a.ROOT / evidence["report"]), evidence["report_sha256"])
        self.assertEqual(a.sha256_file(a.ROOT / evidence["rows"]), evidence["rows_sha256"])
        report = a.read_json(a.ROOT / evidence["report"])
        self.assertEqual(evidence["source_binding"], report["source_binding"])
        groups = [report["summary"]["groups"][arm + "/all"] for arm in ("anchor", "alternate")]
        fields = {"tail_decisions": "tail_decisions", "fill_applicable_decisions": "fill_applicable",
                  "structural_low_id_fill": "structural_exact_low_id_fill",
                  "same_size_full_coverage_alternative_available": "with_same_size_coverage_alternative",
                  "near_outsider_present": "with_near_outsider", "near_filler_present": "with_near_filler",
                  "filler_slots": "filler_slots", "spatial_overlap_filler_slots": "spatial_overlap_filler_slots",
                  "changed_filler_slots": "changed_filler_slots"}
        for key, field in fields.items():
            self.assertEqual(evidence[key], sum(g[field] for g in groups), key)
        self.assertEqual(report["summary"], a.summarize(a.read_json(a.ROOT / evidence["rows"])))


if __name__ == "__main__":
    unittest.main()
