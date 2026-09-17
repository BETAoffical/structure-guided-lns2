import copy
from pathlib import Path
import tempfile
import unittest

from experiments._common import read_json
from experiments.sa_continuation_value import select_roots, best, state_summary, summarize
from scripts.run_sa_continuation_value import record, read_record, branch_jobs


def fixture():
    return dict(frozen={"a":[1]*8, "b":[0]*8, "c":[0]*8, "d":[0]*8},
                paired={"a":[0]*8, "b":[1]*8, "c":[0]*8, "d":[0]*8})


class ContinuationValueTests(unittest.TestCase):
    def test_selection_outcome_blind_and_order_invariant(self):
        rows = [dict(map_id=f"m{i}", phase=p, id=f"{i}-{p}-{j}", decision=4 if p == "early" else 16,
                     agents=300, conflicts=4, completed=j%2) for i in range(8) for p in ("early", "continuing") for j in range(2)]
        ranges = dict(decision=[4,128], agents=[223,484], conflicts=[1,200])
        selected = select_roots(rows, ranges, 2026, set())
        changed = [dict(r, completed=1-r["completed"]) for r in rows[::-1]]
        self.assertEqual(selected, select_roots(changed, ranges, 2026, set()))
        self.assertEqual(sum("early" in s for s in selected), 4)
        self.assertNotIn(selected[0], select_roots(rows, ranges, 2026, {selected[0]}))

    def test_missing_supported_map_does_not_relax(self):
        rows = [dict(map_id=f"m{i}", phase="early" if i%2==0 else "continuing", id=str(i), decision=0,
                     agents=300, conflicts=4) for i in range(8)]
        with self.assertRaises(ValueError):
            select_roots(rows, dict(decision=[4,128], agents=[223,484], conflicts=[1,200]), 1, set())

    def test_stable_reversal_requires_both_halves(self):
        r = state_summary(fixture(), "a", "b")
        self.assertTrue(r["complete"])
        self.assertEqual(len(r["stable_reversals"]), 1)
        self.assertEqual(r["best_jaccard"], 0)
        self.assertEqual(r["cross_half_gain"], 1)
        v = fixture()
        v["frozen"]["a"] = [1]*4+[0]*4
        v["paired"]["b"] = [1]*4+[0]*4
        noisy = state_summary(v, "a", "b")
        self.assertTrue(any(p["reversed"] for p in noisy["pairs"]))
        self.assertFalse(noisy["stable_reversals"])

    def test_all_ties_are_not_reversals(self):
        values = {a:{c:[1]*8 for c in "abcd"} for a in ("frozen", "paired")}
        r = state_summary(values, "b", "a")
        self.assertEqual(r["best_jaccard"], 1)
        self.assertFalse(r["stable_reversals"])
        self.assertEqual(r["cross_half_gain"], 0)
        self.assertEqual(best({"a":1,"b":1}, "b"), "b")

    def test_holdout_values_do_not_select_fit_half_candidate(self):
        v = fixture()
        baseline = state_summary(v, "a", "b")["cross_half"][0]
        for a in v:
            for c in v[a]:
                v[a][c][4:] = [1-x for x in v[a][c][4:]]
        altered = state_summary(v, "a", "b")["cross_half"][0]
        for key in ("frozen_label_choice", "paired_label_choice"):
            self.assertEqual(baseline[key], altered[key])
        self.assertEqual(altered["gain"], -baseline["gain"])

    def test_censoring_is_not_zero_or_filtered(self):
        v = fixture()
        v["paired"]["a"][0] = None
        row = state_summary(v, "a", "b")
        self.assertFalse(row["complete"])
        self.assertIsNone(row["rates"])
        rows = [dict(state_summary(fixture(), "a", "b"), map_id=str(i)) for i in range(8)]
        rows[0] = dict(row, map_id="0")
        self.assertEqual(summarize(rows, 1, 100)["decision"], "incomplete_evidence_resource_censoring")

    def test_trial_grid_and_candidate_identity(self):
        v = fixture()
        v["paired"]["a"].pop()
        with self.assertRaises(ValueError):
            state_summary(v, "a", "b")
        v = fixture()
        del v["paired"]["b"]
        with self.assertRaises(ValueError):
            state_summary(v, "a", "b")

    def test_maps_are_units_not_trial_rows(self):
        rows = [dict(state_summary(fixture(), "a", "b"), map_id=str(i)) for i in range(8)]
        r = summarize(rows, 2, 1000)
        self.assertEqual(r, summarize(rows[::-1], 2, 1000))
        self.assertEqual(r["decision"], "repeatable_candidate_order_shift")
        self.assertFalse(r["promotion"])
        self.assertEqual(r["cross_half_ci95"], [1,1])
        rows[-1]["map_id"] = "0"
        with self.assertRaises(ValueError):
            summarize(rows, 2, 1000)

    def test_atomic_receipt_tampering_and_resume(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"row.json"
            record(path, dict(binding="b", x=1))
            record(path, dict(binding="b", x=1))
            self.assertEqual(read_record(path, {"binding":"b"})["x"], 1)
            with self.assertRaises(ValueError):
                read_record(path, {"binding":"wrong"})
            path.write_text(path.read_text().replace('"x": 1', '"x": 2'), encoding="utf8")
            with self.assertRaises(ValueError):
                read_record(path, {"binding":"b"})

    def test_schedule_preserves_all_candidates_trials(self):
        entry = dict(target=dict(anchor_id="a", selected=list("abcd")))
        jobs = branch_jobs(entry)
        self.assertEqual(len(jobs), 33)
        self.assertEqual(jobs[0], ("control.json", "a", 0, "frozen"))
        self.assertEqual({(c,t) for _,c,t,_ in jobs[1:]}, {(c,t) for c in "abcd" for t in range(8)})

    def test_config_has_no_training_or_promotion(self):
        cfg = read_json(Path(__file__).resolve().parents[2]/"configs/sa_continuation_value.json")
        self.assertFalse(cfg["training"] or cfg["formal_ttf"] or cfg["automatic_promotion"])
        self.assertEqual((cfg["roots"],cfg["trials"],cfg["candidates"],cfg["horizon"],cfg["workers"]), (8,8,4,32,20))


if __name__ == "__main__":
    unittest.main()
