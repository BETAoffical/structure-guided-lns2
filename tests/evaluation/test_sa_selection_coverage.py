import copy
from pathlib import Path
import unittest

from scripts.audit_sa_selection_coverage import (
    COVERAGE, checked_path, grouped_stats, inspect_event, label_comparisons,
)


def fixture():
    def features(size, coverage):
        return dict({k: coverage for k in COVERAGE}, **{
            "proposal.actual_size": size, "state.colliding_pairs": 10.})
    return dict(decision=0, before="fp", pool=[
        dict(candidate_id="unused", agents=[20]),
        dict(candidate_id="anchor", agents=[2, 7]),
        dict(candidate_id="selected", agents=[5, 9])],
        subset=["selected", "anchor"], features=[features(2, .2), features(2, .6)],
        selected_index=2, anchor_id="anchor", ranking=dict(selected="selected"),
        action=dict(agents=[5, 9]))


class SelectionCoverageTests(unittest.TestCase):
    def test_full_pool_index_and_subset_feature_alignment(self):
        r = inspect_event(fixture())
        self.assertEqual(r["selected"], "selected")
        self.assertEqual(r["size_relation"], "equal")
        self.assertTrue(r["changed"])
        self.assertAlmostEqual(r["coverage_delta"][COVERAGE[0]], -.4)
        self.assertEqual(r["nonanchor_not_lower_internal_coverage"], 0)

    def test_permutation_does_not_change_result(self):
        f = fixture()
        expected = inspect_event(f)
        f["subset"].reverse()
        f["features"].reverse()
        f["pool"].reverse()
        f["selected_index"] = 0
        self.assertEqual(expected, inspect_event(f))

    def test_outcomes_and_scores_not_used(self):
        f = fixture()
        expected = inspect_event(f)
        f["metrics"] = dict(colliding_pairs=0, generated=999)
        f["ranking"]["scores"] = {"selected": -999, "anchor": 888}
        self.assertEqual(expected, inspect_event(f))

    def test_rejects_altered_membership_features_and_selection(self):
        for mutate in (
            lambda f: f.update(selected_index=0),
            lambda f: f["ranking"].update(selected="anchor"),
            lambda f: f["action"].update(agents=[2, 7]),
            lambda f: f["features"][0].update({"proposal.actual_size": 3}),
            lambda f: f["features"][0].update({COVERAGE[0]: float("nan")}),
            lambda f: f["features"][0].update({"state.colliding_pairs": 11}),
        ):
            f = fixture()
            mutate(f)
            with self.assertRaises(ValueError):
                inspect_event(f)

    def test_subgroups_and_empty_means_are_explicit(self):
        r = dict(inspect_event(fixture()), job_id="j", map_id="m")
        groups = grouped_stats([r])
        self.assertEqual(groups["changed_same_size"]["decisions"], 1)
        self.assertEqual(groups["changed_smaller"]["decisions"], 0)
        self.assertIsNone(groups["changed_smaller"]["coverage"][COVERAGE[0]]["mean_delta"])
        self.assertEqual(groups["first_changed_same_size"]["maps"], 1)

    def dataset(self):
        f = fixture()
        candidates = []
        for c in f["pool"][1:]:
            cid = c["candidate_id"]
            candidates.append(dict(c, features=f["features"][f["subset"].index(cid)],
                trials=[dict(trial=i, completed=(i % 2 == 0) if cid == "selected" else False)
                        for i in range(8)]))
        return dict(states=[dict(state_id="s", map_id="m", anchor_id="anchor", candidates=candidates)])

    def test_label_halves_and_order(self):
        data = self.dataset()
        expected = label_comparisons(data)
        r = expected["comparisons"][0]
        self.assertEqual(r["completion_delta"], .5)
        self.assertEqual(r["half_deltas"], [.5, .5])
        self.assertEqual(r["both_halves"], "win")
        self.assertEqual(expected["same_size"]["lower"]["map_state_equal_mean_delta"], .5)
        for c in data["states"][0]["candidates"]:
            c["trials"].reverse()
        data["states"][0]["candidates"].reverse()
        self.assertEqual(expected, label_comparisons(data))

    def test_a_positive_aggregate_need_not_agree_between_halves(self):
        data = self.dataset()
        c = data["states"][0]["candidates"][1]
        c["trials"] = [dict(trial=i, completed=i < 4) for i in range(8)]
        self.assertEqual(label_comparisons(data)["comparisons"][0]["both_halves"], "other")

    def test_rejects_trial_gaps_duplicates_and_nonboolean_labels(self):
        for mutate in (
            lambda t: t.pop(), lambda t: t[0].update(trial=1),
            lambda t: t[0].update(completed=1),
        ):
            data = copy.deepcopy(self.dataset())
            mutate(data["states"][0]["candidates"][0]["trials"])
            with self.assertRaises(ValueError):
                label_comparisons(data)

    def test_manifest_cannot_escape_source(self):
        base = Path(__file__).resolve().parent
        self.assertEqual(checked_path(base, "foo"), base / "foo")
        with self.assertRaises(ValueError):
            checked_path(base, "../foo")


if __name__ == "__main__":
    unittest.main()
