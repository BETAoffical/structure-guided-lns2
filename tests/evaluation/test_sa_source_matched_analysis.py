import copy
from itertools import combinations
import json
import random
import unittest

from experiments.sa_source_matched_analysis import (
    BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED, analyze_record, partitions, summarize,
)


def record(a=None, b=None, anchor=None, state="s", map_id="m"):
    return dict(state_id=state, map_id=map_id, pair_ids=["a", "b"], anchor_id="z",
                values=dict(a=[True]*8 if a is None else a,
                            b=[False]*8 if b is None else b,
                            z=[False]*8 if anchor is None else anchor))


class SourceMatchedAnalysisTests(unittest.TestCase):
    def test_all_unordered_partitions_once(self):
        splits = partitions()
        self.assertEqual(len(splits), 35)
        halves = set()
        for left, right in splits:
            self.assertEqual(len(left), 4)
            self.assertEqual(len(right), 4)
            self.assertIn(0, left)
            self.assertFalse(set(left) & set(right))
            self.assertEqual(set(left) | set(right), set(range(8)))
            halves.update((left, right))
        self.assertEqual(halves, set(combinations(range(8), 4)))

    def test_perfect_contrast_and_anchor_separation(self):
        row = analyze_record(record())
        self.assertEqual(row["all_trial_difference"], 1.)
        self.assertEqual(row["all_trial_difference_bounds"], [1., 1.])
        self.assertEqual(row["crossfit_gain"], .5)
        self.assertEqual(row["half_direction_counts"]["strict_same_direction"], 35)
        changed = analyze_record(record(anchor=[True]*8))
        self.assertEqual(changed["partitions"], row["partitions"])
        self.assertEqual(changed["anchor_minus_pair_uniform"], .5)
        self.assertEqual(row["anchor_minus_pair_uniform"], -.5)

    def test_ties_are_retained_and_use_candidate_id(self):
        row = analyze_record(record(b=[True]*8))
        self.assertEqual(row["crossfit_gain"], 0.)
        self.assertEqual(row["half_direction_counts"]["both_tied"], 35)
        for split in row["partitions"]:
            for direction in ("left_to_right", "right_to_left"):
                self.assertTrue(split[direction]["selection_tied"])
                self.assertEqual(split[direction]["selected"], "a")

    def test_noise_can_have_negative_crossfit_despite_all_trial_tie(self):
        row = analyze_record(record([True]*4+[False]*4, [False]*4+[True]*4))
        self.assertEqual(row["all_trial_difference"], 0.)
        self.assertAlmostEqual(row["crossfit_gain"], -9/70)
        self.assertEqual(row["half_direction_counts"], dict(
            strict_same_direction=0, strict_opposite_direction=17, both_tied=18, one_tied=0))

    def test_selection_does_not_use_evaluation_half(self):
        before = record([True]*4+[False]*4, [False]*4+[True]*4)
        after = copy.deepcopy(before)
        after["values"]["a"][4:] = [True]*4
        after["values"]["b"][4:] = [False]*4
        first = analyze_record(before)["partitions"][0]["left_to_right"]
        second = analyze_record(after)["partitions"][0]["left_to_right"]
        self.assertEqual(first["selected"], second["selected"])
        self.assertEqual(first["selection_rates"], second["selection_rates"])
        self.assertEqual((first["gain"], second["gain"]), (-.5, .5))

    def test_matches_independent_seventy_direction_reference(self):
        source = record([True, False, True, True, False, False, True, False],
                        [False, True, False, True, False, True, False, False])
        gains = []
        for train in combinations(range(8), 4):
            test = sorted(set(range(8))-set(train))
            values = source["values"]
            chosen = min(("a", "b"), key=lambda cid: (-sum(values[cid][i] for i in train), cid))
            gains.append(sum(values[chosen][i] for i in test)/4 -
                         sum(values[cid][i] for cid in ("a", "b") for i in test)/8)
        self.assertAlmostEqual(analyze_record(source)["crossfit_gain"], sum(gains)/70)

    def test_pair_order_and_input_mutation(self):
        source = record()
        saved = copy.deepcopy(source)
        expected = analyze_record(source)
        self.assertEqual(source, saved)
        source["pair_ids"].reverse()
        self.assertEqual(analyze_record(source), expected)

    def test_anchor_can_be_pair_member(self):
        source = record()
        source["anchor_id"] = "b"
        del source["values"]["z"]
        row = analyze_record(source)
        self.assertEqual(row["crossfit_gain"], .5)
        self.assertEqual(row["anchor_minus_pair_uniform"], -.5)

    def test_pair_censoring_not_imputed_or_split_filtered(self):
        source = record([None]+[True]*7)
        row = analyze_record(source)
        self.assertFalse(row["pair_complete"])
        self.assertEqual(row["partitions"], [])
        self.assertEqual(row["partition_count"], 0)
        self.assertIsNone(row["crossfit_gain"])
        self.assertIsNone(row["half_direction_counts"])
        self.assertIsNone(row["all_trial_difference"])
        self.assertEqual(row["all_trial_difference_bounds"], [.875, 1.])
        self.assertEqual(row["candidates"]["a"]["censored"], 1)

    def test_paired_trial_counts_include_all_ties_and_unknowns(self):
        row = analyze_record(record(
            [True, False, True, False, None, True, None, False],
            [False, True, True, False, True, None, None, False]))
        self.assertEqual(row["paired_counts"], dict(
            a_wins=1, b_wins=1, both_success=1, both_failure=2, unknown=3))
        self.assertEqual(sum(row["paired_counts"].values()), 8)
        self.assertIsNone(row["crossfit_gain"])

    def test_equal_marginal_rates_are_not_paired_trial_ties(self):
        row = analyze_record(record([True, False]*4, [False, True]*4))
        self.assertEqual(row["paired_counts"], dict(
            a_wins=4, b_wins=4, both_success=0, both_failure=0, unknown=0))
        self.assertEqual(row["all_trial_difference"], 0.)
        counts = analyze_record(record())["paired_counts"]
        self.assertEqual((counts["a_wins"]-counts["b_wins"])/8, 1.)

    def test_fixed_halves_match_registered_first_partition(self):
        for a, b, expected in (
            ([True]*4+[False]*4, [False]*4+[True]*4, [1, -1]),
            ([True]*4+[False]*4, [False]*8, [1, 0]),
            ([False]*8, [True]*8, [-1, -1]),
            ([False]*8, [False]*8, [0, 0]),
        ):
            with self.subTest(expected=expected):
                row = analyze_record(record(a, b))
                fixed = row["fixed_halves"]
                self.assertEqual(fixed["half_directions"], expected)
                for field, value in fixed.items():
                    self.assertEqual(value, row["partitions"][0][field])
                self.assertEqual(row["partition_count"], 35)

    def test_fixed_half_censoring_is_not_a_tie_or_crossfit_eligibility(self):
        row = analyze_record(record([None]+[True]*7))
        self.assertEqual(row["fixed_halves"]["half_differences"], [None, 1.])
        self.assertEqual(row["fixed_halves"]["half_directions"], [None, 1])
        self.assertEqual(row["partitions"], [])
        self.assertIsNone(row["crossfit_gain"])
        unknown = analyze_record(record([None]*8, [None]*8))
        self.assertEqual(unknown["paired_counts"]["unknown"], 8)
        self.assertEqual(unknown["fixed_halves"]["half_directions"], [None, None])

    def test_anchor_censoring_does_not_block_pair(self):
        row = analyze_record(record(anchor=[None]*8))
        self.assertEqual(row["crossfit_gain"], .5)
        self.assertEqual(row["partition_count"], 35)
        self.assertIsNone(row["anchor_completion_rate"])
        self.assertIsNone(row["anchor_minus_pair_uniform"])

    def test_map_equal_state_equal_not_candidate_or_partition_weighted(self):
        data = [record(state="s1", map_id="m1"), record(state="s2", map_id="m1"),
                record(b=[True]*8, state="s3", map_id="m2")]
        result = summarize(data)
        metric = result["metrics"]["crossfit_gain"]
        self.assertEqual(metric["mean"], .25)
        self.assertEqual(metric["ci95"], [0., .5])
        self.assertEqual(metric["per_map"]["m1"]["mean"], .5)
        self.assertEqual(result["states"], 3)
        self.assertEqual(result["bootstrap"]["samples"], BOOTSTRAP_SAMPLES)
        self.assertEqual(result["bootstrap"]["seed"], BOOTSTRAP_SEED)
        self.assertFalse(result["partitions_are_independent_samples"])
        self.assertFalse(result["thresholds_applied"])

    def test_missing_state_makes_full_cohort_metric_unavailable(self):
        data = [record(state="s1", map_id="m1"),
                record([None]*8, state="s2", map_id="m1"), record(state="s3", map_id="m2")]
        result = summarize(data, bootstrap_samples=20)
        metric = result["metrics"]["crossfit_gain"]
        self.assertIsNone(metric["mean"])
        self.assertIsNone(metric["ci95"])
        self.assertEqual(metric["observed_states"], 2)
        self.assertEqual(metric["complete_maps"], 1)
        self.assertIsNone(metric["per_map"]["m1"]["mean"])
        self.assertEqual(result["censored_pair_states"], 1)
        self.assertTrue(result["metrics"]["anchor_completion_rate"]["complete"])

    def test_reproducible_json_safe_and_no_global_rng_or_input_change(self):
        data = [record(state="s1", map_id="m1"), record(b=[True]*8, state="s2", map_id="m2")]
        saved, rng = copy.deepcopy(data), random.getstate()
        result = summarize(data, bootstrap_samples=40)
        self.assertEqual(result, summarize(data[::-1], bootstrap_samples=40))
        self.assertEqual(data, saved)
        self.assertEqual(random.getstate(), rng)
        json.dumps(result, allow_nan=False)

    def test_validation_rejects_invalid_records(self):
        mutations = [lambda r: r.update(state_id=""), lambda r: r.update(map_id=None),
                     lambda r: r.update(pair_ids=["a", "a"]), lambda r: r.update(pair_ids=["a"]),
                     lambda r: r.update(anchor_id="missing"), lambda r: r["values"].pop("a"),
                     lambda r: r["values"].update(a=[True]*7),
                     lambda r: r["values"].update(a=[1]*8),
                     lambda r: r["values"].update(a=[float("nan")]*8)]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                source = record()
                mutation(source)
                with self.assertRaises(ValueError):
                    analyze_record(source)

    def test_summary_validation(self):
        for data in ([], [record(), record()]):
            with self.assertRaises(ValueError):
                summarize(data)
        for kwargs in (dict(seed=-1), dict(seed=True), dict(bootstrap_samples=0), dict(bootstrap_samples=True)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                summarize([record()], **kwargs)


if __name__ == "__main__":
    unittest.main()
