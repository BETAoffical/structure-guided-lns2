from copy import deepcopy
import json
import math
import random
import unittest

from experiments.sa_raw_confirmation import ARMS
from experiments.sa_raw_timing_metrics import summarize


def fixture(maps=("map_a", "map_b"), replicas=1):
    rows = []
    for i, map_id in enumerate(maps):
        for replica in range(replicas):
            for arm in ARMS:
                rows.append(dict(
                    job_id=f"p{i}-{replica}-{arm}", pair_id=f"p{i}", replica=replica,
                    comparison_arm=arm, map_id=map_id, initial_fingerprint=f"initial-{i}",
                    rng_stream_id=f"rng-{i}-{replica}", status="ok", budget_seconds=120.0,
                    feasible=True, success_within_budget=True, ttf_seconds=10.0,
                    delivery_seconds=12.0, delivered_within_budget=True, decisions=5,
                    search_end_seconds=10.5, last_pp_failure_reason="none",
                    generated=50, soc=100, makespan=20, wait_steps=4, final_conflicts=0,
                    stop="feasible"))
    return rows


def row_for(rows, pair="p0", arm="raw_updated", replica=0):
    return next(r for r in rows if (r["pair_id"], r["replica"], r["comparison_arm"])
                == (pair, replica, arm))


def fail(row, stop="deadline"):
    row.update(feasible=False, success_within_budget=False, delivered_within_budget=False,
               ttf_seconds=None, delivery_seconds=123.0, final_conflicts=2, stop=stop,
               search_end_seconds=120.0, last_pp_failure_reason="time_limit" if stop == "pp_deadline" else "none",
               generated=999999, decisions=9999, soc=99999, makespan=9999, wait_steps=999)


def quantile(values, q):
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


class RawTimingMetricsTests(unittest.TestCase):
    def test_defaults_pure_order_independent_and_json_safe(self):
        rows = fixture(replicas=2)
        frozen = deepcopy(rows)
        global_rng = random.getstate()
        report = summarize(rows)
        self.assertEqual(report, summarize(iter(reversed(rows))))
        self.assertEqual(rows, frozen)
        self.assertEqual(global_rng, random.getstate())
        self.assertEqual(json.loads(json.dumps(report, allow_nan=False)), report)
        self.assertEqual((report["pairs"], report["episodes"], report["maps"]), (4, 16, 2))
        self.assertEqual((report["bootstrap"], report["seed"]), (5000, 2026092603))
        self.assertEqual(report["bootstrap_unit"], "map")
        self.assertTrue(report["no_training"] and report["no_promotion"])
        self.assertEqual(set(report["arms"]), set(ARMS))
        self.assertEqual(set(report["contrasts"]), set(ARMS) - {"raw_updated"})
        self.assertEqual(report["four_arm_common_success"]["count"], 4)
        for arm in ARMS:
            self.assertEqual(report["arms"][arm]["denominator"], 4)
            self.assertEqual(report["arms"][arm]["success_count"], 4)
            self.assertEqual(report["arms"][arm]["delivered_count"], 4)
        for contrast in report["contrasts"].values():
            self.assertEqual(contrast["success"]["ci95"], [0.0, 0.0])
            self.assertEqual(contrast["success"]["bootstrap_valid_draws"], 5000)

    def test_success_denominators_gains_losses_and_diagnostic_failures(self):
        rows = fixture(("map_a", "map_a", "map_b", "map_b"))
        fail(row_for(rows, "p0", "raw_parent"))
        fail(row_for(rows, "p1"), "pp_deadline")
        fail(row_for(rows, "p2"))
        fail(row_for(rows, "p2", "raw_parent"))
        report = summarize(rows, bootstrap=80)
        arm = report["arms"]["raw_updated"]
        self.assertEqual((arm["denominator"], arm["success_count"], arm["success_rate"]), (4, 2, .5))
        self.assertEqual(arm["successful"]["soc"]["total"], 200)
        self.assertEqual(arm["unsuccessful_quality_diagnostic"]["soc"]["count"], 2)
        self.assertEqual(arm["unsuccessful_quality_diagnostic"]["soc"]["total"], 199998)
        contrast = report["contrasts"]["raw_parent"]
        success = contrast["success"]
        self.assertEqual(success["denominator"], 4)
        self.assertEqual((success["wins"], success["ties"], success["losses"]), (1, 2, 1))
        self.assertEqual((success["both_success"], success["both_failure"]), (1, 1))
        self.assertEqual(success["gain_pairs"], [["p0", 0]])
        self.assertEqual(success["loss_pairs"], [["p1", 0]])
        self.assertEqual(success["delta"], 0.0)
        self.assertEqual(contrast["common_success"]["pair_ids"], [["p3", 0]])
        self.assertEqual(contrast["common_success"]["metrics"]["generated"]["raw_updated"]["total"], 50)
        self.assertEqual(report["four_arm_common_success"]["count"], 1)
        self.assertEqual(report["by_map"]["map_a"]["pairs"], 2)
        self.assertEqual(report["by_map"]["map_b"]["arms"]["raw_updated"]["success_count"], 1)
        self.assertNotIn("ci95", report["by_map"]["map_b"]["contrasts"]["raw_parent"]["success"])

    def test_raw_metrics_paired_wins_losses_quantiles_and_difference_sign(self):
        rows = fixture(("map_a", "map_a", "map_b"))
        for pair, ttf, delivery, quality in (("p0", 8., 10., 3), ("p1", 10., 12., 5), ("p2", 13., 15., 8)):
            row_for(rows, pair).update(ttf_seconds=ttf, delivery_seconds=delivery, search_end_seconds=ttf,
                                      decisions=quality, generated=quality * 10,
                                      soc=quality * 20, makespan=quality * 4, wait_steps=quality - 1)
        report = summarize(rows, bootstrap=80)
        metrics = report["contrasts"]["raw_parent"]["common_success"]["metrics"]
        for field in ("ttf_seconds", "delivery_seconds", "decisions", "generated", "soc", "makespan", "wait_steps"):
            metric = metrics[field]
            self.assertEqual((metric["wins"], metric["ties"], metric["losses"]), (1, 1, 1))
            self.assertEqual(metric["denominator"], 3)
            self.assertGreater(metric["mean_difference"], 0)
        times = metrics["ttf_seconds"]["raw_updated"]
        self.assertAlmostEqual(metrics["ttf_seconds"]["mean_difference"], 1 / 3)
        self.assertEqual(times["p50"], 10.)
        self.assertAlmostEqual(times["p95"], 12.7)
        self.assertEqual(times["max"], 13.)
        self.assertEqual(report["arms"]["raw_updated"]["successful"]["delivery_seconds"]["max"], 15.)
        self.assertEqual(report["difference_direction"], "raw_updated_minus_baseline")

    def test_deadline_boundary_delivery_overshoot_and_feasible_overshoot(self):
        rows = fixture(("map_a", "map_a", "map_b", "map_b"))
        row_for(rows, "p0").update(ttf_seconds=120., delivery_seconds=120., search_end_seconds=120.)
        row_for(rows, "p1").update(ttf_seconds=119., delivery_seconds=125., search_end_seconds=119.5,
                                   delivered_within_budget=False)
        row_for(rows, "p2").update(ttf_seconds=121., delivery_seconds=123., search_end_seconds=122.,
                                   success_within_budget=False, delivered_within_budget=False)
        fail(row_for(rows, "p3"), "pp_deadline")
        report = summarize(rows, bootstrap=80)
        arm = report["arms"]["raw_updated"]
        self.assertEqual((arm["feasible_count"], arm["success_count"], arm["delivered_count"]), (3, 2, 1))
        self.assertEqual(arm["successful"]["ttf_seconds"]["max"], 120.)
        self.assertEqual(arm["successful"]["delivery_seconds"]["max"], 125.)
        self.assertEqual(arm["late_feasible_times_diagnostic"]["ttf_seconds"]["max"], 121.)
        contrast = report["contrasts"]["raw_parent"]
        self.assertEqual(contrast["common_success"]["count"], 2)
        self.assertEqual(contrast["common_delivered"]["count"], 1)
        self.assertEqual(contrast["success"]["delta"], -.5)
        self.assertEqual(contrast["delivery_success"]["delta"], -.75)
        self.assertEqual(contrast["common_success"]["metrics"]["delivery_seconds"]["raw_updated"]["mean"], 122.5)
        for stop in ("feasible", "deadline"):
            row_for(rows, "p2")["stop"] = stop
            self.assertEqual(summarize(rows, bootstrap=2)["arms"]["raw_updated"]["success_count"], 2)

    def test_modeled_completion_only_common_delivered_and_can_exceed_budget(self):
        rows = fixture(("map_a", "map_b", "map_c"))
        row_for(rows).update(delivery_seconds=20., makespan=100)
        row_for(rows, arm="raw_parent").update(delivery_seconds=15., makespan=20)
        row_for(rows, "p1").update(delivery_seconds=121., delivered_within_budget=False)
        fail(row_for(rows, "p2", "raw_parent"))
        report = summarize(rows, bootstrap=80)
        contrast = report["contrasts"]["raw_parent"]
        modeled = contrast["common_delivered"]["modeled_completion_seconds"]
        self.assertEqual(contrast["common_delivered"]["pair_ids"], [["p0", 0]])
        for step in (.5, 1., 2.):
            metric = modeled[str(step)]
            self.assertEqual(metric["denominator"], 1)
            self.assertEqual(metric["raw_updated"]["mean"], 20 + step * 100)
            self.assertEqual(metric["baseline"]["mean"], 15 + step * 20)
            self.assertEqual(metric["mean_difference"], 5 + step * 80)
            self.assertEqual(metric["ci95"], [5 + step * 80] * 2)
        self.assertEqual(modeled["2.0"]["raw_updated"]["mean"], 220.)

    def test_four_arm_common_success_differs_from_pairwise_cohorts(self):
        rows = fixture(("map_a", "map_b", "map_c"))
        fail(row_for(rows, "p0", "official_sa"))
        fail(row_for(rows, "p1", "dual16_sa"))
        row_for(rows, "p2").update(ttf_seconds=3., delivery_seconds=5., search_end_seconds=4.)
        report = summarize(rows, bootstrap=80)
        self.assertEqual(report["contrasts"]["raw_parent"]["common_success"]["count"], 3)
        four = report["four_arm_common_success"]
        self.assertEqual((four["count"], four["denominator"]), (1, 3))
        self.assertEqual(four["pair_ids"], [["p2", 0]])
        self.assertEqual(four["arms"]["raw_updated"]["ttf_seconds"]["mean"], 3.)
        self.assertEqual(four["contrasts"]["raw_parent"]["ttf_seconds"]["mean_difference"], -7.)

    def test_capped_ttf_is_secondary_all_pair_penalty_not_raw_time(self):
        rows = fixture(("map_a", "map_b"))
        fail(row_for(rows, "p0", "raw_parent"))
        row_for(rows, "p1").update(ttf_seconds=150., delivery_seconds=160., search_end_seconds=155.,
                                   success_within_budget=False, delivered_within_budget=False)
        report = summarize(rows, bootstrap=80)
        secondary = report["secondary_capped_ttf"]
        self.assertIn("secondary", secondary["label"])
        self.assertIn("not_raw_TTF_or_sole_claim", secondary["label"])
        self.assertEqual(secondary["denominator"], 2)
        self.assertEqual(secondary["failure_penalty_seconds"], 120)
        self.assertEqual(secondary["arms"]["raw_updated"]["mean"], 65.)
        self.assertEqual(secondary["arms"]["raw_parent"]["mean"], 65.)
        self.assertEqual(secondary["contrasts"]["raw_parent"]["mean_difference"], 0.)
        self.assertEqual(report["contrasts"]["raw_parent"]["common_success"]["count"], 0)
        self.assertIsNone(report["contrasts"]["raw_parent"]["common_success"]["metrics"]["ttf_seconds"]["mean_difference"])

    def test_all_failed_empty_success_summaries_are_not_zero(self):
        rows = fixture()
        for row in rows:
            fail(row)
        report = summarize(rows, bootstrap=20)
        self.assertEqual(report["four_arm_common_success"]["count"], 0)
        for arm in ARMS:
            stats = report["arms"][arm]["successful"]["ttf_seconds"]
            self.assertEqual(stats["count"], 0)
            for field in ("total", "mean", "p50", "p95", "max"):
                self.assertIsNone(stats[field])
        for contrast in report["contrasts"].values():
            self.assertEqual(contrast["success"]["delta"], 0.)
            for cohort in ("common_success", "common_delivered"):
                metric = contrast[cohort]["metrics"]["ttf_seconds"]
                self.assertEqual(metric["denominator"], 0)
                self.assertIsNone(metric["ci95"])
                self.assertEqual((metric["bootstrap_valid_draws"], metric["bootstrap_empty_draws"]), (0, 20))
            for metric in contrast["common_delivered"]["modeled_completion_seconds"].values():
                self.assertIsNone(metric["raw_updated"]["mean"])
                self.assertIsNone(metric["mean_difference"])
        json.dumps(report, allow_nan=False)

    def test_map_cluster_bootstrap_pair_weighting_matches_manual_draws(self):
        rows = fixture(("map_a", "map_b", "map_b", "map_b"), replicas=2)
        for row in rows:
            if row["comparison_arm"] == "raw_updated":
                row.update(ttf_seconds=11. if row["map_id"] == "map_a" else 19.,
                           delivery_seconds=20., search_end_seconds=20.)
            if row["map_id"] == "map_a" and row["comparison_arm"] == "official_sa":
                fail(row)
        draws, seed = 127, 91
        report = summarize(rows, bootstrap=draws, seed=seed)
        rng = random.Random(seed)
        expected_times, expected_success = [], []
        for _ in range(draws):
            selected = rng.choices(["map_a", "map_b"], k=2)
            n = sum(2 if m == "map_a" else 6 for m in selected)
            expected_times.append(sum(2 if m == "map_a" else 54 for m in selected) / n)
            expected_success.append(sum(2 if m == "map_a" else 0 for m in selected) / n)
        timing = report["contrasts"]["raw_parent"]["common_success"]["metrics"]["ttf_seconds"]
        success = report["contrasts"]["official_sa"]["success"]
        self.assertEqual(timing["mean_difference"], 7.)
        self.assertEqual(success["delta"], .25)
        for metric, expected in ((timing, expected_times), (success, expected_success)):
            for actual, q in zip(metric["ci95"], (.025, .975)):
                self.assertAlmostEqual(actual, quantile(expected, q))
            self.assertEqual(metric["bootstrap_valid_draws"], draws)

    def test_empty_bootstrap_cohorts_counted_and_one_map_interval(self):
        rows = fixture()
        fail(row_for(rows, "p1", "raw_parent"))
        row_for(rows, "p0").update(ttf_seconds=7., delivery_seconds=9., search_end_seconds=8.)
        draws, seed = 97, 34
        report = summarize(rows, bootstrap=draws, seed=seed)
        rng = random.Random(seed)
        empty = sum(all(m == "map_b" for m in rng.choices(["map_a", "map_b"], k=2))
                    for _ in range(draws))
        metric = report["contrasts"]["raw_parent"]["common_success"]["metrics"]["ttf_seconds"]
        self.assertEqual(metric["ci95"], [-3., -3.])
        self.assertEqual(metric["bootstrap_empty_draws"], empty)
        self.assertEqual(metric["bootstrap_valid_draws"], draws - empty)
        single = summarize(fixture(("map_a",)), bootstrap=5)
        self.assertEqual(single["contrasts"]["raw_parent"]["success"]["ci95"], [0., 0.])

    def test_reject_empty_missing_arms_duplicates_and_cross_replica_maps(self):
        rows = fixture(replicas=2)
        duplicate_arm = deepcopy(rows[0])
        duplicate_arm["job_id"] = "another-job"
        for bad in ([], rows[:-1], rows + [rows[0]], rows + [duplicate_arm]):
            with self.subTest(size=len(bad)), self.assertRaises(ValueError):
                summarize(bad, bootstrap=2)
        altered = deepcopy(rows)
        altered[1]["job_id"] = altered[0]["job_id"]
        with self.assertRaisesRegex(ValueError, "duplicate job_id"):
            summarize(altered, bootstrap=2)
        for row in rows:
            if row["pair_id"] == "p0" and row["replica"] == 1:
                row["map_id"] = "different-map"
        with self.assertRaisesRegex(ValueError, "map mismatch"):
            summarize(rows, bootstrap=2)

    def test_reject_pair_identity_budget_schema_and_count_errors(self):
        cases = [("initial_fingerprint", "wrong"), ("rng_stream_id", "wrong"),
                 ("map_id", "wrong"), ("comparison_arm", "unknown"), ("status", "error"),
                 ("budget_seconds", 119.), ("budget_seconds", 121.), ("stop", "node_budget"),
                 ("replica", True), ("replica", -1), ("replica", .5), ("job_id", ""),
                 ("pair_id", []), ("rng_stream_id", 123), ("initial_fingerprint", None)]
        for field in ("decisions", "generated", "soc", "makespan", "wait_steps", "final_conflicts"):
            cases.extend((field, value) for value in (-1, True, 1.5, None, math.inf))
        for field, value in cases:
            with self.subTest(field=field, value=value):
                rows = fixture()
                rows[0][field] = value
                with self.assertRaises(ValueError):
                    summarize(rows, bootstrap=2)
        rows = fixture()
        for row in rows:
            row["budget_seconds"] = 60.
        with self.assertRaises(ValueError):
            summarize(rows, bootstrap=2)
        for field in fixture()[0]:
            rows = fixture()
            del rows[0][field]
            with self.subTest(missing=field), self.assertRaises(ValueError):
                summarize(rows, bootstrap=2)
        with self.assertRaises(ValueError):
            summarize([None], bootstrap=2)

    def test_reject_nonfinite_negative_nonnumeric_times(self):
        for field in ("ttf_seconds", "delivery_seconds", "budget_seconds", "search_end_seconds", "reset_seconds"):
            for value in (-1., math.nan, math.inf, -math.inf, True, "12", 10 ** 400, None):
                rows = fixture()
                rows[0][field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    summarize(rows, bootstrap=2)
        rows = fixture()
        rows[0]["delivery_seconds"] = None
        with self.assertRaises(ValueError):
            summarize(rows, bootstrap=2)

    def test_reject_inconsistent_feasibility_success_delivery_or_stop(self):
        changes = [dict(feasible=False), dict(ttf_seconds=None), dict(final_conflicts=1),
                   dict(success_within_budget=False), dict(delivered_within_budget=False),
                   dict(ttf_seconds=121., delivery_seconds=122.), dict(delivery_seconds=121.),
                   dict(delivery_seconds=9.), dict(stop="deadline"), dict(stop="pp_deadline")]
        for field in ("feasible", "success_within_budget", "delivered_within_budget"):
            changes.extend({field: value} for value in (0, 1, "true", None))
        for change in changes:
            rows = fixture()
            rows[0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                summarize(rows, bootstrap=2)
        for change in (dict(ttf_seconds=1.), dict(final_conflicts=0), dict(stop="feasible"),
                       dict(success_within_budget=True), dict(delivered_within_budget=True)):
            rows = fixture()
            fail(rows[0])
            rows[0].update(change)
            with self.subTest(failure_change=change), self.assertRaises(ValueError):
                summarize(rows, bootstrap=2)

    def test_zero_times_allowed_and_bootstrap_settings_validated(self):
        rows = fixture(("map_a",))
        for row in rows:
            row.update(ttf_seconds=0., delivery_seconds=0., decisions=0, generated=0,
                       soc=0, makespan=0, wait_steps=0, search_end_seconds=0.,
                       last_pp_failure_reason=None)
        report = summarize(rows, bootstrap=1, seed=0)
        self.assertEqual(report["arms"]["raw_updated"]["successful"]["ttf_seconds"]["p95"], 0.)
        for bootstrap in (0, -1, True, 1.5, "5"):
            with self.subTest(bootstrap=bootstrap), self.assertRaises(ValueError):
                summarize(rows, bootstrap=bootstrap)
        for seed in (True, 1.5, "seed", None):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                summarize(rows, bootstrap=2, seed=seed)

    def test_reject_early_deadline_even_when_delivery_exceeds_budget(self):
        for search_end, delivery in ((1., 1.), (1., 123.), (119.999, 123.)):
            rows = fixture()
            fail(rows[0])
            rows[0].update(search_end_seconds=search_end, delivery_seconds=delivery)
            with self.subTest(search_end=search_end, delivery=delivery):
                with self.assertRaisesRegex(ValueError, "deadline before budget"):
                    summarize(rows, bootstrap=2)
        for search_end in (120., 122.):
            rows = fixture()
            fail(rows[0])
            rows[0]["search_end_seconds"] = search_end
            self.assertEqual(summarize(rows, bootstrap=2)["arms"]["raw_parent"]["success_count"], 1)
        rows[0].update(decisions=0, last_pp_failure_reason=None, reset_seconds=120.)
        self.assertEqual(summarize(rows, bootstrap=2)["arms"]["raw_parent"]["success_count"], 1)

    def test_search_end_bounds_and_optional_reset(self):
        for change, message in (
                (dict(search_end_seconds=9.9), "precedes first feasible env-return"),
                (dict(search_end_seconds=12.1), "exceeds delivery_seconds"),
                (dict(reset_seconds=10.6), "precedes reset_seconds")):
            rows = fixture()
            rows[0].update(change)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, message):
                summarize(rows, bootstrap=2)
        rows = fixture()
        fail(rows[0])
        rows[0]["reset_seconds"] = 121.
        with self.assertRaisesRegex(ValueError, "precedes reset_seconds"):
            summarize(rows, bootstrap=2)
        for search_end in (10., 12.):
            rows = fixture()
            rows[0].update(search_end_seconds=search_end, reset_seconds=search_end)
            self.assertEqual(summarize(rows, bootstrap=2)["arms"]["raw_parent"]["success_count"], 2)

    def test_pp_deadline_requires_executed_time_limit_and_no_feasible_paths(self):
        rows = fixture()
        fail(rows[0], "pp_deadline")
        rows[0].update(decisions=1, search_end_seconds=1., delivery_seconds=2.)
        self.assertEqual(summarize(rows, bootstrap=2)["arms"]["raw_parent"]["success_count"], 1)
        for change in (
                dict(last_pp_failure_reason="none"),
                dict(last_pp_failure_reason="not_run"),
                dict(decisions=0, last_pp_failure_reason=None),
                dict(feasible=True, final_conflicts=0, ttf_seconds=121.,
                     search_end_seconds=122., delivery_seconds=123.)):
            altered = deepcopy(rows)
            altered[0].update(change)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "pp_deadline requires"):
                summarize(altered, bootstrap=2)

    def test_last_pp_failure_reason_matches_decision_count(self):
        for decisions, reasons in ((0, ("none", "time_limit", "not_run", "")),
                                   (1, (None, 0, False, [], {}))):
            for reason in reasons:
                rows = fixture()
                rows[0].update(decisions=decisions, last_pp_failure_reason=reason)
                with self.subTest(decisions=decisions, reason=reason):
                    with self.assertRaisesRegex(ValueError, "last_pp_failure_reason"):
                        summarize(rows, bootstrap=2)
        for decisions, reason in ((0, None), (1, "none"), (1, "not_run")):
            rows = fixture()
            rows[0].update(decisions=decisions, last_pp_failure_reason=reason)
            self.assertEqual(summarize(rows, bootstrap=2)["arms"]["raw_parent"]["success_count"], 2)


if __name__ == "__main__":
    unittest.main()
