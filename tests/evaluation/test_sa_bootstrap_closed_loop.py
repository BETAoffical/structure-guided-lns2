"""Pure fixtures and mocked fits only: no solver or real sklearn fitting."""
from collections import Counter
import copy
import json
import random
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

import numpy as np

from experiments import sa_bootstrap_closed_loop as subject
from experiments.sa_linear_closed_loop import training_arrays
from experiments.sa_paired_closed_loop import choose as old_choose
from experiments.sa_paired_completion import MODEL_PARAMS, PairedCompletionModel
from tests.evaluation.test_sa_low_complexity_ranker import fixture


class Estimator:
    def __init__(self, sign=1):
        self.sign = sign
        self.calls = 0

    def predict(self, vectors):
        self.calls += 1
        return [self.sign * row[0] for row in vectors]


class MockFit:
    def __init__(self, **params):
        self.params = params

    def fit(self, x, y, sample_weight):
        self.x, self.y, self.weights = copy.deepcopy((x, y, sample_weight))
        return self

    def predict(self, vectors):
        raise AssertionError("fit test must not infer")


def full_fixture():
    data = fixture()
    template = data["states"][0]
    data["states"] = []
    for i in range(47):
        state = copy.deepcopy(template)
        state.update(state_id=f"s{i:02d}", episode=f"e{i:02d}", map_id=f"m{i % 8}")
        state["candidates"] = state["candidates"][:2 + i % 3]
        state["candidates"][0]["features"]["realized.signal"] = float(i + 10)
        data["states"].append(state)
    return data


def mocked_fit(data, seed=19):
    sklearn = ModuleType("sklearn")
    sklearn.__version__ = "1.5.0"
    ensemble = ModuleType("sklearn.ensemble")
    ensemble.HistGradientBoostingRegressor = MockFit
    with patch.dict(sys.modules, {"sklearn": sklearn, "sklearn.ensemble": ensemble}):
        return subject.fit_full_members(data, {"model_seed": seed})


class BootstrapFitTests(unittest.TestCase):
    def test_map_weights_blocks_and_zero_frequency_exclusion(self):
        data = full_fixture()
        before = copy.deepcopy(data)
        arrays = training_arrays(data)
        result = mocked_fit(data)
        metadata = result["metadata"]
        self.assertEqual(len(result["models"]), 20)
        self.assertEqual(metadata["schema"], subject.SCHEMA)
        json.dumps(metadata, allow_nan=False)
        expected_draws = np.random.default_rng(19).integers(0, 8, (20, 8))
        self.assertEqual(metadata["draws"], [[f"m{i}" for i in row] for row in expected_draws])
        self.assertTrue(any(len(set(draw)) < 8 for draw in metadata["draws"]))
        for index, model in enumerate(result["models"]):
            frequencies = Counter(metadata["draws"][index])
            offset = 0
            x, y, weights, train_ids = [], [], [], []
            for state in sorted(data["states"], key=lambda s: s["state_id"]):
                sid = state["state_id"]
                count = len(state["candidates"])
                end = offset + count * (count - 1)
                frequency = frequencies[state["map_id"]]
                if frequency:
                    train_ids.append(sid)
                    x.extend(arrays["pairwise"][offset:end])
                    y.extend(arrays["y"][offset:end])
                    weights.extend(w * frequency for w in arrays["weights"][offset:end])
                    self.assertAlmostEqual(metadata["state_weights"][index][sid],
                                           arrays["state_weights"][sid] * frequency)
                else:
                    self.assertNotIn(sid, metadata["state_weights"][index])
                offset = end
            self.assertEqual(model.estimator.params, dict(loss="squared_error", **MODEL_PARAMS))
            self.assertEqual(model.estimator.x, x)
            self.assertEqual(model.estimator.y, y)
            self.assertEqual(model.estimator.weights, weights)
            self.assertEqual(metadata["weights"][index], weights)
            self.assertEqual(metadata["train_ids"][index], train_ids)
            self.assertTrue(all(w > 0 for w in weights))
            self.assertAlmostEqual(sum(weights), 47.)
        self.assertEqual(data, before)

    def test_reproducible_order_and_independent_bootstrap_seed(self):
        data = full_fixture()
        first = mocked_fit(data)
        data["states"].reverse()
        for state in data["states"]:
            state["candidates"].reverse()
        second = mocked_fit(data)
        self.assertEqual(first["metadata"], second["metadata"])
        self.assertEqual([m.estimator.x for m in first["models"]],
                         [m.estimator.x for m in second["models"]])
        different = mocked_fit(data, 20)
        self.assertNotEqual(first["metadata"]["draws"], different["metadata"]["draws"])
        self.assertTrue(all(m.estimator.params["random_state"] == MODEL_PARAMS["random_state"]
                            for m in different["models"]))

    def test_existing_validation_and_scope(self):
        for change in (lambda d: d["states"].pop(),
                       lambda d: d["states"][0]["candidates"][0]["trials"].pop(),
                       lambda d: d.update(schema="wrong")):
            data = full_fixture()
            change(data)
            with self.assertRaises(ValueError):
                mocked_fit(data)
        with self.assertRaises(ValueError):
            mocked_fit(full_fixture(), True)

    def test_tied_bootstrap_sample_is_not_redrawn(self):
        data = full_fixture()
        # Keep contrast only on m0; some preregistered draws omit that map.
        for state in data["states"]:
            if state["map_id"] != "m0":
                for candidate in state["candidates"]:
                    for row in candidate["trials"]:
                        row.update(completed=False, stop="horizon", steps=32, final_conflicts=1)
        result = mocked_fit(data)
        unchanged_draws = mocked_fit(full_fixture())["metadata"]["draws"]
        self.assertEqual(result["metadata"]["draws"], unchanged_draws)
        omitted = [i for i, draw in enumerate(unchanged_draws) if "m0" not in draw]
        self.assertTrue(omitted)
        for i in omitted:
            self.assertTrue(all(y == 0 for y in result["models"][i].estimator.y))


class BootstrapChoiceTests(unittest.TestCase):
    def setUp(self):
        self.candidates = [dict(candidate_id="a", agents=[2]), dict(candidate_id="b", agents=[9])]
        self.features = [{"x": 0.}, {"x": 1.}]
        self.gbdt = PairedCompletionModel(["x"], Estimator())
        self.members = [PairedCompletionModel(["x"], Estimator(1 if i < 13 else -1)) for i in range(20)]

    def choose(self, arm, **changes):
        args = dict(arm=arm, candidates=self.candidates, anchor="a", gbdt=self.gbdt,
                    members=self.members, features=self.features, known_ids=[2, 9],
                    pair_id="task-seed", decision=7, seed=31)
        args.update(changes)
        return subject.choose(**args)

    def test_frozen_and_gbdt_reuse_original_ranking(self):
        self.assertEqual(subject.ARMS, ("frozen", "gbdt", "posterior", "uniform"))
        for arm, old_arm in (("frozen", "frozen"), ("gbdt", "paired")):
            expected = old_choose(old_arm, self.candidates, "a", self.gbdt,
                                  self.features, [2, 9], "task-seed", 31)
            result = self.choose(arm)
            self.assertEqual({k: result[k] for k in expected}, expected)
            self.assertIsNone(result["member_index"])

    def test_posterior_only_infers_one_member_and_matches_its_vote(self):
        result = self.choose("posterior")
        index = result["member_index"]
        self.assertIn(index, range(20))
        self.assertEqual(result["selected"], "b" if index < 13 else "a")
        self.assertEqual([m.estimator.calls for m in self.members],
                         [2 if i == index else 0 for i in range(20)])
        self.assertEqual(self.gbdt.estimator.calls, 0)
        self.assertEqual(len(result["selection_draw"]), 64)

    def test_hash_replay_order_labels_and_global_rng_unchanged(self):
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        for arm in subject.ARMS:
            expected = self.choose(arm)
            changed = [dict(c, trials=object(), completed=object(), score=1e99,
                            labels=object()) for c in reversed(self.candidates)]
            result = self.choose(arm, candidates=changed, features=self.features[::-1])
            self.assertEqual(expected, result)
        self.assertEqual(random.getstate(), python_state)
        current = np.random.get_state()
        self.assertEqual(current[0], numpy_state[0])
        np.testing.assert_array_equal(current[1], numpy_state[1])
        self.assertEqual(current[2:], numpy_state[2:])

    def test_uniform_has_no_model_inference(self):
        result = self.choose("uniform", gbdt=None, members=None)
        self.assertEqual(result["selected"], sorted(c["candidate_id"] for c in self.candidates)[result["selection_index"]])
        self.assertEqual(result["scores"], {"a": 0., "b": 0.})
        self.assertIsNone(result["member_index"])
        self.assertEqual(self.gbdt.estimator.calls, 0)
        self.assertTrue(all(m.estimator.calls == 0 for m in self.members))

    def test_hash_keys_and_arms_are_separate(self):
        base = self.choose("posterior")["selection_draw"]
        for changes in (dict(seed=32), dict(decision=8), dict(pair_id="other")):
            self.assertNotEqual(base, self.choose("posterior", **changes)["selection_draw"])
        self.assertNotEqual(base, self.choose("uniform")["selection_draw"])

    def test_single_candidate_never_ranks_or_draws(self):
        with patch.object(PairedCompletionModel, "rank", side_effect=AssertionError("rank")), \
                patch.object(subject, "_selection_index", side_effect=AssertionError("draw")):
            for arm in subject.ARMS:
                result = self.choose(arm, candidates=self.candidates[:1], features=self.features[:1])
                self.assertEqual(result["selected"], "a")
                self.assertEqual(result["scores"], {"a": 0.})
                self.assertIsNone(result["member_index"])

    def test_invalid_actions_features_and_members_are_rejected(self):
        for arm in subject.ARMS:
            for changes in (dict(known_ids=[2]), dict(anchor="unknown"),
                            dict(candidates=[self.candidates[0]] * 2), dict(features=self.features[:1]),
                            dict(features=[{"x": float("nan")}, {"x": 1.}]),
                            dict(candidates=[self.candidates[0], dict(candidate_id="b", agents=[2])])):
                with self.subTest(arm=arm, changes=changes), self.assertRaises(ValueError):
                    self.choose(arm, **changes)
        for members in ([], self.members[:-1], self.members + self.members[:1],
                        [None] + self.members[1:],
                        [PairedCompletionModel(["y"], Estimator())] + self.members[1:]):
            with self.assertRaises(ValueError):
                self.choose("posterior", members=members)
        with self.assertRaises(ValueError):
            self.choose("unknown")
        with patch.object(PairedCompletionModel, "rank", return_value=dict(selected="unknown", scores={"a": 0., "b": 1.})):
            with self.assertRaises(ValueError):
                self.choose("gbdt")

    def test_ties_keep_anchor(self):
        for arm in ("gbdt", "posterior"):
            self.assertEqual(self.choose(arm, features=[{"x": 0.}, {"x": 0.}])["selected"], "a")


def episode_fixture(maps=8, pairs_per_map=2):
    rows = []
    for i in range(maps * pairs_per_map):
        for arm in subject.ARMS:
            rows.append(dict(pair_id=f"p{i:02d}", map_id=f"m{i // pairs_per_map}", arm=arm,
                             solver_seed=(11, 13)[i % 2], initial_fingerprint=f"initial{i}",
                             success=True, stop="feasible", decisions=10 + i, generated=100 + i,
                             final_soc=200 + i, final_makespan=20 + i, physical_revisits=2,
                             changed_from_anchor=0 if arm == "frozen" else 3,
                             consecutive_repeats=1, rollbacks=2, mean_feature_outside=.1))
    return rows


def outcome(row, success, stop="decision_budget"):
    row.update(success=success, stop="feasible" if success else stop)


class BootstrapSummaryTests(unittest.TestCase):
    def config(self, **changes):
        return dict(master_seed=3, bootstrap=100, **changes)

    def test_four_arm_counts_all_six_contrasts_order_and_no_mutation(self):
        rows = episode_fixture()
        before = copy.deepcopy(rows)
        cfg = self.config(maps=8, paired_tasks=16, episodes=64, arms=list(subject.ARMS),
                          tasks_per_map=1, solver_seeds=[11, 13])
        result = subject.summarize(rows, cfg)
        self.assertEqual(rows, before)
        self.assertEqual(result, subject.summarize(rows[::-1], cfg))
        self.assertEqual(set(result["contrasts"]), {a + "_vs_" + b for a, b in subject.CONTRASTS})
        self.assertEqual(result["primary_contrasts"],
                         ["posterior_vs_frozen", "posterior_vs_gbdt", "posterior_vs_uniform"])
        self.assertEqual((result["maps"], result["paired_tasks"], result["episodes"]), (8, 16, 64))
        self.assertTrue(result["no_ttf"])
        self.assertFalse(result["production_changed"])
        self.assertFalse(result["automatic_promotion"])
        json.dumps(result, allow_nan=False)

    def test_censoring_is_unknown_not_failure(self):
        for stop in ("wall_safety", "incomplete_pp", "external_timeout"):
            rows = episode_fixture(2, 1)
            for row in rows:
                if row["arm"] == "frozen" and row["pair_id"] == "p01":
                    outcome(row, False)
                if row["arm"] == "posterior" and row["pair_id"] == "p00":
                    outcome(row, False, stop)
            result = subject.summarize(rows, self.config())
            summary = result["summary"]["posterior"]
            self.assertEqual((summary["success"], summary["censored"], summary["known_noncompletion"]), (1, 1, 0))
            self.assertEqual(summary["completion_bounds"], [.5, 1.])
            contrast = result["contrasts"]["posterior_vs_frozen"]
            self.assertEqual(contrast["observed_map_delta"], 0.)
            self.assertEqual(contrast["censor_delta_bounds"], [0., .5])
            self.assertEqual((contrast["known_pairs"], contrast["known_pair_wins"], contrast["known_pair_losses"]), (1, 1, 0))
            self.assertEqual(contrast["ci_target"], "observed_completion_not_censor_adjusted")
            self.assertEqual(result["decision"], "resource_censored_inconclusive")

    def test_both_unknown_and_baseline_unknown_bounds(self):
        rows = episode_fixture(1, 1)
        for row in rows:
            if row["arm"] in ("posterior", "frozen"):
                outcome(row, False, "wall_safety")
        contrast = subject.summarize(rows, self.config())["contrasts"]["posterior_vs_frozen"]
        self.assertEqual(contrast["censor_delta_bounds"], [-1., 1.])
        self.assertEqual(contrast["known_pairs"], 0)
        outcome(next(r for r in rows if r["arm"] == "posterior"), True)
        contrast = subject.summarize(rows, self.config())["contrasts"]["posterior_vs_frozen"]
        self.assertEqual(contrast["censor_delta_bounds"], [0., 1.])

    def test_common_success_ratios_use_same_pairs_and_sum_not_mean(self):
        rows = episode_fixture(3, 1)
        for row in rows:
            i = int(row["pair_id"][1:])
            if row["arm"] == "frozen":
                outcome(row, i != 2)
            elif row["arm"] == "gbdt":
                outcome(row, i != 0)
            elif row["arm"] == "uniform":
                outcome(row, False)
            elif row["arm"] == "posterior":
                for field in ("generated", "decisions", "final_soc", "final_makespan"):
                    row[field] *= i + 2
        result = subject.summarize(rows, self.config())
        for baseline, indices in (("frozen", (0, 1)), ("gbdt", (1, 2))):
            contrast = result["contrasts"]["posterior_vs_" + baseline]
            self.assertEqual(contrast["common_success_pair_ids"], [f"p{i:02d}" for i in indices])
            for metric, field in (("generated", "generated"), ("decisions", "decisions"),
                                  ("soc", "final_soc"), ("makespan", "final_makespan")):
                shared = [r for r in rows if int(r["pair_id"][1:]) in indices]
                numerator = sum(r[field] for r in shared if r["arm"] == "posterior")
                denominator = sum(r[field] for r in shared if r["arm"] == baseline)
                self.assertAlmostEqual(contrast["common_" + metric + "_ratio"], numerator / denominator)
        empty = result["contrasts"]["posterior_vs_uniform"]
        self.assertEqual(empty["common_success"], 0)
        self.assertTrue(all(empty["common_" + metric + "_ratio"] is None
                            for metric in ("generated", "decisions", "soc", "makespan")))

    def test_diagnostics_missingness_and_zero_denominators(self):
        rows = episode_fixture(1, 2)
        posterior = [r for r in rows if r["arm"] == "posterior"]
        posterior[0].pop("consecutive_repeats")
        posterior[0].pop("final_soc")
        posterior[1]["mean_feature_outside"] = .3
        for row in rows:
            if row["arm"] == "frozen":
                row["generated"] = 0
        result = subject.summarize(rows, self.config())
        summary = result["summary"]["posterior"]
        self.assertEqual(summary["changed_from_anchor"], 6)
        self.assertEqual(summary["revisits"], 4)
        self.assertEqual(summary["stops"], {"feasible": 2})
        self.assertIsNone(summary["repeats"])
        self.assertEqual(summary["diagnostics"]["consecutive_repeats"]["missing_episodes"], 1)
        ranges = summary["diagnostics"]["mean_feature_outside"]
        self.assertAlmostEqual(ranges["episode_mean"], .2)
        self.assertAlmostEqual(ranges["decision_weighted_mean"], (10 * .1 + 11 * .3) / 21)
        contrast = result["contrasts"]["posterior_vs_frozen"]
        self.assertIsNone(contrast["common_generated_ratio"])
        self.assertIsNone(contrast["common_soc_ratio"])
        self.assertEqual(contrast["common_success_metrics"]["soc"]["missing_pairs"], 1)

    def test_map_equal_bounds_differ_from_episode_equal_bounds(self):
        rows = episode_fixture(2, 2)
        rows = [r for r in rows if r["pair_id"] != "p01"]
        for row in rows:
            if row["arm"] == "frozen":
                outcome(row, False)
            if row["arm"] == "posterior" and row["map_id"] == "m0":
                outcome(row, False, "wall_safety")
        contrast = subject.summarize(rows, self.config())["contrasts"]["posterior_vs_frozen"]
        self.assertEqual(contrast["observed_map_delta"], .5)
        self.assertEqual(contrast["censor_delta_bounds"], [.5, 1.])
        self.assertEqual(contrast["episode_censor_delta_bounds"], [2 / 3, 1.])

    def test_no_five_percent_gate_or_promotion(self):
        rows = episode_fixture(1, 32)
        for row in rows:
            outcome(row, row["arm"] == "posterior" and row["pair_id"] == "p00")
        result = subject.summarize(rows, self.config())
        self.assertEqual(result["contrasts"]["posterior_vs_frozen"]["observed_map_delta"], 1 / 32)
        self.assertEqual(result["decision"], "bounded_positive_signal")
        self.assertTrue(result["decision_is_descriptive"])
        self.assertFalse(result["automatic_promotion"])
        self.assertNotIn("passed", result)

    def test_work_signal_and_completion_curve(self):
        rows = episode_fixture(1, 2)
        for row in rows:
            if row["arm"] == "posterior":
                row["generated"] = 1
        result = subject.summarize(rows, self.config())
        self.assertEqual(result["decision"], "bounded_work_signal")
        self.assertEqual(result["summary"]["posterior"]["completion_curve"]["32"], 2)
        self.assertEqual(result["summary"]["posterior"]["successful_decision_median"], 10.5)

    def test_pair_identity_termination_and_counts_rejected(self):
        changes = (lambda r: r.pop(), lambda r: r.append(copy.deepcopy(r[0])),
                   lambda r: r[0].update(initial_fingerprint="wrong"),
                   lambda r: r[0].update(map_id="wrong"),
                   lambda r: r[0].update(solver_seed=999),
                   lambda r: r[0].update(success=False),
                   lambda r: r[0].update(success=False, stop="error"),
                   lambda r: r[0].update(generated=-1),
                   lambda r: r[0].update(mean_feature_outside=float("nan")),
                   lambda r: r[0].update(changed_from_anchor=999))
        for change in changes:
            rows = episode_fixture()
            change(rows)
            with self.assertRaises(ValueError):
                subject.summarize(rows, self.config())
        for counts in (dict(maps=7), dict(paired_tasks=15), dict(pairs=17), dict(episodes=63),
                       dict(arms=["frozen"]), dict(tasks_per_map=2, solver_seeds=[11, 13])):
            with self.assertRaises(ValueError):
                subject.summarize(episode_fixture(), self.config(**counts))
        rows = episode_fixture()
        for row in rows:
            if row["pair_id"] == "p01":
                row["solver_seed"] = 11
        with self.assertRaises(ValueError):
            subject.summarize(rows, self.config(maps=8, paired_tasks=16, solver_seeds=[11, 13]))
        with self.assertRaises(ValueError):
            subject.summarize([], self.config())


if __name__ == "__main__":
    unittest.main()
