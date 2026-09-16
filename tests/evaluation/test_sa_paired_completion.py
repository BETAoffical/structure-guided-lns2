import copy
from collections import defaultdict
import importlib.util
import json
import math
from pathlib import Path
import unittest

from experiments._common import json_fingerprint
from experiments.sa_paired_completion import (
    MODEL_PARAMS, SCHEMA, PairedCompletionModel, inspect_dataset, paired_labels,
    pair_vector, training_matrix, validate_dataset,
)


def trial(t, completed):
    return dict(trial=t, randomization_key=json_fingerprint(["paired", t]),
                stop="feasible" if completed else "horizon", steps=1 if completed else 32,
                final_conflicts=0 if completed else 1, completed=bool(completed))


def fixture(role="prospective_development"):
    states = []
    for i, m in enumerate(("train", "held")):
        states.append(dict(state_id=f"s{i}", map_id=m, episode=f"e{i}", decision=16,
            state_fingerprint=json_fingerprint(["state", i]), history_fingerprint=json_fingerprint(["history", i]),
            agent_ids=[7, 19, 31], anchor_id="b", candidates=[
                dict(candidate_id=c, agents=[agent],
                     features={"realized.signal":value, "sa.log_temperature":3., "state.colliding_pairs":2.},
                     trials=[trial(t, v) for t, v in enumerate(labels)])
                for c, agent, value, labels in (("a", 7, .8, [1]*6+[0]*2), ("b", 19, .2, [1]*2+[0]*6))]))
    return dict(schema=SCHEMA, role=role, sampling="outcome_blind_same_state",
                source_kind="registered_prospective_collection", continuation_binding="a"*64,
                horizon=32, trial_count=8, feature_names=sorted(states[0]["candidates"][0]["features"]), states=states)


class LinearEstimator:
    def predict(self, x):
        return [r[0] for r in x]


class PairedCompletionTests(unittest.TestCase):
    def test_frozen_parameters_and_no_runtime(self):
        path = Path(__file__).resolve().parents[2] / "configs/sa_paired_completion_model.json"
        cfg = json.loads(path.read_text())
        self.assertEqual((cfg["horizon"], cfg["trials"]), (32, 8))
        self.assertEqual(cfg["new_solver_calls"], 0)
        self.assertFalse(cfg["fit_historical_bridge"])
        self.assertFalse(cfg["runtime_integration_allowed"])
        self.assertEqual(MODEL_PARAMS, dict(learning_rate=.05, max_iter=100, max_leaf_nodes=7,
            min_samples_leaf=10, l2_regularization=.1, early_stopping=False, random_state=20260916))

    def test_pair_target_is_completion_difference_not_conflict_drop(self):
        data = fixture()
        validate_dataset(data)
        p = paired_labels(data["states"][0])[0]
        self.assertEqual((p["wins"], p["losses"], p["ties"], p["mean_difference"]), (4, 0, 4, .5))
        for c in data["states"][0]["candidates"]:
            for t in c["trials"]:
                if not t["completed"]:
                    t["final_conflicts"] = 1000
        self.assertEqual(p, paired_labels(data["states"][0])[0])

    def test_tie_trials_retained_and_no_certain_winner_invented(self):
        s = fixture()["states"][0]
        s["candidates"][0]["trials"] = [trial(t, t < 4) for t in range(8)]
        s["candidates"][1]["trials"] = [trial(t, t >= 4) for t in range(8)]
        p = paired_labels(s)[0]
        self.assertEqual((p["wins"], p["losses"], p["ties"], p["mean_difference"]), (4, 4, 0, 0))

    def test_missing_duplicate_and_misaligned_trials_rejected(self):
        for mutation, error in ((lambda t: t.pop(), "trial grid"),
                                (lambda t: t.__setitem__(1, copy.deepcopy(t[0])), "trial grid"),
                                (lambda t: t[0].update(randomization_key="b"*64), "randomization")):
            with self.subTest(error=error):
                d = fixture()
                mutation(d["states"][0]["candidates"][0]["trials"])
                with self.assertRaisesRegex(ValueError, error):
                    validate_dataset(d)

    def test_censoring_is_unknown_and_blocks_fit_not_zero(self):
        d = fixture()
        row = d["states"][0]["candidates"][0]["trials"][0]
        row.update(stop="wall_safety", steps=3, final_conflicts=1, completed=None)
        self.assertIsNone(paired_labels(d["states"][0])[0]["mean_difference"])
        self.assertIn("censoring_policy_not_registered", inspect_dataset(d)["rejection_reasons"])
        with self.assertRaisesRegex(ValueError, "censoring_policy"):
            training_matrix(d, {"held"})
        row["completed"] = False
        with self.assertRaisesRegex(ValueError, "censored outcome"):
            validate_dataset(d)

    def test_invalid_completion_and_truncation_rejected(self):
        mutations = [dict(completed=1), dict(final_conflicts=1), dict(steps=0),
                     dict(stop="horizon", completed=False, final_conflicts=1, steps=31), dict(stop="error")]
        for change in mutations:
            d = fixture()
            d["states"][0]["candidates"][0]["trials"][0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_dataset(d)

    def test_map_episode_occurrence_and_noncontinuous_agent_ids(self):
        validate_dataset(fixture())
        for change, error in ((dict(state_id="s0"), "state id"),
                              (dict(episode="e0"), "occurrence"),
                              (dict(episode="e0", decision=17), "crosses maps")):
            d = fixture()
            d["states"][1].update(change)
            with self.assertRaisesRegex(ValueError, error):
                validate_dataset(d)
        for agents in ([999], [7, 7], [-1], [True]):
            d = fixture()
            d["states"][0]["candidates"][0]["agents"] = agents
            with self.assertRaises(ValueError):
                validate_dataset(d)

    def test_duplicate_physical_candidates_and_missing_anchor(self):
        d = fixture()
        d["states"][0]["candidates"][1]["agents"] = [7]
        with self.assertRaisesRegex(ValueError, "physical"):
            validate_dataset(d)
        d = fixture()
        d["states"][0]["anchor_id"] = "missing"
        with self.assertRaisesRegex(ValueError, "anchor"):
            validate_dataset(d)

    def test_each_state_weight_one_despite_candidate_count(self):
        d = fixture()
        extra = copy.deepcopy(d["states"][0])
        extra.update(state_id="s2", episode="e2")
        c = copy.deepcopy(extra["candidates"][0])
        c.update(candidate_id="c", agents=[31])
        extra["candidates"].append(c)
        d["states"].append(extra)
        matrix = training_matrix(d, {"held"})
        totals = defaultdict(float)
        for sid, w in zip(matrix["state_ids"], matrix["weights"]):
            totals[sid] += w
        self.assertEqual(set(totals), {"s0", "s2"})
        self.assertTrue(all(math.isclose(v, 1.) for v in totals.values()))
        self.assertEqual(len(matrix["y"]), 8)  # 1+3 pairs, two orientations, not 64 trials.
        for i in range(0, len(matrix["y"]), 2):
            self.assertEqual(matrix["y"][i], -matrix["y"][i+1])

    def test_holdout_outcomes_cannot_change_training_or_eligibility(self):
        d = fixture()
        before = training_matrix(d, {"held"})
        for c in d["states"][1]["candidates"]:
            for t in c["trials"]:
                t.update(stop="wall_safety", steps=2, final_conflicts=1, completed=None)
        self.assertEqual(before, training_matrix(d, {"held"}))
        for held in (set(), {"typo"}, {"train", "held"}):
            with self.assertRaisesRegex(ValueError, "held-out"):
                training_matrix(d, held)

    def test_historical_data_cannot_be_admitted_by_changing_role_alone(self):
        d = fixture(role="diagnostic_only")
        d.update(sampling="model_selected", source_kind="historical_candidate_bridge")
        self.assertEqual(len(inspect_dataset(d)["rejection_reasons"]), 3)
        d["role"] = "prospective_development"
        with self.assertRaisesRegex(ValueError, "model_selected_candidate_pool"):
            PairedCompletionModel.fit(d, {"held"})

    def test_equal_state_features_retain_temperature_level(self):
        d = fixture()
        a, b = d["states"][0]["candidates"]
        v = pair_vector(a, b, d["feature_names"])
        self.assertEqual(v[1], 0)
        self.assertEqual(v[4], 3.)

    def test_feature_schema_nonfinite_and_outcome_leakage(self):
        d = fixture()
        m = PairedCompletionModel(d["feature_names"], LinearEstimator())
        s = d["states"][0]
        baseline = m.rank(s)
        for c in s["candidates"]:
            c["trials"] = []
            c["future_paths"] = [999]
            c["runtime"] = 1e9
        self.assertEqual(baseline, m.rank(s))
        s["candidates"][0]["features"]["outcome.completion"] = 1.
        with self.assertRaisesRegex(ValueError, "schema"):
            m.rank(s)
        for value in (float("nan"), float("inf"), True, "1"):
            d = fixture()
            d["states"][0]["candidates"][0]["features"]["realized.signal"] = value
            with self.assertRaisesRegex(ValueError, "nonnumeric"):
                validate_dataset(d)

    def test_ranking_order_invariance_antisymmetry_and_anchor_ties(self):
        d = fixture()
        m = PairedCompletionModel(d["feature_names"], LinearEstimator())
        s = d["states"][0]
        baseline = m.rank(s)
        self.assertEqual(baseline["selected"], "a")
        self.assertAlmostEqual(sum(baseline["scores"].values()), 0.)
        s["candidates"].reverse()
        self.assertEqual(baseline, m.rank(s))
        for c in s["candidates"]:
            c["features"]["realized.signal"] = 0.
        self.assertEqual(m.rank(s)["selected"], "b")

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "sklearn unavailable; no installation")
    def test_synthetic_fit_is_deterministic_and_learns_paired_signal(self):
        import sklearn
        if sklearn.__version__ != "1.5.0":
            self.skipTest("registered sklearn 1.5.0 required")
        d = fixture()
        prototype = d["states"][0]
        d["states"] = []
        for i in range(36):
            s = copy.deepcopy(prototype)
            s.update(state_id=f"s{i:02d}", episode=f"e{i}", map_id=f"m{i%3}")
            d["states"].append(s)
        a = PairedCompletionModel.fit(d, {"m2"})
        b = PairedCompletionModel.fit(d, {"m2"})
        self.assertEqual(a.rank(d["states"][2]), b.rank(d["states"][2]))
        self.assertEqual(a.rank(d["states"][2])["selected"], "a")


if __name__ == "__main__":
    unittest.main()
