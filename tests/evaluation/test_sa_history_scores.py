import copy
import unittest

from scripts.audit_sa_history_scores import analyze_state, bootstrap, shapley_replacement


def sample(c0=4, b_conflicts=1.2):
    rows = []
    for cid, c, p in (("a", 1.0, 1.0), ("b", b_conflicts, 0.0)):
        label = dict(conflicts=c, persistence=p, feasible=0.0)
        rows.append(dict(state_id="s", map_id="m", stratum="history_exact", candidate_id=cid,
                         initial_conflicts=c0, old_selected=cid == "a", labels=label,
                         trial_labels=[dict(label) for _ in range(4)]))
    predictions = {(profile, r["candidate_id"]): dict(r["labels"]) for r in rows for profile in ("dynamic", "history")}
    return rows, predictions


class ScoreDiagnosisTests(unittest.TestCase):
    def test_empirical_score_can_sacrifice_pareto_quality(self):
        rows, predictions = sample()
        result = analyze_state(rows, predictions)
        self.assertEqual(result["choices"]["empirical_conflict"]["candidate_id"], "a")
        self.assertEqual(result["choices"]["empirical_risk"]["candidate_id"], "b")
        self.assertEqual(result["choices"]["empirical_risk"]["pareto_hit"], 0)
        self.assertAlmostEqual(result["profiles"]["history"]["score_cost"], .2)
        self.assertEqual(result["profiles"]["history"]["prediction_residual"], 0)

    def test_decomposition_can_be_signed_not_causal_fraction(self):
        rows, predictions = sample()
        predictions["history", "a"]["conflicts"] = .8
        result = analyze_state(rows, predictions)["profiles"]["history"]
        self.assertAlmostEqual(result["prediction_residual"], -.2)
        self.assertAlmostEqual(result["total_gap"], 0)
        self.assertAlmostEqual(result["score_cost"] + result["prediction_residual"], result["total_gap"])

    def test_predicted_guard_not_actual_guard(self):
        rows, predictions = sample(b_conflicts=2)
        predictions["history", "b"]["conflicts"] = 1.1
        result = analyze_state(rows, predictions)["profiles"]["history"]
        self.assertTrue(result["actual_excess_over_one_pair"])
        self.assertEqual(result["actual_excess_pairs_vs_predicted_conflict"], 4)

    def test_shapley_exact_additivity_and_dummy_heads(self):
        loss = [3.0 - bool(m & 1) * 2 - bool(m & 2) for m in range(8)]
        self.assertEqual(shapley_replacement(loss), dict(conflicts=2.0, persistence=1.0, feasible=0.0))

    def test_split_selection_cannot_see_evaluation_half(self):
        rows, predictions = sample()
        before = analyze_state(rows, predictions)
        for t in rows[1]["trial_labels"][2:]:
            t["conflicts"] = 9.0
        rows[1]["labels"]["conflicts"] = (1.2 + 9.0) / 2
        after = analyze_state(rows, predictions)
        self.assertEqual(before["split_halves"][0]["risk_id"], after["split_halves"][0]["risk_id"])
        self.assertNotEqual(before["split_halves"][0]["delta"], after["split_halves"][0]["delta"])

    def test_candidate_order_and_inputs_unchanged(self):
        rows, predictions = sample()
        original = copy.deepcopy(rows)
        self.assertEqual(analyze_state(rows, predictions), analyze_state(list(reversed(rows)), predictions))
        self.assertEqual(rows, original)

    def test_missing_trial_or_changed_aggregation_rejected(self):
        rows, predictions = sample()
        rows[0]["trial_labels"].pop()
        with self.assertRaisesRegex(ValueError, "four trials"):
            analyze_state(rows, predictions)
        rows, predictions = sample()
        rows[0]["labels"]["conflicts"] = 2
        with self.assertRaisesRegex(ValueError, "aggregation"):
            analyze_state(rows, predictions)

    def test_duplicate_candidates_rejected(self):
        rows, predictions = sample()
        rows[1]["candidate_id"] = "a"
        with self.assertRaisesRegex(ValueError, "duplicate"):
            analyze_state(rows, predictions)

    def test_map_bootstrap_weights_and_determinism(self):
        states = [dict(map_id="a"), dict(map_id="a"), dict(map_id="b")]
        result = bootstrap(states, [1.0, 3.0, 8.0], draws=100)
        self.assertEqual(result["mean"], 4.0)
        self.assertEqual(result["map_means"], {"a": 2.0, "b": 8.0})
        self.assertEqual(result, bootstrap(states, [1.0, 3.0, 8.0], draws=100))

    def test_empirical_feasibility_is_prioritized(self):
        rows, predictions = sample()
        rows[1]["labels"]["feasible"] = .25
        rows[1]["trial_labels"][0]["feasible"] = 1.0
        result = analyze_state(rows, predictions)
        self.assertEqual(result["choices"]["empirical_conflict"]["candidate_id"], "b")
        self.assertAlmostEqual(sum(result["profiles"]["history"]["shapley_gains"].values()),
                               result["profiles"]["history"]["masks"][0]["regret"] -
                               result["profiles"]["history"]["masks"][7]["regret"])


if __name__ == "__main__":
    unittest.main()
