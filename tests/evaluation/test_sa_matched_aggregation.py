import copy
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts import validate_sa_matched_aggregation as a


def group(sid="s", map_id="m", kind="matched_pair", shift=0):
    n = 2 if kind == "matched_pair" else 4
    candidates = [dict(candidate_id=chr(97+i), agents=list(range(i*16, (i+1)*16)),
        selection_families=["collision:16"], values=[bool((i+shift+t) % 3) for t in range(8)]) for i in range(n)]
    g = a.coverage.comparison_group(dict(state_id=sid, map_id=map_id), sid+kind, candidates,
        [sid+kind+str(t) for t in range(8)], kind, "collision:16" if n == 2 else None)
    for i, c in enumerate(g["candidates"]):
        c["features"] = {"x": i + shift, "y": 4}
    return g


def groups():
    return [group(sid, m, k, i) for i, (sid, m) in enumerate((("s", "m"), ("t", "n"), ("u", "n"), ("v", "p")))
            for k in ("old_grid", "matched_pair")]


class AggregationTests(unittest.TestCase):
    def test_contract_is_fixed(self):
        cfg = a.read_json(a.CONFIG)
        a.contract(cfg)
        for key, value in (("solver_calls", 1), ("automatic_promotion", True), ("feature_count", 128),
                           ("expected_roots", 48), ("hyperparameter_search", True)):
            bad = cfg | {key: value}
            with self.assertRaises(ValueError):
                a.contract(bad)
        bad = copy.deepcopy(cfg)
        bad["model"]["max_iter"] = 300
        with self.assertRaises(ValueError):
            a.contract(bad)

    def test_group_validation_retains_ties_and_rejects_censoring(self):
        g = group()
        g["candidates"][1]["values"] = g["candidates"][0]["values"][:]
        self.assertEqual(a.coverage.pair_rows(a.validate_groups([g]))[0]["target"], 0)
        for v in (None, 1, "False"):
            bad = copy.deepcopy(g)
            bad["candidates"][0]["values"][0] = v
            with self.assertRaisesRegex(ValueError, "incomplete labels"):
                a.validate_groups([bad])

    def test_group_identity_cannot_change(self):
        g = group()
        g["source_binding"] = "wrong"
        with self.assertRaisesRegex(ValueError, "identity"):
            a.validate_groups([g])

    def test_7_within_group_pairs_not_15_cross_group_pairs(self):
        g = [group(kind=k) for k in ("old_grid", "matched_pair")]
        rows = a.coverage.pair_rows(g)
        self.assertEqual(len(rows), 7)
        self.assertAlmostEqual(sum(r["state_budget_weight"] for r in rows), 1)
        self.assertAlmostEqual(sum(r["state_budget_weight"] for r in rows if r["kind"] == "matched_pair"), .5)

    def test_reverse_rows_and_map_root_weight(self):
        m = a.matrix(groups(), ["x", "y"], "m")
        self.assertEqual(m["train_ids"], ["t", "u", "v"])
        self.assertEqual(m["train_maps"], ["n", "p"])
        self.assertEqual(len(m["x"]), 3*7*2)
        self.assertAlmostEqual(math.fsum(m["weights"]), 3)
        for i in range(0, len(m["x"]), 2):
            self.assertEqual(m["y"][i], -m["y"][i+1])
            self.assertEqual(m["x"][i][:2], [-v for v in m["x"][i+1][:2]])
            self.assertEqual(m["x"][i][2:], m["x"][i+1][2:])
            self.assertEqual(m["weights"][i], m["weights"][i+1])
        weights = {s: math.fsum(w for w, sid in zip(m["weights"], m["state_ids"]) if sid == s) for s in m["train_ids"]}
        self.assertEqual(weights, dict(t=.75, u=.75, v=1.5))

    def test_held_labels_and_features_cannot_affect_training(self):
        g = groups()
        first = a.matrix(g, ["x", "y"], "m")
        for row in g:
            if row["map_id"] == "m":
                for c in row["candidates"]:
                    c["values"] = [True]*8
                    c["features"] = dict(x=99999, y=-8888)
        self.assertEqual(first, a.matrix(g, ["x", "y"], "m"))

    def test_feature_allowlist_drops_outcomes(self):
        g = group()
        for c in g["candidates"]:
            c.update(runtime=999, generated_nodes=456, future_path=[0])
        state = a.scoring_state(g, dict(anchor_id="absent", agent_ids=list(range(64))))
        for c in state["candidates"]:
            self.assertEqual(set(c), {"candidate_id", "agents", "features"})
        self.assertEqual(state["anchor_id"], "absent")

    def test_pair_without_anchor_not_fabricated(self):
        class Estimator:
            def predict(self, x): return [0., 0.]
        model = a.PairedCompletionModel(["x", "y"], Estimator())
        s = a.scoring_state(group(), dict(anchor_id="not_sampled", agent_ids=list(range(64))))
        result = a.rank(model, s)
        self.assertEqual(result["selected"], "a")
        self.assertEqual(set(result["scores"]), {"a", "b"})
        s["anchor_id"] = "b"
        self.assertEqual(a.rank(model, s)["selected"], "b")

    def test_input_drift_and_unsafe_path_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a").write_text("old")
            with mock.patch.object(a, "ROOT", root):
                pins = {"a": a.sha256_file(root / "a")}
                a.check_inputs(pins)
                (root / "a").write_text("changed")
                with self.assertRaisesRegex(ValueError, "drift"):
                    a.check_inputs(pins)
                with self.assertRaises((ValueError, FileNotFoundError)):
                    a.check_inputs({"../outside": "0"*64})

    def test_conflicting_hashes_rejected(self):
        with self.assertRaisesRegex(ValueError, "conflicting"):
            a.add_pins({"x": "a"}, {"x": "b"})

    def predictions(self, gs):
        return [dict(group_id=g["group_id"], state_id=g["state_id"], map_id=g["map_id"], kind=g["kind"], held=True,
                     **{m: dict(selected="a") for m in ("aggregated_h32", "frozen_h32", "frozen_pool")}) for g in gs]

    def test_evaluation_separates_group_kinds_and_is_deterministic(self):
        gs = groups()
        p = self.predictions(gs)
        first = a.evaluate(gs, p, samples=100)
        self.assertEqual(first, a.evaluate(gs, p, samples=100))
        for k in ("old_grid", "matched_pair"):
            section = first["sections"][k]
            self.assertEqual(section["states"], 4)
            self.assertEqual(section["maps"], 3)
            self.assertEqual(section["metrics"]["gain_vs_frozen_h32"]["ci95"], [0, 0])
            self.assertEqual(section["changed_vs_frozen_h32"], 0)

    def test_evaluation_rejects_training_rows_and_missing_groups(self):
        gs = groups()
        p = self.predictions(gs)
        with self.assertRaisesRegex(ValueError, "coverage"):
            a.evaluate(gs, p[:-1], samples=10)
        p[0]["held"] = False
        with self.assertRaisesRegex(ValueError, "held-map"):
            a.evaluate(gs, p, samples=10)

    def test_evaluation_rejects_unknown_selection_and_unknown_labels(self):
        gs = groups()
        p = self.predictions(gs)
        p[0]["frozen_pool"]["selected"] = "unknown"
        with self.assertRaisesRegex(ValueError, "unknown"):
            a.evaluate(gs, p, samples=10)
        p = self.predictions(gs)
        gs[0]["candidates"][0]["values"][0] = None
        with self.assertRaisesRegex(ValueError, "incomplete"):
            a.evaluate(gs, p, samples=10)

    def test_chosen_rates_are_real_labels_not_scores(self):
        gs = groups()
        p = self.predictions(gs)
        for row in p:
            row["aggregated_h32"] = dict(selected="b", scores={"a": -1000, "b": 1000})
        result = a.evaluate(gs, p, samples=10)
        for row, g in zip(result["rows"], gs):
            rates = {c["candidate_id"]: sum(c["values"])/8 for c in g["candidates"]}
            self.assertEqual(row["aggregated_h32"], rates["b"])
            self.assertEqual(row["gain_vs_uniform"], rates["b"] - sum(rates.values())/len(rates))

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "sklearn not installed in this interpreter")
    def test_fixed_fit_is_deterministic_and_pickle_round_trip(self):
        import pickle
        import sklearn
        from sklearn.ensemble import HistGradientBoostingRegressor
        if sklearn.__version__ != "1.5.0": self.skipTest("requires registered sklearn 1.5.0")
        m = a.matrix(groups(), ["x", "y"], "m")
        def fit():
            return HistGradientBoostingRegressor(loss="squared_error", **a.MODEL_PARAMS).fit(m["x"], m["y"], sample_weight=m["weights"])
        first, second = fit(), fit()
        self.assertEqual(first.predict(m["x"]).tolist(), second.predict(m["x"]).tolist())
        self.assertEqual(first.predict(m["x"]).tolist(), pickle.loads(pickle.dumps(first)).predict(m["x"]).tolist())


if __name__ == "__main__":
    unittest.main()
