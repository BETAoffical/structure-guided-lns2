from copy import deepcopy
from contextlib import nullcontext
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.sa_onpolicy_actor import NumpyActor, initial_bundle, torch_actor, torch_distribution, validate_bundle
from experiments.sa_crossfit_update import tensor_distribution, within_budget
from experiments.sa_parent_update import padded_with_prior, log_distribution, guarded_parent_update
from scripts import train_sa_parent_update as driver


class ScopeTests(unittest.TestCase):
    def test_frozen_scope_rejects_cap_control_training_or_extra_update(self):
        cfg = driver.run.read_json(driver.ROOT/driver.CONFIG)
        driver.fixed_scope(cfg)
        for key, value in dict(max_decisions=256, training_arm="untrained_exploration", maximum_updates=2,
            heldout_evaluation=True, solver_calls=1, parent_iteration=0).items():
            with self.assertRaises(ValueError):
                driver.fixed_scope(dict(cfg, **{key: value}))
        changed = deepcopy(cfg)
        changed["update"]["initial_l2"] = .5
        with self.assertRaises(ValueError):
            driver.fixed_scope(changed)

    def test_recomputed_credit_is_required(self):
        reg = dict(entries=[], parent_policy="a"*64, credits=[])
        with self.assertRaises(ValueError):
            driver.validate_credit(reg, [])
        expected = [dict(episode_id="x", coefficient=.1)]
        reg["credits"] = expected
        with patch.object(driver.credit, "gradient_coefficients", return_value=expected) as compute:
            self.assertEqual(driver.validate_credit(reg, [dict(split="train")]), {"x": expected[0]})
            self.assertIsNone(compute.call_args.kwargs["max_decisions"])
            self.assertEqual(compute.call_args.kwargs["policy_sha256"], "a"*64)
        with patch.object(driver.credit, "gradient_coefficients", return_value=[dict(episode_id="x", coefficient=.2)]):
            with self.assertRaises(ValueError):
                driver.validate_credit(reg, [])

    def test_cache_rejects_duplicate_control_or_changed_source(self):
        entries, rows, credits = [], [], []
        for i in range(96):
            episode = dict(job_id=str(i), episode_id=str(i), split="train", arm="trained_actor", decisions=1)
            weight = dict(episode_id=str(i), coefficient=0.)
            entries.append(dict(job_id=str(i), episode=deepcopy(episode), result_sha256="a"*64))
            rows.append((dict(episode=episode, credit=weight, source_sha256="a"*64), dict(lengths=[2])))
            credits.append(weight)
        reg = dict(entries=entries, credits=credits)
        with patch.object(driver.cache_tools, "cached", return_value=rows):
            self.assertEqual(len(driver.cached(reg, Path("unused"))), 96)
        wrong = [rows[0], *rows[:-1]]
        with patch.object(driver.cache_tools, "cached", return_value=wrong):
            with self.assertRaises(ValueError):
                driver.cached(reg, Path("unused"))
        for key, value in (("split", "development_holdout"), ("arm", "untrained_exploration")):
            wrong = deepcopy(rows)
            wrong[0][0]["episode"][key] = value
            with patch.object(driver.cache_tools, "cached", return_value=wrong):
                with self.assertRaises(ValueError):
                    driver.cached(reg, Path("unused"))
        wrong = deepcopy(rows)
        wrong[0][0]["source_sha256"] = "b"*64
        with patch.object(driver.cache_tools, "cached", return_value=wrong):
            with self.assertRaises(ValueError):
                driver.cached(reg, Path("unused"))


@unittest.skipUnless(importlib.util.find_spec("torch"), "existing Windows Torch environment")
class ParentUpdateTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1)

    def fixture(self, zero=False):
        rng = np.random.default_rng(37)
        fs = [dict(zip([f"f{i:03d}" for i in range(129)], v.tolist())) for v in rng.normal(size=(4, 129))]
        parent = initial_bundle(fs, 11, "test")
        if not zero:
            parent.update(iteration=1, parent_policy="a"*64, w2=rng.normal(0, .08, (1, 32)).tolist(), b2=[.03])
        actor = NumpyActor(parent)
        lengths = np.array([2, 4, 1])
        anchors = np.array([0, 2, 0])
        ids = [["a", "b"], ["a", "b", "c", "d"], ["a"]]
        groups = [fs[:2], fs, fs[:1]]
        probabilities = [actor.probabilities(cs, cs[a], features) for cs, a, features in zip(ids, anchors, groups)]
        data = dict(x=np.asarray([[f[n] for n in parent["feature_names"]] for group in groups for f in group]),
            lengths=lengths, anchors=anchors, probabilities=np.asarray([p[c] for cs, p in zip(ids, probabilities) for c in cs]))
        return parent, ids, groups, data

    def test_nonzero_parent_matches_portable_and_fixed_prior_not_behavior(self):
        parent, _, _, data = self.fixture()
        padded, prior = padded_with_prior(data, parent)
        actual = log_distribution(torch_actor(parent), parent, padded, prior).exp().detach().numpy()
        np.testing.assert_allclose(actual, padded[1], atol=1e-12, rtol=0)
        self.assertEqual(actual[0, 3], 0.)
        self.assertEqual(actual[2, 0], 1.)
        self.assertGreater(float(np.max(np.abs(prior-padded[1]))), 1e-4)
        doubled = tensor_distribution(torch_actor(parent), parent, padded).exp().detach().numpy()
        self.assertGreater(float(np.max(np.abs(actual-doubled))), 1e-4)
        with self.assertRaises(ValueError):
            log_distribution(torch_actor(parent), parent, padded, padded[1])

    def test_gradient_matches_independent_single_state_and_finite_difference(self):
        import torch
        parent, ids, groups, data = self.fixture()
        padded, prior = padded_with_prior(data, parent)
        model = torch_actor(parent)
        weights = torch.tensor([.7, -.2, .1], dtype=torch.float64)
        selected = torch.tensor([1, 3, 0])
        loss = -(weights * log_distribution(model, parent, padded, prior)[torch.arange(3), selected]).sum()
        g = torch.autograd.grad(loss, tuple(model.parameters()))
        def direct():
            return sum(-w * torch_distribution(model, parent, cs, cs[a], fs).log_prob(i)
                for w, cs, a, fs, i in zip(weights, ids, data["anchors"], groups, selected))
        h = torch.autograd.grad(direct(), tuple(model.parameters()))
        for ga, gb in zip(g, h):
            np.testing.assert_allclose(ga, gb, atol=1e-12, rtol=0)
        for p, analytic in zip(model.parameters(), g):
            index = int(analytic.abs().argmax())
            flat = p.view(-1)
            old = float(flat[index].detach())
            with torch.no_grad():
                flat[index] = old + 1e-6
                plus = float(direct())
                flat[index] = old - 1e-6
                minus = float(direct())
                flat[index] = old
            self.assertAlmostEqual((plus-minus)/2e-6, float(analytic.flatten()[index]), places=7)

    def test_zero_residual_reference_remains_compatible(self):
        parent, _, _, data = self.fixture(zero=True)
        padded, prior = padded_with_prior(data, parent)
        model = torch_actor(parent)
        np.testing.assert_allclose(log_distribution(model, parent, padded, prior).detach().numpy(),
            tensor_distribution(model, parent, padded).detach().numpy(), atol=1e-12, rtol=0)

    def test_candidate_order_invariance_and_anchor_validation(self):
        parent, ids, groups, data = self.fixture()
        actor = NumpyActor(parent)
        expected = actor.probabilities(ids[1], "c", groups[1])
        order = [3, 1, 0, 2]
        actual = actor.probabilities([ids[1][i] for i in order], "c", [groups[1][i] for i in order])
        for key in expected:
            self.assertAlmostEqual(expected[key], actual[key], places=14)
        for bad in (np.array([2, 2, 0]), np.array([-1, 2, 0]), np.array([0., 2., 0.])):
            with self.assertRaises(ValueError):
                padded_with_prior(dict(data, anchors=bad), parent)
        with self.assertRaises(ValueError):
            padded_with_prior(dict(data, lengths=data["lengths"].astype(float)), parent)

    def test_guarded_update_uses_parent_kl_preserves_lineage_and_is_deterministic(self):
        import torch
        parent, _, _, data = self.fixture()
        saved = deepcopy(parent)
        padded, prior = padded_with_prior(data, parent)
        model = torch_actor(parent)
        g = torch.autograd.grad(-log_distribution(model, parent, padded, prior)[0, 1], tuple(model.parameters()))
        pack = dict(padded=padded, prior=prior, weight=1., draws=np.array([.96, .99, .1]),
                    lengths=data["lengths"], selected=np.array([1, 3, 0]))
        cfg = dict(initial_l2=.25, backtracks=12, mean_kl=.002, mean_trajectory_kl=.1, max_state_kl=.02)
        child, report = guarded_parent_update(parent, g, [pack], cfg, "next", "second_condition")
        second, repeated = guarded_parent_update(parent, g, [pack], cfg, "next", "second_condition")
        self.assertEqual(child, second)
        self.assertEqual(report, repeated)
        self.assertEqual(parent, saved)
        self.assertEqual(child["iteration"], 2)
        self.assertEqual(child["parent_policy"], validate_bundle(parent))
        self.assertEqual(child["mean"], parent["mean"])
        self.assertEqual(child["scale"], parent["scale"])
        self.assertTrue(within_budget(report["chosen"], cfg))
        _, ids, fs, _ = self.fixture()
        self.assertGreater(NumpyActor(child).probabilities(ids[0], "a", fs[0])["b"],
                           NumpyActor(parent).probabilities(ids[0], "a", fs[0])["b"])
        zero = dict(cfg, mean_kl=0., mean_trajectory_kl=0., max_state_kl=0., backtracks=1)
        self.assertIsNone(guarded_parent_update(parent, g, [pack], zero, "next", "second_condition")[0])
        with self.assertRaises(ValueError):
            guarded_parent_update(parent, [torch.zeros_like(x) for x in g], [pack], cfg, "next", "second_condition")

    def test_recorded_or_partially_published_update_cannot_train_again(self):
        parent, _, _, _ = self.fixture()
        reg = dict(binding="test", parent_policy=validate_bundle(parent))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            driver.run.write_json(out/"cache.json", {})
            driver.run.write_json(out/"numerical_check.json", driver.run.sealed(dict(binding="test",
                parent_policy=reg["parent_policy"], cache_sha256=driver.run.sha256_file(out/"cache.json"))))
            for relative in ("update.json", "models/actor-2.json"):
                driver.run.write_json(out/relative, {})
                with patch.object(driver, "verify", return_value=(reg, {}, out, parent)), \
                     patch.object(driver.recovery, "strict_lock", return_value=nullcontext()), \
                     patch.object(driver, "cached") as load:
                    with self.assertRaises(ValueError):
                        driver.train()
                    load.assert_not_called()
                (out/relative).unlink()


if __name__ == "__main__":
    unittest.main()
