from copy import deepcopy
import importlib.util
import unittest

import numpy as np
from experiments.sa_crossfit_update import (state_columns, state_vector, fold_masks, fit_value,
    predict_value, padded_episode, tensor_distribution, probability_stats, within_budget, guarded_update)
from experiments.sa_onpolicy_actor import initial_bundle, torch_actor, torch_distribution, NumpyActor


class BaselineTests(unittest.TestCase):
    def test_action_independent_input_and_future_feature_exclusion(self):
        names = ["budget.left", "outcome.success", "proposal.seed", "sa.temperature", "state.conflicts"]
        columns = state_columns(names)
        self.assertEqual(columns, [0, 3, 4])
        x = np.array([[1, 0, 3, .1, 6], [1, 1, 9, .1, 6]])
        np.testing.assert_array_equal(state_vector(x, columns), [1, .1, 6])
        x[1, 4] = 7
        with self.assertRaises(ValueError): state_vector(x, columns)

    def test_whole_episode_folds_and_no_heldout_labels(self):
        metadata = [dict(episode_id=str(i), split="train", replica=i%4) for i in range(8)]
        for held in range(4):
            train, test = fold_masks(metadata, held)
            self.assertEqual((sum(train), sum(test)), (6, 2))
            self.assertFalse(np.any(train & test))
        with self.assertRaises(ValueError): fold_masks([dict(metadata[0], split="validation"), *metadata[1:]], 0)
        metadata[1]["episode_id"] = metadata[0]["episode_id"]
        with self.assertRaises(ValueError): fold_masks(metadata, 0)

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "existing global Windows sklearn environment")
    def test_ridge_portable_values_and_weights(self):
        x = np.arange(24, dtype=float).reshape(8, 3)
        y = np.array([0., 0., 0., 1., 0., 1., 1., 1.])
        model = fit_value(x, y, np.ones(8), 1.)
        p = predict_value(model, x)
        self.assertTrue(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all())
        self.assertGreater(p[-1], p[0])
        with self.assertRaises(ValueError): fit_value(x, y, np.ones(8), .1)

    def test_probability_budget_checks_all_limits(self):
        limits = dict(mean_kl=.002, mean_trajectory_kl=.1, max_state_kl=.02)
        self.assertTrue(within_budget(limits, limits))
        for key in limits:
            self.assertFalse(within_budget(dict(limits, **{key:limits[key]+1e-10}), limits))


@unittest.skipUnless(importlib.util.find_spec("torch"), "existing Windows torch environment")
class PackedActorTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1)

    def fixture(self):
        fs = [{f"f{i:03d}": float(i/129 + k) for i in range(129)} for k in (-1, 0, 1)]
        actor = initial_bundle(fs, 11, "test")
        data = dict(x=np.asarray([[f[n] for n in actor["feature_names"]] for f in [*fs[:2], *fs]]),
                    lengths=np.array([2, 3]), probabilities=np.array([.95, .05, 1/30, 14/15, 1/30]))
        return fs, actor, data

    def test_ragged_probabilities_and_gradient_equal_original(self):
        import torch
        fs, actor, data = self.fixture()
        model = torch_actor(actor)
        padded = padded_episode(data, actor)
        logp = tensor_distribution(model, actor, padded)
        self.assertEqual(float(logp.exp()[0, 2].detach()), 0.)
        selected = torch.tensor([1, 2])
        batch_loss = -(logp[torch.arange(2), selected]*torch.tensor([.7, -.2], dtype=torch.float64)).sum()
        a = torch.autograd.grad(batch_loss, tuple(model.parameters()))
        loss = -.7*torch_distribution(model, actor, ["a", "b"], "a", fs[:2]).log_prob(torch.tensor(1))
        loss += .2*torch_distribution(model, actor, ["a", "b", "c"], "b", fs).log_prob(torch.tensor(2))
        b = torch.autograd.grad(loss, tuple(model.parameters()))
        for ga, gb in zip(a, b): np.testing.assert_allclose(ga, gb, atol=1e-12, rtol=0)

    def test_one_bounded_step_preserves_parent_and_accepts_no_future_reward(self):
        import torch
        fs, actor, data = self.fixture()
        old = deepcopy(actor)
        padded = padded_episode(data, actor)
        model = torch_actor(actor)
        loss = -tensor_distribution(model, actor, padded)[0, 1]
        gradient = torch.autograd.grad(loss, tuple(model.parameters()))
        pack = dict(padded=padded, weight=1., draws=np.array([.96, .99]), lengths=data["lengths"], selected=np.array([1, 2]))
        config = dict(initial_l2=.25, backtracks=12, mean_kl=.002, mean_trajectory_kl=.1, max_state_kl=.02)
        updated, report = guarded_update(actor, gradient, [pack], config, "fork", "bounded_state")
        self.assertIsNotNone(updated)
        self.assertTrue(within_budget(report["chosen"], config))
        self.assertEqual(old, actor)
        self.assertEqual(updated["w1"], actor["w1"])
        self.assertGreater(NumpyActor(updated).probabilities(["a", "b"], "a", fs[:2])["b"], .05)
        zero = dict(config, mean_kl=0., mean_trajectory_kl=0., max_state_kl=0., backtracks=1)
        self.assertIsNone(guarded_update(actor, gradient, [pack], zero, "fork", "bounded_state")[0])
        with self.assertRaises(ValueError): guarded_update(updated, gradient, [pack], config, "fork", "bounded_state")

    def test_zero_probability_shift_is_not_success_improvement(self):
        _, actor, data = self.fixture()
        padded = padded_episode(data, actor)
        pack = dict(padded=padded, weight=1., draws=np.array([.96, .99]), lengths=data["lengths"], selected=np.array([1, 2]))
        stats = probability_stats([pack], [padded[1]])
        self.assertEqual(stats["changed_actions"], 0)
        self.assertEqual(stats["mean_kl"], 0.)
        self.assertNotIn("success", stats)


if __name__ == "__main__":
    unittest.main()
