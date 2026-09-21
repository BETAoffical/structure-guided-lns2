import importlib.util
import unittest

import numpy as np
from scripts.audit_sa_onpolicy_gradient import (cosine, directional_summary, require_zero_output,
                                               summarize_rows, zero_output_score)


class GradientAuditTests(unittest.TestCase):
    def rows(self):
        rows = []
        for m in range(2):
            for pair in range(m+1):
                for rep in range(4):
                    score = np.zeros(32)
                    score[0] = rep
                    rows.append(dict(episode_id=f"{m}:{pair}:{rep}", pair_id=f"{m}:{pair}",
                                     map_id=str(m), replica=rep, split="train", return_value=float(rep >= 2), score=score.tolist()))
        return rows

    @unittest.skipUnless(importlib.util.find_spec("torch"), "existing Windows torch environment")
    def test_exact_zero_output_identity_against_autograd(self):
        import torch
        h = torch.tensor([[.7, -.3], [-.2, .9], [.1, .4]], dtype=torch.float64, requires_grad=True)
        w = torch.zeros(2, dtype=torch.float64, requires_grad=True)
        bias = torch.tensor(0., dtype=torch.float64, requires_grad=True)
        p = np.array([.9, .05, .05])
        distribution = torch.distributions.Categorical(logits=torch.tensor(p).log() + 2 * torch.tanh(h @ w + bias))
        for selected in range(3):
            dh, dw, db = torch.autograd.grad(distribution.log_prob(torch.tensor(selected)), (h, w, bias), retain_graph=True)
            np.testing.assert_allclose(dw.numpy(), zero_output_score(h.detach().numpy(), p, selected), atol=1e-14)
            self.assertEqual(float(dh.abs().sum()), 0.)
            self.assertAlmostEqual(float(db), 0., places=14)

    def test_condition_map_weights_and_no_trajectory_length_division(self):
        rows = self.rows()
        _, _, weights, _ = summarize_rows(rows)
        self.assertAlmostEqual(abs(weights["0:0:0"]), 1/12)
        self.assertAlmostEqual(abs(weights["1:0:0"]), 1/24)
        doubled = [dict(r, score=(2*np.asarray(r["score"])).tolist()) for r in rows]
        np.testing.assert_allclose(summarize_rows(doubled)[0], 2*summarize_rows(rows)[0])

    def test_all_equal_returns_zero_credit_and_zero_norm_is_unknown(self):
        rows = [dict(r, return_value=1.) for r in self.rows()]
        report, weights = directional_summary(rows)
        self.assertEqual(report["norm"], 0.)
        self.assertTrue(all(v == 0 for v in weights.values()))
        self.assertTrue(all(s["cosine"] is None for s in report["split_replicas"]))
        self.assertIsNone(cosine(np.zeros(32), np.ones(32)))

    def test_all_three_splits_and_permutation_invariance(self):
        rows = self.rows()
        report, _ = directional_summary(rows)
        reverse, _ = directional_summary(rows[::-1])
        self.assertEqual(report, reverse)
        self.assertEqual(len(report["split_replicas"]), 3)
        self.assertIsNone(report["split_replicas"][0]["cosine"])
        self.assertAlmostEqual(report["split_replicas"][1]["cosine"], 1.)

    def test_reject_heldout_missing_replica_invalid_score_and_nonzero_output(self):
        rows = self.rows()
        with self.assertRaises(ValueError):
            summarize_rows([dict(rows[0], split="validation"), *rows[1:]])
        with self.assertRaises(ValueError):
            summarize_rows(rows[:-1])
        with self.assertRaises(ValueError):
            zero_output_score([[0.], [1.]], [.5, .8], 0)
        with self.assertRaises(ValueError):
            require_zero_output(type("Actor", (), dict(w2=np.array([[.1]]), b2=np.array([0.])))())

    def test_candidate_common_hidden_mode_has_zero_score(self):
        np.testing.assert_array_equal(zero_output_score(np.ones((4, 32)), [.7, .1, .1, .1], 2), np.zeros(32))


if __name__ == "__main__":
    unittest.main()
