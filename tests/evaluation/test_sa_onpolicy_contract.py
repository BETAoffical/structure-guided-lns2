from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import unittest

from experiments._common import json_fingerprint, sha256_file
from experiments.sa_onpolicy_contract import anchor_distribution, terminal_return, gradient_coefficients
from scripts.check_sa_onpolicy_contract import CONFIG, ROOT, inspect

POLICY = "a" * 64


def episode(pair="a1", map_id="a", replica=0, success=True, decisions=1):
    if not success:
        decisions = 3
    return dict(episode_id=f"{pair}-{replica}", pair_id=pair, map_id=map_id, replica=replica,
                policy_sha256=POLICY, split="train", status="ok",
                initial_fingerprint=json_fingerprint(pair), rng_stream_id=json_fingerprint([pair, replica]),
                stop="feasible" if success else "decision_budget", success=success,
                final_conflicts=0 if success else 1, decisions=decisions, generated=10,
                steps=[dict(decision=d, policy_sha256=POLICY, probabilities={"x": .75, "y": .25},
                            selected_id="x", behavior_log_probability=math.log(.75)) for d in range(decisions)])


def coefficients(rows, groups=None):
    return gradient_coefficients(rows, policy_sha256=POLICY, expected_groups=groups or {"a1": "a"},
                                 replicas=2, max_decisions=3, node_budget=100)


class OnPolicyContractTests(unittest.TestCase):
    def test_zero_residual_is_exploration_control_not_frozen_argmax(self):
        result = anchor_distribution(["b", "a"], "a", {"a": 0, "b": 0})
        self.assertAlmostEqual(result["a"], .95)
        self.assertEqual(result["b"], .05)
        self.assertEqual(result, anchor_distribution(["a", "b"], "a", {"b": 0, "a": 0}))
        self.assertEqual(anchor_distribution(["a"], "a", {"a": 2}), {"a": 1.})

    def test_permutation_and_common_logit_shift(self):
        a = anchor_distribution(["a", "b"], "a", {"a": -.5, "b": .5})
        b = anchor_distribution(["b", "a"], "a", {"a": .5, "b": 1.5})
        for key in a:
            self.assertAlmostEqual(a[key], b[key])
        self.assertAlmostEqual(sum(a.values()), 1)

    def test_reject_invalid_action_distributions(self):
        for ids, anchor, residuals, eps in [
            (["a", "a"], "a", {"a": 0}, .1), (["a"], "b", {"a": 0}, .1),
            (["a"], "a", {"a": float("nan")}, .1), (["a"], "a", {"a": 2.1}, .1),
            (["a"], "a", {"a": 0}, 0), (["a"], "a", {"a": 0}, True),
        ]:
            with self.assertRaises(ValueError):
                anchor_distribution(ids, anchor, residuals, eps)

    def test_leave_one_out_credit_is_episode_not_step_weighted(self):
        rows = [episode(), episode(replica=1, success=False)]
        r = coefficients(rows)
        self.assertEqual([v["baseline"] for v in r], [0, 1])
        self.assertEqual([v["coefficient"] for v in r], [.5, -.5])
        rows[0] = episode(decisions=2)
        self.assertEqual(coefficients(rows), r)

    def test_map_weighting_does_not_reward_maps_with_more_conditions(self):
        groups = {"a1": "a", "a2": "a", "b1": "b"}
        rows = [episode(pair, m, r, success=(r == 0)) for pair, m in groups.items() for r in range(2)]
        result = coefficients(rows, groups)
        self.assertAlmostEqual(sum(r["episode_weight"] for r in result), 1)
        self.assertAlmostEqual(sum(r["episode_weight"] for r in result if r["episode_id"].startswith("a")), .5)
        self.assertEqual(result, coefficients(rows[::-1], groups))

    def test_all_equal_returns_have_no_comparison_signal(self):
        for success in (True, False):
            rows = [episode(replica=r, success=success) for r in range(2)]
            self.assertEqual([r["coefficient"] for r in coefficients(rows)], [0, 0])

    def test_all_censoring_reasons_block_entire_update(self):
        for stop in ("wall_safety", "external_timeout", "incomplete_pp", "user_stop"):
            rows = [episode(), episode(replica=1, success=False)]
            rows[1].update(status="censored", stop=stop)
            self.assertIsNone(terminal_return(rows[1], max_decisions=3, node_budget=100))
            with self.assertRaisesRegex(ValueError, "censored batch"):
                coefficients(rows)

    def test_budget_and_terminal_consistency(self):
        row = episode(success=False)
        self.assertEqual(terminal_return(row, max_decisions=3, node_budget=100), 0)
        row.update(stop="node_budget", generated=101, decisions=2)
        self.assertEqual(terminal_return(row, max_decisions=3, node_budget=100), 0)
        row.update(generated=99)
        with self.assertRaises(ValueError):
            terminal_return(row, max_decisions=3, node_budget=100)
        row = episode()
        row["final_conflicts"] = 1
        with self.assertRaises(ValueError):
            terminal_return(row, max_decisions=3, node_budget=100)

    def test_missing_duplicate_or_evaluation_episode_rejected(self):
        good = [episode(), episode(replica=1, success=False)]
        for change in (lambda r: r.pop(), lambda r: r.append(deepcopy(r[0])),
                       lambda r: r[1].update(replica=0), lambda r: r[0].update(split="validation"),
                       lambda r: r[0].update(map_id="held"), lambda r: r[0].update(policy_sha256="b" * 64)):
            rows = deepcopy(good)
            change(rows)
            with self.assertRaises(ValueError):
                coefficients(rows)

    def test_mid_episode_policy_switch_and_unlogged_action_rejected(self):
        for key, value in (("policy_sha256", "b" * 64), ("selected_id", "not-a-candidate"),
                           ("behavior_log_probability", -9), ("decision", True),
                           ("probabilities", {"x": .8, "y": .8})):
            rows = [episode(), episode(replica=1)]
            rows[0]["steps"][0][key] = value
            with self.assertRaises(ValueError):
                coefficients(rows)

    def test_baseline_replicas_share_initial_state_not_random_stream(self):
        rows = [episode(), episode(replica=1)]
        rows[1]["initial_fingerprint"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "initial state"):
            coefficients(rows)
        rows = [episode(), episode(replica=1)]
        rows[1]["rng_stream_id"] = rows[0]["rng_stream_id"]
        with self.assertRaisesRegex(ValueError, "random stream"):
            coefficients(rows)

    def test_zero_conflict_episode_retained_with_zero_action_terms(self):
        rows = [episode(replica=r, decisions=0) for r in range(2)]
        self.assertEqual(len(coefficients(rows)), 2)
        self.assertEqual([r["coefficient"] for r in coefficients(rows)], [0, 0])

    def test_design_preflight_is_not_permission_to_train(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = json.loads((ROOT / CONFIG).read_text(encoding="utf8"))
            source = dict(split=dict(train_maps=[f"m{i}" for i in range(6)], validation_maps=["m6", "m7"]),
                          cases=[dict(task_id=f"m{i}-t{t}", map_id=f"m{i}", solver_seeds=[227, 229])
                                 for i in range(8) for t in range(2)])
            source["binding"] = json_fingerprint(source)
            path = root / cfg["source_plan"]
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(source), encoding="utf8")
            cfg["source_plan_sha256"] = sha256_file(path)
            cfg["historical_documents"] = {}
            config = root / CONFIG
            config.parent.mkdir(parents=True)
            config.write_text(json.dumps(cfg), encoding="utf8")
            report = inspect(root)
            self.assertEqual(report["status"], "design_checked_not_execution_ready")
            self.assertEqual((report["proposed_training_episodes"], report["proposed_evaluation_episodes"]), (192, 64))
            self.assertEqual(report["maximum_proposed_repairs"], 65536)
            self.assertEqual(report["actual_training_updates"], 0)
            cfg["training_authorized"] = True
            config.write_text(json.dumps(cfg), encoding="utf8")
            with self.assertRaises(ValueError):
                inspect(root)

    def test_historical_input_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = json.loads((ROOT / CONFIG).read_text(encoding="utf8"))
            config = root / CONFIG
            config.parent.mkdir(parents=True)
            cfg["historical_documents"] = {"changed.json": "a" * 64}
            config.write_text(json.dumps(cfg), encoding="utf8")
            (root / "changed.json").write_text("{}", encoding="utf8")
            with self.assertRaisesRegex(ValueError, "evidence changed"):
                inspect(root)


if __name__ == "__main__":
    unittest.main()
