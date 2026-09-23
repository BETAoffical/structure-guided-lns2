import copy
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from scripts import probe_sa_terminal_budget as s


class BudgetTests(unittest.TestCase):
    def test_fixed_scope(self):
        cfg = s.config()
        self.assertIsNone(cfg["max_decisions"])
        self.assertEqual(cfg["node_budget"], 30000000)
        self.assertFalse(cfg["training"])
        self.assertFalse(cfg["formal_ttf"])
        self.assertFalse(cfg["automatic_promotion"])

    def test_feature_budget_is_not_stop_budget(self):
        plan = dict(proposal=dict(max_decisions=None, decision_feature_reference=256, node_budget=25000000))
        before = copy.deepcopy(plan)
        f = s.run.budget_features({}, 240, 24000000, s.runtime.feature_plan(plan)["proposal"])
        self.assertEqual(f["budget.remaining_node_fraction"], .04)
        self.assertEqual(s.run.budget_features({}, 400, 27000000, s.runtime.feature_plan(plan)["proposal"]),
                         {"budget.remaining_node_fraction": 0., "budget.remaining_decision_fraction": 0.})
        self.assertEqual(plan, before)
        self.assertIsNone(s.runtime.work_stop(False, 999999, 27000000, s.stop_config(s.config())))
        self.assertEqual(s.runtime.work_stop(False, 999999, 30000001, s.stop_config(s.config())), "node_budget")
        self.assertEqual(s.runtime.work_stop(True, 999999, 30000001, s.stop_config(s.config())), "feasible")

    def test_all_branches_not_only_close_to_zero(self):
        cfg = s.config()
        rows = [dict(trial=i, root=dict(episode_id=cfg["root_episode"], split="train")) for i in range(4)]
        source = dict(config=dict(node_budget=25000000))
        with patch.object(s.support, "jobs_for", return_value=rows):
            self.assertEqual(s.source_jobs(source, cfg), rows)
        with patch.object(s.support, "jobs_for", return_value=rows[:3]):
            with self.assertRaises(ValueError): s.source_jobs(source, cfg)
        rows[3]["root"]["split"] = "validation"
        with patch.object(s.support, "jobs_for", return_value=rows):
            with self.assertRaises(ValueError): s.source_jobs(source, cfg)

    def test_replay_science_ignores_only_clocks(self):
        a = dict(metrics=dict(step_runtime=1, repair_order=[2, 9]), delta=dict(nodes=3), action=dict(agents=[2, 9]), features=[.2])
        b = dict(a, metrics=dict(step_runtime=2, repair_order=[2, 9]))
        self.assertEqual(s.event_science(a), s.event_science(b))
        self.assertNotEqual(s.event_science(a), s.event_science(dict(b, metrics=dict(step_runtime=2, repair_order=[9, 2]))))
        for key in ("delta", "action", "features"):
            self.assertNotEqual(s.event_science(a), s.event_science(dict(b, **{key: None})))

    def test_no_random_stream_reset(self):
        plan = dict(config=dict(stream_seed=17))
        key = (plan, "original-terminal-support-v1", "root", 0, 350)
        self.assertEqual(s.run.stream_draw(*key, "pp"), s.run.stream_draw(*key, "pp"))
        self.assertNotEqual(s.run.stream_draw(*key, "pp"),
            s.run.stream_draw(plan, "budget-diagnostic", "root", 0, 350, "pp"))

    def test_new_success_cannot_change_original_budget(self):
        rows = [dict(trial=i, success=i == 0, original_budget_success=False, status="ok",
                     stop="feasible" if i == 0 else "node_budget") for i in range(4)]
        result = s.summary(rows)
        self.assertEqual(result["original_25m_successes"], 0)
        self.assertEqual(result["extended_30m_successes"], 1)
        self.assertEqual(result["independent_roots"], 1)
        rows[1].update(status="censored", stop="wall_safety")
        with self.assertRaises(ValueError): s.summary(rows)

    def test_missing_and_duplicate_trial_rejected(self):
        rows = [dict(trial=i, success=False, original_budget_success=False, status="ok", stop="node_budget") for i in range(4)]
        self.assertEqual(s.summary(rows)["extended_30m_successes"], 0)
        with self.assertRaises(ValueError): s.summary(rows[:3])
        with self.assertRaises(ValueError): s.summary(rows[:3] + [rows[0]])

    def test_path_quality_and_waits(self):
        self.assertEqual(s.quality(dict(agents=[dict(id=5, path=[1, 1, 2]), dict(id=9, path=[3, 4])])),
                         dict(soc=3, makespan=2, wait_steps=1))

    def test_physical_cycle_does_not_use_work_counters(self):
        a = dict(iteration=2, low_level=dict(generated=10), agents=[dict(id=9, path=[1, 2]), dict(id=3, path=[5])])
        b = dict(a, iteration=3, low_level=dict(generated=20), agents=list(reversed(a["agents"])))
        self.assertEqual(s.physical_signature(a), s.physical_signature(b))
        b["agents"] = [dict(id=3, path=[5]), dict(id=9, path=[1, 1, 2])]
        self.assertNotEqual(s.physical_signature(a), s.physical_signature(b))

    def test_frozen_execution_namespace_is_used(self):
        import ast
        tree = ast.parse((s.ROOT / s.CODE[1]).read_text(encoding="utf8"))
        worker = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "worker")
        call = next(n for n in ast.walk(worker) if isinstance(n, ast.Call) and
                    isinstance(n.func, ast.Attribute) and n.func.attr == "execute_step")
        self.assertIsInstance(call.args[0], ast.Subscript)
        self.assertEqual(call.args[0].slice.value, "source_job")
        stop = next(n for n in ast.walk(worker) if isinstance(n, ast.Call) and
                    isinstance(n.func, ast.Attribute) and n.func.attr == "work_stop")
        self.assertEqual(stop.args[-1].func.id, "stop_config")

    def test_prefix_replay_preserves_boundary_and_rejects_changed_features(self):
        expected = dict(decision=227, features=[.1], metrics=dict(step_runtime=1), delta=dict(nodes=4))
        before = dict(fp="root", low_level=dict(generated=24000010))
        after = dict(fp="boundary", low_level=dict(generated=25000020))
        q = SimpleNamespace(state_fingerprint=lambda state: state["fp"])
        h = SimpleNamespace(decision=227)
        h.observe = lambda *args: setattr(h, "decision", h.decision + 1)
        old = dict(job_id="old", phase="branches", registration=dict(config=dict(output="unused")))
        job = dict(source_job=old, record=dict(row=dict(final_fingerprint="boundary", absolute_decisions=228, generated=25000010)))
        replay = (q, None, before, dict(initial_nodes=10), h, None, None, None,
                  dict(proposal=dict(pp_safety_seconds=20)), None)
        with patch.object(s.support, "replay", return_value=replay), \
             patch.object(s.run, "trace_read", return_value=iter([expected])), \
             patch.object(s.support, "execute_step", return_value=(expected, after)), \
             patch("experiments.sa_onpolicy_noop_runtime.pp_incomplete", return_value=False):
            restored = s.prefix_replay(job)
            self.assertEqual(restored[2], after)
            self.assertEqual(restored[4].decision, 228)
        h.decision = 227
        with patch.object(s.support, "replay", return_value=replay), \
             patch.object(s.run, "trace_read", return_value=iter([expected])), \
             patch.object(s.support, "execute_step", return_value=(dict(expected, features=[.2]), after)):
            with self.assertRaisesRegex(ValueError, "science changed"):
                s.prefix_replay(job)


if __name__ == "__main__":
    unittest.main()
