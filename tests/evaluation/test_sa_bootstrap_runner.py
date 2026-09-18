import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_sa_bootstrap_closed_loop as runner


class RunnerTests(unittest.TestCase):
    def test_blind_selection_order_and_outcomes(self):
        cfg = dict(maps=2, case_seed=42)
        cases = [dict(map_id=m, task_id=m+t) for m in ("a", "b") for t in ("0", "1")]
        selected = runner.select_cases(cases, cfg)
        changed = copy.deepcopy(cases[::-1])
        for c in changed:
            c["success"] = c["task_id"] not in {x["task_id"] for x in selected}
        self.assertEqual([c["task_id"] for c in selected],
                         [c["task_id"] for c in runner.select_cases(changed, cfg)])

    def test_scoped_adapter_counter_and_restore(self):
        original = runner.previous.choose
        old_load = runner.previous.models_load
        job = dict(pair_id="task-s1", plan=dict(config=dict(selection_seed=12)))
        calls = []
        def fake(*args):
            calls.append(args[-3:])
            return dict(selected="a")
        with patch.object(runner, "choose", fake):
            for repeat in range(2):
                with runner.adapted_runner(job):
                    for _ in range(2):
                        runner.previous.choose("posterior", [], "a", dict(gbdt=None, members=[]), [], [])
                self.assertIs(runner.previous.choose, original)
                self.assertIs(runner.previous.models_load, old_load)
        self.assertEqual(calls, [("task-s1", 0, 12), ("task-s1", 1, 12)]*2)

    def test_adapter_exception_restores(self):
        original = runner.previous.choose
        with self.assertRaises(RuntimeError):
            with runner.adapted_runner(dict(pair_id="x", plan=dict(config=dict(selection_seed=1)))):
                raise RuntimeError("interruption")
        self.assertIs(runner.previous.choose, original)

    def test_complete_four_arm_schedule(self):
        p = dict(cases=[dict(task_id="x", map_id="m")], config=dict(solver_seeds=[227,229]),
                 anchors={"x-s227":"a", "x-s229":"b"})
        jobs = runner.schedule(p)
        self.assertEqual(len(jobs), 8)
        self.assertEqual(len({j["job_id"] for j in jobs}), 8)
        for seed in (227,229):
            self.assertEqual({j["arm"] for j in jobs if j["solver_seed"] == seed}, set(runner.ARMS))
            self.assertEqual({j["expected_initial"] for j in jobs if j["solver_seed"] == seed}, {p["anchors"][f"x-s{seed}"]})

    def test_missing_member_receipt_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            with patch.object(runner.previous, "model_receipt", return_value=dict(files={"gbdt.json":"sha"})):
                with self.assertRaises(ValueError):
                    runner.model_receipt({}, Path(root))

    def test_qualification_identity_not_just_seal(self):
        p = dict(binding="b", anchors={"task-s1":"expected"})
        bad = runner.sealed(dict(binding="b", passed=True, results=[dict(status="ok", job_id="task-s1", initial="wrong")]))
        with patch.object(runner, "read_json", return_value=bad):
            with self.assertRaises(ValueError):
                runner.qualification(p, Path("unused"))

    def test_resume_rejects_every_identity_field(self):
        job = dict(job_id="x-frozen", pair_id="x", arm="frozen", solver_seed=1,
                   expected_initial="fp", plan=dict(binding="b"), case=dict(map_id="m"))
        result = dict(binding="b", job_id="x-frozen", pair_id="x", arm="frozen", solver_seed=1,
                      initial_fingerprint="fp", map_id="m")
        runner.validate_result_identity(result, job)
        for key in result:
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.validate_result_identity(result | {key:"altered"}, job)

    def test_incomplete_pp_must_terminate(self):
        self.assertFalse(runner.incomplete(None))
        self.assertFalse(runner.incomplete(dict(metrics=dict(pp_failure_reason="", acceptance_evaluated=True))))
        self.assertTrue(runner.incomplete(dict(metrics=dict(pp_failure_reason="time_limit", acceptance_evaluated=True))))
        self.assertTrue(runner.incomplete(dict(metrics=dict(pp_failure_reason="", acceptance_evaluated=False))))

    def test_expansion_requires_unmodified_smoke(self):
        with tempfile.TemporaryDirectory() as root:
            out = Path(root)
            with self.assertRaises(ValueError):
                runner.require_smoke({}, out)
            runner.once(out / "smoke-audit.json", dict(binding="old"))
            with patch.object(runner, "smoke_record", return_value=dict(binding="new")):
                with self.assertRaises(ValueError):
                    runner.require_smoke({}, out)


if __name__ == "__main__":
    unittest.main()
