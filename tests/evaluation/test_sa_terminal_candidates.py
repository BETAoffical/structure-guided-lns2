import copy
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from scripts import probe_sa_terminal_candidates as p


def target():
    return dict(id="root", fingerprint="fingerprint", trial=0, decision=310, edge=[4, 9],
        event=dict(selected_id="b", action=dict(mode="explicit_neighborhood", agents=[4, 9], random_seed=19),
                   pool=[dict(candidate_id="a", agents=[4]), dict(candidate_id="b", agents=[4, 9])]))


def native_micro():
    import lns2_env
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "tiny.map").write_text("type octile\nheight 3\nwidth 4\nmap\n....\n....\n....\n", encoding="utf8")
        (root / "tiny.scen").write_text("version 1\n0\ttiny.map\t4\t3\t0\t0\t3\t0\t3\n0\ttiny.map\t4\t3\t3\t0\t0\t0\t3\n", encoding="utf8")
        def make():
            env = lns2_env.LNS2RepairEnv(str(root / "tiny.map"), str(root / "tiny.scen"), 2)
            state = p.q._plain(env.reset_paths([[0, 1, 2, 3], [3, 2, 1, 0]], seed=11))
            return env, state
        original, state = make()
        action = dict(mode="explicit_neighborhood", agents=[0, 1], random_seed=31)
        raw = p.q._plain(original.step_experimental_pp(action, 20., "annealed", 100., .4))
        t = dict(id="micro", fingerprint=p.q.state_fingerprint(state), event=dict(action=action,
            selected_id="both", temperature=100., uniform=.4, metrics=raw["metrics"],
            delta=p.q.encode_state_delta(state, raw["observation"]),
            pool=[dict(candidate_id="both", agents=[0, 1]), dict(candidate_id="one", agents=[0])]))
        cfg = dict(p.config(), output=str(root / "out"), workers=2)
        reg = dict(config=cfg, binding="micro")
        env, before = make()
        jobs = p.branch_jobs(reg, t)
        out = Path(cfg["output"])
        p.run.write_json(out / "STOP_AFTER_JOB", {})
        assert not p.run_forks(env, before, t, reg, jobs, False)
        (out / "STOP_AFTER_JOB").unlink()
        assert p.run_forks(env, before, t, reg, jobs, False)
        digests = {}
        for job in jobs:
            row = p.read_branch(reg, t, job)
            assert row["status"] == "ok"
            p.check_branch(before, row, t, job)
            digests[job["job_id"]] = p.run.sha256_file(p.branch_folder(reg, t, job) / "result.json")
        assert p.run_forks(env, before, t, reg, jobs, True)
        assert digests == {j["job_id"]: p.run.sha256_file(p.branch_folder(reg, t, j) / "result.json") for j in jobs}
        assert p.q.state_fingerprint(p.q._plain(env.get_state())) == t["fingerprint"]
        row = p.read_branch(reg, t, jobs[1])
        try:
            p.check_branch(before, dict(row, generated=row["generated"]+1), t, jobs[1])
        except ValueError: pass
        else: raise AssertionError("tampered accounting accepted")
        (p.branch_folder(reg, t, jobs[0]) / "result.json").unlink()
        try: p.run_forks(env, before, t, reg, jobs[:1], True)
        except ValueError: pass
        else: raise AssertionError("partial result accepted")
        timeout_reg = dict(reg, config=dict(cfg, output=str(root / "timeout"), branch_fuse_seconds=.01))
        def pause_child(*args): time.sleep(5.)
        with patch.object(p, "branch_worker", pause_child):
            try: p.run_forks(env, before, t, timeout_reg, jobs[:1], False)
            except ValueError as error: assert "timeout is unknown" in str(error)
            else: raise AssertionError("hard timeout not enforced")
        receipt = p.run.check_seal(p.run.read_json(p.branch_folder(timeout_reg, t, jobs[0]) / "timeout.json"))
        assert receipt["status"] == "censored" and receipt["stop"] == "external_timeout"
        assert p.q.state_fingerprint(p.q._plain(env.get_state())) == t["fingerprint"]


class TerminalCandidateTests(unittest.TestCase):
    def test_scope(self):
        cfg = p.config()
        self.assertEqual((cfg["workers"], cfg["trials"]), (20, 4))
        self.assertIsNone(cfg["max_decisions"])
        self.assertFalse(cfg["training"])
        self.assertFalse(cfg["formal_ttf"])
        self.assertFalse(cfg["automatic_expansion"])

    def test_first_occurrence_per_pair_not_best_outcome(self):
        rows = [dict(trial=t, decision=d, edge=e, split="train", root_episode="origin", outcome=o)
                for t, d, e, o in [(0, 241, [4, 9], 1), (0, 242, [9, 4], 0), (0, 249, [4, 8], 2), (3, 309, [3, 7], 0)]]
        before = copy.deepcopy(rows)
        self.assertEqual([r["decision"] for r in p.select_targets(list(reversed(rows)))], [241, 249, 309])
        self.assertEqual(rows, before)
        for r in rows: r["outcome"] = 999
        self.assertEqual([r["decision"] for r in p.select_targets(rows)], [241, 249, 309])
        rows[0]["split"] = "validation"
        with self.assertRaises(ValueError): p.select_targets(rows)

    def test_all_candidates_four_orders_and_original_control(self):
        t = target()
        jobs = p.branch_jobs(dict(config=p.config()), t)
        self.assertEqual(len(jobs), 9)
        self.assertEqual(jobs[0]["action"], t["event"]["action"])
        seeds = []
        for trial in range(4):
            paired = [j for j in jobs if j["trial"] == trial]
            self.assertEqual({j["candidate_id"] for j in paired}, {"a", "b"})
            self.assertEqual(len({j["action"]["random_seed"] for j in paired}), 1)
            seeds.append(paired[0]["action"]["random_seed"])
        self.assertEqual(len(set(seeds)), 4)

    def test_candidate_order_does_not_change_pp_randomization(self):
        t = target()
        a = p.branch_jobs(dict(config=p.config()), t)
        t["event"]["pool"].reverse()
        b = p.branch_jobs(dict(config=p.config()), t)
        def key(rows): return {(j["candidate_id"], j["trial"]): j["action"] for j in rows}
        self.assertEqual(key(a), key(b))

    def test_claim_requires_feasibility_not_conflict_reduction(self):
        t = target()
        rows = [dict(job=dict(candidate_id=cid, trial=i), status="ok", feasible=False, conflicts=1, generated=100)
                for cid in ("a", "b") for i in range(4)]
        self.assertEqual(p.summarize_target(t, rows)["interpretation"], "no_observed_terminal_option")
        rows[0].update(feasible=True, conflicts=0)
        self.assertEqual(p.summarize_target(t, rows)["interpretation"], "unselected_terminal_witness")
        rows[4].update(feasible=True, conflicts=0)
        self.assertEqual(p.summarize_target(t, rows)["interpretation"], "selected_option_also_order_sensitive")

    def test_missing_duplicate_and_censored_results_rejected(self):
        rows = [dict(job=dict(candidate_id=cid, trial=i), status="ok", feasible=False, conflicts=1, generated=100)
                for cid in ("a", "b") for i in range(4)]
        with self.assertRaises(ValueError): p.summarize_target(target(), rows[:-1])
        with self.assertRaises(ValueError): p.summarize_target(target(), rows+[rows[0]])
        rows[0]["status"] = "censored"
        with self.assertRaises(ValueError): p.summarize_target(target(), rows)

    def test_immutable_root_ignores_only_external_runtime(self):
        a = dict(runtime=1., context={"host": "one"}, iteration=300, low_level={"generated": 999}, agents=[])
        b = dict(a, runtime=2., context={"host": "two"})
        self.assertEqual(p.canonical_state(a), p.canonical_state(b))
        self.assertNotEqual(p.canonical_state(a), p.canonical_state(dict(b, iteration=301)))

    def test_grouped_replay_count(self):
        reg = dict(config=p.config(), occurrences=[1]*6, targets=[
            dict(target(), source_job_id="a", decision=d) for d in (241, 249)] +
            [dict(target(), source_job_id="b", decision=309)])
        d = p.dry_run(reg)
        self.assertEqual(d["replay_repairs"], 558)
        self.assertEqual(d["controls"], 3)

    def test_native_fork_resume_and_parent_isolation(self):
        if sys.platform != "linux": self.skipTest("frozen Linux native required")
        env = dict(os.environ, PYTHONPATH=str(p.ROOT / "build/linux/sa-wall-clock-v1"),
                   OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        result = subprocess.run([sys.executable, "-B", "-c",
            "from tests.evaluation.test_sa_terminal_candidates import native_micro; native_micro()"],
            cwd=p.ROOT, env=env, capture_output=True, text=True, timeout=60.)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)


if __name__ == "__main__":
    unittest.main()
