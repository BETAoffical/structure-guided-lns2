import copy
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts import probe_sa_terminal_augmentation as p


def fixture():
    state = dict(agents=[dict(id=i) for i in (2, 7, 19, 23, 28, 31, 40)])
    t = dict(id="test", fingerprint="abc", event=dict(selected_id="old",
        action=dict(mode="explicit_neighborhood", agents=[2, 7], random_seed=123),
        pool=[dict(candidate_id="old", agents=[2, 7])]))
    cfg = dict(p.config(), resource_agents=[2, 19, 23])
    return state, t, cfg


def micro():
    import lns2_env
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root/"tiny.map").write_text("type octile\nheight 3\nwidth 4\nmap\n....\n....\n....\n", encoding="utf8")
        (root/"tiny.scen").write_text("version 1\n0\ttiny.map\t4\t3\t0\t0\t3\t0\t3\n"
            "0\ttiny.map\t4\t3\t3\t0\t0\t0\t3\n0\ttiny.map\t4\t3\t0\t2\t3\t2\t3\n", encoding="utf8")
        def reset():
            env = lns2_env.LNS2RepairEnv(str(root/"tiny.map"), str(root/"tiny.scen"), 3)
            state = p.q._plain(env.reset_paths([[0,1,2,3],[3,2,1,0],[8,9,10,11]], seed=11))
            return env, state
        original, state = reset()
        action = dict(mode="explicit_neighborhood", agents=[0,1], random_seed=31)
        raw = p.q._plain(original.step_experimental_pp(action,20.,"annealed",100.,.4))
        target = dict(id="micro", fingerprint=p.q.state_fingerprint(state), event=dict(
            action=action, selected_id="old", temperature=100., uniform=.4, metrics=raw["metrics"],
            delta=p.q.encode_state_delta(state,raw["observation"]), pool=[dict(candidate_id="old",agents=[0,1])]))
        cfg = dict(p.config(), output=str(root/"out"), workers=2)
        reg = dict(config=cfg, source=dict(config=p.source.config()), binding="micro",
                   candidates={"micro":[dict(arm="targeted",candidate_id="all",agents=[0,1,2])]})
        jobs = p.branch_jobs(reg,target)
        env,before = reset()
        assert p.source.run_forks(env,before,target,reg,jobs,False)
        for job in jobs:
            row = p.source.read_branch(reg,target,job)
            after = p.source.check_branch(before,row,target,job)
            assert row["status"] == "ok"
            assert set(row["metrics"]["repair_order"]) == set(job["action"]["agents"])
            if job["trial"] >= 0: assert len(after["agents"]) == 3
        assert p.q.state_fingerprint(p.q._plain(env.get_state())) == target["fingerprint"]
        hashes = [p.run.sha256_file(p.source.branch_folder(reg,target,j)/"result.json") for j in jobs]
        assert p.source.run_forks(env,before,target,reg,jobs,True)
        assert hashes == [p.run.sha256_file(p.source.branch_folder(reg,target,j)/"result.json") for j in jobs]


class AugmentationTests(unittest.TestCase):
    def test_scope_and_exact_dry_run(self):
        cfg = p.config()
        self.assertFalse(cfg["training"] or cfg["formal_ttf"] or cfg["automatic_expansion"])
        self.assertIsNone(cfg["max_decisions"])
        dry = p.dry_run(dict(config=cfg,candidates={}))
        self.assertEqual((dry["new_branches"],dry["reused_baseline_trials"],dry["original_controls"]),(16,8,2))
        self.assertEqual(dry["replay_repairs"],249)

    def test_supersets_equal_size_unique_and_noncontinuous_ids(self):
        state,t,cfg = fixture()
        a,b = p.candidate_sets(t,state,cfg)
        self.assertEqual(a["agents"],[2,7,19,23])
        self.assertEqual(a["additions"],[19,23])
        self.assertEqual(len(a["agents"]),len(b["agents"]))
        self.assertNotEqual(a["agents"],b["agents"])
        for row in (a,b):
            self.assertTrue({2,7} <= set(row["agents"]))
            self.assertEqual(len(row["agents"]),len(set(row["agents"])))

    def test_no_outcome_or_agent_order_dependency(self):
        state,t,cfg = fixture()
        prior = copy.deepcopy((state,t,cfg))
        a = p.candidate_sets(t,state,cfg)
        self.assertEqual((state,t,cfg),prior)
        state["agents"].reverse()
        t["event"]["outcome"] = {"winner": [19,23]}
        self.assertEqual(a,p.candidate_sets(t,state,cfg))

    def test_unknown_duplicate_and_empty_addition_rejected(self):
        state,t,cfg = fixture()
        for resources in ([2,99],[2,7]):
            with self.assertRaises(ValueError): p.candidate_sets(t,state,dict(cfg,resource_agents=resources))
        state["agents"].append(state["agents"][0])
        with self.assertRaises(ValueError): p.candidate_sets(t,state,cfg)

    def test_trials_reuse_exact_old_seeds_and_control(self):
        state,t,cfg = fixture()
        old = dict(config=p.source.config())
        reg = dict(config=cfg,source=old,candidates={t["id"]:p.candidate_sets(t,state,cfg)})
        jobs = p.branch_jobs(reg,t)
        baseline = p.baseline_jobs(old,t)
        self.assertEqual(len(jobs),9)
        self.assertEqual(jobs[0],p.source.branch_jobs(old,t)[0])
        for trial,b in enumerate(baseline):
            pair = [j for j in jobs if j["trial"] == trial]
            self.assertEqual(len(pair),2)
            self.assertEqual({j["action"]["random_seed"] for j in pair},{b["action"]["random_seed"]})
        reg["candidates"][t["id"]].reverse()
        self.assertEqual({j["job_id"]:j for j in jobs},{j["job_id"]:j for j in p.branch_jobs(reg,t)})

    def test_missing_original_candidate_rejected(self):
        _,t,_ = fixture()
        t["event"]["selected_id"] = "absent"
        with self.assertRaises(ValueError): p.baseline_jobs(dict(config=p.source.config()),t)

    def test_full_success_not_old_pair_removal(self):
        rows = [dict(trial=i,status="ok",feasible=False,conflicts=1,generated=10,
                     old_pair_removed=True,repair_order=[2,7]) for i in range(4)]
        self.assertEqual(p.summarize(rows)["feasible"],0)
        self.assertEqual(p.summarize(rows)["old_pair_removed"],4)
        for bad in (rows[:-1],rows+[rows[0]],[dict(r,status="censored") for r in rows],
                    [dict(r,feasible=True) for r in rows]):
            with self.assertRaises(ValueError): p.summarize(bad)

    def test_native_augmented_members_controls_and_resume(self):
        if sys.platform != "linux": self.skipTest("frozen Linux native required")
        env = dict(os.environ,PYTHONPATH=str(p.ROOT/"build/linux/sa-wall-clock-v1"),
                   OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1")
        result = subprocess.run([sys.executable,"-B","-c",
            "from tests.evaluation.test_sa_terminal_augmentation import micro; micro()"],
            cwd=p.ROOT,env=env,capture_output=True,text=True,timeout=60.)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)


if __name__ == "__main__": unittest.main()
