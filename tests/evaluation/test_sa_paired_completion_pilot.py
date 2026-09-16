import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from experiments._common import json_fingerprint, write_json
from experiments.sa_paired_sampling import choose_candidates, choose_roots, work_budget
from scripts.run_sa_paired_completion_pilot import expected_files, verify_receipt, fit_fold


def pool():
    return [dict(candidate_id=str(i), agents=list(range(i*20, i*20+n)), score=100-i)
            for i,n in enumerate((16,4,4,8,8,16,16))]


def target(i,m,phase,episode=None):
    return dict(id=str(i), map_id=m, phase=phase, item=dict(job_id=episode or str(i)),
                decision=4 if phase=="early" else 32, selected=["a","b","c","d"])


def training_fixture():
    from experiments.sa_paired_completion import SCHEMA
    states=[]
    for i in range(24):
        cs=[]
        for cid,quality in (("a",True),("b",False)):
            cs.append(dict(candidate_id=cid, agents=[7 if quality else 19], features={"quality":float(quality)},
                trials=[dict(trial=t, randomization_key=json_fingerprint([i,t]), completed=quality,
                             stop="feasible" if quality else "horizon", steps=1 if quality else 32,
                             final_conflicts=0 if quality else 1) for t in range(8)]))
        states.append(dict(state_id=str(i), episode=str(i), map_id=str(i%3), decision=4, phase="early", anchor_id="b",
                           agent_ids=[7,19], state_fingerprint="a"*64, history_fingerprint="b"*64, candidates=cs))
    return dict(schema=SCHEMA, role="prospective_development", sampling="outcome_blind_same_state",
                source_kind="registered_prospective_collection", continuation_binding="c"*64,
                horizon=32, trial_count=8, feature_names=["quality"], states=states)


class PairedPilotTests(unittest.TestCase):
    def test_config_stays_bounded_and_no_timing(self):
        cfg=json.loads((Path(__file__).resolve().parents[2]/"configs/sa_paired_completion_pilot.json").read_text())
        self.assertEqual((cfg["maps"],cfg["states"],cfg["max_candidates"],cfg["trials"],cfg["horizon"]),(8,16,4,8,32))
        self.assertEqual(cfg["workers"],20)
        self.assertTrue(cfg["no_ttf"])
        self.assertFalse(cfg["runtime_integration_allowed"])

    def test_challengers_ignore_scores_results_and_candidate_order(self):
        p=pool()
        ids=choose_candidates(p,"0","root",3)
        for c in p:
            c["score"]=-c["score"]
            c["future_completion"]=1
        self.assertEqual(ids,choose_candidates(list(reversed(p)),"0","root",3))
        sizes=[len(next(c for c in p if c["candidate_id"]==cid)["agents"]) for cid in ids]
        self.assertEqual(len(ids),4)
        self.assertIn(4,sizes)
        self.assertIn(8,sizes)
        self.assertIn(16,sizes)
        self.assertEqual(ids[0],"0")

    def test_small_pool_and_missing_size_not_fabricated(self):
        p=pool()[:2]
        self.assertEqual(set(choose_candidates(p,"0","root",3)),{"0","1"})
        with self.assertRaisesRegex(ValueError,"coverage"):
            choose_candidates(p[:1],"0","root",3)
        p.append(copy.deepcopy(p[0]))
        with self.assertRaisesRegex(ValueError,"duplicate"):
            choose_candidates(p,"0","root",3)

    def test_roots_use_distinct_episodes_exclude_old_bridge(self):
        roots=[target(1,"m","early","shared"),target(2,"m","continuing","shared"),
               target(3,"m","early","other"),target(4,"m","continuing","excluded")]
        selected,coverage=choose_roots(roots,{"m"},{"excluded"},3)
        self.assertEqual({r["id"] for r in selected},{"2","3"})
        self.assertEqual(coverage[0]["selected"],2)
        for r in roots:
            r["future_label"]=999
        other,_=choose_roots(list(reversed(roots)),{"m"},{"excluded"},3)
        self.assertEqual([r["id"] for r in selected],[r["id"] for r in other])

    def test_no_fill_from_other_map_or_phase(self):
        chosen,coverage=choose_roots([target(1,"a","early"),target(2,"a","early"),target(3,"b","continuing")],{"a","b"},set(),3)
        self.assertEqual(len(chosen),2)
        self.assertTrue(all(c["selected"]==1 for c in coverage))

    def test_budget_counts_trials_not_independent_samples(self):
        cfg=dict(trials=8,horizon=32,workers=20,branch_seconds=120.)
        rs=[target(i,str(i//2),"early" if i%2==0 else "continuing") for i in range(16)]
        b=work_budget(rs,cfg)
        self.assertEqual((b["states"],b["branch_jobs"],b["continuation_repairs_upper"]),(16,512,16384))
        self.assertEqual(b["all_repairs_upper"],16384+16+8*36)
        self.assertFalse(b["timing_claim"])

    def test_receipt_requires_root_control_every_trial_and_no_extras(self):
        t=target(1,"m","early")
        plan=dict(binding="b",config=dict(trials=8))
        self.assertEqual(len(expected_files(t,plan["config"])),34)
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            write_json(folder/"receipt.json",dict(binding="b",files={}))
            with self.assertRaisesRegex(ValueError,"coverage"):
                verify_receipt(folder,t,plan)

    @unittest.skipUnless(importlib.util.find_spec("sklearn"),"sklearn unavailable; no install")
    def test_paired_and_pointwise_do_not_train_on_held_labels(self):
        import sklearn
        if sklearn.__version__!="1.5.0":
            self.skipTest("registered sklearn required")
        data=training_fixture()
        first=fit_fold(dict(data=data,map_id="2"))
        changed=copy.deepcopy(data)
        for s in changed["states"]:
            if s["map_id"]=="2":
                for c in s["candidates"]:
                    for t in c["trials"]:
                        t.update(completed=not t["completed"],stop="horizon" if t["completed"] else "feasible",
                                 steps=32 if t["completed"] else 1,final_conflicts=1 if t["completed"] else 0)
        self.assertEqual(first,fit_fold(dict(data=changed,map_id="2")))
        self.assertTrue(all(next(s for s in data["states"] if s["state_id"]==sid)["map_id"]!="2" for sid in first["train_ids"]))


if __name__=="__main__":
    unittest.main()
