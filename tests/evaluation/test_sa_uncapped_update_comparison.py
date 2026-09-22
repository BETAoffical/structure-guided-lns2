from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import compare_sa_uncapped_update as compare
from scripts import run_sa_onpolicy as run


def source():
    return dict(binding="b"*64,config=dict(stream_seed=123),proposal=dict(max_decisions=256,node_budget=25000000,
        pp_safety_seconds=20.),template=dict(environment=dict(max_repair_iterations=0)))


def configuration():
    cfg=run.read_json(compare.ROOT/compare.CONFIG)
    old=dict(config=dict(output=cfg["source_evaluation"],phase="crossfit-comparison"))
    old["jobs"]=[dict(pair_id=f"p{i}",case=dict(map_id=f"m{6+i//4}"),replica=r,phase=old["config"]["phase"],
        split="development_holdout",comparison_arm="untrained_exploration",arm="untrained_exploration",iteration=0,
        job_id=f"p{i}-r{r}",expected_initial=run.json_fingerprint(i)) for i in range(8) for r in range(2)]
    return old,compare.configuration(old,cfg)


def summaries(old=0, dual=0, explore=0):
    return {a:dict(overall=dict(net_success_bounds=[v,v])) for a,v in
        (("bounded_condition",old),("dual16_sa",dual),("untrained_exploration",explore))}


class UncappedComparisonTests(unittest.TestCase):
    def test_schedule_only_adds_frozen_model_not_new_conditions(self):
        old,cfg=configuration()
        jobs=compare.schedule(old,cfg)
        self.assertEqual(len(jobs),16)
        self.assertEqual(len({j["job_id"] for j in jobs}),16)
        for a,b in zip(jobs,old["jobs"]):
            for key in ("case","pair_id","replica","split","expected_initial","phase"):
                self.assertEqual(a[key],b[key])
            self.assertEqual((a["comparison_arm"],a["arm"],a["iteration"]),(compare.ARM,"trained_actor",1))

    def test_scope_rejects_training_ttf_extra_jobs_and_decision_cap(self):
        old,_=configuration()
        cfg=run.read_json(compare.ROOT/compare.CONFIG)
        for k,v in (("max_decisions",10**12),("training",True),("formal_ttf",True),("automatic_promotion",True),
                    ("expected_new",32),("node_budget",50000000),("workers",4)):
            with self.assertRaises(ValueError):compare.configuration(old,dict(cfg,**{k:v}))

    def test_real_runtime_and_model_identity_use_uncapped_plan(self):
        old,cfg=configuration()
        p=compare.previous.runtime_plan(source(),cfg,compare.ARM)
        self.assertIsNone(p["proposal"]["max_decisions"])
        self.assertEqual(p["proposal"]["decision_feature_reference"],256)
        self.assertEqual(p["template"]["environment"]["max_repair_iterations"],0)
        self.assertIsNone(compare.runtime.work_stop(False,10**12,1,p["proposal"]))
        self.assertEqual(compare.previous.prior.expected_policy(cfg,compare.schedule(old,cfg)[0]),cfg["policy_sha256"])

    def test_new_model_name_cannot_change_rng(self):
        old,cfg=configuration()
        p=compare.previous.runtime_plan(source(),cfg,compare.ARM)
        for j in compare.schedule(old,cfg):
            for d in (0,20,255,256,10000):
                for purpose in ("select","pp","accept"):
                    self.assertEqual(run.stream_draw(source(),j["phase"],j["pair_id"],j["replica"],d,purpose),
                                     run.stream_draw(p,j["phase"],j["pair_id"],j["replica"],d,purpose))

    def test_train_or_extra_replica_in_schedule_rejected(self):
        good,cfg=configuration()
        for key,value in (("split","train"),("replica",2),("phase","new-name")):
            old=deepcopy(good)
            old["jobs"][0][key]=value
            with self.assertRaises(ValueError):compare.schedule(old,cfg)

    def test_pairing_rejects_missing_unknown_or_changed_initial(self):
        old,cfg=configuration()
        data={arm:{compare.pair_key(j):dict(j,map_id=j["case"]["map_id"],status="ok",
            initial_fingerprint=j["expected_initial"],rng_stream_id=run.json_fingerprint([j["pair_id"],j["replica"]]))
            for j in compare.schedule(old,cfg)} for arm in (compare.ARM,"dual16_sa")}
        compare.validate_pairs(data)
        first=next(iter(data[compare.ARM]))
        for key,value in (("initial_fingerprint","bad"),("rng_stream_id","bad"),("status","censored"),("split","train")):
            changed=deepcopy(data)
            changed[compare.ARM][first][key]=value
            with self.assertRaises(ValueError):compare.validate_pairs(changed)
        del data[compare.ARM][first]
        with self.assertRaises(ValueError):compare.validate_pairs(data)

    def test_net_gain_not_universal_dominance(self):
        self.assertEqual(compare.interpretation(summaries(1,0,1)),"development_net_gain_needs_fresh_stream_confirmation")
        self.assertEqual(compare.interpretation(summaries(-1,-1,1)),"development_regression_do_not_promote")
        self.assertEqual(compare.interpretation(summaries(0,0,1)),"mixed_or_no_net_gain_do_not_promote")
        result=summaries(1,1,1)
        result["dual16_sa"]["overall"]["net_success_bounds"]=[-1,1]
        with self.assertRaises(ValueError):compare.interpretation(result)

    def test_common_success_not_all_success_or_zero_for_failures(self):
        def row(success,amount):return dict(status="ok",success=success,decisions=amount,generated=amount,
                                          soc=amount,makespan=amount,wait_steps=amount)
        a={0:row(True,10),1:row(True,500),2:row(False,900)}
        b={0:row(True,20),1:row(False,800),2:row(True,400)}
        report=compare.previous.prior.paired_comparison(a,b)
        self.assertEqual((report["wins"],report["losses"],report["common_success"]),(1,1,1))
        self.assertEqual(report["common_success_means"]["generated"],dict(left=10,right=20))

    def test_safe_stop_resume_reuses_finished_jobs_and_refuses_partial(self):
        old,cfg=configuration()
        cfg["output"]="output"
        reg=dict(config=cfg,jobs=compare.schedule(old,cfg),binding="c"*64)
        calls=[]
        with tempfile.TemporaryDirectory() as tmp,patch.object(compare.previous.prior,"ROOT",Path(tmp)):
            out=Path(tmp)/"output"
            def pool(worker,jobs,*args,**kwargs):
                calls.append(len(jobs))
                self.assertTrue(all(j["plan"]["proposal"]["max_decisions"] is None for j in jobs))
                for j in jobs:run.once(compare.previous.prior.folder_for(cfg,j)/"result.json",dict(status="ok"))
                return [dict(status="ok") for _ in jobs]
            actual_require=run.require
            def platform_only(condition,message):
                if message!="use frozen WSL native":actual_require(condition,message)
            with patch.object(compare,"verify",return_value=(reg,source(),out,old)), \
                 patch.object(compare.recovery,"strict_lock",return_value=nullcontext()), \
                 patch.object(run,"require",side_effect=platform_only), \
                 patch.object(compare.previous,"read_result",return_value=dict(status="ok")), \
                 patch.object(compare.previous.prior,"read_result",return_value=dict(status="ok")), \
                 patch("experiments.repair_collection._run_jobs",side_effect=pool):
                run.write_json(out/"STOP_AFTER_BATCH",{})
                self.assertEqual(compare.collect()["status"],"paused")
                self.assertEqual(calls,[])
                self.assertEqual(compare.collect(True)["collected"],16)
                self.assertEqual(compare.collect(True)["collected"],16)
                self.assertEqual(calls,[16])
                (compare.previous.prior.folder_for(cfg,reg["jobs"][0])/"result.json").unlink()
                with self.assertRaisesRegex(ValueError,"partial episode"):compare.collect(True)


if __name__=="__main__":unittest.main()
