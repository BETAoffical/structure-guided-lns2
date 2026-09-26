from copy import deepcopy
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import sa_onpolicy_contract as previous
from experiments import sa_uncapped_training_contract as credit
from scripts import recover_sa_uncapped_training as recover


def episode(pair="p", map_id="m", replica=0, success=True, decisions=1):
    return dict(episode_id=f"{pair}-{replica}",pair_id=pair,map_id=map_id,replica=replica,
        policy_sha256="a"*64,split="train",status="ok",initial_fingerprint=recover.run.json_fingerprint(pair),
        rng_stream_id=recover.run.json_fingerprint([pair,replica]),stop="feasible" if success else "node_budget",
        success=success,final_conflicts=0 if success else 1,decisions=decisions,generated=10 if success else 101,
        steps=[dict(decision=d,policy_sha256="a"*64,probabilities={"a":.75,"b":.25},selected_id="a",
                    behavior_log_probability=math.log(.75)) for d in range(decisions)])


def coefficients(rows, groups=None):
    return credit.gradient_coefficients(rows,policy_sha256="a"*64,expected_groups=groups or {"p":"m"},
                                        replicas=2,max_decisions=None,node_budget=100)


class UncappedTrainingTests(unittest.TestCase):
    def test_shared_credit_preserves_weights_with_distinct_terminal_contracts(self):
        rows=[episode(),episode(replica=1,success=False)]
        self.assertEqual(coefficients(rows),previous.gradient_coefficients(
            rows,policy_sha256='a'*64,expected_groups={'p':'m'},replicas=2,max_decisions=256,node_budget=100))
        with self.assertRaises(ValueError):
            credit.gradient_coefficients(rows,policy_sha256='a'*64,expected_groups={'p':'m'},
                                         replicas=2,max_decisions=256,node_budget=100)

    def test_no_decision_limit_or_length_reweighting(self):
        rows=[episode(decisions=1),episode(replica=1,success=False,decisions=300)]
        expected=coefficients(rows)
        self.assertEqual([v["coefficient"] for v in expected],[.5,-.5])
        rows[1]=episode(replica=1,success=False,decisions=1000)
        self.assertEqual(coefficients(rows),expected)
        self.assertEqual(credit.terminal_return(dict(rows[1],decisions=10**12),max_decisions=None,node_budget=100),0.)

    def test_terminal_unknowns_and_inconsistencies(self):
        row=episode(success=False)
        for changes in (dict(stop="decision_budget"),dict(generated=99),dict(final_conflicts=0),dict(success=True),
                        dict(decisions=-1),dict(decisions=True),dict(status="error")):
            with self.assertRaises(ValueError):credit.terminal_return(dict(row,**changes),max_decisions=None,node_budget=100)
        with self.assertRaises(ValueError):credit.terminal_return(row,max_decisions=100000,node_budget=100)
        for stop in ("wall_safety","incomplete_pp","external_timeout","user_stop"):
            unknown=dict(row,stop=stop,status="censored")
            self.assertIsNone(credit.terminal_return(unknown,max_decisions=None,node_budget=100))
            with self.assertRaisesRegex(ValueError,"censored batch"):coefficients([episode(replica=1),unknown])

    def test_map_equal_and_permutation(self):
        groups={"a1":"a","a2":"a","b":"b"}
        rows=[episode(pair,m,r,r==0) for pair,m in groups.items() for r in range(2)]
        result=coefficients(rows,groups)
        self.assertEqual(result,coefficients(rows[::-1],groups))
        self.assertAlmostEqual(sum(r["episode_weight"] for r in result if r["episode_id"].startswith("a")),.5)
        self.assertAlmostEqual(sum(r["episode_weight"] for r in result),1.)

    def test_isolation_and_complete_replica_batch(self):
        good=[episode(),episode(replica=1,success=False)]
        for edit in (lambda r:r.pop(),lambda r:r[1].update(replica=0),lambda r:r[0].update(split="validation"),
                     lambda r:r[0].update(map_id="heldout"),lambda r:r[1].update(rng_stream_id=r[0]["rng_stream_id"]),
                     lambda r:r[1].update(policy_sha256="b"*64),lambda r:r[1].update(initial_fingerprint="b"*64)):
            rows=deepcopy(good)
            edit(rows)
            with self.assertRaises(ValueError):coefficients(rows)

    def test_signal_gate_does_not_retrain_zero_credit_suffix(self):
        before=[episode(success=False)]
        after=[episode(success=False,decisions=500)]
        old=[dict(episode_id="p-0",coefficient=0)]
        result=credit.signal_change(before,after,old,old)
        self.assertFalse(result["possible_gradient_change"])
        self.assertEqual(result["decision"],"no_new_gradient_information_skip_retraining")

    def test_signal_gate_catches_nonzero_credit_suffix_without_new_success(self):
        before=[episode(success=False)]
        after=[episode(success=False,decisions=500)]
        old=[dict(episode_id="p-0",coefficient=-.5)]
        result=credit.signal_change(before,after,old,old)
        self.assertTrue(result["possible_gradient_change"])
        self.assertEqual(result["changed_returns"],[])
        self.assertEqual(result["credited_suffixes"],{"p-0":499})

    def test_signal_gate_changed_sibling_credit_and_short_prefix(self):
        before=[episode(success=False)]
        after=[episode(success=False)]
        a=[dict(episode_id="p-0",coefficient=0)]
        b=[dict(episode_id="p-0",coefficient=-.5)]
        self.assertEqual(credit.signal_change(before,after,a,b)["changed_coefficients"],["p-0"])
        with self.assertRaises(ValueError):credit.signal_change([episode(decisions=2)],after,a,b)

    def test_partition_only_capped_train_jobs_extended(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(recover,"ROOT",Path(tmp)):
            batch=[]
            for i,stop in enumerate(("feasible","node_budget","decision_budget")):
                folder=Path(tmp)/str(i)
                recover.run.write_json(folder/"result.json",{})
                job=dict(split="train",phase="train-0",iteration=0,job_id=str(i))
                row=dict(job_id=str(i),split="train",status="ok",stop=stop,success=stop=="feasible",decisions=256)
                batch.append((dict(job=job,expected_initial="a"*64),folder,row))
            result=recover.partition(batch)
            self.assertEqual([r["origin"] for r in result.values()],["reuse","reuse","extend"])
            batch[-1][2]["split"]="validation"
            with self.assertRaises(ValueError):recover.partition(batch)

    def test_plan_freezes_scope_and_update(self):
        cfg=recover.run.read_json(recover.ROOT/recover.CONFIG)
        source=dict(binding="b"*64,config=dict(stream_seed=1),proposal=dict(max_decisions=256,node_budget=25000000,pp_safety_seconds=20.))
        p=recover.runtime_plan(source,cfg)
        self.assertIsNone(p["proposal"]["max_decisions"])
        self.assertEqual(source["proposal"]["max_decisions"],256)
        self.assertEqual(p["proposal"]["node_budget"],25000000)
        self.assertEqual(p["binding"],source["binding"])
        for key,value in (("max_decisions",1000),("formal_ttf",True),("heldout_evaluation",True),("maximum_updates",2)):
            with self.assertRaises(ValueError):recover.runtime_plan(source,dict(cfg,**{key:value}))
        bad=deepcopy(cfg)
        bad["update"]["initial_l2"]=.5
        with self.assertRaises(ValueError):recover.runtime_plan(source,bad)

    def test_pinned_inputs_exclude_evaluation_and_include_contract(self):
        self.assertIn("experiments/sa_uncapped_training_contract.py",recover.CODE)
        self.assertIn("tests/evaluation/test_sa_uncapped_training.py",recover.CODE)
        cfg=recover.run.read_json(recover.ROOT/recover.CONFIG)
        self.assertNotIn("comparison",cfg["source"])
        self.assertFalse(cfg["heldout_evaluation"])

    def test_weighted_gradient_matches_sum_not_average(self):
        try:import torch
        except ImportError:self.skipTest("existing Windows Torch environment required")
        x=torch.tensor([.3,-.5,.8],dtype=torch.float64,requires_grad=True)
        advantage=torch.tensor(-2/3,dtype=torch.float64)
        weight=1/96
        a=torch.autograd.grad(-(advantage*x).sum(),x,retain_graph=True)[0]*weight
        b=torch.autograd.grad(-(advantage*weight)*x.sum(),x)[0]
        self.assertTrue(torch.allclose(a,b,atol=1e-15,rtol=0))
        self.assertAlmostEqual(float(a.sum()),2/96)


if __name__=="__main__":unittest.main()
