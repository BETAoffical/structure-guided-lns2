import copy
import unittest
from scripts import run_sa_curriculum_update as s


def entries():
    result=[]
    for m in range(6):
        for c in range(6):
            for r in range(4):
                key=f"m{m}-c{c}-r{r}"
                row=dict(job_id=key,episode_id=key,pair_id=f"m{m}-c{c}",map_id=f"m{m}",split="train",
                    policy_sha256="a"*64,success=True,status="ok",stop="feasible",decisions=300,
                    generated=12,final_conflicts=0,replica=r)
                result.append(dict(job_id=key,origin="original" if c<4 else "transition",episode=row))
    return result


def roots():
    return [dict(case=dict(map_id=f"m{m}",task_id=f"m{m}__task_0001",task_variant="bottleneck_d25"),
        pair_id=f"m{m}__task_0001-s{seed}",solver_seed=seed,expected_initial="a"*64)
        for m in range(6) for seed in (233,239)]


class CurriculumTests(unittest.TestCase):
    def test_fixed_training_and_uncapped_evaluation(self):
        c=s.config()
        self.assertEqual(c["maximum_updates"],1)
        self.assertIsNone(c["max_decisions"])
        self.assertEqual(c["expected_episodes"],144)
        self.assertEqual(c["workers"],20)
        self.assertFalse(c["formal_ttf"])
        self.assertEqual(s.REGISTRATION,"training_registration.json")
        self.assertEqual(s.EVALUATION_REGISTRATION,"evaluation_registration.json")

    def test_full_current_policy_train_only(self):
        e=entries()
        groups=s.checked_entries(e,"a"*64,[f"m{i}" for i in range(6)])
        self.assertEqual(len(groups),36)
        self.assertEqual(len(e),144)

    def test_reject_missing_duplicate_stale_and_evaluation(self):
        for transform in (lambda e:e.pop(),lambda e:e.__setitem__(0,e[1]),
            lambda e:e[0]["episode"].update(split="train_diagnostic_evaluation"),
            lambda e:e[0]["episode"].update(policy_sha256="b"*64),
            lambda e:e[0]["episode"].update(status="censored",stop="wall_safety")):
            e=entries();transform(e)
            with self.assertRaises(ValueError):s.checked_entries(e,"a"*64,[f"m{i}" for i in range(6)])

    def test_failures_retained_and_no_future_quality_label(self):
        e=entries();e[0]["episode"].update(success=False,stop="node_budget",generated=25001000,final_conflicts=7)
        self.assertEqual(len(s.checked_entries(e,"a"*64,[f"m{i}" for i in range(6)])),36)
        e[0]["episode"].update(makespan=999999,runtime=0,one_step_gain=999)
        self.assertEqual(len(s.checked_entries(e,"a"*64,[f"m{i}" for i in range(6)])),36)

    def test_complete_three_way_original_task_schedule(self):
        original=roots();snapshot=copy.deepcopy(original)
        jobs=s.schedule(original,s.config())
        self.assertEqual(original,snapshot)
        self.assertEqual(len(jobs),72)
        self.assertEqual(len({s.stream_key(j) for j in jobs}),24)
        self.assertEqual({j["replica"] for j in jobs},{4,5})
        self.assertEqual({j["comparison_arm"] for j in jobs},set(s.ARMS))
        self.assertTrue(all(j["split"]!="train" for j in jobs))
        self.assertEqual(s.schedule(original,s.config()),jobs)

    def test_subset_tasks_and_bad_coverage_rejected(self):
        for transform in (lambda r:r.pop(),lambda r:r[0]["case"].update(task_id="retain90"),
                          lambda r:r[0].update(solver_seed=51)):
            r=roots();transform(r)
            with self.assertRaises(ValueError):s.schedule(r,s.config())

    def test_group_equal_credit_scaling(self):
        e=entries();rows=[]
        for item in e:
            row=dict(item["episode"],steps=[])
            row.update(decisions=0,initial_fingerprint="b"*64,rng_stream_id=f"{row['replica']+1:064x}")
            if row["pair_id"]=="m4-c5" and row["replica"]==2:
                row.update(success=False,stop="node_budget",generated=25000000,final_conflicts=1)
            rows.append(row)
        groups={r["pair_id"]:r["map_id"] for r in rows}
        result=s.old.credit.gradient_coefficients(rows,policy_sha256="a"*64,expected_groups=groups,
            replicas=4,max_decisions=None,node_budget=25000000)
        self.assertTrue(all(r["episode_weight"]==1/144 for r in result))
        self.assertEqual(sum(r["coefficient"]!=0 for r in result),4)
        self.assertAlmostEqual(sum(r["coefficient"] for r in result),0.)

    def test_merging_reweights_but_does_not_change_returns(self):
        e=entries();original=[];transition=[]
        for item in e:
            count=96 if item["origin"]=="original" else 48
            c=dict(episode_id=item["job_id"],return_value=1.,baseline=2/3,episode_weight=1/count,coefficient=(1-2/3)/count)
            (original if item["origin"]=="original" else transition).append(c)
        merged=s.merged_credits(e,original,transition)
        self.assertEqual(len(merged),144)
        self.assertTrue(all(c["episode_weight"]==1/144 and c["baseline"]==2/3 for c in merged))
        self.assertAlmostEqual(merged[0]["coefficient"],(1-2/3)/144)
        original[0]["return_value"]=0.
        with self.assertRaises(ValueError):s.merged_credits(e,original,transition)

    def test_local_gain_not_automatic_promotion(self):
        def c(a,b):return {"parent":dict(overall=dict(net_success_bounds=[a,a])),"old":dict(overall=dict(net_success_bounds=[b,b]))}
        self.assertEqual(s.interpretation(c(1,2),[1,1]),"local_curriculum_net_gain_needs_independent_confirmation")
        self.assertEqual(s.interpretation(c(-1,0),[1,0]),"localized_hard_task_tradeoff_do_not_promote")
        self.assertEqual(s.interpretation(c(0,0),[0,0]),"no_consistent_curriculum_completion_gain_do_not_promote")


if __name__=="__main__":unittest.main()
