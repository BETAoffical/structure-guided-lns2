import copy
import unittest

from scripts import audit_sa_matched_training_coverage as a


def candidates(n=2):
    return [dict(candidate_id=chr(97+i),agents=list(range(i*16,(i+1)*16)),
                 selection_families=["collision:16"],values=[True]*8 if i==0 else [False]*8) for i in range(n)]


def group(sid="s",map_id="m",binding="x",kind="matched_pair",n=2):
    return a.comparison_group(dict(state_id=sid,map_id=map_id),binding,candidates(n),
                              [binding+str(t) for t in range(8)],kind,"collision:16" if kind=="matched_pair" else None)


class MatchedTrainingCoverageTests(unittest.TestCase):
    def test_old_and_new_groups_do_not_create_cross_stream_pairs(self):
        old=group(binding="old",kind="old_grid",n=4); new=group(binding="new")
        rows=a.pair_rows([old,new])
        self.assertEqual(len(rows),7)
        self.assertAlmostEqual(sum(r["state_budget_weight"] for r in rows),1)
        self.assertAlmostEqual(sum(r["state_budget_weight"] for r in rows if r["kind"]=="old_grid"),.5)
        self.assertEqual(old["candidates"][0]["candidate_id"],new["candidates"][0]["candidate_id"])
        self.assertNotEqual(old["trial_keys"],new["trial_keys"])

    def test_hold_map_excludes_all_its_groups_and_balances_roots(self):
        groups=[group("s","m","old","old_grid",4),group("s","m","new"),
                group("t","n"),group("u","n"),group("v","p")]
        rows=a.training_fold(a.pair_rows(groups),"m")
        self.assertEqual({r["state_id"] for r in rows},{"t","u","v"})
        self.assertAlmostEqual(sum(r["map_balanced_weight"] for r in rows),3)
        self.assertAlmostEqual(sum(r["map_balanced_weight"] for r in rows if r["map_id"]=="n"),1.5)
        self.assertAlmostEqual(sum(r["map_balanced_weight"] for r in rows if r["map_id"]=="p"),1.5)

    def test_duplicate_group_or_cross_map_root_is_rejected(self):
        g=group()
        with self.assertRaisesRegex(ValueError,"duplicate comparison"):
            a.pair_rows([g,g])
        h=group(map_id="other",binding="new")
        with self.assertRaisesRegex(ValueError,"crosses maps"):
            a.pair_rows([g,h])

    def test_matching_size_family_and_members_must_be_real(self):
        for change in (lambda c:c[0]["agents"].pop(),lambda c:c[0].update(selection_families=["target:16"]),
                       lambda c:c[1].update(agents=c[0]["agents"])):
            c=candidates(); change(c)
            with self.assertRaises(ValueError):
                a.comparison_group(dict(state_id="s",map_id="m"),"x",c,list(range(8)),"matched_pair","collision:16")

    def test_censored_and_missing_trials_are_not_coerced(self):
        for value in (None,1):
            c=candidates(); c[0]["values"][0]=value
            with self.assertRaisesRegex(ValueError,"incomplete labels"):
                a.comparison_group(dict(state_id="s",map_id="m"),"x",c,list(range(8)),"matched_pair","collision:16")
        c=candidates(); c[0]["values"].pop()
        with self.assertRaises(ValueError):
            a.comparison_group(dict(state_id="s",map_id="m"),"x",c,list(range(8)),"matched_pair","collision:16")

    def test_tie_neutral_crossfit_does_not_depend_on_candidate_names(self):
        r=dict(pair_ids=["a","b"],values=dict(a=[True]+[False]*7,b=[False]*8))
        self.assertEqual(a.neutral_crossfit(r),0)
        swapped=dict(pair_ids=["a","b"],values=dict(a=r["values"]["b"],b=r["values"]["a"]))
        self.assertEqual(a.neutral_crossfit(swapped),0)
        strong=dict(pair_ids=["a","b"],values=dict(a=[True]*8,b=[False]*5+[True]*3))
        self.assertEqual(a.neutral_crossfit(strong),.3125)

    def test_group_label_and_weight_are_order_invariant(self):
        g=group(); h=copy.deepcopy(g); h["candidates"].reverse()
        rebuilt=a.comparison_group(dict(state_id="s",map_id="m"),"x",h["candidates"],h["trial_keys"],"matched_pair","collision:16")
        self.assertEqual(g,rebuilt)
        r=a.pair_rows([g])[0]
        self.assertEqual((r["target"],r["a_wins"],r["b_wins"],r["ties"]),(1,8,0,0))

    def test_trial_keys_cannot_be_duplicated(self):
        with self.assertRaisesRegex(ValueError,"trial-key"):
            a.comparison_group(dict(state_id="s",map_id="m"),"x",candidates(),["same"]*8,"matched_pair","collision:16")

    def test_zero_contrast_roots_retained_and_folds_do_not_read_held_labels(self):
        roots={}; records=[]; groups=[]
        for sid,m in (("s","m"),("t","n")):
            g=group(sid,m); groups.append(g)
            values={c["candidate_id"]:c["values"] for c in g["candidates"]}
            if sid=="t": values["b"]=[True]*8
            records.append(dict(state_id=sid,map_id=m,pair_ids=["a","b"],anchor_id="a",values=values))
            roots[sid]=dict(state_id=sid,map_id=m,pair_ids=["a","b"],family="collision:16",decision=4)
        first=a.coverage(records,roots,groups)
        self.assertEqual(first["summary"]["states"],2)
        self.assertEqual(first["summary"]["zero_difference_states"],1)
        records[0]["values"]["b"]=[True]*8
        second=a.coverage(records,roots,groups)
        f1=next(f for f in first["folds"] if f["held"]=="m")
        f2=next(f for f in second["folds"] if f["held"]=="m")
        self.assertEqual(f1["matched_train"],f2["matched_train"])

    def test_duplicate_or_misidentified_record_rejected(self):
        r=dict(state_id="s",map_id="m",pair_ids=["a","b"],anchor_id="a",values=dict(a=[True]*8,b=[False]*8))
        with self.assertRaisesRegex(ValueError,"cohort"):
            a.coverage([r,r],{"s":{}},[])
        with self.assertRaisesRegex(ValueError,"identity"):
            a.coverage([r],{"s":dict(map_id="other",pair_ids=["a","b"])},[])


if __name__=="__main__":
    unittest.main()
