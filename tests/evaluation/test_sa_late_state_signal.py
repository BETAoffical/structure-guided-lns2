import unittest

from experiments.sa_late_state_signal import assign_phases, trial_rates, choose_discovery, describe, summarize


CFG = dict(phase_seed=2026091805, root_decisions=[32,64,96],discovery_trials=list(range(8)),
           confirmation_trials=list(range(8,16)),bootstrap=5000)


def root(i=0):
    return dict(id=f"r{i}",map_id=f"m{i}",decision=64,state=dict(num_of_colliding_pairs=5),anchor_id="a",
        candidates=[dict(candidate_id=c) for c in "abcd"],source_event=dict(ranking=dict(selected="c")),
        updated_ranking=dict(selected="d"))


def rows(r, trials, winner="b"):
    return [dict(root_id=r["id"],candidate_id=c,trial=t,arm="frozen",status="ok",
        success=c==winner,stop="feasible" if c==winner else "decision_budget") for c in "abcd" for t in trials]


class LateSignalTest(unittest.TestCase):
    def test_schedule_is_balanced_blind_and_isolated(self):
        maps=[f"m{i}" for i in range(6)]
        a=assign_phases(maps,["v"],CFG)
        self.assertEqual(a,assign_phases(list(reversed(maps)),["v"],CFG))
        self.assertEqual([list(a.values()).count(d) for d in [32,64,96]],[2,2,2])
        with self.assertRaisesRegex(ValueError,"leakage"): assign_phases(maps,["m1"],CFG)

    def test_discovery_cannot_read_confirmation(self):
        r=root(); d=rows(r,range(8))
        c=choose_discovery(r,d,CFG)
        self.assertEqual(c["selected"],"b")
        with self.assertRaisesRegex(ValueError,"grid"):
            choose_discovery(r,d+rows(r,range(8,16)),CFG)
        out=describe(r,c,rows(r,range(8,16),"a"),CFG)
        self.assertEqual(out["confirmation_gain"],-1)
        self.assertFalse(out["confirmed_strict_improvement"])

    def test_ties_keep_anchor_and_unknown_is_not_zero(self):
        r=root(); rs=rows(r,range(8),"none")
        self.assertEqual(choose_discovery(r,rs,CFG)["selected"],"a")
        rs[0].update(stop="wall_safety",status="censored")
        self.assertFalse(choose_discovery(r,rs,CFG)["complete"])
        self.assertIsNone(trial_rates(r,rs,range(8))["a"])

    def test_strict_grid_and_continuation(self):
        r=root(); rs=rows(r,range(8))
        with self.assertRaises(ValueError): trial_rates(r,rs[:-1],range(8))
        with self.assertRaises(ValueError): trial_rates(r,rs+[rs[0]],range(8))
        rs[0]["arm"]="gbdt"
        with self.assertRaises(ValueError): trial_rates(r,rs,range(8))

    def test_positive_signal_does_not_promote_or_require_every_map_win(self):
        result=[]
        for i in range(6):
            r=root(i); choice=choose_discovery(r,rows(r,range(8)),CFG)
            result.append(describe(r,choice,rows(r,range(8,16),"b" if i<4 else "a"),CFG))
        s=summarize(result,CFG)
        self.assertEqual(s["decision"],"development_signal_not_confirmation")
        self.assertEqual(s["negative_maps"],2)
        self.assertFalse(s["automatic_promotion"])
        self.assertEqual(s,summarize(result,CFG))

    def test_floor_is_not_reliable_learning_signal(self):
        result=[]
        for i in range(6):
            r=root(i); c=choose_discovery(r,rows(r,range(8),"none"),CFG)
            result.append(describe(r,c,rows(r,range(8,16),"none"),CFG))
        s=summarize(result,CFG)
        self.assertEqual(s["floor_states"],6)
        self.assertEqual(s["decision"],"no_replicated_selection_gain")
        result[0]["complete"]=False
        self.assertEqual(summarize(result,CFG)["decision"],"resource_censored_inconclusive")


if __name__=="__main__": unittest.main()
