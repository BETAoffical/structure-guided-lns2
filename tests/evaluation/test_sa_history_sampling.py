import copy
import unittest
from pathlib import Path

from experiments._common import read_json
from experiments.sa_history_selector import History
from experiments.sa_history_sampling import classify, choose_history_candidates, select_sources

ROOT = Path(__file__).resolve().parents[2]
CONFIG = read_json(ROOT / "configs/sa_history_selector_corrected.json")


def state():
    return dict(num_of_colliding_pairs=1, conflict_edges=[[2,7]], feasible=False,
                agents=[dict(id=2,path=[0,1]),dict(id=7,path=[2,1]),dict(id=19,path=[3,3])])


class SamplingTests(unittest.TestCase):
    def test_causal_strata_and_external_context(self):
        s = state(); h = History(s)
        pool = [dict(agents=[2,7]), dict(agents=[2,19])]
        self.assertIsNone(classify(h,s,pool,CONFIG))
        for i in range(32):
            e = dict(decision=i,metrics=dict(neighborhood=[2,7], conflicts_before=1,conflicts_after=1,
                                             pp_rolled_back=False, acceptance_evaluated=True))
            h.observe(s,e,s)
        self.assertEqual(classify(h,s,pool,CONFIG), "history_exact")
        changed = copy.deepcopy(s); changed["agents"][-1]["path"] = [3,4]
        self.assertEqual(classify(h,changed,pool,CONFIG), "history_changed")
        self.assertIsNone(classify(h,s,pool[:1],CONFIG))
        self.assertEqual(classify(h,s,pool,CONFIG), "history_exact")

    def test_low_score_historical_candidate_is_retained(self):
        pool = [dict(candidate_id=str(i), agents=[i,i+100], score=-i) for i in range(20)]
        fs = [{"history.same_set_count":0.,"history.valid_context_count":0.} for _ in pool]
        fs[17] = {"history.same_set_count":.0625,"history.valid_context_count":.03125}
        fs[16] = {"history.same_set_count":.03125,"history.valid_context_count":.03125}
        chosen = choose_history_candidates(pool,0,4,fs)
        self.assertEqual(len(chosen),6)
        self.assertEqual(chosen[:2],[0,17])
        self.assertTrue(any(fs[i]["history.same_set_count"]==0 for i in chosen))
        self.assertEqual(chosen,choose_history_candidates(pool,0,4,fs))

    def test_distinct_episode_quota_and_missing_stratum_not_filled_by_other_map(self):
        available = []
        for j in range(5):
            for kind in ("progress","history_exact","history_changed"):
                available.append(dict(map_id="a",item=dict(job_id=str(j)),decision=40,stratum=kind))
        available.append(dict(map_id="b",item=dict(job_id="b"),decision=16,stratum="progress"))
        selected, coverage = select_sources(available,["a","b"],CONFIG)
        self.assertEqual(len(selected),5)
        self.assertEqual(len({r["item"]["job_id"] for r in selected}),5)
        self.assertEqual(sum(r["map_id"]=="a" for r in selected),4)
        self.assertEqual(coverage[-1]["selected"],{"progress":1})
        self.assertEqual(select_sources(list(reversed(available)),["b","a"],CONFIG), (selected,coverage))

    def test_models_and_targets_not_retuned(self):
        old = read_json(ROOT/"configs/sa_history_selector_pilot.json")
        for key in ("model","seed","trials","pp_seconds","history_window","quality_tolerance_pairs"):
            self.assertEqual(old[key],CONFIG[key])
        self.assertNotEqual(old["output"],CONFIG["output"])
        self.assertFalse(CONFIG["runtime_integration_allowed"])


if __name__ == "__main__":
    unittest.main()
