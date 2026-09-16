import unittest

from scripts.preflight_sa_history_candidate_bridge import select_roots,choice,describe,admission


class BridgeTests(unittest.TestCase):
    def test_blind_selection_and_input_order(self):
        targets=[dict(map_id=f"m{m}",id=f"{m}-{s}-{i}",stratum=s,decision=32)
                 for m in range(8) for s in ("history_exact","progress") for i in range(2)]
        cfg=dict(seed=20260918,history_states=4,states=8)
        a=select_roots(targets,cfg)
        self.assertEqual(a,select_roots(list(reversed(targets)),cfg))
        self.assertEqual(sum(t["stratum"]=="history_exact" for t in a),4)
        for t in targets: t["future_success"]=True
        self.assertEqual([t["id"] for t in a],[t["id"] for t in select_roots(targets,cfg)])

    def test_tie_break_candidate_id(self):
        self.assertEqual(choice(["b","a"],[.5,.5]),"a")

    def test_selected_candidate_budget_and_old_control(self):
        r=dict(rows=[dict(candidate_id=x) for x in "abcd"],old_selected_id="c",target=dict(id="root"),
               predictions={k:[.9,.8,.7,.6] for k in ("ordered","temporal_bag","dynamic_progress","dynamic_completion")})
        d=describe(r,dict(seed=1,max_candidates=3))
        self.assertEqual(len(d["selected"]),3)
        self.assertEqual(d["selected"][:2],["c","a"])

    def test_constant_predictor_not_admitted(self):
        rs=[dict(target=dict(map_id=str(i)),choices=dict(ordered="a",temporal_bag="a",dynamic_progress="a",dynamic_completion="a"),
                 old_selected_id="a",spreads=dict(ordered=0,temporal_bag=0)) for i in range(8)]
        self.assertFalse(admission(rs)["passed"])


if __name__=="__main__": unittest.main()
