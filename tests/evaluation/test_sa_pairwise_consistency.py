import copy
from itertools import combinations
import unittest

from experiments.sa_paired_completion import PairedCompletionModel, pair_vector
from experiments.sa_pairwise_consistency import matrix_diagnostics, inspect, evaluate, label_summary, counts


def fixture():
    return dict(anchor_id="b",agent_ids=[2,5,9],candidates=[
        dict(candidate_id=c,agents=[a],features={"x":v}) for c,a,v in (("a",2,0.),("b",5,1.),("c",9,3.))])


class ConsistencyTests(unittest.TestCase):
    def test_borda_winner_can_lose_to_anchor(self):
        r = matrix_diagnostics(list("abc"),{("a","b"):-.1,("a","c"):.8,("b","c"):-.3},"b")
        self.assertEqual(r["selected"],"a")
        self.assertTrue(r["selected_loses_anchor"])
        self.assertEqual(r["direct_anchor_selected"],"c")
        self.assertEqual(r["veto_selected"],"b")
        self.assertEqual(r["cycles"],[["a","b","c"]])
        self.assertEqual(r["deletion_changes"],1)

    def test_true_value_differences_are_consistent(self):
        values = dict(a=.2,b=.5,c=.8,d=.9)
        margins = {(a,b):values[a]-values[b] for a,b in combinations(values,2)}
        r = matrix_diagnostics(list(values),margins,"b")
        self.assertEqual(r["selected"],"d")
        self.assertFalse(r["cycles"])
        self.assertEqual(r["deletion_changes"],0)
        self.assertLess(r["projection_rmse"],1e-15)
        self.assertLess(r["maximum_triangle_residual"],1e-15)

    def test_scalar_projection_cannot_change_borda_order(self):
        import numpy as np
        ids = list("abcd")
        margins = dict(zip(combinations(ids,2),[.2,-.8,.1,.9,-.2,.5]))
        r = matrix_diagnostics(ids,margins,"b")
        matrix = []
        for a,b in margins:
            matrix.append([int(c==a)-int(c==b) for c in ids])
        solution = np.linalg.lstsq(matrix,list(margins.values()),rcond=None)[0]
        self.assertTrue(np.allclose(solution,[r["utilities"][c] for c in ids],atol=1e-14))
        self.assertEqual(max(ids,key=lambda c:r["scores"][c]),max(ids,key=lambda c:r["utilities"][c]))

    def test_ties_prefer_anchor_not_hash(self):
        r = matrix_diagnostics(list("abc"),dict.fromkeys(combinations("abc",2),0.),"c")
        self.assertEqual(r["selected"],"c")
        self.assertEqual(r["direct_anchor_selected"],"c")
        self.assertFalse(r["changed"])

    def test_inspection_exactly_reproduces_rank(self):
        state = fixture()
        margins = {("a","b"):-.1,("a","c"):.8,("b","c"):-.3}
        lookup = {}
        for a,b in combinations(state["candidates"],2):
            value = margins[a["candidate_id"],b["candidate_id"]]
            lookup[tuple(pair_vector(a,b,["x"]))] = value
            lookup[tuple(pair_vector(b,a,["x"]))] = -value
        class Estimator:
            def predict(self,rows):
                return [lookup[tuple(row)] for row in rows]
        model = PairedCompletionModel(["x"],Estimator())
        expected = model.rank(state)
        r = inspect(model,state,expected)
        state["candidates"].reverse()
        self.assertEqual(r,inspect(model,state,expected))
        with self.assertRaises(ValueError):
            inspect(model,state,dict(expected,selected="c"))

    def test_rejects_invalid_pair_matrix(self):
        for margins in ({},{("a","b"):float("nan")},{("a","b"):1.2}):
            with self.assertRaises(ValueError):
                matrix_diagnostics(list("ab"),margins,"a")

    def test_singleton_does_not_call_estimator(self):
        state = fixture()
        state["candidates"] = [state["candidates"][1]]
        r = inspect(PairedCompletionModel(["x"],None),state)
        self.assertEqual(r["selected"],"b")
        self.assertEqual(r["projection_rmse"],0.)

    def test_outcomes_do_not_change_selections(self):
        row = dict(matrix_diagnostics(list("abc"),{("a","b"):-.1,("a","c"):.8,("b","c"):-.3},"b"),anchor_id="b")
        original = copy.deepcopy(row)
        first = evaluate(row,dict(a=[0]*8,b=[1]*8,c=[1]*8))
        second = evaluate(row,dict(a=[1]*8,b=[0]*8,c=[0]*8))
        self.assertEqual(first["choices"],second["choices"])
        self.assertEqual(row,original)
        self.assertTrue(first["loss_in_both_halves"])
        self.assertTrue(second["gain_in_both_halves"])

    def test_rejects_censored_or_incomplete_labels(self):
        row = dict(matrix_diagnostics(list("ab"),{("a","b"):.1},"b"),anchor_id="b")
        for values in (dict(a=[None]*8,b=[1]*8),dict(a=[1]*7,b=[1]*8),dict(a=[2]*8,b=[1]*8),dict(a=[1]*8)):
            with self.assertRaises(ValueError):
                evaluate(row,values)

    def test_map_weighting_not_trial_weighting(self):
        rates = [dict(anchor=0.,model=1.,direct=1.,veto=1.),dict(anchor=1.,model=0.,direct=0.,veto=0.)]
        rows = [dict(map_id=m,labels={"frozen":dict(choice_rates=rates[k],selected_minus_anchor=0.,
                    loss_in_both_halves=False,gain_in_both_halves=False)}) for m,k in [("a",0),("a",0),("b",1)]]
        r = label_summary(rows,"frozen",42,1000)
        self.assertEqual(r["contrasts"]["model_minus_anchor"]["delta"],0.)
        self.assertEqual(r,label_summary(rows[::-1],"frozen",42,1000))

    def test_counts_keep_repeated_visits_as_descriptive_only(self):
        row = matrix_diagnostics(list("ab"),{("a","b"):.1},"b")
        r = counts([row,row])
        self.assertEqual(r["decisions"],2)
        self.assertEqual(r["changed"],2)
        self.assertEqual(r["selected_loses_anchor"],0)


if __name__ == "__main__":
    unittest.main()
