import unittest
import numpy as np
from experiments.sa_direct_onpolicy import score_gradient
from experiments.sa_terminal_efficiency import episode_return


class CompleteCredit(unittest.TestCase):
    def test_score_gradient(self):
        rng=np.random.default_rng(8);x=rng.normal(size=(5,4));off=np.array([0,2,5]);sel=np.array([1,0]);co=np.array([.1,-.2])
        params=[rng.normal(size=(4,3))*.1,np.zeros(3),rng.normal(size=3)*.1,np.zeros(1)]
        _,g,_=score_gradient(x,sel,off,co,params)
        for k,idx in [(0,(1,2)),(1,(1,)),(2,(2,)),(3,(0,))]:
            old=params[k][idx];h=1e-6
            params[k][idx]=old+h;plus=score_gradient(x,sel,off,co,params)[0]
            params[k][idx]=old-h;minus=score_gradient(x,sel,off,co,params)[0]
            params[k][idx]=old
            self.assertAlmostEqual(g[k][idx],(plus-minus)/(2*h),places=7)

    def test_terminal_unknown_and_work_bonus(self):
        row=dict(status='ok',stop='feasible',success=True,decisions=4,generated=10,final_conflicts=0)
        self.assertAlmostEqual(episode_return(row,objective='completion_work',max_decisions=None,node_budget=100),1.09)
        row.update(status='censored',stop='incomplete_pp',success=False,final_conflicts=3)
        self.assertIsNone(episode_return(row,objective='completion_work',max_decisions=None,node_budget=100))


if __name__=='__main__':unittest.main()
