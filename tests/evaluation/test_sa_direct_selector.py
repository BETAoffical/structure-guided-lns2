import unittest
import numpy as np
from experiments.sa_direct_selector import distribution,objective_grad,inputs


class DirectMath(unittest.TestCase):
    def test_gradient_and_ragged_probabilities(self):
        rng=np.random.default_rng(3)
        x=rng.normal(size=(5,4)); target=np.array([.8,.2,.1,.7,.2]); offsets=np.array([0,2,5])
        params=[rng.normal(size=(4,3))*.1,np.zeros(3),rng.normal(size=3)*.1,np.zeros(1)]
        loss,grads=objective_grad(x,target,offsets,params)
        self.assertTrue(np.isfinite(loss))
        for k,idx in [(0,(1,2)),(1,(1,)),(2,(2,)),(3,(0,))]:
            old=params[k][idx]; h=1e-6
            params[k][idx]=old+h; plus=objective_grad(x,target,offsets,params)[0]
            params[k][idx]=old-h; minus=objective_grad(x,target,offsets,params)[0]
            params[k][idx]=old
            self.assertAlmostEqual(grads[k][idx],(plus-minus)/(2*h),places=7)
        p=distribution(np.array([1000.,999.,-1000.,-1001.,-999.]),offsets)
        self.assertAlmostEqual(sum(p[:2]),1);self.assertAlmostEqual(sum(p[2:]),1)

    def test_candidate_order_equivariance(self):
        x=np.array([[1.,3.],[5.,2.],[2.,1.]],dtype=np.float32); order=[2,0,1]
        z=inputs(x,np.zeros(2),np.ones(2)); perm=inputs(x[order],np.zeros(2),np.ones(2))
        np.testing.assert_allclose(perm,z[order],atol=1e-6)


if __name__=='__main__': unittest.main()
