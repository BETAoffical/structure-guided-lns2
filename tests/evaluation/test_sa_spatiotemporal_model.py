import copy
import importlib.util
import unittest

import numpy as np

from experiments.sa_spatiotemporal_input import encode_state
from experiments.sa_spatiotemporal_model import encode_model_input, fit_scaler, make_model, select
from experiments.state_analysis import reconstruct_conflicts
from scripts.run_sa_spatiotemporal_model_probe import tensor_arrays


def fixture(paths=None):
    paths = paths or {3:[3,4,5],8:[1,4,7],20:[6]}
    agents = [dict(id=i,path=p,start=p[0],goal=p[-1]) for i,p in paths.items()]
    pairs = {tuple(sorted((e.left,e.right))) for e in reconstruct_conflicts(agents)}
    return encode_state(dict(rows=3,cols=3,obstacles=[0]*9,agents=agents,
        conflict_edges=[list(e) for e in sorted(pairs)],num_of_colliding_pairs=len(pairs)))


class InputTests(unittest.TestCase):
    def test_dimensions_and_no_result_dependence(self):
        s = fixture()
        before = copy.deepcopy(s)
        tokens = encode_model_input(s,[[3],[8,20]])
        self.assertEqual(tokens["base"].shape,(3,3,27))
        self.assertEqual(tokens["condition"][1].shape,(2,3,32))
        self.assertEqual(s,before)
        s.update(runtime=999,outcome={"success":True},generated=999,map_id="LEAK",layout="LEAK")
        extra = encode_model_input(s,[[3],[8,20]])
        np.testing.assert_array_equal(tokens["base"],extra["base"])
        for a,b in zip(tokens["condition"],extra["condition"]):
            np.testing.assert_array_equal(a,b)

    def test_selected_outsider_and_goal_persistence(self):
        tokens = encode_model_input(fixture({3:[3,4],8:[1,2,5,4,7]}),[[8],[3,8]])
        # For agent8 at t3, outsider3 occupies cell4 permanently.
        self.assertAlmostEqual(float(tokens["condition"][0][0,3,5]),np.log(2),places=6)
        self.assertAlmostEqual(float(tokens["condition"][0][0,3,25]),np.log(2),places=6)
        # Selecting both moves that occupancy from outsider to selected.
        self.assertEqual(float(tokens["condition"][1][1,3,5]),0.)
        self.assertAlmostEqual(float(tokens["condition"][1][1,3,0]),np.log(2),places=6)

    def test_swap_channel(self):
        t = encode_model_input(fixture({3:[3,4],8:[4,3]}),[[3],[8]])
        self.assertAlmostEqual(float(t["condition"][0][0,1,31]),np.log(2),places=6)
        self.assertAlmostEqual(float(t["base"][0,1,26]),np.log(2),places=6)

    def test_membership_order_and_id_values_not_features(self):
        a = fixture()
        original = encode_model_input(a,[[3,8],[20]])
        reordered = encode_model_input(a,[[8,3],[20]])
        np.testing.assert_array_equal(original["condition"][0],reordered["condition"][0])
        changed = copy.deepcopy(a)
        mapping = {3:103,8:108,20:120}
        for agent in changed["agents"]:
            agent["id"] = mapping[agent["id"]]
        renamed = encode_model_input(changed,[[103,108],[120]])
        np.testing.assert_array_equal(original["base"],renamed["base"])
        for x,y in zip(original["condition"],renamed["condition"]):
            np.testing.assert_array_equal(x,y)

    def test_time_information_and_train_only_scaler(self):
        early = encode_model_input(fixture({3:[3,4,5,5],8:[1,4,7,7]}),[[3],[8]])
        late = encode_model_input(fixture({3:[3,3,4,5],8:[1,1,4,7]}),[[3],[8]])
        self.assertFalse(np.array_equal(early["condition"][0],late["condition"][0]))
        states = [dict(map_id=m,candidates=[dict(features={"x":x})]) for m,x in (("a",1),("b",3),("c",1000))]
        mean,std = fit_scaler(states,"c",["x"])
        self.assertEqual(mean.tolist(),[2])
        self.assertEqual(std.tolist(),[1])

    def test_ties_and_tensor_masks(self):
        self.assertEqual(select([.2,.2],["a","b"],"b"),1)
        self.assertEqual(select([.2,.3],["a","b"],"a"),1)
        raw = tensor_arrays([encode_model_input(fixture(),[[3],[8,20]])])
        self.assertEqual(raw["agent_mask"].tolist(),[[[1.,0.],[1.,1.]]])
        self.assertTrue(np.isfinite(raw["condition"]).all())


@unittest.skipUnless(importlib.util.find_spec("torch"),"existing isolated PyTorch environment required")
class ModelTests(unittest.TestCase):
    def test_forward_backward_and_determinism(self):
        import torch
        torch.set_num_threads(1)
        raw = tensor_arrays([encode_model_input(fixture(),[[3],[8,20]])])
        batch = {k:torch.from_numpy(v) for k,v in raw.items()}
        flat = torch.tensor([[[0.,1.],[1.,0.]]])
        for profile in ("flat","time_bag","path_time"):
            torch.manual_seed(17)
            model = make_model(profile,2)
            output = model(batch,flat)
            self.assertEqual(tuple(output.shape),(1,2))
            ((output[:,0]-output[:,1]-.3)**2).mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            torch.manual_seed(17)
            other = make_model(profile,2)
            torch.testing.assert_close(output,other(batch,flat),rtol=0,atol=0)

    def test_candidate_permutation_and_matching_capacity(self):
        import torch
        raw = tensor_arrays([encode_model_input(fixture(),[[3],[8,20]])])
        batch = {k:torch.from_numpy(v) for k,v in raw.items()}
        flat = torch.tensor([[[0.,1.],[1.,0.]]])
        torch.manual_seed(23)
        model = make_model("path_time",2)
        actual = model(batch,flat)
        reverse = dict(batch)
        for key in ("condition","indices","agent_mask"):
            reverse[key] = batch[key][:,[1,0]]
        torch.testing.assert_close(actual[:,[1,0]],model(reverse,flat[:,[1,0]]))
        bag = make_model("time_bag",2)
        self.assertEqual(sum(p.numel() for p in model.parameters()),sum(p.numel() for p in bag.parameters()))


if __name__=="__main__":
    unittest.main()
