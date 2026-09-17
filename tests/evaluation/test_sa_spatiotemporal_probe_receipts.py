import copy
import unittest

from scripts.verify_sa_spatiotemporal_probe import verify_fit_identity


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.states = [dict(state_id="a",map_id="m1"),dict(state_id="b",map_id="m2")]
        self.result = dict(method="path_time",held="m1",seed=3,train_ids=["b"],
                           rows=[dict(state_id="a"),dict(state_id="b")],losses=[.1]*60)

    def check(self,value):
        verify_fit_identity(value,"path_time","m1",3,self.states,60)

    def test_complete(self):
        self.check(self.result)

    def test_wrong_profile_seed_or_fold(self):
        for key,value in (("method","time_bag"),("seed",4),("held","m2")):
            altered = copy.deepcopy(self.result)
            altered[key] = value
            with self.assertRaises(ValueError):
                self.check(altered)

    def test_map_leakage_and_incomplete_epochs(self):
        for key,value in (("train_ids",["a","b"]),("losses",[.1]*59),("rows",[dict(state_id="a")]*2)):
            altered = copy.deepcopy(self.result)
            altered[key] = value
            with self.assertRaises(ValueError):
                self.check(altered)


if __name__=="__main__":
    unittest.main()
