import unittest

from experiments.sa_history_information import TEMPORAL,RESOURCE,profile_features
from scripts.audit_sa_history_matched_controls import matched_features,PROFILES


class MatchedTests(unittest.TestCase):
    def row(self):
        return dict(id="one",base={"dynamic":1,"history":2},
                    records=[{k:i for k in TEMPORAL+RESOURCE} for i in range(32)])

    def test_same_columns_for_matched_shuffle(self):
        r=self.row()
        for profile in PROFILES[1:]:
            self.assertEqual(matched_features(r,profile).keys(),profile_features(r,"ordered").keys())

    def test_shuffle_not_future_labels(self):
        r=self.row()
        a=matched_features(r,PROFILES[1])
        r["labels"]={"completion":1}
        self.assertEqual(a,matched_features(r,PROFILES[1]))

    def test_temporal_bag_excludes_resources(self):
        r=self.row()
        f=matched_features(r,"temporal_bag")
        self.assertEqual(len(f),len(r["base"])+len(TEMPORAL))
        r["records"].reverse()
        self.assertEqual(f,matched_features(r,"temporal_bag"))


if __name__=="__main__":
    unittest.main()
