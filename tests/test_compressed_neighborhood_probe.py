import unittest
from scripts.probe_compressed_neighborhood import make_jobs, seed


class CompressedProbeTests(unittest.TestCase):
    def setUp(self):
        self.sets={'original16':list(range(16)), 'compressed12':list(range(12)), 'compressed11':list(range(11))}

    def test_complete_deterministic_schedule(self):
        jobs=make_jobs(self.sets)
        self.assertEqual(jobs,make_jobs(self.sets))
        self.assertEqual(len(jobs),24)
        self.assertEqual(len({j['id'] for j in jobs}),24)

    def test_common_relative_order_and_seeds(self):
        jobs=make_jobs(self.sets)
        for trial in range(8):
            group=[j for j in jobs if j['trial']==trial]
            full=group[0]['action']['repair_order']
            for job in group:
                action=job['action']
                self.assertEqual(action['repair_order'],[a for a in full if a in action['agents']])
                self.assertEqual(action['pp_random_seed'],group[0]['action']['pp_random_seed'])

    def test_stream_namespaces(self):
        values=[seed(label,trial) for label in ('order','action','pp') for trial in range(8)]
        self.assertEqual(len(set(values)),24)
        self.assertTrue(all(0<=v<2**31 for v in values))

    def test_actions_do_not_supply_witness_paths(self):
        for job in make_jobs(self.sets):
            self.assertEqual(set(job['action']),{'mode','agents','repair_order','random_seed','pp_random_seed'})


if __name__=='__main__':
    unittest.main()
