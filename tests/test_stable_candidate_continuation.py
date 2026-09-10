import unittest
from scripts import diagnose_stable_candidate_continuation as d


def results(after=39, feasible=False):
    rows=[]
    for trial in range(8):
        for arm in d.ARMS:
            is_alt=arm=='alternative_first'
            rows.append(dict(job=dict(trial=trial,arm=arm),feasible=feasible and is_alt,
                final_conflicts=after if is_alt else 40,
                fixed64_conflict_auc=after*64 if is_alt else 40*64,
                stop_reason='decision_budget',native_timeouts=0))
    return rows


class StableCandidateTests(unittest.TestCase):
    def test_schedule_paired_balanced(self):
        jobs=d.schedule({'fingerprint':'test'})
        self.assertEqual(len(jobs),16)
        self.assertEqual(len({j['job_id'] for j in jobs}),16)
        for i in range(8):
            pair=jobs[2*i:2*i+2]
            self.assertEqual(pair[0]['trial_seed'],pair[1]['trial_seed'])
            self.assertEqual(pair[0]['arm'],d.ARMS[i%2])

    def test_seeds_deterministic_unique(self):
        self.assertEqual(len({d.trial_seed(i) for i in range(8)}),8)
        self.assertEqual(d.trial_seed(0),d.trial_seed(0))
        self.assertNotEqual(d.trial_seed(0),d.legacy.seed(d.CID,0))

    def test_one_point_only_no_go(self):
        self.assertEqual(d.summarize(results())['decision'],'no_go_or_inconclusive')

    def test_two_point_signal(self):
        self.assertEqual(d.summarize(results(after=38))['decision'],'bounded_signal_only')

    def test_censoring_blocks_signal(self):
        rows=results(after=38); rows[1]['stop_reason']='wall_budget'
        self.assertFalse(d.summarize(rows)['no_budget_censoring'])
        self.assertEqual(d.summarize(rows)['decision'],'no_go_or_inconclusive')

    def test_feasibility_harm_blocks(self):
        rows=results(after=38); rows[0]['feasible']=True
        self.assertEqual(d.summarize(rows)['decision'],'no_go_or_inconclusive')

    def test_fixed_case_and_candidate(self):
        self.assertEqual(d.CID,'case-3723c8bd272e')
        self.assertEqual(d.CANDIDATE,'neighborhood-0d3103de10e25097')
        self.assertEqual(d.ARMS,('frozen','alternative_first'))


if __name__=='__main__':
    unittest.main()
