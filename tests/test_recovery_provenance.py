import unittest
from scripts.audit_recovery_provenance import reconstruct_groups


class ProvenanceTests(unittest.TestCase):
    def test_request_to_source_mapping(self):
        groups = reconstruct_groups(dict(proposal_count=2, unique_neighborhood_count=1,
            invalid_indices=[], rows=[([4, 7], [0, 1])]),
            [(4, 'random', 4, 0), (7, 'collision', 8, 0)], [101, 102])
        self.assertEqual(groups[0]['sources'], [
            dict(family='random:4', seed_agent=4, proposal_seed=101),
            dict(family='collision:8', seed_agent=7, proposal_seed=102)])

    def test_missing_or_duplicate_request_rejected(self):
        for indices in ([0], [0, 0]):
            with self.assertRaises(AssertionError):
                reconstruct_groups(dict(proposal_count=2, unique_neighborhood_count=1,
                    invalid_indices=[], rows=[([4, 7], indices)]),
                    [(4, 'random', 4, 0), (7, 'collision', 8, 0)], [101, 102])

    def test_deadline_rejected(self):
        with self.assertRaises(AssertionError):
            reconstruct_groups(dict(invalid_indices=[0]), [], [])


if __name__ == '__main__':
    unittest.main()
