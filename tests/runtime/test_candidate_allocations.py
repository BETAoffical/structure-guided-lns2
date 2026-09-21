from copy import deepcopy
import random
import unittest
from unittest.mock import patch

from experiments import neighborhood_candidates as candidate
from scripts.audit_candidate_allocations import reference


def group(agents, family="random:4", seed=1, proposal=5):
    return dict(agents=agents, sources=[dict(family=family, seed_agent=seed, proposal_seed=proposal)])


class CandidateAllocationTests(unittest.TestCase):
    def test_repeated_groups_create_identity_once_and_preserve_all_sources(self):
        groups = [group([7, 2]), group([2, 7], seed=9, proposal=6),
                  group([7, 2], family="target:4", proposal=7)]
        saved = deepcopy(groups)
        with patch.object(candidate, "candidate_id", wraps=candidate.candidate_id) as identity:
            actual = candidate.select_representative_neighborhood_groups(groups, 2)
        self.assertEqual(identity.call_count, 1)
        self.assertEqual(groups, saved)
        self.assertEqual(actual, reference()[0](groups, 2))
        self.assertEqual(actual[0]["proposal_count_by_family"], {"random:4": 2, "target:4": 1})
        self.assertEqual(actual[0]["proposal_seeds"], [5, 6, 7])
        self.assertEqual(actual[0]["seed_agents"], [1, 9])

    def test_jaccard_rank_count_and_hash_ties_match_frozen_code(self):
        groups = [group([2, 7]), group([2, 7]), group([2, 9]), group([7, 10]), group([20, 30])]
        for count in (1, 2, 3, 8):
            self.assertEqual(candidate.select_representative_neighborhood_groups(groups, count),
                             reference()[0](groups, count))

    def test_randomized_reference_equivalence_and_input_isolation(self):
        rng = random.Random(20260921)
        for index in range(300):
            groups = []
            for j in range(rng.randrange(1, 65)):
                agents = rng.sample(range(-5, 400, 3), rng.choice([1, 4, 8, 16, 32]))
                groups.append(group(agents, rng.choice(["target:4", "collision:8", "random:16"]), j, index + j))
                if rng.random() < .4:
                    groups.append(deepcopy(groups[-1]))
            original = deepcopy(groups)
            count = rng.randrange(1, 5)
            self.assertEqual(candidate.select_representative_neighborhood_groups(groups, count), reference()[0](groups, count))
            self.assertEqual(groups, original)

    def test_invalid_input_errors_and_empty_pool_unchanged(self):
        for groups, count in (([], 0), ([group([])], 2), ([group([1, 1])], 2),
                              ([dict(agents=[1])], 2), ([dict(agents=[1], sources=[])], 2)):
            for function in (reference()[0], candidate.select_representative_neighborhood_groups):
                with self.assertRaises(ValueError):
                    function(groups, count)
        self.assertEqual(candidate.select_representative_neighborhood_groups([], 2), [])

    def test_grouped_and_expanded_proposals_equal(self):
        grouped = group([50, 2])
        grouped["sources"] += [dict(family="collision:8", seed_agent=50, proposal_seed=6)]
        expanded = [dict(agents=grouped["agents"], **s) for s in grouped["sources"]]
        self.assertEqual(candidate.select_representative_neighborhood_groups([grouped], 2),
                         candidate.select_representative_neighborhoods(expanded, 2))


if __name__ == "__main__":
    unittest.main()
