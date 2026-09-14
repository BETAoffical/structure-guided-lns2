from itertools import product
import random

from experiments.joint_slot_capacity import hall_witness


def test_three_agents_two_cells_is_not_a_pairwise_singleton():
    witness=hall_witness({4:{0,1},8:{0,1},11:{0,1}})
    assert witness==dict(agents=[4,8,11],cells=[0,1],deficit=1)


def test_matching_certificate_agrees_with_exhaustive_assignments():
    rng=random.Random(20260915)
    for _ in range(400):
        domains={a:{c for c in range(5) if rng.randrange(2)} for a in (3,9,20,25)}
        feasible=any(len(set(xs))==len(xs) for xs in product(*domains.values()))
        witness=hall_witness(domains)
        assert (witness is None)==feasible
        if witness:
            neighbors=set().union(*(domains[a] for a in witness["agents"]))
            assert neighbors==set(witness["cells"]) and len(neighbors)<len(witness["agents"])
