from copy import deepcopy
import random
import unittest
from unittest.mock import patch

from experiments import sa_completion_endpoint_repair as repair
from scripts import run_sa_completion_amended_ttf as cli
from scripts import run_sa_completion_independent_ttf as old


class EndpointMatchingTests(unittest.TestCase):
    def matrix(self, edges, n):
        import numpy as np
        from scipy.sparse import csr_matrix
        rr,cc = zip(*edges) if edges else ([],[])
        return csr_matrix((np.ones(len(edges),dtype=np.int8),(rr,cc)),shape=(n,n))

    def test_augmenting_matching_recovers_greedy_dead_end(self):
        graph = self.matrix([(0,0),(0,1),(1,0),(2,2)],3)
        constrained = self.matrix([(0,0),(0,1),(1,0)],3)
        for seed in (1,2,7):
            pairs,proof = repair.assign_endpoints(graph,constrained,3,2,seed)
            self.assertEqual(set(pairs[:2]),{(0,1),(1,0)})
            self.assertEqual(pairs[2],(2,2))
            self.assertFalse(proof['joint_mapf_feasibility_proven'])
            self.assertFalse(proof['original_sampling_distribution_preserved'])

    def test_deterministic_unique_complete_and_quota(self):
        graph = self.matrix([(i,j) for i in range(12) for j in range(12) if i!=j],12)
        constrained = self.matrix([(i,j) for i in range(5) for j in range(6) if i!=j],12)
        one = repair.assign_endpoints(graph,constrained,10,4,77)
        random.seed(66)
        self.assertEqual(one,repair.assign_endpoints(graph,constrained,10,4,77))
        pairs,_ = one
        self.assertEqual(len({a for a,_ in pairs}),10)
        self.assertEqual(len({b for _,b in pairs}),10)
        self.assertTrue(all(constrained[a,b] for a,b in pairs[:4]))

    def test_impossible_quota_is_not_silently_lowered(self):
        graph = self.matrix([(0,0),(1,1)],2)
        constrained = self.matrix([(0,0)],2)
        with self.assertRaisesRegex(ValueError,'capacity 1 below quota 2'):
            repair.assign_endpoints(graph,constrained,2,2,1)
        with self.assertRaisesRegex(ValueError,'feasibility unknown'):
            repair.assign_endpoints(constrained,constrained,2,1,1,attempts=2)
        with self.assertRaisesRegex(ValueError,'must be legal'):
            repair.assign_endpoints(constrained,graph,2,1,1)

    def test_only_generation_and_output_change(self):
        previous = old.config()
        current = cli.config()
        for key in previous:
            if key not in ('output','schema'):
                self.assertEqual(previous[key],current[key])
        self.assertNotEqual(previous['output'],current['output'])
        self.assertEqual(cli._collect.__code__,cli.first.collect.__code__)
        self.assertIs(cli._collect.__globals__['rt'],old.rt)
        self.assertIsNone(current['max_decisions'])

    def test_existing_sampler_is_not_modified(self):
        from generators import task_flows
        sampler = task_flows.generate_tasks
        with patch.object(cli,'amendment',wraps=cli.amendment):
            cli.config()
        self.assertIs(task_flows.generate_tasks,sampler)

    def test_other_task_protocol_rejected(self):
        with self.assertRaisesRegex(ValueError,'unsupported task configuration'):
            repair.repair_task(None,{},1,'x',{})

    def test_bottleneck_and_distance_metadata_independent_validation(self):
        from generators.models import MapData,TaskData
        md = MapData(map_id='m',seed=1,grid=['.....'],metadata={})
        pools = dict(free=md.free_cells(),storage=[(0,0),(0,1)],station=[(0,3),(0,4)])
        metadata = dict(schema_version=1,agent_count=2,required_bottlenecks=[[0,2],None],
            flow_assignments=['storage->station']*2,realized_flow_counts={'storage->station':2},
            actual_shortest_distances=[3,3])
        task = TaskData(task_id='t',map_id='m',seed=3,starts=[(0,0),(0,1)],
            goals=[(0,3),(0,4)],metadata=metadata)
        config = dict(agent_density=.4,required_bottleneck_crossing_ratio=.5,
            od_matrix={'storage->station':1},minimum_shortest_distance=2,maximum_shortest_distance=4)
        with patch.object(repair,'_candidate_pools',return_value=pools),patch.object(repair,'_bottleneck_candidates',return_value=[(0,2)]):
            repair.validate_constraints(md,task,config)
            changed = deepcopy(task)
            changed.metadata['required_bottlenecks'][0]=[0,4]
            with self.assertRaisesRegex(ValueError,'bottleneck'):
                repair.validate_constraints(md,changed,config)
            changed = deepcopy(task)
            changed.metadata['actual_shortest_distances'][0]=2
            with self.assertRaisesRegex(ValueError,'stale'):
                repair.validate_constraints(md,changed,config)


if __name__=='__main__':
    unittest.main()
