"""Whole-path constraints, cached lookahead, rollback and collector boundaries."""
import copy
from itertools import product
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import whole_pair_feedback as pair
from experiments import whole_pair_collection as collection
from experiments._common import read_json, write_json
from experiments.local_path_search import at, pair_events
from experiments.native_path_compatibility import seal
from tests.evaluation.test_prefix_budget_feedback import state, Probe, CONFIG
from tests.evaluation import test_native_path_compatibility as native_tests


class WholePairTests(unittest.TestCase):
    def test_waits_reverse_edges_and_future_goal_occupancy(self):
        constraints=pair.avoid_path([0,0,1,2],1)
        self.assertIn(('vertex',1,0,0),constraints)
        self.assertIn(('vertex',3,2,2),constraints)
        self.assertIn(('edge',2,1,0),constraints)
        self.assertIn(('vertex',6,2,2),pair.avoid_path([0,1,2],6))
        with self.assertRaises(ValueError): pair.avoid_path([],3)
        with self.assertRaises(ValueError): pair.avoid_path([1],-1)

    def test_constraints_exactly_cover_pair_events_under_cap(self):
        paths=[list(p) for n in range(1,5) for p in product(range(3),repeat=n)
               if all(abs(a-b)<=1 for a,b in zip(p,p[1:]))]
        for other in ([0],[0,0,1,2],[2,1,0],[1,0,1,2]):
            for path in paths:
                constraints=pair.avoid_path(other,len(path)-1)
                violated=any(at(path,t)==u if kind=='vertex' else at(path,t-1)==u and at(path,t)==v
                             for kind,t,u,v in constraints)
                self.assertEqual(violated,bool(pair_events(path,other)),(path,other))

    def test_both_methods_recover_pair_without_external_edits(self):
        for method in pair.METHODS:
            source=state(); before=copy.deepcopy(source)
            row=pair.recover_pair(Probe(),source,[0,1,2],11,method,CONFIG)
            self.assertTrue(row['recovered'])
            self.assertEqual(row['final']['conflicts'],0)
            self.assertEqual(row['final']['paths'][3],source['agents'][3]['path'])
            self.assertEqual(row['attempts'][0]['pair_check']['classification'],'pair_removed')
            self.assertEqual(source,before)
            self.assertEqual(row['cached_insertions'],1)
            victim=row['attempts'][0]['result']['records'][1]['search']
            self.assertEqual(victim['expanded'],0)
            self.assertIsNone(victim['low_level_collisions'])

    def test_returned_path_breaking_pair_contract_is_error(self):
        class BadProbe(Probe):
            def plan(self,aid,fixed,overrides,hard,constraints,cap,seconds):
                return super().plan(aid,fixed,overrides,hard,[],cap,seconds)
        with self.assertRaisesRegex(ValueError,'occupancy violation'):
            pair.recover_pair(BadProbe(),state(),[0,1,2],11,'whole_pair',CONFIG)

    def test_not_found_preserves_baseline(self):
        class EmptyProbe(Probe):
            def plan(self,aid,fixed,overrides,hard,constraints,cap,seconds):
                if constraints:
                    return dict(status='empty',path=[],cost=-1,expanded=1,generated=1,low_level_collisions=-1)
                return super().plan(aid,fixed,overrides,hard,constraints,cap,seconds)
        row=pair.recover_pair(EmptyProbe(),state(),[0,1,2],11,'whole_pair',CONFIG)
        self.assertFalse(row['recovered'])
        self.assertEqual(row['final'],row['base'])
        self.assertFalse(row['budget_exhausted'])

    def test_unknown_is_not_infeasibility(self):
        row=pair.recover_pair(Probe(),state(),[0,1,2],11,'whole_pair',dict(CONFIG,job_seconds=0))
        self.assertEqual(row['status'],'unknown')
        self.assertTrue(row['budget_exhausted'])

    def test_fixed_interleaved_order_and_cap(self):
        feedback=dict(evidence=[dict(blocker=0,victim=2,relaxed={'status':'path'}),
                                dict(blocker=1,victim=2,relaxed={'status':'path'})])
        base=dict(records=[dict(search=dict(path=[0,1])),dict(search=dict(path=[0,1,2]))])
        opts=pair.options_of(feedback,[0,1,2],base,4)
        self.assertEqual([(o['blocker'],o['repeat'],o['cap']) for o in opts],[(0,0,1),(1,0,2),(0,1,1),(1,1,2)])

    def test_external_guard_rejection_is_verified(self):
        source=dict(agents=[dict(id=7,path=[0,1,2]),dict(id=19,path=[2,1,0]),dict(id=50,path=[3,4,5])])
        self.assertEqual(pair.external_pairs([2,4,0],source,[7,19]),{50})
        base=dict(records=[dict(agent=19,search=dict(path=[2,1,0]))])
        attempt=dict(option=dict(blocker=7,victim=19,reference=dict(path=[2,4,0])),
                     result=None,reason='reference_adds_external_pair')
        self.assertEqual(pair.verify_pair_branch(source,[7,19],base,attempt,True)['classification'],'reference_guard_rejected')
        attempt['option']['reference']['path']=[2,1,0]
        with self.assertRaisesRegex(ValueError,'unsupported'):
            pair.verify_pair_branch(source,[7,19],base,attempt,True)


class CollectorTests(unittest.TestCase):
    def test_stop_does_not_dispatch(self):
        m=dict(content_sha256='f',jobs=[dict(job_id='j')],config=dict(collection_session_hours=2))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'STOP').touch()
            with patch.object(collection,'load',return_value=m),patch.object(collection.subprocess,'Popen') as spawn:
                collection.collect(root,20)
            spawn.assert_not_called()
            self.assertEqual(read_json(root/'run_status.json')['status'],'paused')

    def test_identity_mismatch_and_error_resume_rejected(self):
        m=dict(content_sha256='f',jobs=[dict(job_id='j')])
        row=seal(dict(schema=collection.SCHEMA,fingerprint='f',job=dict(job_id='j'),status='error'))
        collection.check_result(m,m['jobs'][0],row)
        with self.assertRaises(ValueError): collection.check_result(m,dict(job_id='other'),row)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); write_json(collection.output_file(root,m['jobs'][0]),row)
            with patch.object(collection,'load',return_value=m):
                with self.assertRaisesRegex(ValueError,'require resume'): collection.collect(root)
                with self.assertRaisesRegex(ValueError,'inspect saved error'): collection.collect(root,resume=True)

    def test_frozen_source_overlap_rejected(self):
        for output in ('build','build/old','build/old/new'):
            with self.assertRaises(ValueError):
                collection.checked_output(dict(output=output,source_collection='build/old'))


@unittest.skipIf(native_tests.extension is None,'isolated WSL probe extension required')
class NativeOccupancyTests(unittest.TestCase):
    def setUp(self):
        self.fixture=native_tests.NativeProbeTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_whole_path_mask_agrees_with_hard_stationary_agent(self):
        probe=self.fixture.probe
        for cap in (2,4):
            hard=probe.plan(0,[1],{},True,max_cost=cap)
            masked=probe.plan(0,[1],{},False,sorted(pair.avoid_path([1,1,1],cap)),cap,5)
            self.assertEqual(masked['status'],hard['status'])
            if masked['status']=='path':
                self.assertFalse(pair_events(masked['path'],[1,1,1]))
                self.assertEqual(masked['cost'],hard['cost'])

    def test_goal_wait_after_path_end_is_not_ignored(self):
        other=[1,4,5,2,1]
        masked=self.fixture.probe.plan(0,[],{},False,sorted(pair.avoid_path(other,2)),2,5)
        self.assertEqual(masked['status'],'empty')


if __name__=='__main__': unittest.main()
