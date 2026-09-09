from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import prefix_budget_feedback as feedback
from experiments import prefix_feedback_collection as collection

CONFIG=dict(job_seconds=10,call_seconds=5,max_attempts=4,max_feedback_blockers=2)


def state():
    paths=[[0,1,4],[2,1,0],[6,9,10,11,8],[5,8,7,6]]
    return dict(rows=4,cols=3,obstacles=[False]*12,conflict_edges=[[0,1]],num_of_colliding_pairs=1,
        agents=[dict(id=i,path=p,start=p[0],goal=p[-1]) for i,p in enumerate(paths)])


class Probe:
    def seed_rng(self,seed):
        self.seed=seed

    def plan(self,aid,fixed,overrides,hard,constraints,cap,seconds):
        if hard:
            raise AssertionError('soft CAT must be preserved')
        if aid==0:
            if constraints:
                if cap!=2:
                    raise AssertionError('same-cost predecessor required')
                path=[0,3,4]
            else:
                path=[0,1,4]
        elif aid==1:
            path=[2,1,0]
        else:
            path=[6,9,10,11,8] if overrides.get(0)==[0,3,4] else [6,7,8]
        return dict(status='path',path=path,cost=len(path)-1,expanded=2,generated=3,low_level_collisions=0)


def gate_fixture():
    cases=[dict(case_id=str(i),map_id=str(i%2),role='failed_long' if i==0 else 'fast_control') for i in range(3)]
    rows=[dict(job=dict(case_id=c['case_id'],seed=s,method=m),status='ok',budget_exhausted=False,
        recovered=m=='prefix_resources',triggered=True,base={'conflicts':1},final={'conflicts':0 if m=='prefix_resources' else 1},
        baseline_matches_frozen=True,native_paths_verified=True,control_matches_frozen=True)
        for c in cases for s in (1,2) for m in feedback.METHODS]
    return cases,rows


class PrefixFeedbackTests(unittest.TestCase):
    def test_same_budget_prefix_recovers_earlier_pair_without_external_edits(self):
        source=state(); before=copy.deepcopy(source)
        old=feedback.recover_prefix(Probe(),source,[0,1,2],11,'directed_resources',CONFIG)
        new=feedback.recover_prefix(Probe(),source,[0,1,2],11,'prefix_resources',CONFIG)
        self.assertFalse(old['recovered'])
        self.assertEqual(old['feedback']['external_blockers'],[3])
        self.assertTrue(new['recovered'])
        self.assertEqual(new['path_check']['conflicts'],0)
        self.assertEqual(new['attempts'][0]['option']['blocker'],0)
        self.assertEqual(new['attempts'][0]['option']['victim'],1)
        self.assertEqual(new['final']['paths'][3],source['agents'][3]['path'])
        self.assertEqual(source,before)
        self.assertLessEqual(len(new['attempts']),4)
        self.assertLessEqual(len(new['feedback']['evidence']),2)

    def test_unique_pair_degree_not_event_frequency(self):
        event=lambda a,b,t:dict(left=a,right=b,time=t,kind='vertex')
        base=dict(records=[dict(agent=0,incident_events=[]),
            dict(agent=1,incident_events=[event(0,1,t) for t in range(100)]),
            dict(agent=2,incident_events=[event(1,2,100)]),
            dict(agent=3,incident_events=[event(1,3,101)])])
        ranked,external=feedback.rank_relations(base,[0,1,2,3])
        self.assertEqual(ranked[0]['blocker'],1)
        self.assertEqual(ranked[0]['blocker_pair_degree'],3)
        self.assertEqual(ranked[-1]['blocker_pair_degree'],1)
        self.assertFalse(external)
        for record in base['records']:
            record['incident_events'].reverse()
        self.assertEqual(ranked,feedback.rank_relations(base,[0,1,2,3])[0])

    def test_future_agent_event_rejected(self):
        base=dict(records=[dict(agent=0,incident_events=[dict(left=0,right=1,time=1)])])
        with self.assertRaisesRegex(ValueError,'unplanned'):
            feedback.rank_relations(base,[0,1])

    def test_failed_alternative_preserves_base_exactly(self):
        class EmptyProbe(Probe):
            def plan(self,aid,fixed,overrides,hard,constraints,cap,seconds):
                if constraints:
                    return dict(status='empty',path=[],cost=-1,expanded=1,generated=1,low_level_collisions=-1)
                return super().plan(aid,fixed,overrides,hard,constraints,cap,seconds)
        row=feedback.recover_prefix(EmptyProbe(),state(),[0,1,2],11,'prefix_resources',CONFIG)
        self.assertFalse(row['recovered'])
        self.assertEqual(row['final'],row['base'])
        self.assertEqual(row['final']['paths'],[a['path'] for a in state()['agents']])

    def test_controls_delegate_unchanged(self):
        with patch.object(feedback,'recover',return_value={'sentinel':True}) as original:
            for method in feedback.METHODS[:-1]:
                self.assertEqual(feedback.recover_prefix(None,None,None,1,method,CONFIG),{'sentinel':True})
        self.assertEqual(original.call_count,3)

    def test_time_is_not_a_scientific_signature_field(self):
        row=feedback.recover_prefix(Probe(),state(),[0,1,2],11,'prefix_resources',CONFIG)
        other=copy.deepcopy(row); other['diagnostic_seconds']+=100
        self.assertEqual(feedback.result_signature(row),feedback.result_signature(other))
        other['final']['paths'][0]=[0,1,4]
        self.assertNotEqual(feedback.result_signature(row),feedback.result_signature(other))

    def test_unknown_budget_is_not_infeasibility(self):
        row=feedback.recover_prefix(Probe(),state(),[0,1,2],11,'prefix_resources',dict(CONFIG,job_seconds=0))
        self.assertEqual(row['status'],'unknown')
        self.assertTrue(row['budget_exhausted'])
        self.assertFalse(row['recovered'])

    def test_gate_keeps_single_trigger_separate_from_double_trigger(self):
        cases,rows=gate_fixture()
        self.assertTrue(feedback.evaluate_gate(rows,cases,[1,2])['passed'])
        for row in rows:
            if row['job']['case_id']=='2' and row['job']['seed']==2:
                row.update(triggered=False,recovered=False)
        gate=feedback.evaluate_gate(rows,cases,[1,2])
        self.assertFalse(gate['passed'])
        self.assertEqual(gate['trigger_strata'],{'2':2,'1':1})
        self.assertEqual(gate['stable_recovered_states']['prefix_resources'],['0','1'])

    def test_gate_rejects_duplicate_censored_and_unverified_results(self):
        cases,rows=gate_fixture()
        rows[-1]=rows[0]
        self.assertFalse(feedback.evaluate_gate(rows,cases,[1,2])['passed'])
        cases,rows=gate_fixture(); rows[0]['budget_exhausted']=True
        self.assertFalse(feedback.evaluate_gate(rows,cases,[1,2])['passed'])
        cases,rows=gate_fixture(); rows[0]['native_paths_verified']=False
        self.assertFalse(feedback.evaluate_gate(rows,cases,[1,2])['passed'])

    def test_gate_requires_failed_tail_and_exclusive_gain(self):
        cases,rows=gate_fixture()
        cases[0]['role']='fast_control'
        self.assertFalse(feedback.evaluate_gate(rows,cases,[1,2])['passed'])
        cases,rows=gate_fixture()
        for row in rows:
            if row['job']['method']=='random_retry':
                row['recovered']=True
        self.assertFalse(feedback.evaluate_gate(rows,cases,[1,2])['passed'])


class CollectionTests(unittest.TestCase):
    def test_stop_finishes_without_launching_or_overwriting_source(self):
        m=dict(content_sha256='f',jobs=[dict(job_id='j')],config=dict(workers=20,minimum_available_memory_gib=3,
               collection_session_hours=2,process_fuse_seconds=60))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'STOP').touch()
            with patch.object(collection,'load',return_value=m),patch.object(collection.subprocess,'Popen') as spawn:
                collection.collect(root)
            self.assertFalse(spawn.called)
            self.assertEqual(collection.read_json(root/'run_status.json')['status'],'paused')
            self.assertFalse((root/'run.lock').exists())

    def test_saved_result_identity_rejected(self):
        m=dict(content_sha256='f'); job=dict(job_id='j')
        row=collection.seal(dict(schema=collection.SCHEMA,fingerprint='wrong',job=job))
        with self.assertRaisesRegex(ValueError,'identity'):
            collection.check_result(m,job,row)

    def test_output_cannot_overlap_frozen_source(self):
        with self.assertRaisesRegex(ValueError,'overlaps'):
            collection.checked_output(dict(output='build/source/subfolder',source_collection='build/source'))


if __name__=='__main__':
    unittest.main()
