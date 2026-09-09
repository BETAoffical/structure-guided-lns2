from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import full_neighborhood_recovery as recovery
from experiments import full_recovery_collection as collection


def tiny_state():
    paths = [[0,1,4], [2,1,0,3], [5,2,1,0]]
    return dict(rows=2,cols=3,obstacles=[False]*6,conflict_edges=[[0,1]],num_of_colliding_pairs=1,
        agents=[dict(id=i,path=p,start=p[0],goal=p[-1]) for i,p in enumerate(paths)])


class FakeProbe:
    def seed_rng(self,seed):
        self.seed=seed

    def plan(self,aid,fixed,overrides,hard,constraints,cap,seconds):
        if seconds<=0:
            return dict(status='unknown',path=[],cost=-1,expanded=0,generated=0,low_level_collisions=-1)
        if hard:
            raise AssertionError('must retain soft CAT')
        if aid==0:
            path=[0,3,4] if constraints else [0,1,4]
            if constraints and cap!=2:
                raise AssertionError('must preserve ordinary predecessor cost cap')
        else:
            path=[2,5,4,3] if 0 in fixed and overrides[0]==[0,1,4] else [2,1,0,3]
        return dict(status='path',path=path,cost=len(path)-1,expanded=2,generated=3,low_level_collisions=0)


CONFIG=dict(job_seconds=10,call_seconds=5,max_attempts=4,max_feedback_blockers=2)


class RecoveryTests(unittest.TestCase):
    def test_ordinary_matches_soft_conflict_bound_and_rolls_back(self):
        state=tiny_state()
        result=recovery.recover(FakeProbe(),state,[0,1],11,'ordinary',CONFIG)
        self.assertEqual(result['base']['status'],'rolled_back')
        self.assertEqual(result['base']['attempted_pairs'],2)
        self.assertEqual(result['final']['paths'],recovery.paths_of(state))

    def test_directed_recovery_changes_only_selected_agents(self):
        state=tiny_state(); before=copy.deepcopy(state)
        for method in ('directed_events','directed_resources'):
            result=recovery.recover(FakeProbe(),state,[0,1],11,method,CONFIG)
            self.assertTrue(result['recovered'])
            self.assertEqual(result['path_check']['conflicts'],0)
            self.assertEqual(result['final']['paths'][2],state['agents'][2]['path'])
            self.assertEqual(result['attempts'][0]['option']['constraint'],['vertex',1,1,1])
        self.assertEqual(state,before)

    def test_random_attempt_limit(self):
        result=recovery.recover(FakeProbe(),tiny_state(),[0,1],11,'random_retry',CONFIG)
        self.assertEqual(len(result['attempts']),4)
        self.assertFalse(result['recovered'])
        self.assertEqual(result['final']['paths'],recovery.paths_of(tiny_state()))

    def test_distinct_resources_do_not_spend_all_slots_on_one_cell(self):
        candidates=[dict(blocker=0,victim=1,constraint=['vertex',t,1,1]) for t in range(1,6)]
        candidates.append(dict(blocker=0,victim=1,constraint=['vertex',6,4,4]))
        events=recovery.choose_feedback(candidates,'directed_events',4)
        resources=recovery.choose_feedback(candidates,'directed_resources',4)
        self.assertEqual(len(events),4)
        self.assertEqual([c['constraint'][3] for c in resources],[1,4])
        self.assertEqual(resources,recovery.choose_feedback(candidates[::-1],'directed_resources',4))

    def test_accepted_equal_but_changed_is_not_a_stall_trigger(self):
        self.assertFalse(recovery.needs_recovery(dict(status='accepted',same_paths=False)))
        self.assertTrue(recovery.needs_recovery(dict(status='accepted',same_paths=True)))
        self.assertFalse(recovery.needs_recovery(dict(status='unknown')))

    def test_zero_budget_does_not_claim_infeasibility(self):
        result=recovery.recover(FakeProbe(),tiny_state(),[0,1],11,'directed_resources',dict(CONFIG,job_seconds=0))
        self.assertEqual(result['status'],'unknown')
        self.assertFalse(result['recovered'])

    def test_reject_invalid_ids_and_prefix(self):
        for order in ([0,0],[99],[]):
            with self.assertRaises(ValueError):
                recovery.validate_order(tiny_state(),order)
        state=tiny_state(); state['agents'][1]['id']=8
        with self.assertRaises(ValueError):
            recovery.validate_order(state,[0,8])
        with self.assertRaisesRegex(ValueError,'prefix'):
            recovery.run_sequence(FakeProbe(),tiny_state(),[0,1],1,recovery.SearchBudget(5,5),prefix=[{'agent':1}])

    def test_reject_external_path_edits(self):
        paths=recovery.paths_of(tiny_state()); paths[2]=[5,4,3,0]
        with self.assertRaisesRegex(ValueError,'external'):
            recovery.check_paths(tiny_state(),paths,[0,1])

    def test_feedback_only_external_does_not_add_agents(self):
        state=tiny_state()
        base=dict(status='rolled_back',records=[dict(agent=0,search={'path':[0,1,4]},
            incident_events=[dict(left=0,right=2,time=1,kind='vertex')])])
        result=recovery.generate_feedback(FakeProbe(),state,[0,1],base,7,recovery.SearchBudget(5,5),2)
        self.assertEqual(result['external_blockers'],[2])
        self.assertEqual(result['candidates'],[])

    def test_gate_requires_stable_recovery_of_a_failed_tail(self):
        cases=[dict(case_id=str(i),map_id=str(i%2),role='failed_long' if i==0 else 'fast_control') for i in range(3)]
        rows=[dict(status='ok',recovered=method=='directed_resources',job=dict(case_id=c['case_id'],seed=seed,method=method))
              for c in cases for seed in (1,2) for method in recovery.METHODS]
        self.assertTrue(recovery.mechanism_gate(rows,cases,len(rows))['passed'])
        cases[0]['role']='fast_control'
        self.assertFalse(recovery.mechanism_gate(rows,cases,len(rows))['passed'])
        rows[-1]=rows[0]
        self.assertFalse(recovery.mechanism_gate(rows,cases,len(rows))['passed'])

    def test_failure_classifier_does_not_conflate_unknown_and_no_path(self):
        self.assertEqual(collection.failure_label({'status':'unknown'}),'unknown')
        row=dict(status='ok',budget_exhausted=False,recovered=False,triggered=True,base={},
                 feedback=None,attempts=[dict(result={'status':'not_found'})])
        self.assertEqual(collection.failure_label(row),'cost_bounded_path_search_failed')

    def test_one_seed_recoveries_and_budget_censoring_cannot_pass(self):
        cases=[dict(case_id=str(i),map_id=str(i%2),role='failed_long') for i in range(3)]
        rows=[dict(status='ok',recovered=method=='directed_resources' and seed==1,
                   job=dict(case_id=c['case_id'],seed=seed,method=method))
              for c in cases for seed in (1,2) for method in recovery.METHODS]
        gate=recovery.mechanism_gate(rows,cases,len(rows))
        self.assertFalse(gate['passed'])
        self.assertEqual(gate['stable_recoveries']['directed_resources'],[])
        rows[0]['budget_exhausted']=True
        self.assertEqual(recovery.mechanism_gate(rows,cases,len(rows))['reason'],
                         'incomplete_error_or_censored_evidence')

    def test_retry_not_found_is_not_source_infeasibility(self):
        class NoAlternative(FakeProbe):
            def plan(self,aid,fixed,overrides,hard,constraints,cap,seconds):
                if constraints:
                    return dict(status='empty',path=[],cost=-1,expanded=2,generated=3,low_level_collisions=-1)
                return super().plan(aid,fixed,overrides,hard,constraints,cap,seconds)
        result=recovery.recover(NoAlternative(),tiny_state(),[0,1],11,'directed_resources',CONFIG)
        self.assertFalse(result['recovered'])
        self.assertEqual(result['status'],'ok')
        self.assertEqual(result['final'],result['base'])
        self.assertTrue(all(a['result']['status']=='not_found' for a in result['attempts']))


class CollectionTests(unittest.TestCase):
    def test_source_sampling_excludes_holdout_maps_and_duplicate_tasks(self):
        anchors=[dict(map_id='development',source_job_id='anchor',task_id='anchor-task')]
        rows=[dict(job_id=str(i),task_id='task'+str(i//2),controller='v2-full',map_id='development',steps=1,
                   success=i%2==0,max_same_state_run=4) for i in range(10)]
        rows.append(dict(rows[0],job_id='holdout',task_id='new-task',map_id='heldout'))
        chosen=collection.choose_sources(rows,anchors,4,71)
        self.assertEqual(chosen,collection.choose_sources(rows[::-1],anchors,4,71))
        self.assertEqual(len({r['task_id'] for r in chosen}),len(chosen))
        self.assertTrue(all(r['map_id']=='development' for r in chosen))

    def test_schedule_has_one_parity_and_four_methods_per_seed(self):
        jobs=collection.make_jobs([{'case_id':'a'}],[1,2])
        self.assertEqual(len(jobs),10)
        self.assertEqual(len({j['job_id'] for j in jobs}),10)

    def test_parity_gate_refuses_unknown_and_changed_identity(self):
        job=dict(job_id='a',method='parity')
        m=dict(content_sha256='f',jobs=[job])
        r=collection.seal(dict(schema=collection.SCHEMA,fingerprint='f',job=job,status='unknown'))
        with patch.object(collection,'read_json',return_value=r):
            with self.assertRaisesRegex(ValueError,'parity'):
                collection.parity_gate(m,Path('unused'))

    def test_stop_file_prevents_launch(self):
        m=dict(content_sha256='f',jobs=[dict(job_id='a',method='parity')],config=dict(workers=20,
               minimum_available_memory_gib=3,collection_session_hours=2))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'STOP').touch()
            with patch.object(collection,'load',return_value=m),patch.object(collection.subprocess,'Popen') as spawn:
                collection.collect(root,'parity')
            self.assertFalse(spawn.called)
            self.assertEqual(collection.read_json(root/'run_status.json')['status'],'paused')

    def test_timing_readiness_remains_closed_without_native_transaction(self):
        m=dict(content_sha256='f',timing_authorized=True,heldout_tasks=[1]*8)
        report=collection.seal(dict(fingerprint='f',gate={'passed':True},timing_blocker='native_transaction_not_integrated'))
        with patch.object(collection,'load',return_value=m),patch.object(collection,'read_json',return_value=report):
            value=collection.timing_readiness(Path('unused'))
        self.assertTrue(value['authorized'])
        self.assertFalse(value['ready'])


if __name__=='__main__':
    unittest.main()
