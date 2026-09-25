from copy import deepcopy
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.sa_history_selector import History
from experiments.sa_raw_selection_fast import dynamic_features_for, FastPolicy, _selection
from experiments import sa_raw_timed_runtime as rt
from scripts import run_sa_onpolicy as run
from scripts.audit_sa_raw_selection_cost import indices, SavedPool, equal_event


class Engine:
    def prepare(self,state):self.state=state
    def realized_rows(self,candidates,state_hash):
        return [dict(features={'realized_dynamic':{'x':1.}}) for _ in candidates],{}


class SelectionFastTests(unittest.TestCase):
    def fixture(self):
        state=dict(agents=[dict(id=4,path=[0,1,2]),dict(id=9,path=[2,1,0]),dict(id=21,path=[3,4,5])],
                   conflict_edges=[[4,9]],num_of_colliding_pairs=1)
        pool=[dict(candidate_id='a',agents=[9,4],score=.7),dict(candidate_id='b',agents=[4,21],score=.3)]
        return state,pool,History(state)

    def test_exact_dynamic_rows_before_and_after_repeated_rejected_actions(self):
        state,pool,h=self.fixture()
        original=deepcopy(state)
        for d in range(40):
            for temp in (0.,.25,10.):
                self.assertEqual(run.features_for(state,pool,Engine(),h,temp,'fp'),
                                 dynamic_features_for(state,pool,Engine(),h,temp,'fp'))
            event=dict(decision=d,metrics=dict(neighborhood=[4,9],conflicts_before=1,conflicts_after=1,
                       pp_rolled_back=True,acceptance_evaluated=True))
            h.observe(state,event,state)
        self.assertEqual(state,original)
        self.assertEqual(h.decision,40)

    def test_filtered_history_is_not_computed_and_phase_is_preserved(self):
        state,pool,h=self.fixture()
        h.decision=600
        with patch.object(h,'features',side_effect=AssertionError('unused history computation')):
            rows=dynamic_features_for(state,pool,Engine(),h,.5,'fp')
        self.assertEqual(rows[0]['sa.log_decision'],math.log1p(600))
        self.assertEqual(rows[0]['sa.log_temperature'],math.log1p(.5))
        self.assertFalse(any(k.startswith('history.') for k in rows[0]))

    def test_candidate_and_history_guards(self):
        state,pool,h=self.fixture()
        for ids in ([],[4,4],[4,99]):
            with self.assertRaises(ValueError):dynamic_features_for(state,[dict(agents=ids,score=0.)],Engine(),h,1.,'fp')
        h.ages={}
        with self.assertRaises(ValueError):dynamic_features_for(state,pool,Engine(),h,1.,'fp')

    def test_private_bindings_leave_frozen_globals_unchanged(self):
        self.assertIs(FastPolicy.choose.__code__,rt.Policy.choose.__code__)
        self.assertIs(_selection.__code__,run.selection.__code__)
        self.assertIs(rt.Policy.choose.__globals__['run'],run)
        self.assertIsNot(run.selection.__globals__['features_for'],dynamic_features_for)
        self.assertIs(_selection.__globals__['features_for'],dynamic_features_for)

    def test_history_model_rejected_and_original_constructor_preserved(self):
        def constructor(self,*args):
            self.actor=SimpleNamespace(bundle={'base':{'feature_names':['history.foo']+['x']*128}})
        with patch.object(rt.Policy,'__init__',constructor):
            with self.assertRaises(ValueError):FastPolicy({},None,{})

    def test_stage_sampling_and_saved_proposal_order(self):
        self.assertEqual(indices(1),[0])
        self.assertEqual(indices(100),[0,49,99])
        with self.assertRaises(ValueError):indices(0)
        e=dict(decision=8,pool=[dict(candidate_id='a'),dict(candidate_id='b')],proposal_order=['b','a'],anchor_id='a')
        index,pool=SavedPool(e).select(None,{},8)
        self.assertEqual(index,1)
        self.assertEqual([c['candidate_id'] for c in pool],['b','a'])
        with self.assertRaises(ValueError):SavedPool(e).select(None,{},9)
        equal_event({'anchor_id':'a'},e)
        with self.assertRaises(ValueError):equal_event({'anchor_id':'b'},e)


if __name__=='__main__':unittest.main()
