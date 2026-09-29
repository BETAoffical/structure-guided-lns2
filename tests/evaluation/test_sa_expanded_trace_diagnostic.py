from copy import deepcopy
import unittest

from experiments import sa_expanded_trace_diagnostic as audit
from experiments.closed_loop_trace_storage import encode_state_delta


def state(paths=None):
    paths = paths or {7:[0,1],19:[1,0]}
    return dict(agents=[dict(id=i,path=p) for i,p in paths.items()],
        initialized=True,initial_solution_complete=True,feasible=False,done=False,
        rows=2,cols=2,obstacles=[0]*4,iteration=0,sum_of_costs=sum(len(p)-1 for p in paths.values()),
        num_of_colliding_pairs=1,conflict_edges=[[7,19]],low_level=dict(generated=0))


def example():
    initial=state()
    final=state({7:[0,2,3,1],19:[1,0]})
    final.update(iteration=1,feasible=True,done=True,num_of_colliding_pairs=0,conflict_edges=[],low_level=dict(generated=10))
    event=dict(decision=0,elapsed_seconds=2.,temperature=1000.,
        action=dict(mode='explicit_neighborhood',agents=[7],random_seed=1),
        metrics=dict(action_valid=True,step_applied=True,conflicts_before=1,conflicts_after=0,
            neighborhood=[7],replan_success=True,pp_failure_reason='none',native_replan_seconds=.5,
            acceptance_evaluated=True,pp_old_conflict_pair_count=1,pp_attempt_conflict_pair_count=0),
        delta=encode_state_delta(initial,final),pool=[dict(candidate_id='one',agents=[7])],
        probabilities={'one':1.},selected_id='one',anchor_id='one',selection_draw=.2,out_of_range_fraction=0.)
    result=dict(job_id='j',pair_id='p',map_id='m',comparison_arm='parent',success_within_budget=True,
        reset_seconds=.1,search_end_seconds=2.1,ttf_seconds=2.,initial_conflicts=1,final_conflicts=0,
        decisions=1,native_pp_seconds=.5,generated=10)
    return initial,final,event,result


class ExpandedTraceTests(unittest.TestCase):
    def test_path_identity_ignores_counters_but_not_paths(self):
        s=state(); cache=audit.PathIdentity(); a=cache.update(s)
        changed=deepcopy(s);changed.update(iteration=999,runtime=7.)
        self.assertEqual(a,cache.update(changed))
        changed['agents'][0]['path']=[0,0,1]
        self.assertNotEqual(a,cache.update(changed))

    def test_agent_order_and_noncontinuous_ids(self):
        s=state(); r=deepcopy(s);r['agents'].reverse()
        self.assertEqual(audit.PathIdentity().update(s),audit.PathIdentity().update(r))
        r['agents'].append(deepcopy(r['agents'][0]))
        with self.assertRaisesRegex(ValueError,'duplicate'):
            audit.PathIdentity().update(r)

    def test_single_success_and_total_checks(self):
        i,f,e,r=example(); data=audit.analyze(i,f,[e],r)
        self.assertEqual(data['best_conflicts'],0)
        self.assertEqual(data['first_at_most_conflicts']['0'],2.)
        self.assertEqual(data['all']['generated'],10)
        self.assertEqual(data['all']['unchanged_paths'],0)
        self.assertEqual(data['final_events']['events'],0)
        for k in ('decisions','generated','native_pp_seconds'):
            bad=r|{k:10000}
            with self.assertRaises(ValueError):audit.analyze(i,f,[e],bad)

    def test_unknown_or_changed_members_rejected(self):
        i,f,e,r=example()
        for members in ([19],[7,7],[404]):
            bad=deepcopy(e);bad['action']['agents']=members
            with self.assertRaises(ValueError):audit.analyze(i,f,[bad],r)

    def test_partial_attempt_not_labeled_worsening(self):
        i,f,e,r=example();e['metrics']['acceptance_evaluated']=False
        e['metrics']['pp_attempt_conflict_pair_count']=100
        self.assertEqual(audit.analyze(i,f,[e],r)['all']['worsening_attempts'],0)

    def test_same_set_is_not_path_cycle(self):
        i,f,e,r=example();d=audit.analyze(i,f,[e],r)
        self.assertEqual(d['all']['unique_neighborhoods'],1)
        self.assertEqual(d['unique_path_states'],2)
        self.assertEqual(d['changed_path_revisits'],0)

    def test_invalid_probabilities_and_pool_rejected(self):
        i,f,e,r=example()
        for probabilities in ({'one':.3},{'missing':1.},{'one':float('nan')}):
            bad=deepcopy(e);bad['probabilities']=probabilities
            with self.assertRaises(ValueError):audit.analyze(i,f,[bad],r)

    def test_goal_arrival_not_goal_hold(self):
        arrival=state({7:[0,1],19:[2,1]})
        self.assertEqual(audit.final_events(arrival)['goal_hold_pairs'],[])
        hold=state({7:[0,1],19:[3,2,1]})
        self.assertEqual(audit.final_events(hold)['goal_hold_pairs'],[(7,19)])

    def test_edge_conflict_and_false_report(self):
        s=state();self.assertEqual(audit.final_events(s)['swap_events'],1)
        s['conflict_edges']=[]
        with self.assertRaises(ValueError):audit.final_events(s)

    def test_actor_divergence_and_budget_boundary(self):
        i,f,e,r=example();a=audit.analyze(i,f,[e],r);b=deepcopy(a)
        self.assertEqual(audit.first_divergence(a,b)['kind'],'identical_until_stop')
        b['decision_signatures'][0]['members']=[19]
        self.assertEqual(audit.first_divergence(a,b)['kind'],'selection')
        b=deepcopy(a);b['decision_signatures'][0]['path_after']='changed'
        self.assertEqual(audit.first_divergence(a,b)['kind'],'same_choice_different_repair')

    def test_zero_episode_and_empty_window(self):
        _,f,_,r=example();r.update(decisions=0,generated=0,native_pp_seconds=0.,initial_conflicts=0,
            reset_seconds=.1,search_end_seconds=.1,ttf_seconds=.1)
        d=audit.analyze(f,f,[],r)
        self.assertEqual(d['best_conflicts'],0)
        self.assertIsNone(d['all']['top_neighborhood_fraction'])
        self.assertEqual(audit.longest_run([1,1,2,2,2,1]),3)


if __name__=='__main__':
    unittest.main()
