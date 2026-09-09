from __future__ import annotations

import copy
import unittest

from experiments import conflict_budget_attribution as audit


def fixture():
    paths={10:[0,1,2],20:[3,2,1,0],30:[8,4,5,6,10],40:[11,10,9,8]}
    state=dict(rows=3,cols=4,obstacles=[False]*12,
        agents=[dict(id=i,path=p,start=p[0],goal=p[-1]) for i,p in paths.items()],
        conflict_edges=[[10,20]],num_of_colliding_pairs=1)
    return state,[10,20,30],{10:paths[10],20:paths[20],30:[8,9,10]}


def sequence(state,order,attempted):
    old=audit.pair_set(audit.events_of(state['agents']))
    selected=set(order)
    allowance=sum(bool(selected&set(e)) for e in old)
    visible={a['id']:a['path'] for a in state['agents'] if a['id'] not in selected}
    pairs,records=set(),[]
    for aid in order:
        path=attempted[aid]
        visible[aid]=path
        events=[e for e in audit.events_of([dict(id=i,path=p) for i,p in visible.items()]) if aid in (e['left'],e['right'])]
        pairs |= audit.pair_set(events)
        records.append(dict(agent=aid,search=dict(status='path',path=path,cost=len(path)-1),
            cumulative_pairs=len(pairs),incident_events=events))
        if len(pairs)>allowance:
            return dict(status='rolled_back',records=records,old_pairs=allowance,attempted_pairs=len(pairs),
                        paths=[a['path'] for a in state['agents']],conflicts=state['num_of_colliding_pairs'])
    paths=[visible[a['id']] for a in state['agents']]
    return dict(status='accepted',records=records,old_pairs=allowance,attempted_pairs=len(pairs),paths=paths,
        conflicts=len(audit.pair_set(audit.events_of([dict(id=a['id'],path=p) for a,p in zip(state['agents'],paths)]))),
        same_paths=paths==[a['path'] for a in state['agents']])


class BudgetAttributionTests(unittest.TestCase):
    def test_earlier_internal_pair_consumes_budget_before_external_overflow(self):
        state,order,paths=fixture()
        value=audit.reconstruct_ledger(state,order,sequence(state,order,paths))
        self.assertEqual(value['allowance'],1)
        self.assertEqual(value['excess'],1)
        self.assertEqual(value['earlier_internal_pairs'],[(10,20)])
        self.assertEqual(value['last_internal_pairs'],[])
        self.assertEqual(value['external_pairs'],[(30,40)])
        self.assertTrue(value['earlier_internal_deletion_could_cover_excess'])
        self.assertEqual([s['budget_after'] for s in value['steps']],[1,0,-1])

    def test_repeated_events_are_one_pair_and_waiting_is_continuous(self):
        events=audit.events_of([dict(id=10,path=[0,1]),dict(id=30,path=[2,1,1,1,1])])
        self.assertEqual(len(events),4)
        self.assertEqual(audit.pair_set(events),{(10,30)})
        swap=audit.events_of([dict(id=10,path=[0,1]),dict(id=30,path=[1,0])])
        self.assertEqual([e['kind'] for e in swap],['edge'])

    def test_external_external_pairs_do_not_raise_neighborhood_allowance(self):
        state,order,paths=fixture()
        path=[7,11,10,9,8,4]
        state['agents'].append(dict(id=50,path=path,start=path[0],goal=path[-1]))
        edges=audit.pair_set(audit.events_of(state['agents']))
        state.update(conflict_edges=sorted(edges),num_of_colliding_pairs=len(edges))
        value=audit.reconstruct_ledger(state,order,sequence(state,order,paths))
        self.assertEqual(value['allowance'],1)
        self.assertGreater(value['initial_global_pairs'],value['allowance'])
        self.assertNotIn((40,50),value['external_pairs'])

    def test_accepted_equal_bound_is_legal_and_no_recovery_claim(self):
        state,order,_=fixture()
        paths={a['id']:a['path'] for a in state['agents']}
        value=audit.reconstruct_ledger(state,order,sequence(state,order,paths))
        self.assertEqual(value['status'],'accepted')
        self.assertEqual(value['excess'],0)
        self.assertFalse(value['earlier_internal_deletion_could_cover_excess'])

    def test_tampered_counts_events_paths_and_order_rejected(self):
        state,order,paths=fixture()
        original=sequence(state,order,paths)
        bad=copy.deepcopy(original); bad['records'][1]['cumulative_pairs']=0
        with self.assertRaisesRegex(ValueError,'unique-pair'):
            audit.reconstruct_ledger(state,order,bad)
        bad=copy.deepcopy(original); bad['records'][-1]['incident_events']=[]
        with self.assertRaisesRegex(ValueError,'incident events'):
            audit.reconstruct_ledger(state,order,bad)
        bad=copy.deepcopy(original); bad['records'][0]['search']['path']=[0,2]
        with self.assertRaisesRegex(ValueError,'geometry'):
            audit.reconstruct_ledger(state,order,bad)
        with self.assertRaisesRegex(ValueError,'order'):
            audit.reconstruct_ledger(state,order[::-1],original)

    def test_two_agent_cover_is_not_double_counted(self):
        value=audit.optimistic_cover([(1,2),(1,3),(2,3),(3,4)])
        self.assertEqual(value,dict(agents=[1,3],covered_pairs=4))
        self.assertEqual(value,audit.optimistic_cover([(3,4),(2,3),(1,3),(1,2)]))
        self.assertEqual(audit.optimistic_cover([]),dict(agents=[],covered_pairs=0))

    def test_common_prefix_comparison_does_not_compare_different_lengths(self):
        a=dict(steps=[dict(internal_pairs=[[1,2]],external_pairs=[]),
                      dict(internal_pairs=[[1,2]],external_pairs=[[1,3]])])
        b=dict(steps=[dict(internal_pairs=[],external_pairs=[[1,3]])])
        value=audit.prefix_difference(a,b)
        self.assertEqual(value['common_processed_agents'],1)
        self.assertEqual(value['internal_pairs']['removed'],[(1,2)])
        self.assertEqual(value['external_pairs']['added'],[(1,3)])
        self.assertTrue(value['not_a_full_outcome_comparison'])

    def test_summary_never_promotes_optimistic_bounds(self):
        state,order,paths=fixture()
        ledger=audit.reconstruct_ledger(state,order,sequence(state,order,paths))
        row=dict(job=dict(case_id='a',seed=1),map_id='m',base=ledger,attempts=[],previous_external_feedback_only=True)
        report=audit.summarize([row])
        self.assertEqual(report['missing_earlier_internal_feedback'],1)
        self.assertEqual(report['earlier_internal_bound_covers_excess'],1)
        self.assertEqual(report['solver_calls'],0)
        self.assertFalse(report['timing_allowed'])


if __name__=='__main__':
    unittest.main()
