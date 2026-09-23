import copy
import unittest

from scripts import audit_sa_terminal_resources as a


def state():
    return dict(rows=2,cols=3,obstacles=[0]*6,agents=[
        dict(id=4,start=0,goal=2,path=[0,1,2]),
        dict(id=9,start=3,goal=1,path=[3,4,1])],
        conflict_edges=[],num_of_colliding_pairs=0,feasible=True)


class ResourceTests(unittest.TestCase):
    def test_arrival_hold_wait_and_early_goal_visit(self):
        self.assertEqual(a.phase([0,1,1,2,1],1),'move')
        self.assertEqual(a.phase([0,1,1,2,1],2),'wait')
        self.assertEqual(a.phase([0,1,1,2,1],4),'goal_arrival')
        self.assertEqual(a.phase([0,1,1,2,1],5),'goal_hold')
        self.assertEqual(a.phase([0],0),'goal_arrival')
        self.assertEqual(a.phase([0],1),'goal_hold')

    def test_vertex_late_goal_and_noncontinuous_ids(self):
        s=state(); s['agents'][0]['path']=[0,0,1,2]
        s.update(conflict_edges=[[9,4]],num_of_colliding_pairs=1,feasible=False)
        events=a.describe_events(s,s,[4])
        self.assertEqual(len(events),1)
        self.assertEqual(events[0]['time'],2)
        self.assertTrue(events[0]['goal_arrival'])
        self.assertFalse(events[0]['goal_hold'])
        s['agents'][0]['path']=[0,0,0,1,2]
        events=a.describe_events(s,s,[4])
        self.assertTrue(events[0]['goal_hold'])
        self.assertEqual(events[0]['relation'],'boundary')
        self.assertEqual(events[0]['coordinates'],[[0,1]])

    def test_swap_not_misclassified_as_parking(self):
        s=state(); s['agents'][1].update(start=2,goal=0,path=[2,2,1,0])
        s.update(conflict_edges=[[4,9]],num_of_colliding_pairs=1,feasible=False)
        events=a.describe_events(s,s,[9,4])
        self.assertEqual([(e['kind'],e['time']) for e in events],[('edge',2)])
        self.assertFalse(events[0]['goal_hold'])
        self.assertEqual(events[0]['relation'],'internal')
        self.assertEqual([x['repair_order_index'] for x in events[0]['endpoints']],[1,0])

    def test_outsider_change_and_unknown_members_rejected(self):
        s=state(); after=copy.deepcopy(s); after['agents'][1]['path']=[3,3,4,1]
        with self.assertRaises(ValueError): a.describe_events(s,after,[4])
        with self.assertRaises(ValueError): a.describe_events(s,s,[123])
        with self.assertRaises(ValueError): a.describe_events(s,s,[4,4])

    def test_reported_edge_mismatch_rejected(self):
        s=state(); s['conflict_edges']=[[4,9]]
        with self.assertRaises(ValueError): a.checked_events(s)

    def test_goal_neighbors_not_an_infeasibility_certificate(self):
        s=state(); c=a.goal_context(s,4)
        self.assertTrue(c['not_an_infeasibility_certificate'])
        self.assertFalse(c['all_free_neighbors_have_final_owners'])
        self.assertEqual({x['cell'] for x in c['neighbors']},{1,5})
        self.assertEqual(next(x for x in c['neighbors'] if x['cell']==1)['final_owners'],[dict(agent=9,terminal_time=2)])

    def test_signatures_ignore_runtime_not_paths_or_ids(self):
        s=state(); t=copy.deepcopy(s); t['runtime']=100; t['agents'].reverse()
        self.assertEqual(a.task_signature(s,True),a.task_signature(t,True))
        t['agents'][0]['path']=[3,3,4,1]
        self.assertEqual(a.task_signature(s),a.task_signature(t))
        self.assertNotEqual(a.task_signature(s,True),a.task_signature(t,True))

    def test_branch_vs_event_denominators(self):
        s=state(); s['agents'][0]['path']=[0,0,0,1,1,2]
        s.update(conflict_edges=[[4,9]],num_of_colliding_pairs=1,feasible=False)
        ev=a.describe_events(s,s,[4])
        rows=[dict(events=ev,transferred=False,rolled_back=False)]
        result=a.tally(rows)
        self.assertEqual(result['events'],2)
        self.assertEqual(result['branches_with_external_goal_hold'],1)
        self.assertEqual(result['parked_agents_by_branch'],[dict(agent=9,branches=1)])


if __name__ == '__main__': unittest.main()
