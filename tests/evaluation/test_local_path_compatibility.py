import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from experiments.local_path_search import (
    Budget, LimitReached, cbs_pair, directed_constraints, graph_of, low_level,
    pair_events, select_pairs, sequential_method, validate_state, validate_witness,
)
from experiments.local_path_compatibility import (
    check_result, contained, digest, dry_run, identity, load_manifest, run_lock, seal,
    source_analysis, diagnose,
)
from experiments._common import write_json, sha256_file
from experiments.state_analysis import reconstruct_conflicts


def state(rows, cols, agents, obstacles=None):
    result = dict(rows=rows, cols=cols, obstacles=obstacles or [0]*(rows*cols), agents=agents)
    result['conflict_edges'] = [list(p) for p in sorted({(e.left,e.right) for e in reconstruct_conflicts(agents)})]
    result['num_of_colliding_pairs'] = len(result['conflict_edges'])
    return result


def agent(i, path):
    return dict(id=i,start=path[0],goal=path[-1],path=path)


class SearchTests(unittest.TestCase):
    def test_free_grid_shortest(self):
        s=state(2,3,[agent(4,[0,1,2])])
        found=low_level(graph_of(s),0,5,[],Budget())
        self.assertEqual(found['cost'],3)

    def test_vertex_edge_and_terminal_wait(self):
        graph=graph_of(state(2,2,[agent(4,[0,1])]))
        # Other agent permanently occupies 1: entering it is never legal.
        self.assertEqual(low_level(graph,0,1,[[3,1]],Budget())['status'],'infeasible')
        result=low_level(graph,0,1,[[1,0,2]],Budget())
        self.assertEqual(result['status'],'feasible')
        self.assertFalse(pair_events(result['path'],[1,0,2]))

    def test_future_goal_visit_requires_leaving(self):
        graph=graph_of(state(2,3,[agent(1,[0,1])]))
        result=low_level(graph,0,1,[[5,4,1,2]],Budget())
        self.assertGreater(result['cost'],2)
        self.assertFalse(pair_events(result['path'],[5,4,1,2]))

    def test_edge_constraint(self):
        graph=graph_of(state(1,3,[agent(1,[0,1,2])]))
        result=low_level(graph,0,2,[],Budget(),[('edge',1,0,1)])
        self.assertEqual(result['path'],[0,0,1,2])

    def test_wait_edge_constraint_at_goal(self):
        graph=graph_of(state(1,2,[agent(1,[0])]))
        result=low_level(graph,0,0,[],Budget(),[('edge',2,0,0)])
        self.assertGreater(result['cost'],1)
        self.assertFalse(result['path'][1]==result['path'][2]==0)

    def test_static_tail_no_artificial_horizon(self):
        graph=graph_of(state(1,20,[agent(1,list(range(20)))]))
        self.assertEqual(low_level(graph,0,19,[],Budget())['cost'],19)

    def test_node_and_time_limits(self):
        graph={0:[0,1],1:[0,1,2],2:[1,2]}
        with self.assertRaises(LimitReached):
            low_level(graph,0,2,[],Budget(max_expanded=1))
        with self.assertRaises(LimitReached):
            low_level(graph,0,2,[],Budget(seconds=0))

    def test_cost_cap_and_constraints(self):
        graph={0:[0,1],1:[0,1,2],2:[1,2]}
        result=low_level(graph,0,2,[],Budget(),[('vertex',1,1,1)],max_cost=2)
        self.assertEqual(result['reason'],'cost_bounded_exhausted')

    def test_same_cost_random_ties_deterministic(self):
        graph=graph_of(state(4,4,[agent(1,[0,1,2,3])]))
        a=low_level(graph,0,15,[],Budget(),tie_seed=19)
        b=low_level(graph,0,15,[],Budget(),tie_seed=19)
        self.assertEqual(a,b)
        self.assertEqual(a['cost'],6)

    def test_constraints_derived_from_paths(self):
        self.assertEqual(directed_constraints([2,1,0],[0,1,2]),[('vertex',1,1,1)])
        self.assertEqual(directed_constraints([0,1],[1,0]),[('edge',1,0,1)])

    def test_noncontiguous_ids_and_input_order(self):
        s=state(2,3,[agent(17,[0,1,2]),agent(93,[2,1,0])])
        self.assertEqual(select_pairs(s),[[17,93]])
        reversed_state=dict(s,agents=list(reversed(s['agents'])))
        self.assertEqual(select_pairs(s),select_pairs(reversed_state))
        a=sequential_method(s,[17,93],'directed_paths',Budget())
        b=sequential_method(reversed_state,[17,93],'directed_paths',Budget())
        self.assertEqual(a,b)

    def test_cbs_small_joint_success(self):
        s=state(2,3,[agent(17,[0,1,2]),agent(93,[2,1,0])])
        result=cbs_pair(s,[17,93],Budget())
        self.assertEqual(result['status'],'feasible')
        self.assertTrue(validate_witness(s,result['paths'])['globally_feasible'])

    def test_cbs_fixed_external_impossibility(self):
        s=state(1,4,[agent(17,[0,1,2]),agent(93,[2,1,0]),agent(52,[1])])
        result=cbs_pair(s,[17,93],Budget())
        self.assertEqual(result['status'],'infeasible')
        self.assertEqual(result['reason'],'single_agent_infeasible_without_partner')

    def test_budget_is_not_joint_infeasibility(self):
        s=state(1,3,[agent(17,[0,1,2]),agent(93,[2,1,0])])
        result=sequential_method(s,[17,93],'directed_paths',Budget(max_expanded=1))
        self.assertEqual(result['status'],'unknown')

    def test_external_external_conflicts_retained(self):
        s=state(2,4,[agent(7,[0,1]),agent(8,[1,0]),agent(31,[6,7]),agent(32,[7,6])])
        found={'7':[0,4,5,1],'8':[1,0]}
        checked=validate_witness(s,found)
        self.assertEqual(checked['external_conflict_pairs'],[[31,32]])
        self.assertFalse(checked['globally_feasible'])

    def test_unknown_invalid_witness(self):
        s=state(2,2,[agent(7,[0,1]),agent(8,[1,0])])
        for paths in ({'99':[0]}, {'7':[0,3,1]}, {'7':[0,1]}):
            with self.assertRaises(ValueError):
                validate_witness(s,paths)

    def test_invalid_state(self):
        s=state(2,2,[agent(7,[0,1]),agent(8,[1,0])])
        for modified in (dict(s,obstacles=[0]),dict(s,agents=[s['agents'][0]]*2),dict(s,conflict_edges=[])):
            with self.assertRaises(ValueError):
                validate_state(modified)

    def test_goal_boundary_not_automatically_infeasible(self):
        s=state(2,2,[agent(7,[0,1]),agent(8,[3,2])])
        result=source_analysis(s,set())
        self.assertEqual(result,[])  # Only actual conflict pairs are diagnosed.

    def test_no_state_mutation(self):
        s=state(2,3,[agent(17,[0,1,2]),agent(93,[2,1,0])])
        frozen=copy.deepcopy(s)
        for method in ('ordinary','random_paths','directed_paths'):
            sequential_method(s,[17,93],method,Budget())
        self.assertEqual(s,frozen)

    def test_random_small_reference_enumeration(self):
        # Compare finite-tail BFS against an explicit, uncompressed time expansion.
        rng=random.Random(20260909)
        graph=graph_of(state(2,3,[agent(1,[0,1])]))
        for _ in range(100):
            fixed=[rng.randrange(6)]
            for tick in range(4):
                fixed.append(rng.choice(graph[fixed[-1]]))
            start,goal=rng.sample(range(6),2)
            constraint=('vertex',rng.randrange(1,5),rng.randrange(6),0)
            constraint=(constraint[0],constraint[1],constraint[2],constraint[2])
            actual=low_level(graph,start,goal,[fixed],Budget(),[constraint])
            reachable={start} if start!=fixed[0] else set()
            expected=None
            last=max([t for t,v in enumerate(fixed) if v==goal]+[constraint[1] if constraint[2]==goal else -1])
            for t in range(25):
                if goal in reachable and goal!=fixed[-1] and t>last:
                    expected=t
                    break
                current=fixed[min(t+1,4)]
                previous=fixed[min(t,4)]
                reachable={v for u in reachable for v in graph[u]
                    if v!=current and not (u==current and v==previous and u!=v)
                    and not (t+1==constraint[1] and v==constraint[2])}
            self.assertEqual(actual.get('cost'),expected)


class StorageTests(unittest.TestCase):
    def test_containment(self):
        for value in ('../escape','C:/outside','a\\b'):
            with self.assertRaises(ValueError):
                contained(value)

    def test_exclusive_lock_cleanup(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)
            with run_lock(out):
                with self.assertRaises(FileExistsError):
                    with run_lock(out):
                        pass
            self.assertFalse((out/'run.lock').exists())

    def test_manifest_hash_and_input_change(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)
            m=dict(schema='lns2.local_path_compatibility.v1',files={},jobs=[])
            m['fingerprint']=identity(m)
            write_json(out/'manifest.json',m)
            self.assertEqual(load_manifest(out),m)
            m['jobs'].append('changed')
            write_json(out/'manifest.json',m)
            with self.assertRaises(ValueError):
                load_manifest(out)

    def test_output_checksum_tamper(self):
        job=dict(job_id='x',case_id='c',pair=[1,2],method='ordinary')
        m=dict(fingerprint='f')
        r=seal(dict(fingerprint='f',job=job,status='not_found'))
        r['status']='feasible'
        with self.assertRaises(ValueError):
            check_result(m,job,r)

    def test_dry_run_no_extra_trials(self):
        m=dict(cases=[1]*6,jobs=[dict(method=v) for v in ('ordinary','random_paths','directed_paths','cbs_pair')]*10,
               config=dict(workers=4,seconds_per_pair_method=180,scope='diagnostic'))
        result=dry_run(m)
        self.assertEqual(result['maximum_jobs'],40)
        self.assertEqual(result['primary_jobs'],30)
        self.assertEqual(result['serial_budget_upper_seconds'],7200)

    def test_json_key_and_order_canonicalization(self):
        self.assertEqual(digest({'a':1,'b':2}),digest({'b':2,'a':1}))
        self.assertEqual(digest(json.loads(json.dumps({'a':[1,2]}))),digest({'a':[1,2]}))
        self.assertEqual(digest({2:'a',10:'b'}),digest({'2':'a','10':'b'}))

    def test_reject_false_infeasibility_and_empty_success(self):
        job=dict(job_id='x',case_id='c',pair=[1,2],method='ordinary')
        for status in ('infeasible','feasible','skipped','unknown_status'):
            with self.assertRaises(ValueError):
                check_result(dict(fingerprint='f'),job,seal(dict(fingerprint='f',job=job,status=status)))

    def test_atomic_replace_has_no_temporary_leftovers(self):
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'result.json'
            write_json(target,{'step':1})
            write_json(target,{'step':2})
            self.assertEqual(json.loads(target.read_text()),{'step':2})
            self.assertEqual(list(Path(folder).iterdir()),[target])

    def test_safe_stop_does_not_start_new_job(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)
            m=dict(fingerprint='f',jobs=[dict(job_id='one',method='ordinary')])
            write_json(out/'historical_native.json',dict(fingerprint='f',passed=True,count=3))
            (out/'STOP').touch()
            with patch('experiments.local_path_compatibility.load_manifest',return_value=m), \
                 patch('experiments.local_path_compatibility.subprocess.Popen') as spawn:
                diagnose(out)
            spawn.assert_not_called()
            self.assertEqual(json.loads((out/'run_status.json').read_text())['status'],'paused')
            self.assertFalse((out/'run.lock').exists())

    def test_resume_preserves_completed_result(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)
            job=dict(job_id='one',method='ordinary')
            m=dict(fingerprint='f',jobs=[job])
            write_json(out/'historical_native.json',dict(fingerprint='f',passed=True,count=3))
            write_json(out/'results/one.json',seal(dict(fingerprint='f',job=job,status='not_found')))
            before=sha256_file(out/'results/one.json')
            with patch('experiments.local_path_compatibility.load_manifest',return_value=m), \
                 patch('experiments.local_path_compatibility.check_result'), \
                 patch('experiments.local_path_compatibility._native_filesystem_path',side_effect=lambda p:p), \
                 patch('experiments.local_path_compatibility.subprocess.Popen') as spawn:
                diagnose(out,resume=True)
            spawn.assert_not_called()
            self.assertEqual(before,sha256_file(out/'results/one.json'))
            self.assertEqual(json.loads((out/'run_status.json').read_text())['status'],'completed')

    def test_external_fuse_is_unknown_and_reaped(self):
        class Child:
            pid=123
            returncode=None
            def poll(self): return self.returncode
            def kill(self): self.returncode=-9
            def wait(self): return self.returncode
        child=Child()
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)
            job=dict(job_id='one',method='ordinary',case_id='test',pair=[4,9])
            m=dict(fingerprint='f',jobs=[job],config=dict(seconds_per_pair_method=0,external_fuse_buffer_seconds=0))
            write_json(out/'historical_native.json',dict(fingerprint='f',passed=True,count=3))
            with patch('experiments.local_path_compatibility.load_manifest',return_value=m), \
                 patch('experiments.local_path_compatibility.check_result'), \
                 patch('experiments.local_path_compatibility._native_filesystem_path',side_effect=lambda p:p), \
                 patch('experiments.local_path_compatibility.subprocess.Popen',return_value=child):
                diagnose(out)
            result=json.loads((out/'results/one.json').read_text())
            self.assertEqual(result['status'],'unknown')
            self.assertEqual(result['reason'],'external_fuse')
            self.assertEqual(child.poll(),-9)


if __name__ == '__main__':
    unittest.main()
