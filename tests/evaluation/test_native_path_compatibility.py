from __future__ import annotations

import copy
import importlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import native_path_compatibility as audit

try:
    extension = importlib.import_module('lns2_path_probe_native')
except ImportError:
    extension = None


class ProtocolTests(unittest.TestCase):
    def test_seal_detects_tampering(self):
        value = audit.seal(dict(status='pass', paths=[[0, 1]]))
        audit.check_seal(value)
        value['paths'][0].append(2)
        with self.assertRaisesRegex(ValueError, 'hash'):
            audit.check_seal(value)

    def test_schedule_is_deterministic_and_orders_are_paired(self):
        cases = [dict(case_id='a')]
        state = dict(agents=[dict(id=0, path=[0, 1]), dict(id=1, path=[1, 0])])
        cfg = dict(parity_seeds=[10, 11], path_seed=12)
        jobs = audit.make_jobs(cases, {'a': state}, cfg)
        self.assertEqual(jobs, audit.make_jobs(cases, {'a': state}, cfg))
        self.assertEqual(len(jobs), 8)
        self.assertEqual(len({j['job_id'] for j in jobs}), 8)
        self.assertEqual({tuple(j['order']) for j in jobs if j['kind'] == 'hard_pair'}, {(0, 1), (1, 0)})

    def test_selected_edges_ignore_external_external_collisions(self):
        paths = {0: [0, 0], 1: [2, 1], 2: [1, 2]}
        self.assertEqual(audit.edge_set(paths, [0]), set())
        self.assertEqual(audit.edge_set(paths), {(1, 2)})

    def test_soft_pp_rolls_back_exactly_on_bound_exceedance(self):
        class FakeProbe:
            def seed_rng(self, seed):
                self.seed = seed
            def plan(self, aid, fixed, overrides, hard, **kwargs):
                return dict(status='path', path=[0, 1, 2], cost=2, low_level_collisions=1)
        state = dict(agents=[dict(path=[0, 3, 4, 5, 2]), dict(path=[1, 1])])
        result = audit.soft_pp(FakeProbe(), state, [0], 7, 1)
        self.assertTrue(result['rolled_back'])
        self.assertEqual(result['paths'], audit.paths_of(state))
        self.assertEqual(result['attempted_pairs'], 1)

    def test_soft_pp_accepts_equal_conflicts_not_just_zero(self):
        class FakeProbe:
            def seed_rng(self, seed):
                pass
            def plan(self, aid, fixed, overrides, hard, **kwargs):
                return dict(status='path', path=[0, 1, 2], cost=2, low_level_collisions=1)
        state = dict(agents=[dict(path=[0, 1, 2]), dict(path=[1, 1])])
        result = audit.soft_pp(FakeProbe(), state, [0], 7, 1)
        self.assertFalse(result['rolled_back'])
        self.assertEqual(result['attempted_pairs'], 1)

    def test_parity_gate_refuses_unknown(self):
        m = dict(content_sha256='f', jobs=[dict(job_id='a', kind='parity')])
        result = audit.seal(dict(fingerprint='f', job=m['jobs'][0], status='unknown'))
        with patch.object(audit, 'load', return_value=m), patch.object(audit, 'read_json', return_value=result):
            with self.assertRaisesRegex(ValueError, 'every parity'):
                audit.collect(Path('unused'), 'hard_pair')

    def test_result_identity_checks_job(self):
        result = audit.seal(dict(job={'job_id': 'a'}, fingerprint='f'))
        with self.assertRaisesRegex(ValueError, 'identity'):
            audit.check_result({'content_sha256': 'f'}, {'job_id': 'b'}, result)


@unittest.skipIf(extension is None, 'isolated WSL probe extension required')
class NativeProbeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.map = root/'tiny.map'
        self.scen = root/'tiny.scen'
        self.map.write_text('type octile\nheight 2\nwidth 3\nmap\n...\n...\n')
        self.scen.write_text('version 1\n0\ttiny.map\t3\t2\t0\t0\t2\t0\t2\n'
                             '0\ttiny.map\t3\t2\t1\t0\t1\t0\t0\n')
        self.paths = [[0, 1, 2], [1, 1, 1]]
        self.probe = extension.NativePathProbe(str(self.map), str(self.scen), self.paths)

    def test_terminal_wait_has_no_hard_occupancy_gap(self):
        result = self.probe.plan(0, [1], {}, True, max_cost=2)
        self.assertEqual(result['status'], 'empty')
        result = self.probe.plan(0, [1], {}, True)
        self.assertEqual(result['status'], 'path')
        self.assertEqual(result['cost'], 4)
        self.assertNotIn(1, result['path'])

    def test_vertex_edge_and_cost_constraints(self):
        for constraint in [('vertex', 1, 1, 1), ('edge', 1, 0, 1)]:
            result = self.probe.plan(0, [], {}, True, [constraint], max_cost=2)
            self.assertEqual(result['status'], 'empty')
        self.assertEqual(self.probe.plan(0, [], {}, True, max_cost=2)['cost'], 2)

    def test_zero_budget_is_unknown_not_infeasible(self):
        result = self.probe.plan(0, [], {}, True, seconds=0)
        self.assertEqual(result['status'], 'unknown')
        self.assertEqual(result['expanded'], 0)

    def test_seed_determinism_and_source_not_modified(self):
        results = []
        before = copy.deepcopy(self.paths)
        for _ in range(2):
            self.probe.seed_rng(71)
            results.append(self.probe.plan(0, [1], {}, False))
        for key in ('path', 'cost', 'expanded', 'generated', 'low_level_collisions'):
            self.assertEqual(results[0][key], results[1][key])
        self.assertEqual(self.paths, before)

    def test_reject_invalid_ids_constraints_paths_and_budgets(self):
        for args in [(2, [], {}, True), (0, [0], {}, True), (0, [1, 1], {}, True),
                     (0, [7], {}, True), (0, [], {1: [1, 5, 1]}, True),
                     (0, [], {}, True, [('edge', 0, 0, 1)]),
                     (0, [], {}, True, [('vertex', 1, 9, 9)])]:
            with self.assertRaises(ValueError):
                self.probe.plan(*args)
        for budget in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                self.probe.plan(0, [], {}, True, seconds=budget)

    def test_hard_swap_is_avoided(self):
        self.scen.write_text('version 1\n0\ttiny.map\t3\t2\t0\t0\t2\t0\t2\n'
                             '0\ttiny.map\t3\t2\t2\t0\t0\t0\t2\n')
        probe = extension.NativePathProbe(str(self.map), str(self.scen), [[0, 1, 2], [2, 1, 0]])
        result = probe.plan(0, [1], {}, True)
        self.assertEqual(result['status'], 'path')
        self.assertEqual(audit.edge_set({0: result['path'], 1: [2, 1, 0]}), set())


if __name__ == '__main__':
    unittest.main()
