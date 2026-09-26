from copy import deepcopy
from collections import Counter
import gzip
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments import sa_lean_timing as rt
from experiments.sa_raw_selection_fast import _bind
from scripts import run_sa_lean_ttf as cli
from tests.evaluation.test_sa_raw_fast_timing import old_jobs
from tests.evaluation.test_sa_shared_feature_timing import rows


def state(n=0, feasible=False):
    return dict(agents=[dict(id=7, path=[0, 1], path_cost=1)], obstacles=[0, 0],
                low_level=dict(generated=n), num_of_colliding_pairs=0 if feasible else 1,
                sum_of_costs=1, feasible=feasible, initial_solution_complete=True,
                conflict_edges=[], iteration=n)


class LeanTimingTests(unittest.TestCase):
    def test_snapshot_sharing_without_changing_states(self):
        initial = state()
        initial['agents'].append(dict(id=99, path=[3, 4], path_cost=1))
        after = deepcopy(initial)
        after['agents'][0]['path'] = [0, 0, 1]
        after['agents'][0]['path_cost'] = 2
        expected = deepcopy(after)
        journal = rt.DeferredTrace(initial)
        journal.append({'decision': 0}, after)
        saved = journal.last
        self.assertEqual(after, expected)
        self.assertEqual(saved, after)
        self.assertIs(saved['agents'][1], initial['agents'][1])
        self.assertIs(saved['obstacles'], initial['obstacles'])
        self.assertIsNot(saved['agents'][0], initial['agents'][0])
        self.assertEqual(initial['agents'][0]['path'], [0, 1])

    def test_recording_does_not_call_scientific_audit_or_fingerprint(self):
        observed = []
        policy = SimpleNamespace(sha='frozen', observe=lambda *args: observed.append(args))
        before, after = state(), state(1)
        event = dict(action=dict(mode='explicit_neighborhood', agents=[7]))
        metrics = dict(action_valid=True, step_applied=True, neighborhood=[7])
        journal = rt.DeferredTrace(before)
        rt.record_step(policy, journal, before, after, event, 0, metrics, 1., .2, .3)
        self.assertEqual(len(observed), 1)
        self.assertEqual(journal.last, after)
        self.assertNotIn('delta', event)
        self.assertNotIn('before', event)
        with self.assertRaisesRegex(ValueError, 'altered'):
            rt.record_step(policy, journal, before, after, event, 1, metrics | {'neighborhood': [99]}, 1., .2, .3)

    def exercise_worker(self, *, export_delay=0., final_delay=0., steps=1, zero=False, pp_timeout=False):
        clock = SimpleNamespace(value=0.)
        seen = dict(choose=0, observations=0, exports=0, validations=0)
        def advance(value):
            clock.value += value
        def reset(seed):
            advance(.1)
            return state(feasible=zero)
        class FakePolicy:
            sha = 'official'
            def __init__(self, *args): pass
            def start(self, state): pass
            def choose(self, env, state, d):
                seen['choose'] += 1
                advance(.05)
                return {'action': {'mode': 'official'}}
            def observe(self, *args):
                seen['observations'] += 1
                advance(.01)
        def transition(job, q, env, before, event, d, seconds):
            advance(seconds if pp_timeout else .2)
            return state(d+1, not pp_timeout and d+1 >= steps), dict(
                action_valid=True, step_applied=True, neighborhood=[7],
                native_replan_seconds=.2, pp_failure_reason='time_limit' if pp_timeout else 'none'), 1., .2
        def validate_final(after):
            seen['validations'] += 1
            self.assertTrue(pp_timeout or after['feasible'])
            advance(final_delay)
        def forbidden(*args):
            self.fail('scientific validation entered the timed loop')
        q = SimpleNamespace(_plain=lambda x:x, validate_transition=forbidden,
                            state_fingerprint=lambda s:'fp'+str(s['iteration']),
                            encode_state_delta=lambda before, after: {'after':after})
        original_export = rt.DeferredTrace.export
        def export(journal, path, runtime):
            self.assertEqual(seen['validations'], 1)
            seen['exports'] += 1
            advance(export_delay)
            original_export(journal, path, runtime)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            job = dict(parent_pid=0, timing_mode=rt.TIMING_MODE, output='result', job_id='job',
                       timing_binding='frozen', pair_id='pair', replica=0, comparison_arm='official',
                       case=dict(map_id='map', task_id='task'), solver_seed=0, expected_initial='fp0',
                       plan=dict(config=dict(stream_seed=1)), phase='test', budget_seconds=120.)
            worker = _bind(rt.lean_worker, Policy=FakePolicy,
                prepare_environment=lambda j:(q, SimpleNamespace(reset=reset), {}),
                transition=transition, time=SimpleNamespace(monotonic=lambda:clock.value),
                run=SimpleNamespace(**(vars(rt.run) | {'ROOT':root})))
            with patch('scripts.train_sa_history_selector.die_with_parent'), \
                 patch('scripts.run_feedback_exploration_diagnostics.validate_final', validate_final), \
                 patch.object(rt.DeferredTrace, 'export', export):
                worker(job)
                with self.assertRaisesRegex(ValueError, 'partial episode'):
                    worker(job)
            folder = root/'result/episodes/job'
            row = json.loads((folder/'result.json').read_text())
            with gzip.open(folder/'trace.jsonl.gz', 'rt') as f:
                trace = [json.loads(line) for line in f]
            self.assertEqual(len(trace), row['decisions'])
            self.assertEqual(row['files']['trace.jsonl.gz'], rt.run.sha256_file(folder/'trace.jsonl.gz'))
            return row, seen

    def test_export_and_audit_time_not_in_search_or_delivery(self):
        base, _ = self.exercise_worker()
        costly, seen = self.exercise_worker(export_delay=100.)
        self.assertEqual(base['ttf_seconds'], costly['ttf_seconds'])
        self.assertEqual(base['delivery_seconds'], costly['delivery_seconds'])
        self.assertEqual(costly['post_delivery_trace_seconds'], 100.)
        self.assertEqual(costly['trace_seconds'], 0.)
        self.assertEqual(seen, dict(choose=1, observations=1, exports=1, validations=1))

    def test_final_path_validation_charged_to_delivery_not_ttf(self):
        base, _ = self.exercise_worker()
        costly, _ = self.exercise_worker(final_delay=3.)
        self.assertEqual(base['ttf_seconds'], costly['ttf_seconds'])
        self.assertAlmostEqual(costly['delivery_seconds']-base['delivery_seconds'], 3.)

    def test_no_decision_limit_and_terminal_cases(self):
        row, _ = self.exercise_worker(steps=105)
        self.assertEqual(row['decisions'], 105)
        self.assertTrue(row['success_within_budget'])
        row, seen = self.exercise_worker(zero=True)
        self.assertEqual(row['ttf_seconds'], .1)
        self.assertEqual(seen['choose'], 0)
        row, _ = self.exercise_worker(pp_timeout=True)
        self.assertIsNone(row['ttf_seconds'])
        self.assertEqual(row['stop'], 'pp_deadline')
        self.assertFalse(row['success_within_budget'])

    def test_new_identity_old_models_and_streams_unchanged(self):
        original = cli.old.selected_jobs(old_jobs(), cli.old.config())
        saved = deepcopy(original)
        jobs = cli.selected_jobs(original, cli.config())
        self.assertEqual(original, saved)
        self.assertEqual(len(jobs), 192)
        self.assertEqual(set(j['comparison_arm'] for j in jobs), set(rt.ARMS))
        self.assertNotIn('raw_reference', rt.WORKERS)
        self.assertTrue({j['job_id'] for j in original}.isdisjoint(j['job_id'] for j in jobs))
        source = {j['job_id']: j for j in original}
        for after in jobs:
            before = source[after['source_shared_job_id']]
            for key in ('model', 'phase', 'pair_id', 'replica', 'solver_seed', 'case', 'comparison_arm'):
                self.assertEqual(before[key], after[key])
            self.assertEqual(after['timing_mode'], rt.TIMING_MODE)
        self.assertEqual(cli.phase_jobs(dict(jobs=jobs, binding='b'), 'pair'), [])
        with patch.object(rt, 'pair_worker', side_effect=AssertionError('unnecessary engineering recheck')):
            self.assertTrue(cli.phase('pair')['skipped'])
        self.assertIs(cli.collect.__code__, cli.old.collect.__code__)
        self.assertIs(cli.collect.__globals__['rt'], rt)
        self.assertIs(rt.WORKERS['official'].__globals__['transition'], rt.previous.standard_transition)
        self.assertIs(rt.WORKERS['official_sa'].__globals__['transition'], rt.reference.transition)
        self.assertIs(rt.previous.WORKERS['official_sa'], rt.reference.timed_worker)

    def test_four_method_schedule_balances_positions_and_predecessors(self):
        original = cli.old.selected_jobs(old_jobs(), cli.old.config())
        jobs = cli.selected_jobs(original, cli.config())
        self.assertEqual(jobs, cli.selected_jobs(reversed(original), cli.config()))
        self.assertEqual([j['schedule_index'] for j in jobs], list(range(192)))
        positions, edges, coverage = Counter(), Counter(), Counter()
        for i in range(0, len(jobs), 4):
            group = jobs[i:i+4]
            self.assertEqual(len({(j['pair_id'], j['replica']) for j in group}), 1)
            self.assertEqual({j['comparison_arm'] for j in group}, set(rt.ARMS))
            names = [j['comparison_arm'] for j in group]
            positions.update((a, position) for position, a in enumerate(names))
            edges.update(zip(names, names[1:]))
            coverage.update((j['case']['map_id'], j['comparison_arm']) for j in group)
        self.assertEqual(len(positions), 16)
        self.assertEqual(set(positions.values()), {12})
        self.assertEqual(len(edges), 12)
        self.assertEqual(set(edges.values()), {12})
        self.assertEqual(len(coverage), 24)
        self.assertEqual(set(coverage.values()), {8})
        with self.assertRaisesRegex(ValueError, 'missing source arm'):
            cli.selected_jobs(original[:-1], cli.config())
        with self.assertRaisesRegex(ValueError, 'duplicate source arm'):
            cli.selected_jobs(original+[original[0]], cli.config())

    def test_summary_rejects_old_clock(self):
        with self.assertRaisesRegex(ValueError, 'mixed timing'):
            rt.summarize(rows(), 20, 1)
        values = [r | {'timing_mode': rt.TIMING_MODE} for r in rows() if r['comparison_arm'] in rt.ARMS]
        result = rt.summarize(values, 20, 1)
        self.assertTrue(result['scientific_audit_outside_ttf'])
        self.assertTrue(result['engineering_comparison_reused_not_rerun'])
        self.assertNotIn('same_raw_model', result)
        self.assertNotIn('raw_reference', result['contrasts'])
        self.assertEqual(set(result['arms']), set(rt.ARMS))
        prior = rt.previous.summarize(rows(), 20, 1)
        for arm in ('official', 'official_sa', 'dual16_sa'):
            self.assertEqual(result['contrasts'][arm], prior['contrasts'][arm])
        self.assertEqual(result['official_sa_vs_official'], prior['official_sa_vs_official'])
        with self.assertRaisesRegex(ValueError, 'missing arm'):
            rt.summarize(values[:-1], 20, 1)
        with self.assertRaisesRegex(ValueError, 'unknown comparison_arm'):
            rt.summarize([r | {'timing_mode': rt.TIMING_MODE} for r in rows()], 20, 1)


@unittest.skipUnless(os.environ.get('LNS2_LEAN_NATIVE_SMOKE') == '1', 'opt-in frozen native prefix check')
class NativeLeanTimingTests(unittest.TestCase):
    def test_active_arms_same_actions_paths_and_deferred_trace(self):
        registration = rt.run.read_json(rt.run.ROOT/'build/sa-shared-feature-ttf-v1/registration.json')
        source = {arm:dict(next(j for j in registration['jobs'] if j['comparison_arm']==arm),
                           timing_binding=registration['binding']) for arm in rt.ARMS}
        signatures = {}
        for arm, job in source.items():
            outputs = []
            for lean in (False, True):
                g = rt.WORKERS[arm].__globals__
                q, env, ctx = g['prepare_environment'](job)
                policy = g['Policy'](job, q, ctx)
                current = q._plain(env.reset(seed=job['solver_seed']))
                self.assertEqual(q.state_fingerprint(current), job['expected_initial'])
                policy.start(current)
                journal = rt.DeferredTrace(current)
                result = []
                for d in range(3):
                    self.assertFalse(current['feasible'])
                    event = policy.choose(env, current, d)
                    after, m, temp, uniform = g['transition'](job, q, env, current, event, d, 20.)
                    self.assertNotEqual(m['pp_failure_reason'], 'time_limit')
                    if lean:
                        rt.record_step(policy, journal, current, after, event, d, m, temp, uniform, 0.)
                        self.assertEqual(journal.last, after)
                    else:
                        rt.reference.finish_event(q, current, after, event, d, m, temp, uniform, policy.sha)
                        policy.observe(current, event, after)
                    result.append(deepcopy(dict(after=q.state_fingerprint(after),
                        selection={k:event[k] for k in ('action', 'pool', 'proposal_order', 'anchor_id',
                            'selected_id', 'features', 'probabilities', 'candidate_ids', 'selection_draw') if k in event},
                        metrics={k:m[k] for k in ('neighborhood', 'repair_order', 'replan_success', 'pp_failure_reason')})))
                    current = after
                if lean:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        path = Path(temp_dir)/'trace.jsonl.gz'
                        journal.export(path, q)
                        replay = journal.initial
                        with gzip.open(path, 'rt') as f:
                            for line, (_, expected) in zip(f, journal.entries, strict=True):
                                event = json.loads(line)
                                for key, value in result[event['decision']]['selection'].items():
                                    self.assertEqual(event[key], value)
                                self.assertEqual(event['before'], q.state_fingerprint(replay))
                                replay = q.apply_state_delta(replay, event['delta'])
                                self.assertEqual(q.state_fingerprint(replay), q.state_fingerprint(expected))
                                q.validate_transition(journal.initial if event['decision']==0 else prior,
                                    expected, event['metrics'], event['metrics']['neighborhood'],
                                    'annealed', event['temperature'], event['uniform'])
                                prior = expected
                outputs.append(result)
            self.assertEqual(*outputs, arm)
            signatures[arm] = rt.run.json_fingerprint(outputs[0])
        rt.run.write_json(rt.run.ROOT/'build/sa-lean-timing-verification-v1/native-prefix-four-arm.json',
                          dict(passed=True, native_repairs=6*len(rt.ARMS), arms=signatures, no_ttf_claim=True))


if __name__ == '__main__':
    unittest.main()
