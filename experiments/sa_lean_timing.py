"""Deferred scientific auditing for the frozen five-arm first-feasible runtime."""
import gzip
import json
import time

from experiments import sa_raw_timed_runtime as reference
from experiments import sa_shared_feature_timing as previous
from experiments.sa_raw_selection_fast import _bind, FastPolicy
from experiments.sa_shared_features import SharedFeaturePolicy
from scripts import run_sa_onpolicy as run

ARMS = previous.ARMS
LABELS = previous.LABELS
TIMING_MODE = 'deferred_scientific_audit_v1'
Policy = reference.Policy
prepare_environment = reference.prepare_environment
transition = reference.transition
preflight_worker = previous.preflight_worker
pair_worker = previous.pair_worker


class DeferredTrace:
    """Retain immutable native snapshots, sharing unchanged paths until export."""

    def __init__(self, initial):
        self.initial = initial
        self.last = initial
        self.entries = []

    def append(self, event, after):
        previous_agents = {a['id']: a for a in self.last['agents']}
        agents = []
        for agent in after['agents']:
            old = previous_agents.get(agent['id'])
            if old == agent:
                agents.append(old)
            elif old is not None and old['path'] == agent['path']:
                agents.append(dict(agent, path=old['path']))
            else:
                agents.append(agent)
        saved = dict(after, agents=agents)
        for key in ('obstacles', 'context', 'conflict_edges'):
            if key in after and after[key] == self.last.get(key):
                saved[key] = self.last[key]
        self.entries.append((event, saved))
        self.last = saved

    def export(self, path, q):
        # Hashes, state diffs, JSON encoding and compression never run in TTF.
        before = self.initial
        with gzip.open(path, 'xt', encoding='utf8', compresslevel=3) as stream:
            for event, after in self.entries:
                row = dict(event, before=q.state_fingerprint(before),
                           delta=q.encode_state_delta(before, after))
                stream.write(json.dumps(row, separators=(',', ':'), allow_nan=False)+'\n')
                before = after


def record_step(policy, journal, state, after, event, d, metrics, temp, uniform, elapsed):
    run.require(metrics['action_valid'] and metrics['step_applied'], 'invalid or unapplied action')
    requested = event['action'].get('agents')
    if requested is not None:
        run.require(sorted(requested) == sorted(metrics['neighborhood']), 'explicit action altered')
    event.update(decision=d, policy_sha256=policy.sha, metrics=metrics,
                 temperature=temp, uniform=uniform, elapsed_seconds=elapsed)
    # Keep actual policy state updates. Only diagnostic reconstruction is deferred.
    policy.observe(state, event, after)
    journal.append(event, after)


def lean_worker(job):
    from scripts.train_sa_history_selector import die_with_parent
    from scripts.run_feedback_exploration_diagnostics import validate_final
    die_with_parent(job['parent_pid'])
    run.require(job['timing_mode'] == TIMING_MODE, 'unregistered lean clock')
    run.require(job.get('max_decisions') is None and job.get('execution_node_budget') is None,
                'lean timer has no decision or node limit')
    cold = time.monotonic()
    folder = run.ROOT/job['output']/'episodes'/job['job_id']
    run.require(not folder.exists(), 'partial episode; inspect, do not overwrite')
    folder.mkdir(parents=True)
    q, env, ctx = prepare_environment(job)
    policy = Policy(job, q, ctx)
    setup = time.monotonic()-cold
    start = time.monotonic()
    state = q._plain(env.reset(seed=job['solver_seed']))
    reset_seconds = time.monotonic()-start
    run.require(state['initial_solution_complete'], 'initial PP incomplete')
    initial = state
    policy.start(state)
    journal = DeferredTrace(state)
    ttf = reset_seconds if state['feasible'] else None
    d, pp_calls, noops = 0, 0, 0
    selection_seconds = step_wall_seconds = native_pp_seconds = bookkeeping_seconds = 0.
    last_pp_reason = None
    stop = 'feasible' if state['feasible'] else 'deadline'
    while not state['feasible']:
        if time.monotonic()-start >= job['budget_seconds']:
            break
        t = time.monotonic()
        event = policy.choose(env, state, d)
        selection_seconds += time.monotonic()-t
        remaining = job['budget_seconds']-(time.monotonic()-start)
        if remaining <= 0:
            break
        t = time.monotonic()
        after, m, temp, uniform = transition(job, q, env, state, event, d, remaining)
        ended = time.monotonic()
        step_wall_seconds += ended-t
        native_pp_seconds += m['native_replan_seconds']
        last_pp_reason = m['pp_failure_reason']
        if after['feasible']:
            ttf = ended-start
        record_step(policy, journal, state, after, event, d, m, temp, uniform, ended-start)
        bookkeeping_seconds += time.monotonic()-ended
        pp_calls += m['pp_failure_reason'] != 'not_run'
        noops += m['pp_failure_reason'] == 'not_run'
        state = after
        d += 1
        if state['feasible']:
            stop = 'feasible'
        elif m['pp_failure_reason'] == 'time_limit':
            stop = 'pp_deadline'
            break
    search_end = time.monotonic()-start
    # Validate and deliver the final paths once. This is not the all-step audit.
    validate_final(state)
    paths = [a['path'] for a in state['agents']]
    run.require(sum(len(p)-1 for p in paths) == state['sum_of_costs'], 'SOC mismatch')
    initial_fp = q.state_fingerprint(initial)
    run.require(initial_fp == job['expected_initial'], 'initial fingerprint mismatch')
    final_fp = q.state_fingerprint(state)
    run.once(folder/'final.json', state)
    final_sha = run.sha256_file(folder/'final.json')
    delivery = time.monotonic()-start

    # Evidence production is explicitly after both TTF and path delivery.
    post = time.monotonic()
    run.once(folder/'initial.json', initial)
    journal.export(folder/'trace.jsonl.gz', q)
    files = {'final.json': final_sha} | {
        n: run.sha256_file(folder/n) for n in ('initial.json', 'trace.jsonl.gz')}
    post_delivery_trace_seconds = time.monotonic()-post
    row = dict(schema='lns2.sa_raw_ttf.episode.v1', timing_mode=TIMING_MODE,
        binding=job['timing_binding'], status='ok', job_id=job['job_id'],
        pair_id=job['pair_id'], replica=job['replica'], comparison_arm=job['comparison_arm'],
        map_id=job['case']['map_id'], task_id=job['case']['task_id'], solver_seed=job['solver_seed'],
        policy_sha256=policy.sha, initial_fingerprint=initial_fp, final_fingerprint=final_fp,
        rng_stream_id=run.json_fingerprint([job['plan']['config']['stream_seed'],job['phase'],job['pair_id'],job['replica']]),
        budget_seconds=job['budget_seconds'], feasible=state['feasible'],
        success_within_budget=ttf is not None and ttf <= job['budget_seconds'],
        delivered_within_budget=state['feasible'] and delivery <= job['budget_seconds'],
        ttf_seconds=ttf, delivery_seconds=delivery, setup_seconds=setup,
        cold_worker_seconds=time.monotonic()-cold, reset_seconds=reset_seconds,
        selection_seconds=selection_seconds, step_wall_seconds=step_wall_seconds,
        native_pp_seconds=native_pp_seconds, trace_seconds=0.,
        bookkeeping_seconds=bookkeeping_seconds, finalization_seconds=delivery-search_end,
        post_delivery_trace_seconds=post_delivery_trace_seconds,
        search_end_seconds=search_end, last_pp_failure_reason=last_pp_reason,
        stop=stop, decisions=d, pp_calls=pp_calls, legal_noops=noops,
        generated=state['low_level']['generated']-initial['low_level']['generated'],
        final_conflicts=state['num_of_colliding_pairs'], initial_conflicts=initial['num_of_colliding_pairs'],
        soc=state['sum_of_costs'], makespan=max(len(p)-1 for p in paths),
        wait_steps=sum(a==b for p in paths for a,b in zip(p,p[1:])), files=files,
        max_decisions=None, execution_node_budget=None)
    run.once(folder/'result.json', run.sealed(row))
    return dict(status='ok', job_id=job['job_id'], success=row['success_within_budget'],
                ttf=ttf, delivery=delivery, decisions=d)


WORKERS = {'raw_reference': _bind(lean_worker, Policy=FastPolicy),
           'raw_fast': _bind(lean_worker, Policy=SharedFeaturePolicy),
           'dual16_sa': lean_worker, 'official_sa': lean_worker,
           'official': _bind(lean_worker, Policy=previous.OfficialPolicy,
                            prepare_environment=previous.standard_environment,
                            transition=previous.standard_transition)}


def timed_worker(job):
    run.require(job['comparison_arm'] in WORKERS, 'unknown lean arm')
    return WORKERS[job['comparison_arm']](job)


def audit_worker(job):
    row = run.check_seal(run.read_json(run.ROOT/job['output']/'episodes'/job['job_id']/'result.json'))
    run.require(row.get('timing_mode') == job['timing_mode'] == TIMING_MODE, 'mixed timing semantics')
    run.require(row['trace_seconds'] == 0 and row['post_delivery_trace_seconds'] >= 0,
                'trace export was charged to search')
    return previous.audit_worker(job)


def summarize(rows, bootstrap, seed):
    rows = list(rows)
    run.require(all(r.get('timing_mode') == TIMING_MODE for r in rows), 'mixed timing semantics')
    return previous.summarize(rows, bootstrap, seed) | {
        'timing_mode': TIMING_MODE, 'scientific_audit_outside_ttf': True,
        'evidence_export_outside_delivery': True,
        'snapshot_buffer_overhead_included': True}
