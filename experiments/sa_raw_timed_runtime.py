"""Opt-in serial wall-clock lane; frozen policy inputs, no work/step cap."""
import gzip
import json
import math
import time

from scripts import run_sa_onpolicy as run
from experiments.sa_raw_residual_runtime import RuntimeActor
from experiments.sa_uncapped_runtime import feature_plan


def prepare_environment(job):
    q = run.native_runtime(job['plan'])
    item = dict(task_id=job['case']['task_id'], solver_seed=job['solver_seed'],
                controller='dual16_sa', budget_seconds=100000.)
    mapped = q.worker_job(job['case'], item, job['plan']['template'],
                          run.ROOT/job['output'], job['timing_binding'])
    run.require(mapped['environment']['max_repair_iterations'] == 0, 'hidden native step limit')
    env = q._make_environment(mapped['dataset_root'], mapped['row'], mapped['environment'], 'Adaptive')
    ctx = dict(case_id=mapped['sa_case_id'], task_id=job['case']['task_id'],
               solver_seed=job['solver_seed'], proposal=mapped['sa_proposal'])
    return q, env, ctx


class Policy:
    def __init__(self, job, q, ctx):
        from scripts.run_sa_raw_residual import load_model
        self.job, self.q, self.ctx = job, q, ctx
        self.actor = RuntimeActor(load_model(job)) if job['model'] else None
        self.sha = self.actor.sha if self.actor else job['arm']
        self.selector = q.SingleFullCheckPool(ctx) if job['arm'] != 'official_sa' else None
        self.history, self.engine = None, None
        self.plan = feature_plan(job['plan'])

    def start(self, state):
        if self.actor:
            from experiments.sa_history_selector import History
            from experiments.online_feature_engine import OnlineFeatureEngine
            self.history = History(state)
            self.engine = OnlineFeatureEngine(state, backend='native')
            self.ctx = dict(self.ctx, initial_nodes=state['low_level']['generated'])

    def choose(self, env, state, d):
        j = self.job
        if self.actor:
            e = run.selection(env, state, self.ctx, self.history, self.engine,
                              self.selector, self.plan, self.actor)
        elif self.selector:
            # Baselines must not pay for the new actor's unused 129-D features.
            index, pool = self.selector.select(env, state, d)
            anchor = pool[index]['candidate_id']
            e = dict(pool=sorted(pool, key=lambda c:c['candidate_id']),
                     proposal_order=[c['candidate_id'] for c in pool], anchor_id=anchor)
        else:
            return dict(action=dict(mode='official'))
        draw = run.stream_draw(j['plan'], j['phase'], j['pair_id'], j['replica'], d, 'select')
        cid = run.select_with_draw(e['probabilities'], draw) if self.actor else e['anchor_id']
        chosen = next(c for c in e['pool'] if c['candidate_id'] == cid)
        e.update(selected_id=cid, selection_draw=draw, action=dict(mode='explicit_neighborhood',
                 agents=chosen['agents'], random_seed=run.stream_draw(
                     j['plan'], j['phase'], j['pair_id'], j['replica'], d, 'pp')))
        if self.actor:
            e['behavior_log_probability'] = math.log(e['probabilities'][cid])
        return e

    def observe(self, before, event, after):
        if self.history:
            self.history.observe(before, event, after)


def transition(job, q, env, state, event, d, seconds):
    temp = q.temperature(d)
    uniform = run.stream_draw(job['plan'], job['phase'], job['pair_id'], job['replica'], d, 'accept')
    raw = q._plain(env.step_experimental_pp(event['action'], seconds, 'annealed', temp, uniform))
    after, m = raw['observation'], raw['metrics']
    return after, m, temp, uniform


def finish_event(q, state, after, event, d, metrics, temp, uniform, sha):
    selected = event['action'].get('agents', metrics['neighborhood'])
    q.validate_transition(state, after, metrics, selected, 'annealed', temp, uniform)
    event.update(decision=d, policy_sha256=sha, before=q.state_fingerprint(state),
                 metrics=metrics, temperature=temp, uniform=uniform, delta=q.encode_state_delta(state, after))


def preflight_worker(job):
    from scripts.train_sa_history_selector import die_with_parent
    die_with_parent(job['parent_pid'])
    for name,sha in job['source_files'].items():
        run.require(run.sha256_file(run.ROOT/job['source_folder']/name)==sha,'source prefix artifact changed')
    q, env, ctx = prepare_environment(job)
    policy = Policy(job, q, ctx)
    state = q._plain(env.reset(seed=job['solver_seed']))
    run.require(q.state_fingerprint(state) == job['expected_initial'], 'preflight initial')
    policy.start(state)
    count = 0
    for old in run.trace_read(run.ROOT/job['source_folder']):
        d = count
        e = policy.choose(env, state, d)
        for key in ('action', 'pool', 'proposal_order', 'anchor_id', 'selected_id', 'selection_draw'):
            if key in old:
                run.require(e[key] == old[key], 'timed selection differs: '+key)
        if policy.actor:
            for key in ('features', 'probabilities', 'candidate_ids', 'behavior_log_probability'):
                run.require(e[key] == old[key], 'timed actor differs: '+key)
        after, m, temp, uniform = transition(job, q, env, state, e, d, 20.)
        finish_event(q, state, after, e, d, m, temp, uniform, policy.sha)
        expected = q.apply_state_delta(state, old['delta'])
        run.require(q.state_fingerprint(after) == q.state_fingerprint(expected), 'timed PP prefix differs')
        for key in ('repair_order', 'neighborhood', 'pp_failure_reason', 'replan_success'):
            run.require(m[key] == old['metrics'][key], 'timed PP differs: '+key)
        run.require(temp == old['temperature'] and uniform == old['uniform'], 'SA differs')
        policy.observe(state, e, after)
        state = after
        count += 1
        if count == job['preflight_steps']:
            break
    run.require(count == min(job['preflight_steps'],job['source_decisions']), 'shortened prefix proof')
    return dict(status='ok', job_id=job['job_id'], prefix_steps=count, action_feature_path_equal=True)


def timed_worker(job):
    from scripts.train_sa_history_selector import die_with_parent
    from scripts.run_feedback_exploration_diagnostics import validate_final
    die_with_parent(job['parent_pid'])
    cold = time.monotonic()
    folder = run.ROOT/job['output']/'episodes'/job['job_id']
    run.require(not folder.exists(), 'partial episode; no automatic retry')
    folder.mkdir(parents=True)
    q, env, ctx = prepare_environment(job)
    policy = Policy(job, q, ctx)
    setup = time.monotonic()-cold
    start = time.monotonic()
    state = q._plain(env.reset(seed=job['solver_seed']))
    reset_seconds = time.monotonic()-start
    run.require(state['initial_solution_complete'], 'initial PP incomplete')
    initial = state
    initial_fp = q.state_fingerprint(state)
    run.require(initial_fp == job['expected_initial'], 'timed initial mismatch')
    policy.start(state)
    run.once(folder/'initial.json', state)
    ttf = reset_seconds if state['feasible'] else None
    d, pp_calls, noops = 0, 0, 0
    selection_seconds, step_wall_seconds, native_pp_seconds = 0., 0., 0.
    trace_seconds, bookkeeping_seconds, last_pp_reason = 0., 0., None
    stop = 'feasible' if state['feasible'] else 'deadline'
    with gzip.open(folder/'trace.jsonl.gz', 'xt', encoding='utf8', compresslevel=3) as stream:
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
            finish_event(q, state, after, event, d, m, temp, uniform, policy.sha)
            event['elapsed_seconds'] = ended-start
            policy.observe(state, event, after)
            bookkeeping_seconds += time.monotonic()-ended
            t = time.monotonic()
            stream.write(json.dumps(event, separators=(',', ':'), allow_nan=False)+'\n')
            stream.flush()
            trace_seconds += time.monotonic()-t
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
    validate_final(state)
    paths = [a['path'] for a in state['agents']]
    run.require(sum(len(p)-1 for p in paths) == state['sum_of_costs'], 'SOC mismatch')
    run.once(folder/'final.json', state)
    files = {n:run.sha256_file(folder/n) for n in ('initial.json','final.json','trace.jsonl.gz')}
    delivery = time.monotonic()-start
    row = dict(schema='lns2.sa_raw_ttf.episode.v1', binding=job['timing_binding'], status='ok',
        job_id=job['job_id'], pair_id=job['pair_id'], replica=job['replica'],
        comparison_arm=job['comparison_arm'], map_id=job['case']['map_id'],
        task_id=job['case']['task_id'], solver_seed=job['solver_seed'],
        policy_sha256=policy.sha, initial_fingerprint=initial_fp, final_fingerprint=q.state_fingerprint(state),
        rng_stream_id=run.json_fingerprint([job['plan']['config']['stream_seed'],job['phase'],job['pair_id'],job['replica']]),
        budget_seconds=job['budget_seconds'], feasible=state['feasible'],
        success_within_budget=ttf is not None and ttf <= job['budget_seconds'],
        delivered_within_budget=state['feasible'] and delivery <= job['budget_seconds'],
        ttf_seconds=ttf, delivery_seconds=delivery, setup_seconds=setup,
        cold_worker_seconds=time.monotonic()-cold, reset_seconds=reset_seconds,
        selection_seconds=selection_seconds, step_wall_seconds=step_wall_seconds,
        native_pp_seconds=native_pp_seconds, trace_seconds=trace_seconds,
        bookkeeping_seconds=bookkeeping_seconds, finalization_seconds=delivery-search_end,
        search_end_seconds=search_end,last_pp_failure_reason=last_pp_reason,
        stop=stop, decisions=d, pp_calls=pp_calls, legal_noops=noops,
        generated=state['low_level']['generated']-initial['low_level']['generated'],
        final_conflicts=state['num_of_colliding_pairs'], initial_conflicts=initial['num_of_colliding_pairs'],
        soc=state['sum_of_costs'], makespan=max(len(p)-1 for p in paths),
        wait_steps=sum(a==b for p in paths for a,b in zip(p,p[1:])), files=files,
        max_decisions=None, execution_node_budget=None)
    run.once(folder/'result.json', run.sealed(row))
    return dict(status='ok', job_id=job['job_id'], success=row['success_within_budget'],
                ttf=ttf, delivery=delivery, decisions=d)


def audit_worker(job):
    from scripts.run_feedback_exploration_diagnostics import validate_final
    from experiments.online_feature_engine import OnlineFeatureEngine
    from lns2_selector.runtime.online_selection import score_online_candidates
    from experiments.compact_controller_model import load_controller_bundle
    folder = run.ROOT/job['output']/'episodes'/job['job_id']
    row = run.check_seal(run.read_json(folder/'result.json'))
    run.require(row['binding'] == job['timing_binding'] and row['job_id'] == job['job_id'], 'timing identity')
    for name, digest in row['files'].items():
        run.require(run.sha256_file(run.contained_file(folder,name,field='timing artifact')) == digest, 'changed timing file')
    q = run.native_runtime(job['plan'])
    state = run.read_json(folder/'initial.json')
    initial_nodes = state['low_level']['generated']
    run.require(q.state_fingerprint(state) == row['initial_fingerprint'] == job['expected_initial'], 'audit initial')
    policy = Policy(job, q, dict(task_id=job['case']['task_id'], solver_seed=job['solver_seed'],
        case_id=f"{job['case']['task_id']}-seed{job['solver_seed']}", proposal=job['plan']['template']['proposal']))
    policy.start(state)
    anchor = load_controller_bundle(run.ROOT/'artifacts/initlns-closed-loop-controller-v2').main_models['realized_dynamic'] if policy.selector else None
    engine = OnlineFeatureEngine(state,backend='native') if policy.selector else None
    count, noops, last_pp_reason = 0, 0, None
    last_time = row['reset_seconds']
    feasible_time = last_time if state['feasible'] else None
    for event in run.trace_read(folder):
        d = count
        run.require(not state['feasible'] and last_time < row['budget_seconds'], 'action after terminal')
        run.require(last_pp_reason != 'time_limit','continued after PP deadline')
        run.require(event['decision'] == d and event['before'] == q.state_fingerprint(state), 'audit sequence')
        run.require(event['policy_sha256'] == row['policy_sha256'] == policy.sha, 'audit policy')
        if policy.selector:
            pool = event['pool']
            ids = [c['candidate_id'] for c in pool]
            run.require(ids == sorted(set(ids)) and set(event['proposal_order']) == set(ids), 'candidate identity')
            by_id = {c['candidate_id']:c for c in pool}
            original = [by_id[c] for c in event['proposal_order']]
            engine.prepare(state)
            fs,_ = engine.realized_rows(original, state_hash=event['before'])
            index,scores,_ = score_online_candidates(fs,anchor)
            run.require(original[index]['candidate_id'] == event['anchor_id'] and
                all(abs(c['score']-s)<=1e-12 for c,s in zip(original,scores)), 'anchor mismatch')
            draw = run.stream_draw(job['plan'],job['phase'],job['pair_id'],job['replica'],d,'select')
            if policy.actor:
                fs = run.features_for(state,pool,policy.engine,policy.history,q.temperature(d),event['before'])
                supplied = [run.budget_features(f,d,state['low_level']['generated']-initial_nodes,policy.plan['proposal']) for f in fs]
                run.require(supplied == event['features'], 'feature drift')
                probs = policy.actor.probabilities(ids,event['anchor_id'],supplied)
                run.require(probs == event['probabilities'], 'probability drift')
                cid = run.select_with_draw(probs,draw)
            else:
                run.require('features' not in event, 'baseline charged actor features')
                cid = event['anchor_id']
            expected = dict(mode='explicit_neighborhood',agents=by_id[cid]['agents'],random_seed=run.stream_draw(
                job['plan'],job['phase'],job['pair_id'],job['replica'],d,'pp'))
            run.require(cid == event['selected_id'] and draw == event['selection_draw'] and event['action'] == expected, 'action drift')
        else:
            run.require(event['action'] == dict(mode='official'), 'official RNG altered')
        run.require(event['temperature'] == q.temperature(d) and event['uniform'] == run.stream_draw(
            job['plan'],job['phase'],job['pair_id'],job['replica'],d,'accept'), 'acceptance stream')
        after = q.apply_state_delta(state,event['delta'])
        q.validate_transition(state,after,event['metrics'],event['action'].get('agents',event['metrics']['neighborhood']),
                              'annealed',event['temperature'],event['uniform'])
        policy.observe(state,event,after)
        run.require(math.isfinite(event['elapsed_seconds']) and event['elapsed_seconds']>=last_time, 'clock order')
        last_time = event['elapsed_seconds']
        if after['feasible']:feasible_time = last_time
        noops += event['metrics']['pp_failure_reason'] == 'not_run'
        last_pp_reason = event['metrics']['pp_failure_reason']
        state = after
        count += 1
    final = run.read_json(folder/'final.json')
    validate_final(final)
    run.require(q.state_fingerprint(state) == q.state_fingerprint(final) == row['final_fingerprint'], 'final mismatch')
    run.require(count == row['decisions'] and noops == row['legal_noops'] and count-noops == row['pp_calls'], 'work counts')
    run.require(row['generated'] == state['low_level']['generated']-initial_nodes and row['feasible'] == state['feasible'], 'outcome counts')
    paths = [a['path'] for a in final['agents']]
    run.require(row['soc'] == sum(len(p)-1 for p in paths) and row['makespan'] == max(len(p)-1 for p in paths) and
        row['wait_steps'] == sum(a==b for p in paths for a,b in zip(p,p[1:])), 'path quality mismatch')
    run.require(row['ttf_seconds'] == feasible_time and row['delivery_seconds'] >= last_time, 'TTF not from trace')
    run.require(row['last_pp_failure_reason']==last_pp_reason and
        last_time <= row['search_end_seconds'] <= row['delivery_seconds'],'terminal clock/reason')
    run.require(row['success_within_budget']==(feasible_time is not None and feasible_time<=row['budget_seconds']) and
        row['delivered_within_budget']==(state['feasible'] and row['delivery_seconds']<=row['budget_seconds']),'success/delivery flags')
    if row['stop']=='deadline':
        run.require(not state['feasible'] and row['search_end_seconds']>=row['budget_seconds'],'premature deadline')
    elif row['stop']=='pp_deadline':
        run.require(not state['feasible'] and count>0 and last_pp_reason=='time_limit','not a PP deadline')
    else:
        run.require(row['stop']=='feasible' and state['feasible'],'invalid terminal')
    return dict(status='ok',job_id=job['job_id'],decisions=count,legal_noops=noops,
                result_sha256=run.sha256_file(folder/'result.json'))
