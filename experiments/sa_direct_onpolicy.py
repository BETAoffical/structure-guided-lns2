"""Complete-trajectory updates for the frozen direct-selector architecture."""
from copy import deepcopy
import gzip
import json
import math
import time

import numpy as np
from scripts import run_sa_onpolicy as run
from experiments import sa_direct_selector as direct
from experiments import sa_raw_timed_runtime as reference
from experiments import sa_terminal_efficiency as terminal
from experiments.sa_onpolicy_noop_runtime import pp_incomplete


def score_gradient(x, selected, offsets, coefficients, params):
    """Ascent gradient of coefficient-weighted sums of action log probabilities."""
    w1,b1,w2,b2=params
    hidden=np.tanh(x @ w1+b1)
    p=direct.distribution(hidden @ w2+b2[0],offsets)
    chosen=offsets[:-1]+np.asarray(selected)
    objective=float(np.sum(coefficients*np.log(p[chosen])))
    dl=-p.copy();dl[chosen]+=1.
    dl=(dl*np.repeat(coefficients,np.diff(offsets))).astype(x.dtype)
    dh=(dl[:,None]*w2[None,:])*(1-hidden*hidden)
    return objective,[x.T @ dh,dh.sum(axis=0),hidden.T @ dl,np.asarray([dl.sum()],dtype=x.dtype)],p


def rollout_worker(job):
    from scripts.train_sa_history_selector import die_with_parent
    from scripts.run_feedback_exploration_diagnostics import validate_final
    die_with_parent(job['parent_pid'])
    cfg=job['config'];folder=run.ROOT/job['output']/'rollouts'/job['job_id']
    run.require(not folder.exists(),'partial training episode; inspect')
    folder.mkdir(parents=True)
    q,env,ctx=reference.prepare_environment(job)
    state=q._plain(env.reset(seed=job['solver_seed']))
    run.require(state['initial_solution_complete'] and q.state_fingerprint(state)==job['expected_initial'],'train initial')
    initial=state;policy=direct.DirectPolicy(job,q,ctx);policy.start(state)
    began=time.monotonic();d=0;stop=None
    run.once(folder/'initial.json',initial)
    with gzip.open(folder/'trace.jsonl.gz','xt',encoding='utf8',compresslevel=1) as stream:
        while True:
            used=state['low_level']['generated']-initial['low_level']['generated']
            if state['feasible']: stop='feasible';break
            if used>=cfg['node_budget']: stop='node_budget';break
            remaining=cfg['episode_safety_seconds']-(time.monotonic()-began)
            if remaining<=0:stop='wall_safety';break
            e=policy.choose(env,state,d)
            remaining=cfg['episode_safety_seconds']-(time.monotonic()-began)
            if remaining<=0:stop='wall_safety';break
            after,m,temp,draw=reference.transition(job,q,env,state,e,d,min(cfg['pp_safety_seconds'],remaining))
            run.require(m['action_valid'] and m['step_applied'],'invalid training action')
            run.require(sorted(e['action']['agents'])==sorted(m['neighborhood']),'native changed neighborhood')
            e.update(decision=d,policy_sha256=policy.sha,metrics=m,temperature=temp,uniform=draw)
            policy.observe(state,e,after)
            stream.write(json.dumps(e,separators=(',',':'),allow_nan=False)+'\n')
            d+=1
            interrupted=pp_incomplete(state,after,m)
            state=after
            if interrupted:stop='incomplete_pp';break
    validate_final(state)
    run.once(folder/'final.json',state)
    row=dict(status='censored' if stop in ('wall_safety','incomplete_pp') else 'ok',
        binding=job['timing_binding'],job_id=job['job_id'],episode_id=job['job_id'],
        pair_id=job['pair_id'],map_id=job['case']['map_id'],split='train',replica=job['replica'],
        round=job['round'],policy_sha256=policy.sha,initial_fingerprint=q.state_fingerprint(initial),
        rng_stream_id=run.json_fingerprint([job['plan']['config']['stream_seed'],job['phase'],job['pair_id'],job['replica']]),
        stop=stop,success=state['feasible'],final_conflicts=state['num_of_colliding_pairs'],decisions=d,
        generated=state['low_level']['generated']-initial['low_level']['generated'],
        seconds_diagnostic=time.monotonic()-began,no_ttf=True)
    reward=terminal.episode_return(row,objective=cfg['objective'],max_decisions=None,node_budget=cfg['node_budget'])
    row['reward']=reward
    run.once(folder/'result.json',run.sealed(row))
    return dict(status='ok',episode_status=row['status'],job_id=job['job_id'],round=job['round'],
                success=row['success'],reward=reward,decisions=d,stop=stop)


def pack_trajectory(row, folder, actor):
    steps=[]
    with gzip.open(folder/'trace.jsonl.gz','rt') as stream:
        for line in stream:steps.append(json.loads(line))
    run.require(len(steps)==row['decisions'],'missing training actions')
    x,selected,offsets,old=[],[],[0],[]
    for d,e in enumerate(steps):
        run.require(e['decision']==d and e['policy_sha256']==actor.sha,'mixed policy or trajectory order')
        ids=e['candidate_ids'];fs=e['features']
        raw=np.asarray([[f[n] for n in actor.names] for f in fs],dtype=np.float32)
        x.append(direct.inputs(raw,actor.mean,actor.scale))
        offsets.append(offsets[-1]+len(ids));selected.append(ids.index(e['selected_id']))
        old.extend(e['probabilities'][c] for c in ids)
    return dict(row=row|{'steps':steps},x=np.concatenate(x) if x else np.empty((0,256),dtype=np.float32),
        selected=np.asarray(selected),offsets=np.asarray(offsets,dtype=np.int64),old=np.asarray(old))


def step_statistics(packs,params):
    state_means,trajectory_sums,maximum=[],[],0.
    for pack in packs:
        if len(pack['selected'])==0:state_means.append(0.);trajectory_sums.append(0.);continue
        w1,b1,w2,b2=params
        new=direct.distribution(np.tanh(pack['x'] @ w1+b1) @ w2+b2[0],pack['offsets'])
        old=pack['old']
        kl=np.maximum(0.,np.add.reduceat(old*(np.log(old)-np.log(new)),pack['offsets'][:-1]))
        state_means.append(float(kl.mean()));trajectory_sums.append(float(kl.sum()))
        maximum=max(maximum,float(kl.max()))
    return dict(mean_state_kl=float(np.mean(state_means)),mean_trajectory_kl=float(np.mean(trajectory_sums)),max_state_kl=maximum)


def update(parent,rows,root,cfg,expected_groups,round_index):
    actor=direct.DirectActor(parent)
    run.require(len(rows)==len(expected_groups)*cfg['replicas'],'training coverage')
    if any(row['status']!='ok' for row in rows):
        return deepcopy(parent),dict(updated=False,decision='censored_batch_no_update',
            censored=[row['job_id'] for row in rows if row['status']!='ok'])
    packs=[pack_trajectory(row,root/'rollouts'/row['job_id'],actor) for row in rows]
    coefficients=terminal.coefficients([p['row'] for p in packs],objective=cfg['objective'],
        policy_sha256=actor.sha,expected_groups=expected_groups,replicas=cfg['replicas'],node_budget=cfg['node_budget'])
    credit={c['episode_id']:c['coefficient'] for c in coefficients}
    grads=[np.zeros_like(a,dtype=np.float64) for a in actor.params]
    max_error=0.
    for pack in packs:
        if len(pack['selected'])==0:continue
        co=np.full(len(pack['selected']),credit[pack['row']['episode_id']])
        _,g,p=score_gradient(pack['x'],pack['selected'],pack['offsets'],co,actor.params)
        max_error=max(max_error,float(np.max(abs(p-pack['old']))))
        for target,value in zip(grads,g):target+=value
    run.require(max_error<=5e-6,'behavior probability replay mismatch')
    norm=math.sqrt(sum(float(np.sum(g*g)) for g in grads))
    run.require(math.isfinite(norm),'nonfinite terminal gradient')
    proof=dict(gradient_norm=norm,behavior_replay_max_absolute_error=max_error,credit=coefficients,
               credited_episodes=sum(c['coefficient']!=0 for c in coefficients),attempts=[])
    if norm==0:return deepcopy(parent),proof|dict(updated=False,decision='zero_gradient')
    for step in cfg['parameter_steps']:
        params=[(p.astype(np.float64)+step*g/norm).astype(np.float32) for p,g in zip(actor.params,grads)]
        stats=step_statistics(packs,params);proof['attempts'].append(dict(parameter_step=step,**stats))
        if all(stats[k]<=cfg[k] for k in ('mean_state_kl','mean_trajectory_kl','max_state_kl')):
            model=deepcopy(parent)
            model.update(training='on_policy_complete_trajectory_reinforce',onpolicy_round=round_index,
                         parent_policy_sha256=actor.sha)
            model.update({k:p.tolist() for k,p in zip(('w1','b1','w2','b2'),params)})
            direct.DirectActor(model)
            return model,proof|dict(updated=True,decision='guarded_terminal_update',selected=proof['attempts'][-1])
    return deepcopy(parent),proof|dict(updated=False,decision='no_step_within_kl_limits')


def qualification_worker(job):
    q,env,_=reference.prepare_environment(job)
    state=q._plain(env.reset(seed=job['solver_seed']))
    run.require(state['initial_solution_complete'],'incomplete new-seed initial PP')
    from scripts.run_feedback_exploration_diagnostics import validate_final
    validate_final(state)
    return dict(pair_id=job['pair_id'],initial=q.state_fingerprint(state),conflicts=state['num_of_colliding_pairs'])


def timed_worker(job):
    if job['comparison_arm'] in ('updated','frozen'):
        return direct._bind(direct.lean.lean_worker,Policy=direct.DirectPolicy)(job)
    return direct._bind(direct.lean.lean_worker,Policy=reference.Policy)(job)
