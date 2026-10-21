"""Single neural selector: no tree score, recommendation, or residual at runtime."""
from copy import deepcopy
import gzip
import heapq
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from scripts import run_sa_onpolicy as run
from experiments.sa_raw_residual_actor import RawResidualActor
from experiments.sa_raw_selection_fast import _bind
from experiments import sa_lean_timing as lean
from experiments import sa_raw_timed_runtime as reference
from experiments.sa_shared_features import SharedFeaturePolicy
from experiments.sa_history_selector import History
from experiments.online_feature_engine import OnlineFeatureEngine, TopologyAnalysisCache
from experiments.feature_schema_v2 import PROFILE_FEATURE_NAMES
from experiments.sa_policy_aligned_update import budget_features
from experiments.sa_uncapped_runtime import feature_plan
from lns2_selector.runtime.online_selection import generate_online_candidates
from lns2_selector.runtime.structshell_dual16 import (
    generate_structshell_dual16_runtime_candidates, structshell_dual16_augmentation,
)

SCHEMA = 'lns2.sa_direct_selector.v1'


def inputs(values, mean, scale):
    z = (np.asarray(values, dtype=np.float32)-mean)/scale
    return np.concatenate((z, z-z.mean(axis=0, keepdims=True)), axis=1)


def distribution(logits, offsets):
    """Segmented softmax, one candidate set per segment."""
    starts = offsets[:-1]
    counts = np.diff(offsets)
    x = np.asarray(logits, dtype=np.float64)
    peak = np.maximum.reduceat(x, starts)
    mass = np.exp(x-np.repeat(peak, counts))
    return mass/np.repeat(np.add.reduceat(mass, starts), counts)


def objective_grad(x, target, offsets, params):
    w1, b1, w2, b2 = params
    hidden = np.tanh(x @ w1 + b1)
    logits = hidden @ w2 + b2[0]
    probs = distribution(logits, offsets)
    n = len(offsets)-1
    loss = -float(np.sum(target*np.log(np.maximum(probs, 1e-300)))/n)
    dl = ((probs-target)/n).astype(x.dtype)
    dh = (dl[:, None]*w2[None, :])*(1-hidden*hidden)
    grads = [x.T @ dh, dh.sum(axis=0), hidden.T @ dl,
             np.asarray([dl.sum()], dtype=x.dtype)]
    return loss, grads


class DirectActor:
    def __init__(self, bundle):
        run.require(bundle['schema'] == SCHEMA and 'source.score' not in bundle['feature_names'], 'direct schema')
        self.bundle, self.sha = bundle, run.json_fingerprint(bundle)
        self.names = bundle['feature_names']
        self.mean = np.asarray(bundle['mean'], dtype=np.float32)
        self.scale = np.asarray(bundle['scale'], dtype=np.float32)
        self.params = [np.asarray(bundle[k], dtype=np.float32) for k in ('w1','b1','w2','b2')]
        d, h = 2*len(self.names), len(bundle['b1'])
        run.require(len(self.names)==128 and self.scale.shape==(128,) and np.all(self.scale>0), 'direct inputs')
        run.require([a.shape for a in self.params]==[(d,h),(h,),(h,),(1,)] and
                    all(np.isfinite(a).all() for a in self.params), 'direct weights')

    def probabilities(self, ids, features):
        run.require(ids and len(ids)==len(features) and len(set(ids))==len(ids), 'candidate alignment')
        run.require(all(set(f)==set(self.names) for f in features), 'direct feature schema')
        values = np.asarray([[f[n] for n in self.names] for f in features], dtype=np.float32)
        run.require(np.isfinite(values).all(), 'nonfinite input')
        x = inputs(values, self.mean, self.scale)
        w1,b1,w2,b2 = self.params
        p = distribution(np.tanh(x @ w1+b1) @ w2+b2[0], np.asarray([0,len(ids)]))
        run.require(np.isfinite(p).all() and np.all(p>0), 'invalid probability')
        return dict(zip(ids,p.tolist()))


class DirectPool:
    """Same generators and full dynamic features, with no GBDT constructed."""
    def __init__(self, ctx):
        self.ctx, self.engine, self.topology = ctx, None, None

    def candidates(self, env, state, d):
        from experiments.repair_collection import state_fingerprint
        before = state_fingerprint(state)
        if self.engine is None:
            self.engine = OnlineFeatureEngine(state, backend='native', dense_output=True,
                required_features={'realized_dynamic':set(PROFILE_FEATURE_NAMES['realized_dynamic'])})
        if self.topology is None:
            self.topology = TopologyAnalysisCache(state,static_grid=self.engine.static_grid,backend='native')
        else:
            self.topology.prepare(state)
        candidates,_ = generate_online_candidates(env,state,task_id=self.ctx['task_id'],
            solver_seed=self.ctx['solver_seed'],decision_index=d,proposal_config=self.ctx['proposal'],
            state_hash=before,verify_full_state=False,proposal_backend='optimized')
        pool = list(generate_structshell_dual16_runtime_candidates(state,self.topology.analysis,
            v2_candidates=candidates,config=structshell_dual16_augmentation()).candidates)
        self.engine.prepare(state,prepared_native_analysis=self.topology.last_native_prepared)
        rows,_ = self.engine.realized_rows(pool,state_hash=before)
        run.require(state_fingerprint(env.get_state())==before,'proposal mutated state')
        values = {r['candidate_id']:dict(zip(r['feature_names'],r['feature_values'])) for r in rows}
        return pool,values


class DirectPolicy:
    def __init__(self, job, q, ctx):
        self.job,self.q,self.ctx = job,q,ctx
        run.require(run.sha256_file(run.ROOT/job['model']['path'])==job['model']['sha256'],'direct model changed')
        self.actor = DirectActor(run.read_json(run.ROOT/job['model']['path']))
        self.sha = self.actor.sha
        self.pool = DirectPool(ctx)
        self.history = None
        self.plan = feature_plan(job['plan'])

    def start(self, state):
        self.history = History(state)
        self.initial_nodes = state['low_level']['generated']

    def choose(self, env, state, d):
        pool,values = self.pool.candidates(env,state,d)
        order = [c['candidate_id'] for c in pool]
        pool = sorted(pool,key=lambda c:c['candidate_id'])
        ids = [c['candidate_id'] for c in pool]
        phase = {'sa.log_temperature':math.log1p(self.q.temperature(d)), 'sa.log_decision':math.log1p(d)}
        fs = [budget_features(values[cid]|phase,d,state['low_level']['generated']-self.initial_nodes,
                              self.plan['proposal']) for cid in ids]
        fs = [{n:f[n] for n in self.actor.names} for f in fs]
        p = self.actor.probabilities(ids,fs)
        j = self.job
        draw = run.stream_draw(j['plan'],j['phase'],j['pair_id'],j['replica'],d,'select')
        cid = run.select_with_draw(p,draw)
        chosen = next(c for c in pool if c['candidate_id']==cid)
        return dict(pool=pool,proposal_order=order,candidate_ids=ids,features=fs,probabilities=p,
            selected_id=cid,selection_draw=draw,behavior_log_probability=math.log(p[cid]),
            dual_scoring_calls=0,action=dict(mode='explicit_neighborhood',agents=chosen['agents'],
                random_seed=run.stream_draw(j['plan'],j['phase'],j['pair_id'],j['replica'],d,'pp')))

    def observe(self, before, event, after):
        self.history.observe(before,event,after)


def timed_worker(job):
    classes = {'direct':DirectPolicy,'a':SharedFeaturePolicy,'dual':reference.Policy}
    return _bind(lean.lean_worker,Policy=classes[job['comparison_arm']])(job)


def extract_map(job):
    """Use frozen source traces; sample identities before querying teacher labels."""
    cfg,root = job['config'],run.ROOT/job['config']['output']
    limit = cfg['train_states_per_map'] if job['split']=='train' else cfg['validation_states_per_map']
    teacher = RawResidualActor(run.read_json(run.ROOT/cfg['teacher']))
    names = [n for n in teacher.bundle['base']['feature_names'] if n!='source.score']
    heap,seen = [],set()
    for source_job in job['jobs']:
        folder = run.ROOT/cfg['training_source']/'episodes'/source_job['job_id']
        result = run.read_json(folder/'result.json')
        run.require(result['status']=='ok','incomplete source')
        with gzip.open(folder/'trace.jsonl.gz','rt') as stream:
            for line in stream:
                e = json.loads(line)
                identity = run.json_fingerprint([e['before'],e['decision']])
                if identity in seen:
                    continue
                seen.add(identity)
                priority = int(run.json_fingerprint([cfg['seed'],job['map_id'],identity]),16)
                if len(heap)<limit or priority < -heap[0][0]:
                    item = (-priority,identity,e)
                    if len(heap)<limit:
                        heapq.heappush(heap,item)
                    else:
                        heapq.heapreplace(heap,item)
    xs,ys,offsets,identities = [],[],[0],[]
    for _,identity,e in sorted(heap,key=lambda z:z[1]):
        ids = e['candidate_ids']
        p = teacher.probabilities(ids,e['anchor_id'],e['features'])
        xs.append(np.asarray([[f[n] for n in names] for f in e['features']],dtype=np.float32))
        ys.append(np.asarray([p[c] for c in ids],dtype=np.float64))
        offsets.append(offsets[-1]+len(ids)); identities.append(identity)
    run.require(xs,'empty map')
    dest = root/'data'/f"{job['index']:02d}.npz"
    np.savez_compressed(dest,x=np.concatenate(xs),target=np.concatenate(ys),offsets=offsets,
                        identities=identities)
    result = dict(map_id=job['map_id'],split=job['split'],states=len(xs),available=len(seen),
                  rows=offsets[-1],path=dest.relative_to(run.ROOT).as_posix(),sha256=run.sha256_file(dest),feature_names=names)
    run.once(dest.with_suffix('.json'),result)
    return result


def load_data(root, split):
    meta = run.read_json(root/'dataset.json')
    xs,ys,offsets = [],[],[0]
    for entry in meta['maps']:
        if entry['split']!=split:
            continue
        with np.load(run.ROOT/entry['path']) as d:
            xs.append(d['x']); ys.append(d['target'])
            offsets.extend((d['offsets'][1:]+offsets[-1]).tolist())
    return np.concatenate(xs),np.concatenate(ys),np.asarray(offsets,dtype=np.int64)


def prepared_x(raw, offsets, mean, scale):
    return np.concatenate([inputs(raw[a:b],mean,scale) for a,b in zip(offsets,offsets[1:])])


def evaluate(x, target, offsets, params):
    w1,b1,w2,b2 = params
    p = distribution(np.tanh(x @ w1+b1) @ w2+b2[0],offsets)
    ce = -float(np.sum(target*np.log(np.maximum(p,1e-300)))/(len(offsets)-1))
    entropy = -float(np.sum(target*np.log(np.maximum(target,1e-300)))/(len(offsets)-1))
    wins = sum(np.argmax(p[a:b])==np.argmax(target[a:b]) for a,b in zip(offsets,offsets[1:]))
    return dict(cross_entropy=ce,kl=ce-entropy,top1= wins/(len(offsets)-1),
                states=len(offsets)-1,mean_total_variation=float(np.sum(abs(p-target))/2/(len(offsets)-1)))


def train_worker(job):
    cfg = job['config']; root = run.ROOT/cfg['output']
    normal = run.read_json(root/'normalization.json')
    mean,scale = np.asarray(normal['mean'],dtype=np.float32),np.asarray(normal['scale'],dtype=np.float32)
    raw,target,offsets = load_data(root,'train')
    x = prepared_x(raw,offsets,mean,scale)
    vr,vt,vo = load_data(root,'validation'); vx = prepared_x(vr,vo,mean,scale)
    rng = np.random.default_rng(job['seed']); width = job['width']
    params = [rng.normal(0,1/math.sqrt(x.shape[1]),(x.shape[1],width)).astype(np.float32),
              np.zeros(width,dtype=np.float32),rng.normal(0,.01,width).astype(np.float32),
              np.zeros(1,dtype=np.float32)]
    m,v = [np.zeros_like(a) for a in params],[np.zeros_like(a) for a in params]
    best,best_metrics,best_epoch,step = None,None,None,0
    began = time.monotonic()
    for epoch in range(cfg['epochs']):
        order = rng.permutation(len(offsets)-1)
        for start in range(0,len(order),cfg['batch_states']):
            group = order[start:start+cfg['batch_states']]
            indices = np.concatenate([np.arange(offsets[i],offsets[i+1]) for i in group])
            bo = np.concatenate(([0],np.cumsum([offsets[i+1]-offsets[i] for i in group])))
            _,grads = objective_grad(x[indices],target[indices],bo,params)
            step += 1
            norm = math.sqrt(sum(float(np.sum(g*g)) for g in grads))
            for k,g in enumerate(grads):
                g = g*min(1.,5./max(norm,1e-30))
                if k in (0,2): g = g+cfg['weight_decay']*params[k]
                m[k] = .9*m[k]+.1*g; v[k] = .999*v[k]+.001*g*g
                params[k] -= cfg['learning_rate']*(m[k]/(1-.9**step))/(np.sqrt(v[k]/(1-.999**step))+1e-8)
        metrics = evaluate(vx,vt,vo,params)
        if best_metrics is None or metrics['cross_entropy']<best_metrics['cross_entropy']:
            best = [a.copy() for a in params]; best_metrics = metrics; best_epoch=epoch+1
    ident = f"h{width}-s{job['seed']}"
    bundle = dict(schema=SCHEMA,feature_names=normal['feature_names'],mean=normal['mean'],scale=normal['scale'],
                  width=width,seed=job['seed'],epoch=best_epoch,input_transform='normalized_and_pool_centered',
                  training='frozen_A_policy_distillation',dual_score_input=False,dual_anchor_input=False,
                  **{k:a.tolist() for k,a in zip(('w1','b1','w2','b2'),best)})
    run.once(root/'models'/(ident+'.json'),bundle)
    result = dict(id=ident,width=width,seed=job['seed'],epoch=best_epoch,validation=best_metrics,
                  train=evaluate(x,target,offsets,best),seconds=time.monotonic()-began,
                  path=(root/'models'/(ident+'.json')).relative_to(run.ROOT).as_posix())
    run.once(root/'models'/(ident+'-result.json'),result)
    return result


def smoke(job):
    q,env,ctx = reference.prepare_environment(job)
    state = q._plain(env.reset(seed=job['solver_seed']))
    run.require(q.state_fingerprint(state)==job['expected_initial'],'smoke initial')
    direct = DirectPolicy(job,q,ctx); direct.start(state)
    event = direct.choose(env,state,0)
    other = deepcopy(job)
    other.update(comparison_arm='a',model=job['teacher_spec'],arm='trained_actor',iteration=2)
    baseline = SharedFeaturePolicy(other,q,ctx); baseline.start(state)
    old = baseline.choose(env,state,0)
    strip = lambda c:{k:v for k,v in c.items() if k!='score'}
    run.require([strip(c) for c in old['pool']]==event['pool'],'candidate membership/provenance changed')
    run.require(old['candidate_ids']==event['candidate_ids'],'IDs differ')
    run.require([{k:v for k,v in f.items() if k!='source.score'} for f in old['features']]==event['features'],
                'direct features differ')
    after,metrics,temp,draw = reference.transition(job,q,env,state,event,0,30.)
    q.validate_transition(state,after,metrics,event['action']['agents'],'annealed',temp,draw)
    from scripts.run_feedback_exploration_diagnostics import validate_final
    validate_final(after)
    return dict(status='ok',job_id=job['job_id'],candidate_feature_equal=True,
                probabilities_sum=sum(event['probabilities'].values()),dual_scoring_calls=0,
                transition_valid=True)
