"""Conditional terminal-score replication; no optimizer, reward redesign or model export."""
import math

import numpy as np

from experiments.sa_completion_credit_transfer import cosine
from experiments.sa_paired_completion import require


def conditional_loss_gradient(returns, scores):
    """Unit condition mass, independent other-replica baseline, whole-trajectory score."""
    y, s = np.asarray(returns,dtype=float),np.asarray(scores,dtype=float)
    require(y.ndim==1 and len(y)>=2 and s.ndim==2 and len(s)==len(y), 'incomplete condition')
    require(np.isfinite(s).all() and np.isin(y,[0.,1.]).all(), 'invalid complete-terminal input')
    advantages = y-(y.sum()-y)/(len(y)-1)
    contributions = -advantages[:,None]*s/len(y)
    return contributions.sum(axis=0),contributions


def wilson(successes, count):
    require(0<=successes<=count and count>0, 'invalid binomial counts')
    z, p = 1.959963984540054,successes/count
    denominator = 1+z*z/count
    center = (p+z*z/(2*count))/denominator
    radius = z*math.sqrt(p*(1-p)/count+z*z/(4*count*count))/denominator
    return [max(0.,center-radius),min(1.,center+radius)]


def condition_report(old, fresh, *, bootstrap, seed):
    require(len(old)==4 and len(fresh)==16 and
            len({r['episode_id'] for r in old+fresh})==20, 'fixed independent repetitions')
    require(len({r['pair_id'] for r in old+fresh})==1 and
            len({r['initial_fingerprint'] for r in old+fresh})==1 and
            len({r['policy_sha256'] for r in old+fresh})==1, 'changed condition/state/policy')
    require(len({r['rng_stream_id'] for r in old+fresh})==20, 'reused random stream')
    require(all(r['status']=='ok' and r['split']=='train' for r in old+fresh), 'unknown or heldout is not a label')
    gy,contributions = conditional_loss_gradient([r['success'] for r in fresh],[r['score'] for r in fresh])
    go,_ = conditional_loss_gradient([r['success'] for r in old],[r['score'] for r in old])
    norms = np.linalg.norm(contributions,axis=1)
    total = float(norms.sum())
    index = int(np.argmax(norms))
    def distribution(rows):
        decisions = np.asarray([r['decisions'] for r in rows])
        successes = sum(r['success'] for r in rows)
        return dict(episodes=len(rows),successes=successes,failures=len(rows)-successes,
            conditional_success_rate=successes/len(rows),wilson_interval95=wilson(successes,len(rows)),
            decisions=dict(min=int(decisions.min()),median=float(np.median(decisions)),
                           p90=float(np.quantile(decisions,.9)),max=int(decisions.max())),
            failed_ge_512=sum(not r['success'] and r['decisions']>=512 for r in rows),
            successful_ge_512=sum(r['success'] and r['decisions']>=512 for r in rows),
            generated=dict(mean=float(np.mean([r['generated'] for r in rows])),
                           max=max(r['generated'] for r in rows)))
    y,s = np.asarray([r['success'] for r in fresh],dtype=float),np.asarray([r['score'] for r in fresh])
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(y),np.full(len(y),1/len(y)),size=bootstrap)
    mean_y,mean_s = counts @ y/len(y),counts @ s/len(y)
    mean_ys = counts @ (y[:,None]*s)/len(y)
    sampled = -len(y)/(len(y)-1)*(mean_ys-mean_y[:,None]*mean_s)
    # All-success/all-failure draws have exactly zero credit, including floating roundoff.
    mixed = (mean_y>0)&(mean_y<1)
    usable = mixed & (np.linalg.norm(sampled,axis=1)>0) & (np.linalg.norm(go)>0)
    values = sampled[usable] @ go/(np.linalg.norm(sampled[usable],axis=1)*np.linalg.norm(go))
    summary = dict(old=distribution(old),fresh=distribution(fresh),
        old_conditional_gradient_norm=float(np.linalg.norm(go)),fresh_conditional_gradient_norm=float(np.linalg.norm(gy)),
        old_fresh_direction_cosine=cosine(go,gy),
        largest_episode_contribution=dict(episode_id=fresh[index]['episode_id'],replica=fresh[index]['replica'],
            success=fresh[index]['success'],decisions=fresh[index]['decisions'],
            norm=float(norms[index]),norm_share=float(norms[index]/total) if total else None,
            alignment_with_fresh=cosine(contributions[index],gy)),
        leave_one_episode_cosines=[cosine(conditional_loss_gradient(
            np.delete(y,i),np.delete(s,i,axis=0))[0],gy) for i in range(len(y))],
        bootstrap=dict(unit='independent_replica_within_fixed_condition',samples=bootstrap,seed=seed,
            uninformative_draws=int((~usable).sum()),
            old_direction_cosine_interval95=np.quantile(values,[.025,.975]).tolist() if len(values) else None,
            positive_fraction=float(np.mean(values>0)) if len(values) else None,
            not_a_success_probability=True))
    return summary,go,gy
