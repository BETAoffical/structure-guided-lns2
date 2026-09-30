"""Same-parent terminal-gradient transfer diagnostics, never a policy update."""
from collections import defaultdict
import math

import numpy as np

from experiments.sa_paired_completion import require

ORDER = ('w1', 'b1', 'w2', 'b2')


def parameters(bundle):
    return np.concatenate([np.asarray(bundle['correction'][k], dtype=float).ravel() for k in ORDER])


def log_probabilities(bundle, pack):
    x, _, valid = pack['padded']
    c = bundle['correction']
    h = np.tanh(x @ np.asarray(c['w1']).T + np.asarray(c['b1']))
    prior = pack['base']
    require(prior.shape == valid.shape and np.all(prior[valid] > 0), 'invalid base support')
    logits = np.zeros_like(prior)
    logits[valid] = np.log(prior[valid])
    logits += (h @ np.asarray(c['w2']).T).squeeze(-1) + float(c['b2'][0])
    logits[~valid] = -np.inf
    centered = logits - logits.max(axis=1, keepdims=True)
    logs = centered - np.log(np.exp(centered).sum(axis=1, keepdims=True))
    require(np.isfinite(logs[valid]).all(), 'nonfinite log probability')
    return logs, h


def trajectory_score(bundle, pack):
    """Exact summed log-policy derivative for a nonzero two-layer tanh head."""
    x, behavior, valid = pack['padded']
    logs, h = log_probabilities(bundle, pack)
    probability = np.exp(logs)
    require(np.max(np.abs(probability - behavior)) <= 1e-12, 'behavior replay mismatch')
    delta = -probability
    delta[np.arange(len(delta)), pack['selected']] += 1
    hidden_derivative = delta[..., None] * np.asarray(bundle['correction']['w2'])[0] * (1 - h*h)
    flat_x = x.reshape(-1, x.shape[-1])
    flat_hidden = hidden_derivative.reshape(-1, hidden_derivative.shape[-1])
    gradient = [flat_hidden.T @ flat_x, flat_hidden.sum(axis=0),
                np.einsum('ij,ijk->k', delta, h)[None, :], np.asarray([delta.sum()])]
    score = np.concatenate([g.ravel() for g in gradient])
    require(np.isfinite(score).all(), 'nonfinite trajectory score')
    return score, logs


def cosine(a, b):
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b)/denominator) if denominator else None


def loss_gradient(rows):
    require(rows and len({r['episode_id'] for r in rows}) == len(rows), 'duplicate or empty episodes')
    return np.sum([np.asarray(r['loss_gradient'], dtype=float) for r in rows], axis=0)


def grouped_vectors(rows, key):
    groups = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    return {k:loss_gradient(v) for k, v in sorted(groups.items())}


def concentration(vectors):
    norms = {k:float(np.linalg.norm(v)) for k, v in vectors.items()}
    total = math.fsum(norms.values())
    mass = {k:v/total if total else 0. for k, v in norms.items()}
    return dict(norms=norms, norm_shares=mass,
                effective_norm_groups=1/math.fsum(v*v for v in mass.values()) if total else None,
                largest_norm_share=max(mass.values()),
                cancellation_ratio=float(np.linalg.norm(sum(vectors.values())))/total if total else None)


def summarize(rows, parent, proposed, expected_norm, *, other_direction=None):
    g = loss_gradient(rows)
    norm = float(np.linalg.norm(g))
    require(norm > 0 and math.isclose(norm, expected_norm, rel_tol=1e-10, abs_tol=1e-12),
            'saved gradient norm mismatch')
    delta = parameters(proposed) - parameters(parent)
    distance = float(np.linalg.norm(delta))
    reconstructed = -distance * g/norm
    error = float(np.max(np.abs(delta - reconstructed)))
    require(error <= 1e-11, 'saved update direction mismatch')
    maps = grouped_vectors(rows, 'map_id')
    conditions = grouped_vectors(rows, 'pair_id')
    by_map = {}
    for m, v in maps.items():
        kept = g - v
        selected = [r for r in rows if r['map_id'] == m]
        by_map[m] = dict(episodes=len(selected), successes=sum(r['success'] for r in selected),
            credited_episodes=sum(r['coefficient'] != 0 for r in selected),
            gradient_norm=float(np.linalg.norm(v)), alignment_with_batch=cosine(v, g),
            leave_map_out_alignment=cosine(kept, g),
            other_update_return_derivative=None if other_direction is None else float(-v @ other_direction),
            leave_map_out_other_alignment=None if other_direction is None else cosine(-kept, other_direction))
    return dict(episodes=len(rows), successes=sum(r['success'] for r in rows),
        decisions=sum(r['decisions'] for r in rows),
        credited_episodes=sum(r['coefficient'] != 0 for r in rows),
        maps=len(maps), conditions=len(conditions), gradient_norm=norm,
        saved_update_l2=distance, update_reconstruction_max_error=error,
        map_concentration=concentration(maps), condition_concentration=concentration(conditions),
        by_map=by_map,
        by_density={k:dict(episodes=sum(r['task_variant']==k for r in rows),
                          gradient_norm=float(np.linalg.norm(v)))
                    for k, v in grouped_vectors(rows, 'task_variant').items()},
        surrogate_delta={arm:math.fsum(r['surrogate_delta'][arm] for r in rows)
                         for arm in sorted(rows[0]['surrogate_delta'])})


def transfer(left, right, *, samples, seed):
    """Map resampling describes observed gradient sensitivity, not solver success."""
    a, b = loss_gradient(left), loss_gradient(right)
    require(np.linalg.norm(a)>0 and np.linalg.norm(b)>0, 'zero transfer direction')
    ua, ub = -a/np.linalg.norm(a), -b/np.linalg.norm(b)
    av, bv = grouped_vectors(left, 'map_id'), grouped_vectors(right, 'map_id')
    require(not set(av)&set(bv), 'overlapping batch map IDs')
    rng = np.random.default_rng(seed)
    aa, bb = np.stack(list(av.values())), np.stack(list(bv.values()))
    ac = rng.multinomial(len(aa), np.full(len(aa), 1/len(aa)), size=samples)
    bc = rng.multinomial(len(bb), np.full(len(bb), 1/len(bb)), size=samples)
    sampled_a, sampled_b = ac @ aa, bc @ bb
    denominator = np.linalg.norm(sampled_a, axis=1) * np.linalg.norm(sampled_b, axis=1)
    valid = denominator > 0
    values = np.einsum('ij,ij->i', sampled_a[valid], sampled_b[valid])/denominator[valid]
    return dict(direction_cosine=cosine(a,b),
        left_update_on_left_derivative=float(-a @ ua),
        right_update_on_left_derivative=float(-a @ ub),
        left_update_on_right_derivative=float(-b @ ua),
        right_update_on_right_derivative=float(-b @ ub),
        left_maps_supporting_right=sum(float(-v @ ub)>0 for v in av.values()),
        left_maps_opposing_right=sum(float(-v @ ub)<0 for v in av.values()),
        right_maps_supporting_left=sum(float(-v @ ua)>0 for v in bv.values()),
        right_maps_opposing_left=sum(float(-v @ ua)<0 for v in bv.values()),
        bootstrap=dict(unit='map', samples=samples, seed=seed,
            zero_gradient_draws=int((~valid).sum()),
            direction_cosine_interval95=np.quantile(values,[.025,.975]).tolist() if len(values) else None,
            negative_fraction=float(np.mean(values<0)) if len(values) else None,
            interpretation='empirical_training_direction_sensitivity_not_success_probability'),
        decision='posthoc_cross_batch_credit_transfer_not_performance_evidence')
