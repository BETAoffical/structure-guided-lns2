"""A fixed state-coverage intervention, not a production selector."""
from statistics import mean

from experiments._common import json_fingerprint
from experiments.sa_paired_completion import (
    MODEL_PARAMS, PairedCompletionModel, require, validate_dataset,
)


def choose_new_roots(available, maps, excluded, seed):
    require(len({r['id'] for r in available}) == len(available), 'duplicate occurrence')
    roots, coverage = [], []
    for map_id in sorted(maps):
        pool = [r for r in available if r['map_id'] == map_id and r['item']['job_id'] not in excluded]
        chosen, used = [], set()
        # Sample episodes before occurrences so long traces get no extra tickets.
        for slot, phase in enumerate(('continuing', 'early', None, None)):
            options = [r for r in pool if r['item']['job_id'] not in used and
                       (phase is None or r['phase'] == phase)]
            episodes = {r['item']['job_id'] for r in options}
            if not episodes:
                continue
            ep = min(episodes, key=lambda e: (json_fingerprint([seed, map_id, slot, e]), e))
            pick = min((r for r in options if r['item']['job_id'] == ep),
                       key=lambda r: (json_fingerprint([seed, slot, r['id']]), r['id']))
            chosen.append(pick)
            used.add(ep)
        roots.extend(chosen)
        coverage.append(dict(map_id=map_id, available_episodes=len({r['item']['job_id'] for r in pool}),
                             selected=len(chosen), phases=[r['phase'] for r in chosen],
                             admitted=len(chosen) == 4 and {r['phase'] for r in chosen} == {'early', 'continuing'}))
    require(len({r['item']['job_id'] for r in roots}) == len(roots), 'episode reuse')
    return sorted(roots, key=lambda r: (r['map_id'], r['id'])), coverage


def merge_data(old, new, binding):
    validate_dataset(old)
    validate_dataset(new)
    for key in ('schema', 'feature_names', 'horizon', 'trial_count', 'role', 'sampling', 'source_kind'):
        require(old[key] == new[key], 'dataset contract mismatch: ' + key)
    require(not ({s['episode'] for s in old['states']} & {s['episode'] for s in new['states']}),
            'old/new episode overlap')
    require({s['map_id'] for s in old['states']} == {s['map_id'] for s in new['states']}, 'map cohort mismatch')
    data = dict(old, continuation_binding=binding, states=old['states'] + new['states'])
    validate_dataset(data)
    return data


def fit_models(data, held):
    import sklearn
    from sklearn.ensemble import HistGradientBoostingRegressor
    require(sklearn.__version__ == '1.5.0', 'registered sklearn required')
    paired = PairedCompletionModel.fit(data, {held})
    names = data['feature_names']
    training = sorted((s for s in data['states'] if s['map_id'] != held), key=lambda s: s['state_id'])
    x, y, w = [], [], []
    for state in training:
        for c in sorted(state['candidates'], key=lambda c: c['candidate_id']):
            require(all(t['completed'] is not None for t in c['trials']), 'censoring forbids fitting')
            x.append([c['features'][n] for n in names])
            y.append(mean(int(t['completed']) for t in c['trials']))
            w.append(1 / len(state['candidates']))
    direct = HistGradientBoostingRegressor(loss='squared_error', **MODEL_PARAMS).fit(x, y, sample_weight=w)
    return dict(paired=paired, direct=direct, names=names, train_ids=[s['state_id'] for s in training],
                train_maps=sorted({s['map_id'] for s in training}), held=held)


def predict(models, state):
    require(state['map_id'] == models['held'] and state['map_id'] not in models['train_maps'], 'held-map leakage')
    cs = sorted(state['candidates'], key=lambda c: c['candidate_id'])
    # No trial or outcome is consumed here.
    scores = models['direct'].predict([[c['features'][n] for n in models['names']] for c in cs])
    scores = {c['candidate_id']: float(v) for c, v in zip(cs, scores)}
    selected = min(scores, key=lambda c: (-scores[c], c != state['anchor_id'], c))
    return dict(paired=models['paired'].rank(state), direct=dict(selected=selected, scores=scores))


def contrast(rows, method, baseline, seed, samples=5000):
    import numpy as np
    maps = sorted({r['map_id'] for r in rows})
    delta = np.array([mean(r['rates'][method] - r['rates'][baseline] for r in rows if r['map_id'] == m) for m in maps])
    draws = np.random.default_rng(seed).integers(0, len(maps), (samples, len(maps)))
    return dict(delta=float(delta.mean()), ci95=np.quantile(delta[draws].mean(axis=1), [.025, .975]).tolist(),
                wins=int(sum(delta > 0)), losses=int(sum(delta < 0)), ties=int(sum(delta == 0)),
                map_deltas=dict(zip(maps, delta.tolist())))


def signal_gate(contrasts):
    frozen, random, old = (contrasts[k] for k in ('frozen', 'uniform', 'old_same_model'))
    return (frozen['delta'] >= .05 and frozen['ci95'][0] >= 0 and frozen['wins'] >= 5 and
            random['delta'] > 0 and random['ci95'][0] >= 0 and
            old['delta'] > 0 and old['ci95'][0] >= 0)
