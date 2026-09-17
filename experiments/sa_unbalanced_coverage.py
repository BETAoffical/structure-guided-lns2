"""Map-balanced weights and summaries for a fixed, unequal source cohort."""
from collections import Counter
from statistics import mean

from experiments.sa_paired_completion import MODEL_PARAMS, PairedCompletionModel, require, training_matrix


def state_weights(states, held):
    train = [s for s in states if s['map_id'] != held]
    counts = Counter(s['map_id'] for s in train)
    require(counts and held in {s['map_id'] for s in states}, 'invalid held map')
    require(len({s['state_id'] for s in train}) == len(train), 'duplicate training state')
    # Keep total weight equal to state count, preserving the old L2 scale.
    return {s['state_id']: len(train)/(len(counts)*counts[s['map_id']]) for s in train}


def fit_models(data, held):
    import sklearn
    from sklearn.ensemble import HistGradientBoostingRegressor
    require(sklearn.__version__ == '1.5.0', 'registered sklearn required')
    matrix = training_matrix(data, {held})
    weights = state_weights(data['states'], held)
    estimator = HistGradientBoostingRegressor(loss='squared_error', **MODEL_PARAMS)
    estimator.fit(matrix['x'], matrix['y'], sample_weight=[
        w*weights[sid] for w,sid in zip(matrix['weights'],matrix['state_ids'])])
    paired = PairedCompletionModel(list(data['feature_names']), estimator)
    train = sorted((s for s in data['states'] if s['map_id'] != held), key=lambda s:s['state_id'])
    names = data['feature_names']
    x,y,w = [],[],[]
    for s in train:
        for c in sorted(s['candidates'], key=lambda c:c['candidate_id']):
            require(all(t['completed'] is not None for t in c['trials']), 'censoring forbids training')
            x.append([c['features'][n] for n in names])
            y.append(mean(int(t['completed']) for t in c['trials']))
            w.append(weights[s['state_id']]/len(s['candidates']))
    direct = HistGradientBoostingRegressor(loss='squared_error', **MODEL_PARAMS).fit(x,y,sample_weight=w)
    return dict(paired=paired,direct=direct,names=names,held=held,train_ids=[s['state_id'] for s in train],
                train_maps=sorted({s['map_id'] for s in train}),state_weights=weights)


def summarize(rows):
    require(rows and len({r['state_id'] for r in rows}) == len(rows), 'missing or duplicate evaluation states')
    maps = sorted({r['map_id'] for r in rows})
    keys = set(rows[0]['rates'])
    require(all(set(r['rates']) == keys for r in rows), 'rate columns differ')
    per_map = {m:{k:mean(r['rates'][k] for r in rows if r['map_id']==m) for k in sorted(keys)} for m in maps}
    return dict(map_equal={k:mean(per_map[m][k] for m in maps) for k in sorted(keys)},
                state_equal={k:mean(r['rates'][k] for r in rows) for k in sorted(keys)},per_map=per_map)


def validate_cohort(roots, expected_counts, excluded):
    require(dict(Counter(r['map_id'] for r in roots)) == expected_counts, 'registered map counts changed')
    require(len({r['id'] for r in roots}) == len(roots), 'duplicate root')
    episodes = {r['item']['job_id'] for r in roots}
    require(len(episodes) == len(roots) and not episodes & set(excluded), 'episode overlap')
    for m in expected_counts:
        require({r['phase'] for r in roots if r['map_id']==m} == {'early','continuing'}, 'phase coverage changed')
    require(all(len(r['selected']) == 4 and len(set(r['selected'])) == 4 for r in roots), 'candidate grid changed')
