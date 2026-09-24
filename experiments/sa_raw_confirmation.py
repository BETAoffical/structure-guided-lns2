"""Paired, map-disjoint evaluation; no optimization or model selection."""
from collections import defaultdict
import random

from experiments._common import json_fingerprint
from experiments.sa_paired_completion import require

ARMS = ('raw_parent', 'raw_updated', 'dual16_sa', 'official_sa')


def schedule(conditions, phase, replicas):
    require(conditions and replicas > 0, 'empty confirmation')
    require(all(c['split'] == 'independent_confirmation' for c in conditions), 'confirmation only')
    require(len({c['pair_id'] for c in conditions}) == len(conditions), 'duplicate condition')
    return [dict(c, phase=phase, replica=r, comparison_arm=arm,
                 arm='trained_actor' if arm.startswith('raw_') else arm,
                 iteration=int(arm == 'raw_updated'),
                 job_id=json_fingerprint([phase, c['pair_id'], r, arm])[:24])
            for c in conditions for r in range(replicas) for arm in ARMS]


def summarize(rows, conditions, replicas, bootstrap=5000, seed=2026092602):
    expected = {(c['pair_id'], r) for c in conditions for r in range(replicas)}
    lookup = {c['pair_id']: c for c in conditions}
    paired = defaultdict(dict)
    for row in rows:
        key = (row['pair_id'], row['replica'])
        arm = row['comparison_arm']
        require(key in expected and arm in ARMS and arm not in paired[key], 'unknown/duplicate pair')
        require(row['split'] == 'independent_confirmation' and row['status'] == 'ok', 'incomplete confirmation')
        require(row['map_id'] == lookup[row['pair_id']]['case']['map_id'], 'map mismatch')
        require((row['success'] and row['stop'] == 'feasible' and row['final_conflicts'] == 0) or
                (not row['success'] and row['stop'] == 'node_budget' and row['final_conflicts'] > 0), 'terminal mismatch')
        paired[key][arm] = row
    require(set(paired) == expected, 'missing conditions')
    for values in paired.values():
        require(set(values) == set(ARMS), 'missing arm')
        require(len({r['initial_fingerprint'] for r in values.values()}) == 1, 'initial mismatch')
        require(len({r['rng_stream_id'] for r in values.values()}) == 1, 'unpaired RNG')
    maps = sorted({c['case']['map_id'] for c in conditions})
    grouped = {m: [v for v in paired.values() if v[ARMS[0]]['map_id'] == m] for m in maps}
    successes = {a: sum(v[a]['success'] for v in paired.values()) for a in ARMS}
    by_map = {m: dict(pairs=len(vs), success={a: sum(v[a]['success'] for v in vs) for a in ARMS})
              for m, vs in grouped.items()}
    contrasts = {}
    for baseline in ('raw_parent', 'dual16_sa', 'official_sa'):
        differences = {m: sum(int(v['raw_updated']['success'])-int(v[baseline]['success']) for v in vs)
                       for m, vs in grouped.items()}
        rng = random.Random(seed)
        draws = []
        for _ in range(bootstrap):
            selected = rng.choices(maps, k=len(maps))
            draws.append(sum(differences[m] for m in selected)/sum(len(grouped[m]) for m in selected))
        draws.sort()
        ci = [draws[int((len(draws)-1)*q)] for q in (.025, .975)]
        common = [v for v in paired.values() if v['raw_updated']['success'] and v[baseline]['success']]
        totals = {a: {k: sum(v[a][k] for v in common) for k in
                      ('generated', 'decisions', 'soc', 'makespan', 'wait_steps')}
                  for a in (baseline, 'raw_updated')}
        gains = [list(k) for k,v in paired.items() if v['raw_updated']['success'] and not v[baseline]['success']]
        losses = [list(k) for k,v in paired.items() if v[baseline]['success'] and not v['raw_updated']['success']]
        delta = (len(gains)-len(losses))/len(paired)
        contrasts[baseline] = dict(success_delta=delta, ci95=ci, gains=gains, losses=losses,
            common_success=len(common), common_totals=totals, map_net_success=differences,
            evidence=('positive_interval' if ci[0] > 0 else 'negative_interval' if ci[1] < 0 else
                      'positive_point_uncertain' if delta > 0 else 'no_positive_success_signal'))
    return dict(pairs=len(paired), maps=len(maps), episodes=len(rows), success=successes,
                by_map=by_map, contrasts=contrasts, bootstrap=bootstrap, bootstrap_unit='map',
                no_training=True, no_promotion=True, no_ttf=True,
                interpretation='fixed_work_same_family_new_maps_not_wall_clock_or_broad_OOD')
