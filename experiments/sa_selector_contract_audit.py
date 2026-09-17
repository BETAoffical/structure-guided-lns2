"""Read-only label and candidate-contrast diagnostics, never a policy gate."""
from collections import Counter
from itertools import combinations
import math

from experiments.sa_paired_completion import require, validate_dataset


def feature_support(data, linear):
    validate_dataset(data)
    names = data["feature_names"]
    require(names == linear["feature_names"], "linear feature identity")
    absolute = {n: [math.inf, -math.inf] for n in names}
    contrast = dict.fromkeys(names, 0.)
    varying = Counter()
    for state in data["states"]:
        rows = [c["features"] for c in state["candidates"]]
        for n in names:
            lo, hi = min(r[n] for r in rows), max(r[n] for r in rows)
            absolute[n] = [min(absolute[n][0], lo), max(absolute[n][1], hi)]
            contrast[n] = max(contrast[n], hi-lo)
            varying[n] += hi != lo
    never = [n for n in names if not varying[n]]
    return dict(names=names, absolute=absolute, max_abs_contrast=contrast,
                varying_states=dict(varying), never_varying=never,
                never_varying_coefficients={n: linear["coefficients"][names.index(n)] for n in never},
                history_features=[n for n in names if n.startswith("history.")],
                rare_varying={n: varying[n] for n in names if 0 < varying[n] < 4})


def outside(value, lower, upper):
    tolerance = 1e-12 * max(1., abs(lower), abs(upper))
    return value < lower-tolerance or value > upper+tolerance


def contrast_diagnostic(features, ids, anchor, selected, support):
    require(ids and len(ids) == len(set(ids)) and len(ids) == len(features), "candidate identities")
    require(anchor in ids and selected in ids, "missing anchor/selection")
    names = support["names"]
    require(all(set(f) == set(names) and all(math.isfinite(v) for v in f.values()) for f in features),
            "invalid feature schema or values")
    a, b = features[ids.index(selected)], features[ids.index(anchor)]
    violations = [n for n in names if outside(abs(a[n]-b[n]), 0, support["max_abs_contrast"][n])]
    absolute = sum(outside(v, *support["absolute"][n]) for f in features for n, v in f.items())
    # This is a coordinate-wise support diagnostic, not a causal or joint-OOD test.
    return dict(changed=selected != anchor, contrast_outside=violations,
                previously_unvarying=[n for n in violations if support["max_abs_contrast"][n] == 0],
                absolute_outside_fraction=absolute/(len(features)*len(names)))


def label_curves(data, predictions):
    validate_dataset(data)
    require(data["horizon"] == 32, "only registered H32 labels")
    require(all(t["completed"] is not None for s in data["states"] for c in s["candidates"] for t in c["trials"]),
            "censored trials cannot become incomplete labels")
    require(set(predictions) == {s["state_id"] for s in data["states"]}, "prediction coverage")
    rows, ties, hidden, saturated = [], 0, 0, []
    for s in data["states"]:
        curves = {c["candidate_id"]: [sum(t["completed"] and t["steps"] <= h for t in c["trials"])/data["trial_count"]
                                      for h in range(1, 33)] for c in s["candidates"]}
        for a, b in combinations(curves.values(), 2):
            ties += a[-1] == b[-1]
            hidden += a[-1] == b[-1] and a != b
        selections = dict(predictions[s["state_id"]], frozen=s["anchor_id"])
        require(all(c in curves for c in selections.values()), "unknown selected candidate")
        rates = {k: {str(h): curves[c][h-1] for h in (8, 16, 32)} for k, c in selections.items()}
        rates["uniform"] = {str(h): sum(c[h-1] for c in curves.values())/len(curves) for h in (8, 16, 32)}
        rows.append(dict(state_id=s["state_id"], map_id=s["map_id"], rates=rates))
        if all(c[-1] == 1 for c in curves.values()):
            means = [sum(t["steps"] for t in c["trials"])/data["trial_count"] for c in s["candidates"]]
            saturated.append(dict(state_id=s["state_id"], min_mean_steps=min(means), max_mean_steps=max(means)))
    maps = sorted({s["map_id"] for s in rows})
    methods = sorted(rows[0]["rates"])
    per_map = {m: {k: {str(h): sum(r["rates"][k][str(h)] for r in rows if r["map_id"] == m)/sum(r["map_id"] == m for r in rows)
                          for h in (8, 16, 32)} for k in methods} for m in maps}
    macro = {k: {str(h): sum(v[k][str(h)] for v in per_map.values())/len(maps) for h in (8, 16, 32)} for k in methods}
    return dict(rows=rows, per_map=per_map, map_macro=macro, tied_h32_pairs=ties,
                tied_h32_pairs_with_different_completion_curves=hidden, all_success_states=saturated,
                horizon_selected=False, new_labels_for_training=False)
