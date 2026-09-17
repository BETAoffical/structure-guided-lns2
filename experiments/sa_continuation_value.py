"""Policy-conditional candidate value diagnostics; no training or promotion."""
from itertools import combinations

from experiments._common import json_fingerprint
from experiments.sa_paired_completion import require


def select_roots(rows, ranges, seed, excluded_ids):
    """One per map, alternating early/continuing; never inspect outcomes."""
    maps = sorted({r["map_id"] for r in rows})
    require(len(maps) == 8, "eight maps required")
    selected = []
    for index, map_id in enumerate(maps):
        phase = "early" if index % 2 == 0 else "continuing"
        eligible = [r for r in rows if r["map_id"] == map_id and r["phase"] == phase and
                    r["id"] not in excluded_ids and
                    all(ranges[k][0] <= r[k] <= ranges[k][1] for k in ranges)]
        require(eligible, "no eligible pre-action state: " + map_id)
        selected.append(min(eligible, key=lambda r:(json_fingerprint([seed, r["id"]]), r["id"])))
    return [r["id"] for r in selected]


def mean(values):
    require(values, "empty observations")
    return sum(values) / len(values)


def best(rates, anchor):
    return min(rates, key=lambda c:(-rates[c], c != anchor, c))


def winner_set(rates):
    return {c for c in rates if rates[c] == max(rates.values())}


def state_summary(values, anchor, model_choice):
    """Each arm has all four candidates and eight matched trials, including ties."""
    require(set(values) == {"frozen", "paired"}, "two continuations required")
    ids = sorted(values["frozen"])
    require(len(ids) == 4 and set(ids) == set(values["paired"]) and anchor in ids and model_choice in ids,
            "candidate coverage")
    require(all(len(values[a][c]) == 8 and all(v in (0, 1, None) for v in values[a][c])
                for a in values for c in ids), "eight binary/censored trials required")
    missing = sum(v is None for a in values for c in ids for v in values[a][c])
    if missing:
        return dict(complete=False, censored=missing, rates=None, stable_reversals=[], pairs=[], cross_half=None)
    rates = {a:{c:mean(values[a][c]) for c in ids} for a in values}
    halves = [{a:{c:mean(values[a][c][offset:offset+4]) for c in ids} for a in values} for offset in (0, 4)]
    pairs = []
    for left, right in combinations(ids, 2):
        delta = {a:rates[a][left]-rates[a][right] for a in values}
        half_delta = [{a:h[a][left]-h[a][right] for a in values} for h in halves]
        reverse = delta["frozen"] * delta["paired"] < 0
        stable = reverse and all(h[a] * delta[a] > 0 for h in half_delta for a in values)
        pairs.append(dict(left=left, right=right, differences=delta, half_differences=half_delta,
                          interaction=delta["paired"]-delta["frozen"], reversed=reverse, stable=stable))
    sets = {a:winner_set(rates[a]) for a in values}
    jaccard = len(sets["frozen"] & sets["paired"]) / len(sets["frozen"] | sets["paired"])
    cross = []
    for fit in (0, 1):
        held = 1-fit
        old_choice, new_choice = best(halves[fit]["frozen"], anchor), best(halves[fit]["paired"], anchor)
        cross.append(dict(fit_half=fit, frozen_label_choice=old_choice, paired_label_choice=new_choice,
            paired_test_old_choice=halves[held]["paired"][old_choice],
            paired_test_new_choice=halves[held]["paired"][new_choice],
            gain=halves[held]["paired"][new_choice]-halves[held]["paired"][old_choice]))
    return dict(complete=True, censored=0, rates=rates, pairs=pairs,
        stable_reversals=[p for p in pairs if p["stable"]], best_sets={a:sorted(s) for a,s in sets.items()},
        best_jaccard=jaccard, cross_half=cross, cross_half_gain=mean([c["gain"] for c in cross]),
        fixed_choices={a:dict(anchor=rates[a][anchor], model=rates[a][model_choice], uniform=mean(list(rates[a].values())),
                             empirical_oracle=max(rates[a].values())) for a in values},
        candidate_mean_change=mean([rates["paired"][c]-rates["frozen"][c] for c in ids]))


def summarize(rows, seed, bootstrap):
    import numpy as np
    require(len(rows) == 8 and len({r["map_id"] for r in rows}) == 8, "one state per map required")
    if not all(r["complete"] for r in rows):
        return dict(decision="incomplete_evidence_resource_censoring", censored=sum(r["censored"] for r in rows), promotion=False)
    rows = sorted(rows, key=lambda r:r["map_id"])
    gains = np.asarray([r["cross_half_gain"] for r in rows])
    draws = np.random.default_rng(seed).integers(0, 8, (bootstrap, 8))
    stable = sum(bool(r["stable_reversals"]) for r in rows)
    any_reversed = sum(any(p["reversed"] for p in r["pairs"]) for r in rows)
    decision = ("repeatable_candidate_order_shift" if stable >= 2 else
                "weak_or_noisy_order_shift" if any_reversed else "no_observed_strict_order_reversal")
    return dict(decision=decision, promotion=False, strict_reversal_states=any_reversed,
        split_half_stable_reversal_states=stable, mean_best_jaccard=mean([r["best_jaccard"] for r in rows]),
        cross_half_gain=float(gains.mean()), cross_half_ci95=np.quantile(gains[draws].mean(axis=1), [.025, .975]).tolist(),
        cross_half_positive_maps=int(sum(gains > 0)), cross_half_negative_maps=int(sum(gains < 0)),
        mean_candidate_completion_change=mean([r["candidate_mean_change"] for r in rows]),
        selection_rates={a:{k:mean([r["fixed_choices"][a][k] for r in rows])
            for k in ("anchor", "model", "uniform", "empirical_oracle")} for a in ("frozen", "paired")})
