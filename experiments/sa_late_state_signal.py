"""Candidate discrimination under a frozen continuation, without model training."""
from itertools import combinations
import random

from experiments._common import json_fingerprint
from experiments.sa_paired_completion import require
from experiments.sa_policy_aligned_update import outcome


def assign_phases(train_maps, validation_maps, cfg):
    require(len(train_maps) == len(set(train_maps)) == 6, "six distinct training maps required")
    require(set(train_maps).isdisjoint(validation_maps), "held-out map leakage")
    require(cfg["root_decisions"] == [32, 64, 96], "registered phases changed")
    maps = sorted(train_maps, key=lambda m: (json_fingerprint([cfg["phase_seed"], m]), m))
    return {m:cfg["root_decisions"][i % 3] for i,m in enumerate(maps)}


def trial_rates(root, rows, trial_ids):
    ids = [c["candidate_id"] for c in root["candidates"]]
    keys = [(r["candidate_id"], r["trial"]) for r in rows]
    require(len(keys) == len(set(keys)) and set(keys) == {(c,t) for c in ids for t in trial_ids}, "trial grid mismatch")
    require(all(r["root_id"] == root["id"] and r["arm"] == "frozen" for r in rows), "root or continuation changed")
    values = {key:outcome(r) for key,r in zip(keys, rows)}
    return {c:None if any(values[c,t] is None for t in trial_ids) else
            sum(values[c,t] for t in trial_ids)/len(trial_ids) for c in ids}


def choose_discovery(root, rows, cfg):
    rates = trial_rates(root, rows, cfg["discovery_trials"])
    selected = None if any(v is None for v in rates.values()) else min(
        rates, key=lambda c: (-rates[c], c != root["anchor_id"], c))
    return dict(root_id=root["id"], rates=rates, selected=selected, complete=selected is not None)


def describe(root, choice, rows, cfg):
    test = trial_rates(root, rows, cfg["confirmation_trials"])
    require(choice["root_id"] == root["id"], "choice root mismatch")
    complete = choice["complete"] and all(v is not None for v in test.values())
    result = dict(root_id=root["id"], map_id=root["map_id"], decision=root["decision"],
        initial_conflicts=root["state"]["num_of_colliding_pairs"], discovery=choice["rates"],
        confirmation=test, complete=complete)
    if not complete:
        return result
    anchor, selected = root["anchor_id"], choice["selected"]
    d = choice["rates"]
    best = {c for c in test if test[c] == max(test.values())}
    first_best = {c for c in d if d[c] == max(d.values())}
    pairs = [(d[a]-d[b], test[a]-test[b]) for a,b in combinations(sorted(test), 2)]
    result.update(anchor=anchor, selected=selected, changed=selected != anchor,
        discovery_gain=d[selected]-d[anchor], confirmation_gain=test[selected]-test[anchor],
        confirmed_strict_improvement=d[selected] > d[anchor] and test[selected] > test[anchor],
        uniform=sum(test.values())/len(test), sample_oracle=max(test.values()),
        old_model=test[root["source_event"]["ranking"]["selected"]],
        updated_model=test[root["updated_ranking"]["selected"]],
        all_zero=all(v == 0 for v in [*d.values(), *test.values()]),
        all_one=all(v == 1 for v in [*d.values(), *test.values()]),
        discovery_nonflat=len(set(d.values())) > 1, confirmation_nonflat=len(set(test.values())) > 1,
        best_set_jaccard=len(best & first_best)/len(best | first_best),
        informative_pairs=sum(x != 0 or y != 0 for x,y in pairs),
        repeated_strict_pairs=sum(x*y > 0 for x,y in pairs))
    return result


def summarize(rows, cfg):
    require(len(rows) == 6 and len({r["map_id"] for r in rows}) == 6, "one root per training map required")
    if not all(r["complete"] for r in rows):
        return dict(decision="resource_censored_inconclusive", complete=False, model_fits=0)
    gains = [r["confirmation_gain"] for r in rows]
    mean = sum(gains)/len(gains)
    rng = random.Random(cfg["phase_seed"])
    draws = sorted(sum(rng.choice(gains) for _ in gains)/len(gains) for _ in range(cfg["bootstrap"]))
    ci = [draws[int((len(draws)-1)*q)] for q in (.025, .975)]
    repeated = sum(r["confirmed_strict_improvement"] for r in rows)
    decision = ("development_signal_not_confirmation" if mean > 0 and repeated >= 2 else
                "positive_but_weak_signal" if mean > 0 else "no_replicated_selection_gain")
    completion = {"anchor":sum(r["confirmation"][r["anchor"]] for r in rows)/6,
                  "discovery_selected":sum(r["confirmation"][r["selected"]] for r in rows)/6}
    completion.update({k:sum(r[k] for r in rows)/6 for k in ("uniform", "old_model", "updated_model", "sample_oracle")})
    return dict(decision=decision, complete=True, confirmation_gain=mean, map_bootstrap_ci95=ci,
        completion=completion, positive_maps=sum(x > 0 for x in gains), negative_maps=sum(x < 0 for x in gains),
        equal_maps=sum(x == 0 for x in gains), repeated_improvement_maps=repeated,
        floor_states=sum(r["all_zero"] for r in rows), ceiling_states=sum(r["all_one"] for r in rows),
        changed_choices=sum(r["changed"] for r in rows),
        informative_pairs=sum(r["informative_pairs"] for r in rows),
        repeated_strict_pairs=sum(r["repeated_strict_pairs"] for r in rows),
        model_fits=0, automatic_promotion=False, no_ttf=True, independent_confirmation=False)
