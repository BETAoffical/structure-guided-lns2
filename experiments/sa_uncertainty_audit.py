"""Map-bootstrap disagreement is a hypothesis, not calibrated confidence."""
from collections import Counter
from itertools import combinations
import math
import random

from experiments._common import json_fingerprint
from experiments.sa_paired_completion import (
    MODEL_PARAMS, PairedCompletionModel, pair_vector, require, validate_dataset,
)


def map_draws(data, held, member, cfg):
    maps = sorted({s["map_id"] for s in data["states"]})
    require(held in maps and len(maps) >= 3, "invalid held map")
    require(type(member) is int and -1 <= member < cfg["members"], "invalid member")
    train = [m for m in maps if m != held]
    if member == -1:
        return train
    rng = random.Random(json_fingerprint([cfg["seed"], held, member]))
    return [rng.choice(train) for _ in train]


def training_matrix(data, held, member, cfg):
    validate_dataset(data)
    draws = map_draws(data, held, member, cfg)
    counts = Counter(draws)
    train = sorted((s for s in data["states"] if s["map_id"] != held), key=lambda s: s["state_id"])
    sizes = Counter(s["map_id"] for s in train)
    rows = [s for s in train if counts[s["map_id"]]]
    x, y, weights, owners = [], [], [], []
    state_weights = {}
    for s in rows:
        require(all(t["completed"] is not None for c in s["candidates"] for t in c["trials"]),
                "censored training labels")
        rates = rates_for(s)
        state_weights[s["state_id"]] = len(train) * counts[s["map_id"]] / (len(draws) * sizes[s["map_id"]])
        pairs = list(combinations(sorted(s["candidates"], key=lambda c: c["candidate_id"]), 2))
        for a, b in pairs:
            for left, right in ((a, b), (b, a)):
                x.append(pair_vector(left, right, data["feature_names"]))
                y.append(rates[left["candidate_id"]] - rates[right["candidate_id"]])
                weights.append(state_weights[s["state_id"]] / (2 * len(pairs)))
                owners.append(s["state_id"])
    require(x, "empty training fold")
    return dict(x=x, y=y, weights=weights, owners=owners, state_weights=state_weights,
                map_draws=draws, train_ids=[s["state_id"] for s in rows],
                train_maps=sorted(counts), held=held)


def rates_for(state, trial_ids=None):
    result = {}
    for c in state["candidates"]:
        rows = c["trials"] if trial_ids is None else [t for t in c["trials"] if t["trial"] in trial_ids]
        require(rows and all(type(t["completed"]) is bool for t in rows), "missing or censored evaluation")
        require(trial_ids is None or {t["trial"] for t in rows} == set(trial_ids), "trial half missing")
        result[c["candidate_id"]] = sum(t["completed"] for t in rows) / len(rows)
    return result


def inference_state(state):
    return {k: state[k] for k in ("state_id", "anchor_id", "agent_ids")} | dict(
        candidates=[{k: c[k] for k in ("candidate_id", "agents", "features")} for c in state["candidates"]])


def fit_member(data, held, member, cfg):
    import sklearn
    from sklearn.ensemble import HistGradientBoostingRegressor
    require(sklearn.__version__ == "1.5.0", "registered sklearn required")
    matrix = training_matrix(data, held, member, cfg)
    estimator = HistGradientBoostingRegressor(loss="squared_error", **MODEL_PARAMS).fit(
        matrix["x"], matrix["y"], sample_weight=matrix["weights"])
    model = PairedCompletionModel(data["feature_names"], estimator)
    predictions = [dict(state_id=s["state_id"], **model.rank(inference_state(s)))
                   for s in sorted(data["states"], key=lambda s: s["state_id"]) if s["map_id"] == held]
    return dict(held=held, member=member, map_draws=matrix["map_draws"], train_ids=matrix["train_ids"],
                train_maps=matrix["train_maps"], state_weights=matrix["state_weights"], predictions=predictions,
                constant_targets=len(set(matrix["y"])) == 1, sklearn=sklearn.__version__)


def ensemble_prediction(state, predictions):
    ids = sorted(c["candidate_id"] for c in state["candidates"])
    require(predictions and state["anchor_id"] in ids, "invalid ensemble state")
    for r in predictions:
        require(r["state_id"] == state["state_id"] and sorted(r["scores"]) == ids, "prediction identity")
        require(all(math.isfinite(v) for v in r["scores"].values()), "nonfinite scores")
        expected = min(ids, key=lambda c: (-r["scores"][c], c != state["anchor_id"], c))
        require(r["selected"] == expected, "member selection mismatch")
    scores = {c: math.fsum(r["scores"][c] for r in predictions) / len(predictions) for c in ids}
    selected = min(ids, key=lambda c: (-scores[c], c != state["anchor_id"], c))
    votes = Counter(r["selected"] for r in predictions)
    probabilities = {c: votes[c] / len(predictions) for c in ids}
    return dict(selected=selected, scores=scores, vote_probabilities=probabilities,
                agreement=probabilities[selected], calibrated_confidence=False)


def state_result(state, members, baseline, cfg):
    e = ensemble_prediction(state, members)
    require(baseline["state_id"] == state["state_id"] and baseline["selected"] in e["scores"], "baseline identity")
    rates = rates_for(state)
    chosen, anchor = e["selected"], state["anchor_id"]
    methods = dict(frozen=rates[anchor], uniform=math.fsum(rates.values())/len(rates),
                   gbdt=rates[baseline["selected"]], ensemble_mean=rates[chosen],
                   vote_expectation=math.fsum(e["vote_probabilities"][c]*rates[c] for c in rates))
    halves = []
    for trials in (range(4), range(4, 8)):
        values = rates_for(state, trials)
        halves.append(values[chosen] - values[anchor])
    return dict(state_id=state["state_id"], map_id=state["map_id"], decision=state["decision"],
                prediction=e, rates=rates, methods=methods, high_agreement=e["agreement"] >= cfg["high_agreement"],
                changed=chosen != anchor, anchor=anchor, anchor_gain=methods["ensemble_mean"]-methods["frozen"],
                half_gains=halves, repeated_positive=all(g > 0 for g in halves),
                repeated_negative=all(g < 0 for g in halves))


def grouped_mean(rows, fn):
    maps = sorted({r["map_id"] for r in rows})
    values = [math.fsum(fn(r) for r in rows if r["map_id"] == m) /
              sum(r["map_id"] == m for r in rows) for m in maps]
    return math.fsum(values)/len(values) if values else None


def agreement_group(rows):
    return dict(states=len(rows), maps=len({r["map_id"] for r in rows}),
                anchor_gain=grouped_mean(rows, lambda r: r["anchor_gain"]),
                harmful_fraction=grouped_mean(rows, lambda r: r["anchor_gain"] < 0),
                helpful_fraction=grouped_mean(rows, lambda r: r["anchor_gain"] > 0),
                changed=sum(r["changed"] for r in rows),
                repeated_positive=sum(r["repeated_positive"] for r in rows),
                repeated_negative=sum(r["repeated_negative"] for r in rows))


def summarize(rows, cfg):
    import numpy as np
    require(len({r["state_id"] for r in rows}) == len(rows) == cfg["states"], "state coverage")
    maps = sorted({r["map_id"] for r in rows})
    require(len(maps) == cfg["maps"], "map coverage")
    methods = sorted(rows[0]["methods"])
    per_map = {m: {k: grouped_mean([r for r in rows if r["map_id"] == m], lambda r: r["methods"][k])
                   for k in methods} for m in maps}
    means = {k: math.fsum(per_map[m][k] for m in maps)/len(maps) for k in methods}
    draws = np.random.default_rng(cfg["seed"]).integers(0, len(maps), (cfg["bootstrap"], len(maps)))
    contrasts = {}
    for method in ("ensemble_mean", "vote_expectation"):
        for base in ("frozen", "uniform", "gbdt"):
            differences = np.asarray([per_map[m][method]-per_map[m][base] for m in maps])
            contrasts[method+"_vs_"+base] = dict(gain=float(differences.mean()),
                ci95=np.quantile(differences[draws].mean(1), [.025, .975]).tolist(),
                wins=int(sum(differences > 0)), losses=int(sum(differences < 0)), ties=int(sum(differences == 0)))
    high = agreement_group([r for r in rows if r["high_agreement"]])
    low = agreement_group([r for r in rows if not r["high_agreement"]])
    return dict(means=means, per_map=per_map, contrasts=contrasts,
                agreement_groups=dict(high=high, low=low),
                positive_point_estimates=[k for k in ("ensemble_mean", "vote_expectation")
                    if means[k] > means["frozen"] and means[k] > means["uniform"]],
                decision="development_audit_only_no_automatic_collection",
                confidence_calibrated=False, closed_loop_tested=False, production_changed=False,
                new_solver_calls=0, no_ttf=True, rows=rows)
