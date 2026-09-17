"""Isolated frozen paired-completion inference and bounded evaluation rules."""
import random

from experiments._common import json_fingerprint
from experiments.sa_paired_completion import (
    MODEL_PARAMS, PairedCompletionModel, paired_labels, pair_vector, require,
    validate_dataset,
)
from lns2_selector.runtime.portable_scalar import load_portable_scalar_model
from lns2_selector.runtime.fingerprints import semantic_fingerprint
from experiments.sa_paired_sampling import choose_candidates


def subset(pool, anchor, key, seed, limit):
    if len(pool) == 1:
        c = pool[0]
        require(c["candidate_id"] == anchor and c["agents"] and len(set(c["agents"])) == len(c["agents"]), "invalid single candidate")
        return [anchor]
    return choose_candidates(pool, anchor, key, seed, limit)


def fit_complete(data):
    """Same old-16 recipe; all states enter training, with total weight one each."""
    import sklearn
    from sklearn.ensemble import HistGradientBoostingRegressor
    validate_dataset(data)
    require(sklearn.__version__ == "1.5.0", "registered sklearn required")
    require(len(data["states"]) == 16, "only the frozen old sixteen states may train")
    x, y, w = [], [], []
    for state in sorted(data["states"], key=lambda s: s["state_id"]):
        lookup = {c["candidate_id"]: c for c in state["candidates"]}
        pairs = paired_labels(state)
        for p in pairs:
            require(not p["censored_trials"], "censored training labels")
            a, b = lookup[p["left"]], lookup[p["right"]]
            for left, right, target in ((a,b,p["mean_difference"]), (b,a,-p["mean_difference"])):
                x.append(pair_vector(left, right, data["feature_names"]))
                y.append(target)
                w.append(1/(2*len(pairs)))
    model = HistGradientBoostingRegressor(loss="squared_error", **MODEL_PARAMS).fit(x,y,sample_weight=w)
    return PairedCompletionModel(data["feature_names"], model)


def portable_payload(model):
    from lns2_selector.training.tree_utils import histogram_trees
    names = ["delta."+n for n in model.feature_names] + ["mean."+n for n in model.feature_names]
    p = dict(name="paired_completion_old16", profile="paired_completion", feature_names=names,
             baseline=float(model.estimator._baseline_prediction[0,0]),
             trees=histogram_trees(model.estimator), transform="identity", input_precision="float64")
    return dict(schema="lns2.portable_scalar_hist_gbdt.v1", **p,
                semantic_fingerprint=semantic_fingerprint(p))


class VectorEstimator:
    def __init__(self, payload, native=True):
        self.model = load_portable_scalar_model(payload)
        if not native:
            self.model.native_predictor = None

    def predict(self, vectors):
        return self.model.predict([dict(feature_profile=self.model.profile,
            feature_names=self.model.feature_names, feature_values=v) for v in vectors])


def portable_model(payload, names, native=True):
    require(payload["feature_names"] == ["delta."+n for n in names]+["mean."+n for n in names], "paired schema mismatch")
    return PairedCompletionModel(names, VectorEstimator(payload, native))


def choose(arm, candidates, anchor, model, features, known_ids, key, seed):
    ids = [c["candidate_id"] for c in candidates]
    require(ids and len(ids) == len(set(ids)) and anchor in ids, "candidate identity")
    require(arm in {"frozen", "uniform", "paired"}, "unknown arm")
    require(all(c["agents"] and set(c["agents"]) <= set(known_ids) and len(c["agents"])==len(set(c["agents"])) for c in candidates), "invalid candidate agents")
    require(len({tuple(sorted(c["agents"])) for c in candidates})==len(candidates), "duplicate physical candidate")
    state = dict(anchor_id=anchor, agent_ids=known_ids, candidates=[
        dict(candidate_id=c["candidate_id"], agents=c["agents"], features=f)
        for c,f in zip(candidates,features,strict=True)])
    if len(candidates) == 1:
        return dict(selected=anchor, scores={anchor:0.}, calibrated_confidence=False)
    ranking = model.rank(state)
    if arm == "frozen":
        ranking["selected"] = anchor
    elif arm == "uniform":
        # A separate stream: selecting a candidate must not consume PP/SA RNG.
        rng = random.Random(int(json_fingerprint([key,seed,"uniform"]),16))
        ranking["selected"] = rng.choice(sorted(ids))
    return ranking


def stop_reason(state, decisions, generated, elapsed, cfg):
    if state["feasible"]:
        return "feasible"
    if generated >= cfg["node_budget"]:
        return "node_budget"
    if decisions >= cfg["max_decisions"]:
        return "decision_budget"
    if elapsed >= cfg["episode_seconds"]:
        return "wall_safety"
    return None


def gate(contrasts, node_ratio, censored, cfg):
    tests = dict(
        no_resource_censoring=censored == 0,
        completion_gain=contrasts["frozen"]["delta"] >= cfg["minimum_success_gain"],
        completion_uncertainty=contrasts["frozen"]["ci95"][0] >= 0,
        map_nonloss=contrasts["frozen"]["map_nonlosses"] >= cfg["minimum_map_nonlosses"],
        beats_random=contrasts["uniform"]["delta"] > 0 and contrasts["uniform"]["ci95"][0] >= 0,
        common_success_work=node_ratio is not None and node_ratio <= cfg["maximum_common_success_node_ratio"],
    )
    return dict(tests=tests, passed=all(tests.values()), automatic_promotion=False,
                decision="eligible_for_separate_serial_ttf" if all(tests.values()) else "do_not_promote_closed_loop")


def contrast(rows, baseline, bootstrap, seed):
    import numpy as np
    maps = sorted({r["map_id"] for r in rows})
    values = np.array([sum(r["success"]["paired"]-r["success"][baseline] for r in rows if r["map_id"]==m)
        / sum(r["map_id"]==m for r in rows) for m in maps])
    draws = np.random.default_rng(seed).integers(0,len(maps),(bootstrap,len(maps)))
    return dict(delta=float(values.mean()), ci95=np.quantile(values[draws].mean(axis=1),[.025,.975]).tolist(),
                map_nonlosses=int(sum(values>=0)), map_wins=int(sum(values>0)), map_losses=int(sum(values<0)))
