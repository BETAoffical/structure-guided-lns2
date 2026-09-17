"""Frozen full-development rankers and exploratory closed-loop summaries."""
from collections import Counter
from itertools import combinations
import math

from experiments.sa_paired_completion import (
    MODEL_PARAMS, PairedCompletionModel, pair_vector, require, validate_dataset,
)
from experiments.sa_low_complexity_ranker import completion_rates, fit_scaler

ARMS = ("frozen", "gbdt", "linear")
CENSORED = {"wall_safety", "incomplete_pp", "external_timeout"}


def training_arrays(data, held=None):
    import numpy as np
    validate_dataset(data)
    train = sorted((s for s in data["states"] if s["map_id"] != held), key=lambda s:s["state_id"])
    require(train, "no training states")
    require(all(t["completed"] is not None for s in train for c in s["candidates"] for t in c["trials"]), "censored training")
    counts = Counter(s["map_id"] for s in train)
    weights = {s["state_id"]:len(train)/(len(counts)*counts[s["map_id"]]) for s in train}
    names = data["feature_names"]
    mean, scale = fit_scaler(train, names, weights)
    linear, pairwise, y, w = [], [], [], []
    for s in train:
        rates = completion_rates(s)
        pairs = list(combinations(sorted(s["candidates"], key=lambda c:c["candidate_id"]), 2))
        for a,b in pairs:
            for left,right in ((a,b),(b,a)):
                linear.append((np.array([left["features"][n] for n in names])-np.array([right["features"][n] for n in names]))/scale)
                pairwise.append(pair_vector(left,right,names))
                y.append(rates[left["candidate_id"]]-rates[right["candidate_id"]])
                w.append(weights[s["state_id"]]/(2*len(pairs)))
    require(any(v != 0 for v in y), "no training contrast")
    return dict(linear=linear, pairwise=pairwise, y=y, weights=w, mean=mean, scale=scale,
        train_ids=[s["state_id"] for s in train], train_maps=sorted(counts), state_weights=weights)


def fit_full(data):
    import sklearn
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    require(sklearn.__version__ == "1.5.0", "registered sklearn required")
    a = training_arrays(data)
    linear = Ridge(alpha=1., fit_intercept=False, solver="svd").fit(a["linear"],a["y"],sample_weight=a["weights"])
    gbdt = HistGradientBoostingRegressor(loss="squared_error", **MODEL_PARAMS).fit(a["pairwise"],a["y"],sample_weight=a["weights"])
    payload = dict(schema="lns2.sa.linear_h32_runtime_probe.v1",feature_names=list(data["feature_names"]),
        mean=a["mean"].tolist(),scale=a["scale"].tolist(),coefficients=linear.coef_.tolist(),
        train_ids=a["train_ids"],train_maps=a["train_maps"],state_weights=a["state_weights"],alpha=1.,
        production_allowed=False)
    return payload, PairedCompletionModel(list(data["feature_names"]),gbdt)


class LinearRanker:
    """Scalar inference in a fixed summation order; no sklearn at runtime."""
    def __init__(self, payload):
        self.payload = payload
        self.names = payload["feature_names"]
        require(payload["schema"] == "lns2.sa.linear_h32_runtime_probe.v1", "linear schema")
        require(self.names == sorted(set(self.names)), "feature names")
        require(all(len(payload[k]) == len(self.names) for k in ("mean","scale","coefficients")), "linear shape")
        require(all(math.isfinite(v) for k in ("mean","scale","coefficients") for v in payload[k])
                and all(v>0 for v in payload["scale"]), "invalid normalization")

    def rank(self,state):
        scores = {}
        for c in sorted(state["candidates"],key=lambda c:c["candidate_id"]):
            require(c["candidate_id"] not in scores and set(c["features"]) == set(self.names), "candidate/schema")
            values = [c["features"][n] for n in self.names]
            require(all(type(v) in (int,float) and math.isfinite(v) for v in values), "nonfinite features")
            scores[c["candidate_id"]] = math.fsum(((v-m)/s)*w for v,m,s,w in zip(
                values,self.payload["mean"],self.payload["scale"],self.payload["coefficients"],strict=True))
        require(scores and state["anchor_id"] in scores and all(math.isfinite(v) for v in scores.values()), "invalid scores")
        return dict(selected=min(scores,key=lambda c:(-scores[c],c!=state["anchor_id"],c)),
                    scores=scores,calibrated_confidence=False)


def choose(arm,candidates,anchor,models,features,known_ids):
    from experiments.sa_paired_closed_loop import choose as old_choose
    require(arm in ARMS, "unknown arm")
    # Reuse membership, single-candidate and anchor validation from the frozen runner.
    return old_choose("frozen" if arm=="frozen" else "paired",candidates,anchor,
        models["linear" if arm=="linear" else "gbdt"],features,known_ids,"unused",0)


def summarize(episodes,config):
    import numpy as np
    keys = sorted({r["pair_id"] for r in episodes})
    require(len(episodes)==len(keys)*len(ARMS), "incomplete cohort")
    pairs=[]
    for key in keys:
        records=[r for r in episodes if r["pair_id"]==key]
        require(len(records)==3 and {r["arm"] for r in records}==set(ARMS),"duplicate/missing arm")
        require(len({r["initial_fingerprint"] for r in records})==len({r["map_id"] for r in records})==1,"paired initial mismatch")
        require(all(r["success"]==(r["stop"]=="feasible") for r in records),"termination mismatch")
        pairs.append(dict(pair_id=key,map_id=records[0]["map_id"],arms={r["arm"]:r for r in records}))
    maps=sorted({p["map_id"] for p in pairs})
    draws=np.random.default_rng(config["master_seed"]).integers(0,len(maps),(config["bootstrap"],len(maps)))
    summaries={}
    for arm in ARMS:
        rs=[p["arms"][arm] for p in pairs]
        complete=[r for r in rs if r["success"]]
        summaries[arm]=dict(episodes=len(rs),success=len(complete),censored=sum(r["stop"] in CENSORED for r in rs),
            stops=dict(Counter(r["stop"] for r in rs)),decisions=sum(r["decisions"] for r in rs),
            generated=sum(r["generated"] for r in rs),revisits=sum(r["physical_revisits"] for r in rs),
            completion_curve={str(k):sum(r["success"] and r["decisions"]<=k for r in rs) for k in (32,64,128,256,512)},
            successful_decision_median=float(np.median([r["decisions"] for r in complete])) if complete else None)
    contrasts={}
    for challenger,base in (("linear","frozen"),("linear","gbdt"),("gbdt","frozen")):
        deltas=[]
        bounds=[]
        per_map={}
        common=[]
        wins=losses=0
        for m in maps:
            points=[]
            for p in (p for p in pairs if p["map_id"]==m):
                a,b=p["arms"][challenger],p["arms"][base]
                points.append(int(a["success"])-int(b["success"]))
                lo=(0 if a["stop"] in CENSORED else int(a["success"]))-(1 if b["stop"] in CENSORED else int(b["success"]))
                hi=(1 if a["stop"] in CENSORED else int(a["success"]))-(0 if b["stop"] in CENSORED else int(b["success"]))
                bounds.append((lo,hi))
                if a["stop"] not in CENSORED and b["stop"] not in CENSORED:
                    wins+=a["success"] and not b["success"]
                    losses+=b["success"] and not a["success"]
                if a["success"] and b["success"]:
                    common.append((a,b))
            per_map[m]=sum(points)/len(points)
            deltas.append(per_map[m])
        delta=np.asarray(deltas)
        denominator=sum(b["generated"] for a,b in common)
        contrasts[challenger+"_vs_"+base]=dict(observed_map_delta=float(delta.mean()),
            observed_ci95=np.quantile(delta[draws].mean(1),[.025,.975]).tolist(),per_map=per_map,
            map_wins=int(sum(delta>0)),map_losses=int(sum(delta<0)),map_ties=int(sum(delta==0)),
            known_pair_wins=int(wins),known_pair_losses=int(losses),
            censor_delta_bounds=[sum(b[i] for b in bounds)/len(bounds) for i in (0,1)],
            common_success=len(common),common_generated_ratio=sum(a["generated"] for a,b in common)/denominator if denominator else None)
    positive=[arm for arm in ("linear","gbdt") if summaries[arm]["success"]>summaries["frozen"]["success"]]
    work_signals=[arm for arm in ("linear","gbdt") if summaries[arm]["success"]==summaries["frozen"]["success"]
        and contrasts[arm+"_vs_frozen"]["common_generated_ratio"] is not None
        and contrasts[arm+"_vs_frozen"]["common_generated_ratio"]<1]
    censored=sum(v["censored"] for v in summaries.values())
    return dict(summary=summaries,contrasts=contrasts,positive_observed_completion=positive,
        positive_observed_work=work_signals,
        decision="resource_censored_inconclusive" if censored else ("bounded_positive_signal" if positive else
            ("bounded_work_signal" if work_signals else "no_positive_completion_or_work_signal")),
        no_ttf=True,production_changed=False,automatic_promotion=False,paired_cases=pairs)
