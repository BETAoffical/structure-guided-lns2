"""Small transitive H32 rankers, isolated from production controllers."""
from itertools import combinations
import warnings

import numpy as np

from experiments.sa_paired_completion import require, validate_dataset
from experiments.sa_unbalanced_coverage import state_weights

METHODS = ("linear_difference", "coarse_rank_svm")


def completion_rates(state):
    return {c["candidate_id"]:sum(int(t["completed"]) for t in c["trials"])/len(c["trials"])
            for c in state["candidates"]}


def coarse_labels(rates):
    # Linear interpolation is the default in both registered NumPy environments.
    median,upper = np.quantile(list(rates.values()),[.5,.75])
    return {c:int(v>=median)+int(v>=upper) for c,v in rates.items()}


def fit_scaler(train,names,weights):
    x,w = [],[]
    for s in train:
        for c in sorted(s["candidates"],key=lambda c:c["candidate_id"]):
            x.append([c["features"][n] for n in names])
            w.append(weights[s["state_id"]]/len(s["candidates"]))
    x,w = np.asarray(x,float),np.asarray(w,float)
    mean = np.average(x,axis=0,weights=w)
    scale = np.sqrt(np.average((x-mean)**2,axis=0,weights=w))
    scale[scale<1e-8] = 1.
    return mean,scale


def prepare_matrix(data,held,method):
    validate_dataset(data)
    require(method in METHODS,"unknown model")
    require(all(t["completed"] is not None for s in data["states"] for c in s["candidates"] for t in c["trials"]),
            "censoring cannot be converted to failure")
    train = sorted((s for s in data["states"] if s["map_id"]!=held),key=lambda s:s["state_id"])
    require(train and any(s["map_id"]==held for s in data["states"]),"held map missing")
    names,weights = data["feature_names"],state_weights(data["states"],held)
    mean,scale = fit_scaler(train,names,weights)
    x,y,w,owners = [],[],[],[]
    constraints = {}
    for s in train:
        candidates = sorted(s["candidates"],key=lambda c:c["candidate_id"])
        rates = completion_rates(s)
        labels = coarse_labels(rates) if method=="coarse_rank_svm" else rates
        pairs = [(a,b) for a,b in combinations(candidates,2)
                 if method=="linear_difference" or labels[a["candidate_id"]]!=labels[b["candidate_id"]]]
        constraints[s["state_id"]] = len(pairs)*2
        for a,b in pairs:
            delta = (np.array([a["features"][n] for n in names])-np.array([b["features"][n] for n in names]))/scale
            target = labels[a["candidate_id"]]-labels[b["candidate_id"]]
            if method=="coarse_rank_svm":
                target = 1 if target>0 else -1
            for sign in (1,-1):
                x.append(sign*delta)
                y.append(sign*target)
                w.append(weights[s["state_id"]]/(2*len(pairs)))
                owners.append(s["state_id"])
    require(x and any(v!=0 for v in y),"no training contrasts")
    return dict(x=np.asarray(x),y=np.asarray(y),weights=w,owners=owners,mean=mean,scale=scale,
        names=list(names),train_ids=[s["state_id"] for s in train],train_maps=sorted({s["map_id"] for s in train}),
        state_weights=weights,constraints=constraints)


def fit(data,held,method,config):
    import sklearn
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import Ridge
    from sklearn.svm import LinearSVC
    require(sklearn.__version__=="1.5.0","frozen sklearn required")
    matrix = prepare_matrix(data,held,method)
    if method=="linear_difference":
        estimator = Ridge(alpha=config["ridge_alpha"],fit_intercept=False,solver="svd")
    else:
        estimator = LinearSVC(C=config["svm_C"],fit_intercept=False,dual=False,loss="squared_hinge",
            tol=config["svm_tol"],max_iter=config["svm_max_iter"],random_state=config["seed"])
    with warnings.catch_warnings():
        warnings.simplefilter("error",ConvergenceWarning)
        estimator.fit(matrix["x"],matrix["y"],sample_weight=matrix["weights"])
    coefficients = estimator.coef_.reshape(-1)
    require(np.isfinite(coefficients).all(),"invalid fitted weights")
    model = dict(schema="lns2.sa.linear_h32_diagnostic.v1",method=method,held=held,feature_names=matrix["names"],
        mean=matrix["mean"].tolist(),scale=matrix["scale"].tolist(),coefficients=coefficients.tolist(),
        train_ids=matrix["train_ids"],train_maps=matrix["train_maps"],state_weights=matrix["state_weights"],
        constraints=matrix["constraints"],config=dict(config),runtime_integration_allowed=False)
    rows = []
    for s in sorted(data["states"],key=lambda s:s["state_id"]):
        rows.append(dict(state_id=s["state_id"],held=s["map_id"]==held,**predict(model,s)))
    return dict(model=model,rows=rows)


def predict(model,state):
    names = model["feature_names"]
    candidates = sorted(state["candidates"],key=lambda c:c["candidate_id"])
    require(len(candidates)==len({c["candidate_id"] for c in candidates}) and
            state["anchor_id"] in {c["candidate_id"] for c in candidates},"candidate identity")
    require(all(set(c["features"])==set(names) for c in candidates),"feature schema")
    x = np.array([[c["features"][n] for n in names] for c in candidates],dtype=float)
    values = ((x-np.array(model["mean"]))/np.array(model["scale"])) @ np.array(model["coefficients"])
    require(np.isfinite(values).all(),"nonfinite score")
    scores = {c["candidate_id"]:float(v) for c,v in zip(candidates,values,strict=True)}
    return dict(selected=min(scores,key=lambda c:(-scores[c],c!=state["anchor_id"],c)),scores=scores)


def summarize(data,predictions,config):
    states = sorted(data["states"],key=lambda s:s["state_id"])
    require(set(predictions)=={s["state_id"] for s in states},"prediction coverage")
    rows = []
    for s in states:
        rates = completion_rates(s)
        pred = predictions[s["state_id"]]
        require(set(pred)==set(METHODS)|{"gbdt"},"model coverage")
        require(all(c in rates for c in pred.values()),"unknown selected candidate")
        rows.append(dict(state_id=s["state_id"],map_id=s["map_id"],selected=pred,
            rates={**{k:rates[c] for k,c in pred.items()},"frozen":rates[s["anchor_id"]],
                   "uniform":sum(rates.values())/len(rates),"oracle":max(rates.values())},
            sizes={k:len(next(c["agents"] for c in s["candidates"] if c["candidate_id"]==cid)) for k,cid in pred.items()}))
    maps = sorted({r["map_id"] for r in rows})
    methods = ["gbdt",*METHODS,"frozen","uniform","oracle"]
    per_map = {m:{k:float(np.mean([r["rates"][k] for r in rows if r["map_id"]==m])) for k in methods} for m in maps}
    means = {k:float(np.mean([v[k] for v in per_map.values()])) for k in methods}
    draws = np.random.default_rng(config["seed"]).integers(0,len(maps),(config["bootstrap"],len(maps)))
    contrasts = {}
    for model in METHODS:
        contrasts[model] = {}
        for base in ("frozen","uniform","gbdt"):
            delta = np.array([v[model]-v[base] for v in per_map.values()])
            contrasts[model][base] = dict(delta=float(delta.mean()),ci95=np.quantile(delta[draws].mean(1),[.025,.975]).tolist(),
                wins=int(sum(delta>0)),losses=int(sum(delta<0)),ties=int(sum(delta==0)))
    # Exploration signal only. No 5% gate, all-map dominance or CI exclusion rule.
    signals = [m for m in METHODS if means[m]>means["frozen"]]
    return dict(rows=rows,per_map=per_map,means=means,contrasts=contrasts,positive_point_estimates=signals,
        decision="development_signal_needs_closed_loop" if signals else "no_positive_development_signal",
        promotion=False,independent_confirmation=False,no_ttf=True)
