"""One budget-aligned policy update; independent of production controllers."""
from collections import Counter
from itertools import combinations

from experiments._common import json_fingerprint
from experiments.sa_paired_completion import require, pair_vector, MODEL_PARAMS, PairedCompletionModel

ARMS = ("frozen", "gbdt", "updated")
CENSORED = {"wall_safety", "incomplete_pp", "external_timeout"}


def split_sources(episodes, cfg):
    rows = [e for e in episodes if e["arm"] == cfg["source_arm"]]
    maps = sorted({r["map_id"] for r in rows}, key=lambda m: (json_fingerprint([cfg["seed"], m]), m))
    require(len(maps) == cfg["train_maps"]+cfg["validation_maps"], "map inventory changed")
    require(len({r["pair_id"] for r in rows}) == len(rows), "duplicate source")
    train, validation = maps[:cfg["train_maps"]], maps[cfg["train_maps"]:]
    selected = [min((r for r in rows if r["map_id"] == m),
                    key=lambda r: (json_fingerprint([cfg["seed"], r["pair_id"]]), r["pair_id"])) for m in train]
    return dict(train_maps=train, validation_maps=validation,
                training_episode_ids=[r["job_id"] for r in selected],
                validation_pair_ids=sorted(r["pair_id"] for r in rows if r["map_id"] in validation))


def work_stop(feasible, decision, generated, cfg):
    if feasible:
        return "feasible"
    if generated >= cfg["node_budget"]:
        return "node_budget"
    if decision >= cfg["max_decisions"]:
        return "decision_budget"
    return None


def budget_features(base, decision, generated, cfg):
    require(decision >= 0 and generated >= 0, "negative accumulated work")
    return dict(base, **{
        "budget.remaining_decision_fraction": max(0, cfg["max_decisions"]-decision)/cfg["max_decisions"],
        "budget.remaining_node_fraction": max(0, cfg["node_budget"]-generated)/cfg["node_budget"],
    })


def outcome(row):
    require(row["status"] in ("ok", "censored"), "unexplained branch error")
    stop = row["stop"]
    if stop in CENSORED:
        return None
    require(stop in ("feasible", "node_budget", "decision_budget"), "unknown stop")
    require(row["success"] == (stop == "feasible"), "success/stop mismatch")
    return int(row["success"])


def aggregate(root, results, cfg):
    ids = [c["candidate_id"] for c in root["candidates"]]
    expected = {(cid, arm, t) for cid in ids for arm in cfg["continuations"] for t in range(cfg["trials"])}
    keys = [(r["candidate_id"], r["arm"], r["trial"]) for r in results]
    require(len(keys) == len(set(keys)) and set(keys) == expected, "branch grid incomplete or duplicated")
    require(all(r["root_id"] == root["id"] for r in results), "root identity mismatch")
    values = {key: outcome(r) for key, r in zip(keys, results)}
    pairs = {(cid, t): [r for r in results if r["candidate_id"] == cid and r["trial"] == t]
             for cid in ids for t in range(cfg["trials"])}
    for group in pairs.values():
        if all(r["stop"] not in CENSORED for r in group):
            require(len({r["first_successor"] for r in group}) == 1, "continuations changed first action")
    rates = {arm: {cid: None if any(values[cid, arm, t] is None for t in range(cfg["trials"])) else
                         sum(values[cid, arm, t] for t in range(cfg["trials"]))/cfg["trials"] for cid in ids}
             for arm in cfg["continuations"]}
    complete = all(v is not None for arm in rates.values() for v in arm.values())
    return dict(root_id=root["id"], map_id=root["map_id"], rates=rates, complete=complete,
                censored=sum(v is None for v in values.values()),
                target={cid: cfg["reference_weight"]*rates["frozen"][cid]+(1-cfg["reference_weight"])*rates["gbdt"][cid]
                        for cid in ids} if complete else None)


def training_matrix(roots, aggregates, cfg, train_maps, validation_maps):
    require(set(train_maps).isdisjoint(validation_maps), "train/validation overlap")
    require({r["map_id"] for r in roots} == set(train_maps), "training coverage/role mismatch")
    lookup = {a["root_id"]: a for a in aggregates}
    require(len(lookup) == len(aggregates) == len(roots) and set(lookup) == {r["id"] for r in roots}, "label coverage")
    require(all(a["complete"] for a in aggregates), "censored collection: do not drop hard states or impute labels")
    names = sorted(roots[0]["features"][0])
    counts = Counter(r["map_id"] for r in roots)
    x, y, weights = [], [], []
    for r in sorted(roots, key=lambda r: r["id"]):
        require(lookup[r["id"]]["map_id"] == r["map_id"], "label map mismatch")
        candidates = [dict(c, features=f) for c, f in zip(r["candidates"], r["features"], strict=True)]
        require(all(sorted(c["features"]) == names for c in candidates), "feature schema")
        pairs = list(combinations(sorted(candidates, key=lambda c: c["candidate_id"]), 2))
        require(pairs, "no candidate comparison")
        state_weight = len(roots)/(len(counts)*counts[r["map_id"]])
        target = lookup[r["id"]]["target"]
        for a, b in pairs:
            for left, right in ((a, b), (b, a)):
                x.append(pair_vector(left, right, names))
                y.append(target[left["candidate_id"]]-target[right["candidate_id"]])
                weights.append(state_weight/(2*len(pairs)))
    require(any(v != 0 for v in y), "no completion contrast; no new fit")
    return dict(x=x, y=y, weights=weights, names=names)


def fit_once(matrix):
    import sklearn
    from sklearn.ensemble import HistGradientBoostingRegressor
    require(sklearn.__version__ == "1.5.0", "registered sklearn required")
    model = HistGradientBoostingRegressor(loss="squared_error", **MODEL_PARAMS).fit(
        matrix["x"], matrix["y"], sample_weight=matrix["weights"])
    return PairedCompletionModel(matrix["names"], model)
