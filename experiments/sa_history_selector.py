"""Causal history features and bounded transition targets, not a tail oracle."""
from collections import deque
import math

from experiments._common import json_fingerprint


def edges(state):
    return {tuple(sorted(map(int, pair))) for pair in state["conflict_edges"]}


def members(candidate):
    values = tuple(sorted(candidate["agents"]))
    if not values or len(set(values)) != len(values):
        raise ValueError("empty or duplicate neighborhood")
    return values


def choose_candidates(pool, selected, seed, limit=6):
    """Source score plus size-stratified outcome-blind coverage; no new policy."""
    if not 0 <= selected < len(pool):
        raise ValueError("selected index")
    if len({members(c) for c in pool}) != len(pool):
        raise ValueError("duplicate source sets")
    chosen = [selected]
    score_order = sorted(range(len(pool)), key=lambda i: (-pool[i]["score"], pool[i]["candidate_id"]))
    def take(indices):
        for i in indices:
            if i not in chosen and len(chosen) < limit:
                chosen.append(i)
                break
    take(score_order)
    hashed = sorted(range(len(pool)), key=lambda i: json_fingerprint([seed, pool[i]["candidate_id"]]))
    for size in (4, 8, 16):
        take(i for i in hashed if len(members(pool[i])) == size)
    for i in hashed:
        take([i])
    return chosen


class History:
    def __init__(self, initial, window=32):
        self.window = window
        self.recent = deque(maxlen=window)
        self.ages = {e: 0 for e in edges(initial)}
        self.best = initial["num_of_colliding_pairs"]
        self.since_best = 0
        self.decision = 0

    @staticmethod
    def context(state, selected):
        selected = set(selected)
        return json_fingerprint({"outside": [(a["id"], a["path"]) for a in sorted(state["agents"], key=lambda a: a["id"]) if a["id"] not in selected],
                                 "affected": sorted(e for e in edges(state) if selected.intersection(e))})

    def features(self, state, candidate, temperature):
        ids = members(candidate)
        known = {a["id"] for a in state["agents"]}
        if not set(ids) <= known:
            raise ValueError("unknown candidate agent")
        incident = [e for e in edges(state) if set(ids).intersection(e)]
        ages = [min(self.window, self.ages[e]) / self.window for e in incident]
        same = [r for r in self.recent if r["members"] == ids]
        ctx = self.context(state, ids)
        valid = [r for r in same if r["context"] == ctx]
        overlap = [len(set(ids).intersection(r["members"])) / len(set(ids).union(r["members"])) for r in self.recent]
        f = {"history.window_fill": len(self.recent) / self.window,
             "history.same_set_count": len(same) / self.window,
             "history.valid_context_count": len(valid) / self.window,
             "history.stale_context_count": (len(same) - len(valid)) / self.window,
             "history.unknown_context": float(not valid),
             "history.overlap_mean": sum(overlap) / max(1, len(overlap)),
             "history.since_best": min(self.since_best, self.window) / self.window,
             "history.best_gap": (state["num_of_colliding_pairs"] - self.best) / max(1, state["num_of_colliding_pairs"]),
             "history.incident_age_mean": sum(ages) / max(1, len(ages)),
             "history.incident_age_max": max(ages, default=0.0),
             "history.old_incident_fraction": sum(v == 1 for v in ages) / max(1, len(ages))}
        for prefix, rows in (("same_set", same), ("valid_context", valid)):
            for flag in ("reduced", "edges_changed", "rejected", "incomplete"):
                f[f"history.{prefix}_{flag}"] = sum(r[flag] for r in rows) / max(1, len(rows))
        return f | {"sa.log_temperature": math.log1p(temperature), "sa.log_decision": math.log1p(self.decision)}

    def observe(self, before, event, after):
        if event["decision"] != self.decision or edges(before) != set(self.ages):
            raise ValueError("history/prefix discontinuity")
        m = event["metrics"]
        if (m["conflicts_before"], m["conflicts_after"]) != (len(edges(before)), len(edges(after))):
            raise ValueError("conflict count mismatch")
        ids = tuple(sorted(m["neighborhood"]))
        self.recent.append(dict(members=ids, context=self.context(before, ids),
                                reduced=m["conflicts_after"] < m["conflicts_before"],
                                edges_changed=edges(before) != edges(after),
                                rejected=bool(m["pp_rolled_back"]), incomplete=not m["acceptance_evaluated"]))
        self.ages = {e: self.ages.get(e, -1) + 1 for e in edges(after)}
        c = after["num_of_colliding_pairs"]
        self.since_best = 0 if c < self.best else self.since_best + 1
        self.best = min(self.best, c)
        self.decision += 1

    def edge_weights(self):
        return [(list(e), 1.0 + min(age, self.window) / self.window) for e, age in sorted(self.ages.items())]


def targets(before, after, edge_weights, metrics):
    if metrics["pp_failure_reason"] == "time_limit" or not metrics["acceptance_evaluated"]:
        return None
    initial_edges = edges(before)
    weights = {tuple(e): float(w) for e, w in edge_weights}
    if set(weights) != initial_edges or any(w <= 0 for w in weights.values()):
        raise ValueError("edge weight identity")
    remaining = edges(after)
    return {"conflicts": len(remaining) / max(1, len(initial_edges)),
            "persistence": sum(weights[e] for e in initial_edges & remaining) / max(1.0, sum(weights.values())),
            "feasible": float(after["feasible"])}


def aggregate_trials(rows, expected):
    if len(rows) != expected or {r["trial"] for r in rows} != set(range(expected)):
        raise ValueError("missing or duplicate trials")
    if len({json_fingerprint(r["members"]) for r in rows}) != 1:
        raise ValueError("trial neighborhood changed")
    if any(r["target"] is None for r in rows):
        return None
    return {k: sum(r["target"][k] for r in rows) / expected for k in ("conflicts", "persistence", "feasible")}


def select_prediction(predictions, candidate_ids, initial_conflicts, tolerance=1.0, risk=True):
    """Quality-bounded persistent-edge selection, not predicted completion time."""
    if len(predictions) != len(candidate_ids) or not predictions:
        raise ValueError("prediction shape")
    if any(not math.isfinite(float(v)) for p in predictions for v in p.values()):
        raise ValueError("nonfinite prediction")
    # Four-trial labels resolve feasibility only in quarters; no tuned threshold.
    feasible = [round(min(1.0, max(0.0, p["feasible"])) * 4) for p in predictions]
    eligible = [i for i in range(len(predictions)) if feasible[i] == max(feasible)]
    best = min(predictions[i]["conflicts"] for i in eligible)
    if risk:
        eligible = [i for i in eligible if predictions[i]["conflicts"] <= best + tolerance / max(1, initial_conflicts)]
        return min(eligible, key=lambda i: (predictions[i]["persistence"], predictions[i]["conflicts"], candidate_ids[i]))
    return min(eligible, key=lambda i: (predictions[i]["conflicts"], candidate_ids[i]))


def pareto_indices(labels):
    return [i for i, a in enumerate(labels) if not any(
        b["feasible"] >= a["feasible"] and b["conflicts"] <= a["conflicts"] and
        (b["feasible"] > a["feasible"] or b["conflicts"] < a["conflicts"])
        for b in labels)]
