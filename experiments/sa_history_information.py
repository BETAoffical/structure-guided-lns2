"""Causal, observational history audit; never an action-value/controller API."""
from collections import Counter, deque
import math
import random

from experiments._common import json_fingerprint
from experiments.sa_history_selector import edges, members

BUCKETS = ((0, 1), (1, 4), (4, 8), (8, 16), (16, 32))
TEMPORAL = ("overlap", "selected_changed", "edge_similarity", "relative_conflicts",
            "accepted_worse", "new_best", "rejected", "incomplete")
RESOURCE = ("boundary_changed", "goal_peers_changed", "boundary_path_difference",
            "goal_peers_path_difference")


def ratio(a, b):
    return a / max(1, b)


def path_ids(state):
    return {a["id"]: json_fingerprint(a["path"]) for a in state["agents"]}


def peers(state, selected):
    selected = set(selected)
    boundary = {v for e in edges(state) if selected.intersection(e) for v in e} - selected
    cells = {v for a in state["agents"] if a["id"] in selected for v in a["path"]}
    # Spatial goal overlap is a potential interaction, not a blocking certificate.
    goals = {a["id"] for a in state["agents"] if a["id"] not in selected and a["goal"] in cells}
    return boundary, goals


class OrderedHistory:
    def __init__(self, window=32):
        if window != 32:
            raise ValueError("fixed audit window")
        self.recent = deque(maxlen=window)

    def observe(self, before, event, after, previous_best):
        bp, ap = path_ids(before), path_ids(after)
        if bp.keys() != ap.keys():
            raise ValueError("agent identities changed")
        m = event["metrics"]
        self.recent.append(dict(selected=set(m["neighborhood"]),
            changed={i for i in bp if bp[i] != ap[i]}, paths=ap,
            edges=edges(after), conflicts=after["num_of_colliding_pairs"],
            accepted_worse=float(m["conflicts_after"] > m["conflicts_before"]),
            new_best=float(m["conflicts_after"] < previous_best),
            rejected=float(m["pp_rolled_back"]), incomplete=float(not m["acceptance_evaluated"])))

    def records(self, state, candidate):
        selected = set(members(candidate))
        current_paths = path_ids(state)
        if not selected <= current_paths.keys():
            raise ValueError("unknown candidate agent")
        boundary, goals = peers(state, selected)
        incident = {e for e in edges(state) if selected.intersection(e)}
        records = []
        for r in reversed(self.recent):
            prior = {e for e in r["edges"] if selected.intersection(e)}
            values = {"overlap": ratio(len(selected & r["selected"]), len(selected | r["selected"])),
                "selected_changed": ratio(len(selected & r["changed"]), len(selected)),
                "edge_similarity": ratio(len(incident & prior), len(incident | prior)),
                "relative_conflicts": r["conflicts"] / max(1, state["num_of_colliding_pairs"])}
            values.update({k: r[k] for k in TEMPORAL[4:]})
            for name, ids in (("boundary", boundary), ("goal_peers", goals)):
                values[name + "_changed"] = ratio(len(ids & r["changed"]), len(ids))
                values[name + "_path_difference"] = ratio(sum(r["paths"][i] != current_paths[i] for i in ids), len(ids))
            records.append(values)
        return records


def bag_features(records):
    return {"resource_bag." + k: ratio(sum(r[k] for r in records), len(records)) for k in TEMPORAL + RESOURCE}


def ordered_features(records, resource=False, shuffle_seed=None):
    records = list(records)
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(records)
    result = {}
    for start, end in BUCKETS:
        rows = records[start:end]
        result[f"ordered.{start}-{end}.available"] = len(rows) / (end-start)
        for k in TEMPORAL + (RESOURCE if resource else ()):
            result[f"ordered.{start}-{end}.{k}"] = ratio(sum(r[k] for r in rows), len(rows))
    return result


def future_labels(counts, decision, previous_best, horizon=32, sustain=8):
    """Counts are post-action counts; missing nonterminal suffixes are censored."""
    future = list(counts[decision:decision+horizon])
    if not future:
        raise ValueError("no observed action")
    if 0 in future:
        first = future.index(0)
        if any(c != 0 for c in future[first:]):
            raise ValueError("trajectory continued after feasibility")
        future = future[:first+1] + [0] * (horizon-first-1)
    if len(future) < horizon:
        return None
    return {"sustained_progress": int(all(c < previous_best for c in future[-sustain:])),
            "completion": int(future[-1] == 0)}


def profile_features(row, profile):
    dynamic = {k: v for k, v in row["base"].items() if not k.startswith("history.")}
    if profile == "dynamic":
        return dynamic
    base = row["base"]
    if profile == "aggregate":
        return base
    bag = bag_features(row["records"])
    if profile == "resource_bag":
        return base | bag
    if profile == "ordered":
        return base | ordered_features(row["records"])
    shuffle_seed = None
    if profile.startswith("shuffled_"):
        shuffle_seed = int(json_fingerprint([row["id"], profile, 20260917])[:12], 16)
    elif profile != "resource_ordered":
        raise ValueError("unknown profile")
    return base | bag | ordered_features(row["records"], True, shuffle_seed)


def episode_weights(rows):
    counts = Counter(r["episode"] for r in rows)
    raw = [1 / counts[r["episode"]] for r in rows]
    scale = len(rows) / max(1, len(counts))
    return [w * scale for w in raw]


def metric(rows, probabilities, target):
    if not rows or len(rows) != len(probabilities):
        raise ValueError("metric shape")
    weights = episode_weights(rows)
    total = sum(weights)
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError("invalid probability")
    y = [r["labels"][target] for r in rows]
    return {"brier": sum(w*(p-v)**2 for w,p,v in zip(weights, probabilities, y))/total,
            "prevalence": sum(w*v for w,v in zip(weights,y))/total,
            "log_loss": -sum(w*(v*math.log(max(1e-12,p))+(1-v)*math.log(max(1e-12,1-p)))
                             for w,p,v in zip(weights, probabilities, y))/total}
