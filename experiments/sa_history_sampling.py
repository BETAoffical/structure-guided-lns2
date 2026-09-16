"""Outcome-blind sampling of history-bearing SA occurrences and controls."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from experiments._common import json_fingerprint, read_json, sha256_file
from experiments.closed_loop_trace_storage import apply_state_delta
from experiments.repair_collection import state_fingerprint
from experiments.sa_history_selector import History, members


def candidate_history(history, state, pool, temperature):
    return [history.features(state, c, temperature) for c in pool]


def classify(history, state, pool, config):
    sampling = config["sampling"]
    if history.decision < sampling["min_control_decision"]:
        return None
    counts = Counter(r["members"] for r in history.recent)
    repeated = [i for i,c in enumerate(pool) if counts[members(c)] >= sampling["repeat_count"]]
    fresh = [i for i,c in enumerate(pool) if not counts[members(c)]]
    if history.decision >= config["history_window"] and history.since_best >= sampling["stall_steps"] and repeated and fresh:
        for i in repeated:
            context = history.context(state, members(pool[i]))
            if any(r["members"] == members(pool[i]) and r["context"] == context for r in history.recent):
                return "history_exact"
        return "history_changed"
    if history.since_best <= sampling["control_stall_max"] and len(fresh) == len(pool):
        return "progress"
    return None


def choose_history_candidates(pool, selected, seed, features, limit=6, required_repeat_fraction=2/32):
    if len(pool) != len(features) or len({members(c) for c in pool}) != len(pool):
        raise ValueError("invalid source pool")
    if not 0 <= selected < len(pool) or limit < 3:
        raise ValueError("invalid candidate sampling budget")
    chosen = [selected]
    def take(indices):
        for i in indices:
            if i not in chosen and len(chosen) < limit:
                chosen.append(i)
                return
    historical = [i for i,f in enumerate(features) if f["history.same_set_count"] >= required_repeat_fraction]
    take(sorted(historical, key=lambda i: (-features[i]["history.valid_context_count"],
        -features[i]["history.same_set_count"], pool[i]["candidate_id"])))
    scores = sorted(range(len(pool)), key=lambda i: (-pool[i]["score"], pool[i]["candidate_id"]))
    take(i for i in scores if features[i]["history.same_set_count"] == 0)
    take(scores)
    hashed = sorted(range(len(pool)), key=lambda i: json_fingerprint([seed, pool[i]["candidate_id"]]))
    for size in (4,8,16):
        take(i for i in hashed if len(pool[i]["agents"]) == size)
    for i in hashed:
        take([i])
    return chosen


def scan_source(job):
    root = Path(job["root"])
    initial, trace = [root/job[k] for k in ("initial", "trace")]
    for path, key in ((initial,"initial_sha"),(trace,"trace_sha")):
        if sha256_file(path) != job[key]:
            raise ValueError("source input changed")
    state = read_json(initial)["payload"]["observation"]
    cfg = job["config"]
    history = History(state, cfg["history_window"])
    found = {}
    with trace.open(encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            if event["decision"] > cfg["sampling"]["max_decision"]:
                break
            if event["decision"] != history.decision:
                raise ValueError("nonsequential source")
            kind = classify(history, state, event["pool"], cfg)
            if kind and kind not in found:
                identifier = f"{job['item']['job_id']}-d{event['decision']:04d}"
                features = candidate_history(history, state, event["pool"], event["temperature"])
                indices = choose_history_candidates(event["pool"], event["selected_index"],
                    [cfg["seed"],identifier], features, cfg["max_candidates"])
                found[kind] = dict(item=job["item"], map_id=job["map_id"], decision=event["decision"], id=identifier,
                    stratum=kind, preflight=dict(state_fingerprint=state_fingerprint(state),
                    pool_sha=json_fingerprint(event["pool"]), selected_index=event["selected_index"],
                    candidate_ids=[event["pool"][i]["candidate_id"] for i in indices],
                    history_features=[features[i] for i in indices]))
            if len(found) == 3:
                break
            after = apply_state_delta(state, event["delta"])
            history.observe(state, event, after)
            state = after
    if sha256_file(trace) != job["trace_sha"]:
        raise ValueError("source changed while scanning")
    return list(found.values())


def select_sources(available, map_ids, config):
    chosen, coverage = [], []
    for map_id in sorted(map_ids):
        pool = [r for r in available if r["map_id"] == map_id]
        order = sorted(pool, key=lambda r: (json_fingerprint([config["seed"],r["item"]["job_id"]]), r["decision"]))
        used = set()
        selected = []
        def take(rows, count):
            for row in rows:
                if row["item"]["job_id"] not in used and count:
                    selected.append(row)
                    used.add(row["item"]["job_id"])
                    count -= 1
        # Prefer one of each history type, then fill absent types without replacing maps.
        take([r for r in order if r["stratum"] == "history_exact"], 1)
        take([r for r in order if r["stratum"] == "history_changed"], 1)
        take([r for r in order if r["stratum"].startswith("history_")], 2-len(selected))
        take([r for r in order if r["stratum"] == "progress"], 2)
        chosen.extend(selected)
        coverage.append(dict(map_id=map_id, available=dict(Counter(r["stratum"] for r in pool)),
                             selected=dict(Counter(r["stratum"] for r in selected))))
    return chosen, coverage


def sampling_gate(targets, config):
    history = [t for t in targets if t["stratum"].startswith("history_")]
    exact = [t for t in targets if t["stratum"] == "history_exact"]
    controls = [t for t in targets if t["stratum"] == "progress"]
    def has_contrast(t):
        fs = t["preflight"]["history_features"]
        return any(f["history.same_set_count"] >= config["sampling"]["repeat_count"]/config["history_window"] for f in fs) and any(f["history.same_set_count"] == 0 for f in fs)
    return dict(complete_maps=len({t["map_id"] for t in targets}) == 8,
        distinct_episodes=len({t["item"]["job_id"] for t in targets}) == len(targets),
        bounded_states=24 <= len(targets) <= config["max_states"],
        history_states=len(history) >= 10, history_maps=len({t["map_id"] for t in history}) >= 5,
        exact_states=len(exact) >= 4, exact_maps=len({t["map_id"] for t in exact}) >= 2,
        controls=len(controls) >= 12, candidate_contrast=all(has_contrast(t) for t in history))


def build_history_sample(jobs, config, map_ids):
    with ProcessPoolExecutor(max_workers=config["workers"]) as executor:
        available = [row for rows in executor.map(scan_source, jobs) for row in rows]
    chosen, coverage = select_sources(available, map_ids, config)
    return chosen, coverage, sampling_gate(chosen, config)
