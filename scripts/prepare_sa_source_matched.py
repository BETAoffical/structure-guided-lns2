"""Prepare outcome-blind matched-source pairs; never launch a solver or train."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments._common import read_json, sha256_file, json_fingerprint
from experiments.sa_paired_completion import require, validate_dataset
from scripts.run_sa_paired_closed_loop import once

CONFIG = "configs/sa_source_matched_preparation.json"
CODE = ["scripts/prepare_sa_source_matched.py", "tests/evaluation/test_sa_source_matched.py",
        "docs/SA_SOURCE_MATCHED_PROTOCOL_ZH.md", "experiments/_common.py",
        "experiments/sa_paired_completion.py", "scripts/run_sa_paired_closed_loop.py",
        "scripts/audit_sa_history_information.py"]


def json_snapshot(path):
    content = path.read_bytes()
    return json.loads(content), hashlib.sha256(content).hexdigest()


def verify_inputs(inputs):
    for name, digest in inputs.items():
        require(sha256_file(ROOT / name) == digest, "input changed during preparation: " + name)


def check_root_identity(state, root):
    source = root["source"]
    require(root["state_fingerprint"] == source["state_fingerprint"] == state["state_fingerprint"]
            and source["id"] == state["state_id"] and source["map_id"] == state["map_id"]
            and root["old_selected_id"] == source["anchor_id"] == state["anchor_id"]
            and source["item"]["job_id"] == state["episode"]
            and source["decision"] == state["decision"] == root["control_event"]["decision"],
            "root/index source identity mismatch")
    full = root["control_event"]["pool"]
    grouped(full, 16)
    by_id = {c["candidate_id"]: c for c in full}
    labeled = {c["candidate_id"]: c for c in root["candidates"]}
    indexed = {c["candidate_id"]: c for c in state["candidates"]}
    require(len(labeled) == len(root["candidates"]) and len(indexed) == len(state["candidates"])
            and set(indexed) == set(labeled) == set(source["selected"]) <= set(by_id), "labeled candidate identity")
    for cid, c in labeled.items():
        require(sorted(c["agents"]) == sorted(by_id[cid]["agents"]) == sorted(indexed[cid]["agents"])
                and signature(c) == signature(by_id[cid])
                and indexed[cid]["features"]["proposal.actual_size"] == len(c["agents"]), "cross-artifact candidate mismatch")
    event = root["control_event"]
    i = event["selected_index"]
    require(type(i) is int and 0 <= i < len(full)
            and full[i]["candidate_id"] == state["anchor_id"]
            and event["action"]["agents"] == full[i]["agents"], "root anchor action mismatch")


def check_config(cfg):
    require(cfg["new_solver_calls"] == 0 and cfg["training_allowed"] is False
            and cfg["collection_ready"] is False, "preparation-only boundary changed")
    for key, expected in (("maps", 8), ("roots_per_map", 2), ("actual_size", 16),
                          ("planned_trials", 8), ("planned_horizon", 32),
                          ("planned_trial_timeout_seconds", 180), ("planned_pp_safety_seconds", 5),
                          ("planned_workers", 20)):
        require(type(cfg[key]) is int and cfg[key] == expected, "fixed protocol changed: " + key)
    require(cfg["role"] == "previously_viewed_development" and
            sorted(cfg["allowed_families"]) == ["collision:16", "random:16", "target:16"], "source/role changed")
    out = (ROOT / cfg["output"]).resolve()
    build_dir = (ROOT / "build").resolve()
    require(out != build_dir and out.is_relative_to(build_dir), "unsafe output directory")


def signature(candidate):
    agents = candidate["agents"]
    families = candidate["selection_families"]
    require(agents and len(agents) == len(set(agents))
            and all(type(a) is int and a >= 0 for a in agents), "invalid agents")
    require(families and all(isinstance(f, str) and f for f in families)
            and len(families) == len(set(families)), "invalid source signature")
    require(candidate["actual_size"] == len(agents), "actual size mismatch")
    return len(agents), tuple(sorted(families))


def grouped(pool, size):
    require(len({c["candidate_id"] for c in pool}) == len(pool), "duplicate candidate ID")
    require(len({tuple(sorted(c["agents"])) for c in pool}) == len(pool), "duplicate membership")
    groups = defaultdict(list)
    for c in pool:
        n, families = signature(c)
        if n == size:
            groups[families].append(c)
    return {k: sorted(v, key=lambda c: c["candidate_id"]) for k, v in sorted(groups.items())}


def blind_root(record):
    """Whitelist is also the selection boundary: no outcome, feature or model score."""
    return {k: record[k] for k in ("state_id", "map_id", "episode", "decision", "anchor_id")} | {
        "pool": [{k: c[k] for k in ("candidate_id", "agents", "actual_size", "selection_families")}
                 for c in record["pool"]]}


def select(records, cfg):
    records = [blind_root(r) for r in records]
    require(len({r["state_id"] for r in records}) == len(records), "duplicate roots")
    seed = cfg["sampling_seed"]
    def order(kind, *parts):
        return json_fingerprint(["source-matched-v1", seed, kind, *parts])
    by_map = defaultdict(list)
    for r in records:
        groups = grouped(r["pool"], cfg["actual_size"])
        eligible = {g: cs for g, cs in groups.items() if len(cs) >= 2 and len(g) == 1
                    and g[0] in cfg["allowed_families"]}
        require(eligible, "root has no matched pure-family pair: " + r["state_id"])
        require(r["anchor_id"] in {c["candidate_id"] for c in r["pool"]}, "missing anchor")
        by_map[r["map_id"]].append((r, eligible))
    require(len(by_map) == cfg["maps"], "map count mismatch")
    selected = []
    for map_id, entries in sorted(by_map.items()):
        used = set()
        for r, eligible in sorted(entries, key=lambda item: (order("root", item[0]["state_id"]), item[0]["state_id"])):
            if r["episode"] in used:
                continue
            used.add(r["episode"])
            family = min(eligible, key=lambda g: (order("family", r["state_id"], g), g))
            pair = sorted(eligible[family], key=lambda c: (order("candidate", r["state_id"], c["candidate_id"]), c["candidate_id"]))[:2]
            pair_ids = sorted(c["candidate_id"] for c in pair)
            ids = sorted(set(pair_ids + [r["anchor_id"]]))
            lookup = {c["candidate_id"]: c for c in r["pool"]}
            selected.append({k: r[k] for k in ("state_id", "map_id", "episode", "decision", "anchor_id")} |
                dict(family=family[0], pair_ids=pair_ids, candidates=[lookup[c] for c in ids]))
            if len(used) == cfg["roots_per_map"]:
                break
        require(len(used) == cfg["roots_per_map"], "insufficient distinct source episodes")
    return selected


def work_budget(roots, cfg):
    jobs = sum(len(r["candidates"]) for r in roots) * cfg["planned_trials"]
    prefix = sum(r["decision"] * len(r["candidates"]) for r in roots) * cfg["planned_trials"]
    return dict(roots=len(roots), maps=len({r["map_id"] for r in roots}),
                candidate_evaluations=sum(len(r["candidates"]) for r in roots),
                matched_pair_trial_comparisons=len(roots) * cfg["planned_trials"],
                trial_jobs=jobs, independent_resets=jobs, maximum_rollout_repairs=jobs * cfg["planned_horizon"],
                maximum_prefix_replay_repairs=prefix,
                serial_timeout_budget_seconds=jobs * cfg["planned_trial_timeout_seconds"],
                ideal_parallel_budget_seconds=jobs * cfg["planned_trial_timeout_seconds"] / cfg["planned_workers"],
                predicted_runtime_seconds=None, executed_jobs=0)


def build():
    cfg, config_sha = json_snapshot(ROOT / CONFIG)
    check_config(cfg)
    inputs = {name: sha256_file(ROOT / name) for name in CODE} | {CONFIG: config_sha}
    def load(name, expected):
        path = (ROOT / name).resolve()
        require(path.is_relative_to(ROOT), "unsafe input path")
        value, actual = json_snapshot(path)
        require(actual == expected, "input changed: " + name)
        inputs[name] = expected
        return value
    receipt = load(cfg["source_receipt"], cfg["source_receipt_sha256"])
    data = validate_dataset(load(cfg["index"], cfg["index_sha256"]))
    coverage = load(cfg["coverage_audit"], cfg["coverage_audit_sha256"])
    pred_receipt = load(cfg["prediction_receipt"], cfg["prediction_receipt_sha256"])
    roots, root_data, sources = [], {}, {}
    for name, digest in sorted(receipt["inputs"].items()):
        if name.endswith("/plan.json") and "/sa-" in name:
            sources[str(Path(name).parent).replace("\\", "/")] = load(name, digest)
    require(len(data["states"]) == 47, "development state count changed")
    for state in data["states"]:
        sid = state["state_id"]
        names = [n for n in receipt["inputs"] if n.endswith("/roots/" + sid + "/root.json")]
        require(len(names) == 1, "ambiguous root provenance")
        name = names[0]
        root = load(name, receipt["inputs"][name])
        check_root_identity(state, root)
        folder = str(Path(name).parent).replace("\\", "/")
        origin = folder.split("/roots/")[0]
        require(origin in sources and sources[origin]["binding"] == root["binding"], "source plan/root binding")
        rc_name = folder + "/receipt.json"
        rc = load(rc_name, receipt["inputs"][rc_name])
        require(rc["files"]["root.json"] == inputs[name] and rc["binding"] == root["binding"], "root receipt mismatch")
        for c in root["control_event"]["pool"]:
            require(set(c["agents"]) <= set(state["agent_ids"]), "unknown candidate agent")
        record = {k: state[k] for k in ("state_id", "map_id", "episode", "decision", "anchor_id")}
        record["pool"] = root["control_event"]["pool"]
        roots.append(record)
        root_data[sid] = (root, folder, rc)
    inventory = []
    for r in roots:
        root = root_data[r["state_id"]][0]
        row = {k: r[k] for k in ("state_id", "map_id", "episode", "decision")}
        for label, pool in (("labeled", root["candidates"]), ("full", r["pool"])):
            row[label + "_size16_same_signature_pairs"] = sum(len(cs) * (len(cs) - 1) // 2
                for cs in grouped(pool, cfg["actual_size"]).values())
        row["eligible_pure_family_size16_pairs"] = sum(len(cs) * (len(cs) - 1) // 2
            for sig, cs in grouped(r["pool"], cfg["actual_size"]).items()
            if len(sig) == 1 and sig[0] in cfg["allowed_families"])
        inventory.append(row)
    selected = select(roots, cfg)
    for r in selected:
        root, folder, _ = root_data[r["state_id"]]
        r.update(source_root=folder + "/root.json", root_fingerprint=root["state_fingerprint"],
                 source_binding=root["binding"])

    # Held-map scores already existed before this diagnostic; no new fitting.
    held_predictions = {}
    pred_base = str(Path(cfg["prediction_receipt"]).parent).replace("\\", "/")
    for name, digest in pred_receipt["files"].items():
        if not name.startswith("fits/gbdt-"):
            continue
        fit = load(pred_base + "/" + name, digest)
        require(fit["binding"] == pred_receipt["binding"], "prediction binding")
        held_ids = {s["state_id"] for s in data["states"] if s["map_id"] == fit["held"]}
        require(not held_ids & set(fit["train_ids"]), "held-map training leakage")
        for row in fit["rows"]:
            if row["held"]:
                require(row["state_id"] in held_ids and row["state_id"] not in held_predictions, "prediction map identity")
                held_predictions[row["state_id"]] = row
    require(set(held_predictions) == set(root_data), "incomplete held-map predictions")
    comparisons = []
    casebook = []
    for comparison in coverage["labels"]["comparisons"]:
        if comparison["size_relation"] != "equal" or comparison["coverage_relation"] != "lower":
            continue
        row = dict(comparison)
        scores = held_predictions[row["state_id"]]["scores"]
        row["oof_score_delta"] = scores[row["candidate_id"]] - scores[row["anchor"]]
        comparisons.append(row)
        if row["both_halves"] == "other":
            continue
        root, folder, rc = root_data[row["state_id"]]
        details = {}
        for role, cid in (("anchor", row["anchor"]), ("candidate", row["candidate_id"])):
            branches = []
            for trial in range(8):
                name = cid + "-t" + str(trial) + ".json"
                branch = load(folder + "/" + name, rc["files"][name])
                require((branch["binding"], branch["candidate_id"], branch["trial"], branch["root_id"])
                        == (root["binding"], cid, trial, row["state_id"]), "branch identity")
                require(branch["status"] == "ok" and branch["stop"] in ("feasible", "horizon"), "censored case")
                indexed = next(s for s in data["states"] if s["state_id"] == row["state_id"])
                labeled = next(c for c in indexed["candidates"] if c["candidate_id"] == cid)
                expected_trial = next(t for t in labeled["trials"] if t["trial"] == trial)
                require(expected_trial["completed"] == (branch["stop"] == "feasible")
                        and expected_trial["steps"] == len(branch["events"])
                        and expected_trial["final_conflicts"] == branch["final_conflicts"], "index/branch outcome mismatch")
                metrics = [e["metrics"] for e in branch["events"]]
                require(metrics and metrics[0]["conflicts_before"] == root["state"]["num_of_colliding_pairs"], "initial conflict mismatch")
                sequence = [metrics[0]["conflicts_before"]] + [m["conflicts_after"] for m in metrics]
                require(all(m["conflicts_before"] == sequence[i] for i, m in enumerate(metrics)), "conflict trace discontinuity")
                require(sequence[-1] == branch["final_conflicts"] and (sequence[-1] == 0) == (branch["stop"] == "feasible"), "terminal mismatch")
                branches.append(dict(trial=trial, completed=branch["stop"] == "feasible", conflicts=sequence))
            details[role] = dict(candidate_id=cid, completed=sum(b["completed"] for b in branches),
                                 first_conflicts_mean=statistics.fmean(b["conflicts"][1] for b in branches), branches=branches)
        casebook.append(dict(state_id=row["state_id"], map_id=row["map_id"],
                             initial_conflicts=root["state"]["num_of_colliding_pairs"], details=details))
    native = {(p["native_file"], p["native_sha256"]) for p in sources.values() if "native_file" in p}
    require(len(native) == 1, "source native disagreement")
    native_path, native_sha = next(iter(native))
    require(sha256_file(ROOT / native_path) == native_sha, "native file changed")
    inputs[native_path] = native_sha
    require(all(cfg["trial_seed"] not in range(p["config"]["seed"], p["config"]["seed"] + 8)
                for p in sources.values() if "seed" in p.get("config", {})), "old trial namespace reused")
    verify_inputs(inputs)
    plan = dict(schema=cfg["schema"], config=cfg, inputs=inputs, inventory=inventory, roots=selected,
        budget=work_budget(selected, cfg), collection_ready=False, new_solver_calls=0,
        training_allowed=False, native_file=native_path, native_sha256=native_sha,
        role=cfg["role"], descriptive_comparisons=comparisons, posthoc_casebook=casebook,
        blockers=["No collector registered for this new selection contract",
                  "Prefix/pool replay and independent trial-stream tests required before collection"],
        selection_uses_outcomes=False, no_model_shortcut_causal_claim=True)
    plan["binding"] = json_fingerprint(plan)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "verify", "dry-run"))
    args = parser.parse_args()
    plan = build()
    path = ROOT / plan["config"]["output"] / "plan.json"
    if args.phase == "prepare":
        once(path, plan)
    else:
        require(read_json(path) == plan, "prepared plan changed")
    print(json.dumps(dict(binding=plan["binding"], budget=plan["budget"],
                          collection_ready=False, new_solver_calls=0), sort_keys=True))


if __name__ == "__main__":
    main()
