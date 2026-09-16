"""Fixed post-hoc objective/representation controls; no solver or policy export."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments._common import json_fingerprint, read_json, sha256_file
from experiments.sa_history_information import episode_weights
from scripts.audit_sa_history_information import atomic, require, save_result, receipt_result, progress
from scripts.collect_sa_history_candidate_bridge import verify as verify_bridge, check_root_receipt
from scripts.preflight_sa_history_candidate_bridge import features, choice, MODELS, verify_information, verify_sampling

CONFIG = ROOT / "configs/sa_history_objective_factorial.json"


def prepare():
    cfg = read_json(CONFIG)
    out = ROOT / cfg["output"]
    require(not out.exists(), "output exists; preserve previous attempt")
    require(cfg["profiles"] == ["dynamic", "ordered", "temporal_bag"] and
            cfg["targets"] == ["sustained_progress", "completion"], "frozen factorial grid")
    bridge, source = verify_bridge()
    verify_sampling(ROOT / bridge["config"]["sampling_config"])
    info, info_out = verify_information()
    require(sha256_file(source / "report.json") == cfg["bridge_report_sha256"], "bridge report changed")
    for root in bridge["roots"]:
        check_root_receipt(source / "branches" / root["target"]["id"], root, bridge)
    inputs = dict(bridge["inputs"])
    for path in (CONFIG, Path(__file__), ROOT / "tests/evaluation/test_sa_history_objective_factorial.py",
                 ROOT / "docs/SA_HISTORY_OBJECTIVE_FACTORIAL_ZH.md", source / "report.json",
                 source / "collection_plan.json", info_out / "index.json"):
        inputs[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    plan = dict(config=cfg, inputs=inputs, model=info["config"]["model"],
                source=source.relative_to(ROOT).as_posix(), index=(info_out / "index.json").relative_to(ROOT).as_posix(),
                information_binding=info["binding"], bridge_binding=bridge["binding"],
                commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    plan["binding"] = json_fingerprint(plan)
    atomic(out / "plan.json", plan)
    return dict(binding=plan["binding"], fits=48, solver_calls=0)


def verify():
    cfg = read_json(CONFIG)
    out = ROOT / cfg["output"]
    plan = read_json(out / "plan.json")
    require(plan["config"] == cfg and plan["binding"] == json_fingerprint(
        {k:v for k,v in plan.items() if k != "binding"}), "plan/config changed")
    for name, digest in plan["inputs"].items():
        require(sha256_file(ROOT / name) == digest, "registered input changed: " + name)
    return plan, out


def training_rows(rows, root):
    train = [r for r in rows if r["map_id"] != root["target"]["map_id"]]
    require(bool(train), "empty train fold")
    require(all(r["episode"] != root["target"]["item"]["job_id"] for r in train), "episode leakage")
    return train


def fit(job):
    import numpy as np
    import sklearn
    from sklearn.ensemble import HistGradientBoostingClassifier
    require(sklearn.__version__ == "1.5.0", "registered sklearn version required")
    train = training_rows(job["rows"], job["root"])
    a = [features(job["profile"], r) for r in train]
    b = [features(job["profile"], r) for r in job["root"]["rows"]]
    names = sorted({k for row in a for k in row})
    require(all(set(row) <= set(names) for row in b), "candidate feature schema mismatch")
    def matrix(rows):
        return np.array([[r.get(k, 0.) for k in names] for r in rows], dtype=float)
    y = [r["labels"][job["target"]] for r in train]
    require(set(y) == {0, 1}, "single class fold")
    model = HistGradientBoostingClassifier(**job["model"]).fit(matrix(a), y, sample_weight=episode_weights(train))
    return dict(root_id=job["root"]["target"]["id"], map_id=job["root"]["target"]["map_id"],
        profile=job["profile"], target=job["target"],
        probabilities=model.predict_proba(matrix(b))[:, 1].tolist(),
        candidate_ids=[r["candidate_id"] for r in job["root"]["rows"]], feature_count=len(names),
        train_rows=len(train), train_ids_sha=json_fingerprint([r["id"] for r in train]),
        train_maps=sorted({r["map_id"] for r in train}))


def rates(labels, candidate, indices=range(8)):
    require(candidate in labels, "unobserved candidate has no outcome")
    indices = list(indices)
    return {t:sum(labels[candidate][i][t] for i in indices) / len(indices)
            for t in ("completion", "sustained_progress")}


def empirical_choice(labels, target, indices, baseline):
    values = {c:rates(labels, c, indices)[target] for c in labels}
    best = max(values.values())
    return baseline if values[baseline] == best else min(c for c in values if values[c] == best)


def split_reference(labels, target, baseline):
    require(len(labels) == 3 and all(len(v) == 8 for v in labels.values()), "complete paired trial grid required")
    halves = []
    for train, test in ((range(4), range(4, 8)), (range(4, 8), range(4))):
        selected = empirical_choice(labels, target, train, baseline)
        halves.append(dict(selected=selected, selection_trials=list(train), evaluation_trials=list(test),
                           observed=rates(labels, selected, test)))
    return dict(halves=halves, rates={t:sum(h["observed"][t] for h in halves) / 2
                                    for t in ("completion", "sustained_progress")})


def interval(values, cfg):
    import numpy as np
    a = np.array(values, dtype=float)
    draws = np.random.default_rng(cfg["seed"]).integers(0, len(a), size=(cfg["bootstrap"], len(a)))
    return dict(mean=float(a.mean()), ci95=np.quantile(a[draws].mean(axis=1), [.025, .975]).tolist(),
                wins=int(sum(a > 0)), losses=int(sum(a < 0)), ties=int(sum(a == 0)))


def analyze(preflight, source_report, folds, cfg):
    require(len(folds) == len(preflight["roots"]) * 6, "fold coverage")
    lookup = {(f["root_id"], f["profile"], f["target"]):f for f in folds}
    require(len(lookup) == len(folds), "duplicate fold")
    require(len({r["target"]["map_id"] for r in preflight["roots"]}) == len(preflight["roots"]), "one root per map required")
    outcomes = {s["id"]:s for s in source_report["states"]}
    states, index = [], []
    for root in preflight["roots"]:
        root_id = root["target"]["id"]
        source = outcomes[root_id]
        require(not source["censored"], "censored labels are not failures")
        labels = source["labels"]
        require(set(labels) == set(root["selected"]), "collected candidate identity")
        for name, (profile, target) in MODELS.items():
            require(lookup[root_id, profile, target]["probabilities"] == root["predictions"][name], "old prediction parity: " + name)
        policies = dict(frozen=dict(selected=root["old_selected_id"], rates=rates(labels, root["old_selected_id"])))
        policies["uniform"] = dict(rates={t:sum(rates(labels, c)[t] for c in labels) / len(labels)
                                           for t in ("completion", "sustained_progress")})
        predictions = {}
        for profile in cfg["profiles"]:
            for target in cfg["targets"]:
                key = profile + "/" + target
                f = lookup[root_id, profile, target]
                require(f["candidate_ids"] == [r["candidate_id"] for r in root["rows"]], "prediction candidate order")
                ps = dict(zip(f["candidate_ids"], f["probabilities"], strict=True))
                selected = choice(sorted(labels), [ps[c] for c in sorted(labels)])
                full = choice(f["candidate_ids"], f["probabilities"])
                policies[key] = dict(selected=selected, rates=rates(labels, selected),
                    full_pool_selected=full, full_pool_observed=full in labels,
                    full_pool_rates=rates(labels, full) if full in labels else None)
                predictions[key] = ps
        for target in cfg["targets"]:
            policies["split_reference/" + target] = split_reference(labels, target, root["old_selected_id"])
        for candidate in sorted(labels):
            index.append(dict(root_id=root_id, map_id=source["map_id"], candidate_id=candidate,
                trial_labels=labels[candidate], rates=rates(labels, candidate),
                probabilities={p:ps[candidate] for p, ps in predictions.items()}, diagnostic_only=True))
        states.append(dict(id=root_id, map_id=source["map_id"], stratum=source["stratum"], policies=policies))
    policy_names = sorted(states[0]["policies"])
    summary = {p:{t:sum(s["policies"][p]["rates"][t] for s in states) / len(states)
                  for t in cfg["targets"]} for p in policy_names}
    contrasts = [(p + "/completion", p + "/sustained_progress") for p in cfg["profiles"]]
    contrasts += [(p + "/" + t, "dynamic/" + t) for p in ("ordered", "temporal_bag") for t in cfg["targets"]]
    contrasts += [("split_reference/" + t, "frozen") for t in cfg["targets"]]
    comparisons = {a + " vs " + b:{t:interval([s["policies"][a]["rates"][t] - s["policies"][b]["rates"][t]
                        for s in states], cfg) for t in cfg["targets"]} for a, b in contrasts}
    return dict(states=states, summary=summary, comparisons=comparisons, candidate_index=index,
        old_probability_parity=True, post_hoc=True, no_promotion=True, new_solver_calls=0,
        decision="diagnostic_only_no_model_selection", interval_scope="exploratory_unadjusted_map_bootstrap",
        restricted_pool_warning="all six policies compared only on three observed candidates; no full-pool claim")


def run():
    plan, out = verify()
    cfg = plan["config"]
    source = ROOT / plan["source"]
    preflight = read_json(source / "preflight.json")
    rows = [r for r in receipt_result(ROOT / plan["index"], plan["information_binding"])["rows"] if r["labels"] is not None]
    require(len(rows) == 204, "source landmark count changed")
    lock = out / "run.lock"
    with lock.open("x", encoding="utf8") as f:
        f.write(str(os.getpid()))
    try:
        atomic(out / "run_status.json", dict(status="running", binding=plan["binding"]))
        folds, pending = [], {}
        with ProcessPoolExecutor(max_workers=cfg["workers"]) as pool:
            for root in preflight["roots"]:
                for profile in cfg["profiles"]:
                    for target in cfg["targets"]:
                        path = out / "folds" / (json_fingerprint([root["target"]["id"], profile, target])[:20] + ".json")
                        if path.exists():
                            folds.append(receipt_result(path, plan["binding"]))
                        else:
                            pending[pool.submit(fit, dict(rows=rows, root=root, profile=profile, target=target, model=plan["model"]))] = path
            for future in as_completed(pending):
                result = future.result()
                save_result(pending[future], plan["binding"], result)
                folds.append(result)
                progress(out, "fit", completed=len(folds), total=48)
        report = analyze(preflight, read_json(source / "report.json"), folds, cfg)
        report["binding"] = plan["binding"]
        verify()
        if (out / "report.json").exists():
            require(read_json(out / "report.json") == report, "recomputed report differs")
        else:
            atomic(out / "report.json", report)
        atomic(out / "run_status.json", dict(status="completed", binding=plan["binding"], report_sha256=sha256_file(out / "report.json")))
        return dict(summary=report["summary"], decision=report["decision"], sha256=sha256_file(out / "report.json"))
    except BaseException as exc:
        atomic(out / "run_status.json", dict(status="failed", binding=plan["binding"], error=repr(exc)))
        raise
    finally:
        lock.unlink()


def main():
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run"))
    args = parser.parse_args()
    print(json.dumps(prepare() if args.phase == "prepare" else run(), sort_keys=True))


if __name__ == "__main__":
    main()
