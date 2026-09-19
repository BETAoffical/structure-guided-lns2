"""One fixed H32 grouped-label fit per held map; no solver or runtime changes."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import pickle
import random
from statistics import mean
import subprocess
import sys
import time

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments._common import contained_file, json_fingerprint, read_json, sha256_file
from experiments.sa_paired_completion import MODEL_PARAMS, PairedCompletionModel, pair_vector, require, validate_dataset
from experiments.repair_collection import _CollectionRunLock
from scripts.audit_sa_history_information import atomic
from scripts import audit_sa_matched_training_coverage as coverage
from scripts import audit_sa_source_matched_prediction as prediction
from scripts import reconstruct_sa_recent_model as reconstruction
from scripts.collect_sa_source_matched import NATIVE, NATIVE_SHA
from scripts.run_sa_paired_closed_loop import once

CONFIG = ROOT / "configs/sa_matched_aggregation.json"
COVERAGE = "build/sa-matched-training-coverage-v1"
REMAINING = "build/sa-matched-remaining-collection-v1"
RECENT = "build/sa-recent-model-reconstruction-v1"
OLD_MATCHED = "build/sa-source-matched-collection-v1"
OLD_RECORDS = "build/sa-source-matched-resource-recovery-v1/analysis_records.json"


def sealed(value):
    return reconstruction.sealed(value)


def contract(cfg):
    require(cfg["model"] == MODEL_PARAMS and cfg["solver_calls"] == 0 and
            not any(cfg[k] for k in ("new_collection_allowed", "formal_ttf_allowed",
                                     "automatic_promotion", "hyperparameter_search")), "experiment boundary changed")
    require((cfg["expected_roots"], cfg["expected_maps"], cfg["expected_groups"],
             cfg["expected_unordered_pairs"], cfg["feature_count"], cfg["trials"], cfg["horizon"]) ==
            (47, 8, 94, 329, 127, 8, 32), "cohort or representation changed")
    require(cfg["workers"] == 20 and cfg["primary_group"] == "matched_pair" and
            cfg["secondary_group"] == "old_grid", "execution or estimand changed")


def add_pins(inputs, pins):
    for name, digest in pins.items():
        require(name not in inputs or inputs[name] == digest, "conflicting input: " + name)
        inputs[name] = digest


def check_inputs(inputs):
    for name, digest in inputs.items():
        require(sha256_file(contained_file(ROOT, name, field="aggregation input")) == digest, "input drift: " + name)


def bound_plan(path):
    plan = read_json(path)
    require(plan["binding"] == json_fingerprint({k: v for k, v in plan.items() if k != "binding"}), "source plan binding")
    return plan


def prepare():
    cfg = read_json(CONFIG)
    contract(cfg)
    out = ROOT / cfg["output"]
    require(not out.exists() and out.resolve().is_relative_to((ROOT / "build").resolve()), "existing or unsafe output")
    inputs = dict(cfg["evidence"])
    for name, digest in cfg["evidence"].items():
        require(sha256_file(ROOT / name) == digest, "evidence drift")
        add_pins(inputs, read_json(ROOT / name)["files"])
    plans = {}
    for directory in (COVERAGE, REMAINING, RECENT, OLD_MATCHED):
        name = directory + "/plan.json"
        require(name in inputs, "unregistered source plan")
        plans[directory] = bound_plan(ROOT / name)
        add_pins(inputs, plans[directory]["inputs"])
    old = validate_dataset(read_json(ROOT / plans[RECENT]["training_index"]))
    require(len(old["states"]) == 47 and len(old["feature_names"]) == 127 and
            all(len(s["candidates"]) == 4 for s in old["states"]), "old cohort changed")
    roots = plans[OLD_MATCHED]["roots"] + plans[REMAINING]["roots"]
    require(len({r["state_id"] for r in roots}) == len(roots) == 47 and
            {r["state_id"] for r in roots} == {s["state_id"] for s in old["states"]}, "complement overlap or gap")
    roots = sorted(roots, key=lambda r: (r["map_id"], r["state_id"]))
    maps = sorted({r["map_id"] for r in roots})
    require(len(maps) == 8 and maps == sorted(f["held"] for f in plans[RECENT]["folds"]), "fold coverage changed")
    receipt = prediction.seal_json(ROOT / RECENT / "model_receipt.json", plans[RECENT]["binding"])
    add_pins(inputs, {RECENT + "/" + name: digest for name, digest in receipt["files"].items()})
    for name in (CONFIG.relative_to(ROOT).as_posix(), Path(__file__).relative_to(ROOT).as_posix(),
                 "tests/evaluation/test_sa_matched_aggregation.py", "docs/SA_MATCHED_AGGREGATION_PROTOCOL_ZH.md",
                 "scripts/audit_sa_source_matched_prediction.py", "scripts/audit_sa_matched_training_coverage.py",
                 "scripts/reconstruct_sa_recent_model.py"):
        add_pins(inputs, {name: sha256_file(ROOT / name)})
    check_inputs(inputs)
    require(inputs[NATIVE] == NATIVE_SHA, "native identity changed")
    plan = dict(config=cfg, inputs=inputs, roots=roots, maps=maps, feature_names=old["feature_names"],
                old_index=plans[RECENT]["training_index"], native_sha256=NATIVE_SHA,
                recent_plan=RECENT + "/plan.json", model_params=MODEL_PARAMS,
                commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    plan["binding"] = json_fingerprint(plan)
    once(out / "plan.json", plan)
    return dict(binding=plan["binding"], features=47, new_model_fits=8, registered_inputs=len(inputs), solver_calls=0)


def verify():
    cfg = read_json(CONFIG)
    contract(cfg)
    out = ROOT / cfg["output"]
    require(out.resolve().is_relative_to((ROOT / "build").resolve()), "unsafe output")
    plan = bound_plan(out / "plan.json")
    require(plan["config"] == cfg, "config differs from registration")
    check_inputs(plan["inputs"])
    return plan, out


def phase_status(out, binding, phase, **extra):
    atomic(out / "run_status.json", dict(binding=binding, phase=phase, **extra))


def features():
    plan, out = verify()
    with _CollectionRunLock(out, plan["binding"], "aggregation-features"):
        pending = []
        for root in plan["roots"]:
            path = out / "features" / (root["state_id"] + ".json")
            if path.exists():
                prediction.seal_json(path, plan["binding"])
            else:
                pending.append(dict(entry=root, plan=dict(binding=plan["binding"], feature_names=plan["feature_names"],
                                                        native_sha256=plan["native_sha256"])))
        phase_status(out, plan["binding"], "features", status="running", remaining=len(pending))
        try:
            if pending:
                with ProcessPoolExecutor(max_workers=min(20, len(pending))) as pool:
                    futures = [pool.submit(prediction.feature_job, job) for job in pending]
                    for count, future in enumerate(as_completed(futures), 1):
                        row = future.result()
                        once(out / "features" / (row["state_id"] + ".json"), row)
                        print("FEATURE", count, "/", len(pending), row["state_id"], flush=True)
            files = {"features/" + r["state_id"] + ".json": sha256_file(out / "features" / (r["state_id"] + ".json"))
                     for r in plan["roots"]}
            once(out / "feature_receipt.json", sealed(dict(binding=plan["binding"], files=files, solver_calls=0)))
            index = build_index(plan, out)
            once(out / "training_index.json", sealed(index))
            once(out / "index_receipt.json", sealed(dict(binding=plan["binding"],
                 files={name: sha256_file(out / name) for name in ("training_index.json", "feature_receipt.json")})))
            check_inputs(plan["inputs"])
            phase_status(out, plan["binding"], "features", status="completed")
            return dict(roots=len(plan["roots"]), groups=len(index["groups"]),
                        pairs=len(coverage.pair_rows(index["groups"])), golden_vectors=index["golden_vectors"], solver_calls=0)
        except BaseException as exc:
            phase_status(out, plan["binding"], "features", status="error", error=repr(exc))
            raise


def validate_groups(groups):
    rebuilt = []
    for g in groups:
        clean = [{k: c[k] for k in ("candidate_id", "agents", "values", "selection_families") if k in c}
                 for c in g["candidates"]]
        new = coverage.comparison_group(g, g["source_binding"], clean, g["trial_keys"], g["kind"], g["family"])
        require(new["group_id"] == g["group_id"], "group identity changed")
        rebuilt.append(new)
    coverage.pair_rows(rebuilt)
    return rebuilt


def build_index(plan, out):
    prediction.verify_features(plan, out)
    old = validate_dataset(read_json(ROOT / plan["old_index"]))
    states = {s["state_id"]: s for s in old["states"]}
    groups = read_json(ROOT / COVERAGE / "comparison_groups.json")["groups"]
    groups = validate_groups(groups)
    new_plan = bound_plan(ROOT / REMAINING / "plan.json")
    records = read_json(ROOT / REMAINING / "analysis_records.json")
    labels = {r["state_id"]: r for r in records}
    require(len(labels) == len(records) == 31 and set(labels) == {r["state_id"] for r in new_plan["roots"]}, "new label coverage")
    for r in new_plan["roots"]:
        s, label = states[r["state_id"]], labels[r["state_id"]]
        require((r["root_fingerprint"], r["map_id"], r["episode"], r["decision"]) ==
                (s["state_fingerprint"], s["map_id"], s["episode"], s["decision"]), "root identity mismatch")
        require(label["map_id"] == r["map_id"] and label["pair_ids"] == r["pair_ids"] and
                set(label["values"]) == set(r["pair_ids"]), "label identity mismatch")
        candidates = [c | dict(values=label["values"][c["candidate_id"]]) for c in r["candidates"]]
        keys = [json_fingerprint([new_plan["binding"], s["state_id"], t]) for t in range(8)]
        groups.append(coverage.comparison_group(s, new_plan["binding"], candidates, keys, "matched_pair", r["family"]))
    groups = sorted(groups, key=lambda g: (g["state_id"], g["kind"], g["group_id"]))
    validate_groups(groups)
    golden = 0
    entries = {}
    for r in plan["roots"]:
        row = prediction.seal_json(out / "features" / (r["state_id"] + ".json"), plan["binding"])
        s = states[r["state_id"]]
        require(row["state_id"] == r["state_id"] and row["map_id"] == r["map_id"] == s["map_id"] and
                row["pair_ids"] == r["pair_ids"] and row["anchor_id"] == s["anchor_id"], "feature identity mismatch")
        members = {}
        for c in row["control_candidates"] + row["candidates"]:
            require(c["candidate_id"] not in members or c == members[c["candidate_id"]], "conflicting feature rows")
            require(sorted(c["features"]) == plan["feature_names"] and
                    all(type(v) in (float, int) and math.isfinite(v) for v in c["features"].values()), "invalid features")
            require(c["agents"] and len(set(c["agents"])) == len(c["agents"]) and
                    set(c["agents"]) <= set(row["agent_ids"]), "invalid feature members")
            members[c["candidate_id"]] = c
        for c in s["candidates"]:
            require(members[c["candidate_id"]]["features"] == c["features"] and
                    members[c["candidate_id"]]["agents"] == c["agents"], "old feature golden mismatch")
            golden += 1
        entries[r["state_id"]] = dict(state_id=r["state_id"], map_id=s["map_id"], anchor_id=s["anchor_id"],
             agent_ids=s["agent_ids"], decision=s["decision"], source_root=r["source_root"],
             original_pool_scores=row["original_pool_scores"], candidates=members)
    for g in groups:
        e = entries[g["state_id"]]
        require(e["map_id"] == g["map_id"], "group map mismatch")
        for c in g["candidates"]:
            f = e["candidates"][c["candidate_id"]]
            require(c["agents"] == f["agents"], "label/feature membership mismatch")
            c["features"] = f["features"]
    for sid in states:
        selected = [g for g in groups if g["state_id"] == sid]
        require(len(selected) == 2 and {g["kind"] for g in selected} == {"old_grid", "matched_pair"}, "root group coverage")
        require(not set(selected[0]["trial_keys"]) & set(selected[1]["trial_keys"]), "cross-group streams overlap")
    require(len(groups) == 94 and len(coverage.pair_rows(groups)) == 329 and golden == 188, "index coverage")
    return dict(schema="lns2.sa.matched_aggregation_index.v1", binding=plan["binding"],
                feature_names=plan["feature_names"], entries=entries, groups=groups, golden_vectors=golden,
                independent_confirmation=False, label="within_group_mean_paired_H32_completion_difference")


def load_index(plan, out):
    receipt = prediction.seal_json(out / "index_receipt.json", plan["binding"])
    require(set(receipt["files"]) == {"training_index.json", "feature_receipt.json"}, "index receipt coverage")
    for name, digest in receipt["files"].items():
        require(sha256_file(out / name) == digest, "index changed")
    prediction.verify_features(plan, out)
    return prediction.seal_json(out / "training_index.json", plan["binding"])


def matrix(groups, names, held):
    # Never form labels across collection groups, including shared candidate IDs.
    rows = coverage.training_fold(coverage.pair_rows(groups), held)
    lookup = {g["group_id"]: {c["candidate_id"]: c for c in g["candidates"]} for g in groups if g["map_id"] != held}
    x, y, w, ids = [], [], [], []
    for r in rows:
        a, b = lookup[r["group_id"]][r["left"]], lookup[r["group_id"]][r["right"]]
        for left, right, target in ((a, b, r["target"]), (b, a, -r["target"])):
            x.append(pair_vector(left, right, names))
            y.append(target)
            w.append(r["map_balanced_weight"] / 2)
            ids.append(r["state_id"])
    require(x and any(t != 0 for t in y), "empty or all-tied training fold")
    return dict(x=x, y=y, weights=w, state_ids=ids,
                train_ids=sorted(set(ids)), train_maps=sorted({r["map_id"] for r in rows}),
                group_ids=sorted({r["group_id"] for r in rows}))


def scoring_state(group, entry):
    # Explicit allow-list excludes labels and any post-action fields.
    return dict(state_id=group["state_id"], map_id=group["map_id"], anchor_id=entry["anchor_id"],
                agent_ids=entry["agent_ids"], candidates=[{k: c[k] for k in ("candidate_id", "agents", "features")}
                                                       for c in group["candidates"]])


def rank(model, state):
    ids = [c["candidate_id"] for c in state["candidates"]]
    if len(ids) == 2:
        return prediction.paired_preference(model, state, ids)
    return model.rank(state)


def fit_job(job):
    plan, index, held, directory = job
    import sklearn
    from sklearn.ensemble import HistGradientBoostingRegressor
    require(sklearn.__version__ == "1.5.0", "registered sklearn required")
    out = Path(directory)
    path = out / "folds" / (held + ".pkl")
    require(not path.exists() and not path.with_suffix(".pkl.tmp").exists(), "partial fold requires inspection")
    m = matrix(index["groups"], index["feature_names"], held)
    start = time.monotonic()
    estimator = HistGradientBoostingRegressor(loss="squared_error", **MODEL_PARAMS)
    estimator.fit(m["x"], m["y"], sample_weight=m["weights"])
    model = PairedCompletionModel(list(index["feature_names"]), estimator)
    old_plan = bound_plan(ROOT / plan["recent_plan"])
    old_fold = next(f for f in old_plan["folds"] if f["held"] == held)
    frozen, old_receipt = reconstruction.load_fold(ROOT / RECENT, old_fold, old_plan)
    rows = []
    for g in index["groups"]:
        entry = index["entries"][g["state_id"]]
        state = scoring_state(g, entry)
        pool = {c["candidate_id"]: entry["original_pool_scores"][c["candidate_id"]] for c in g["candidates"]}
        rows.append(dict(group_id=g["group_id"], state_id=g["state_id"], map_id=g["map_id"], kind=g["kind"],
             held=g["map_id"] == held, aggregated_h32=rank(model, state), frozen_h32=rank(frozen, state),
             frozen_pool=dict(scores=pool, selected=prediction.choose(pool, entry["anchor_id"], rounded=True))))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pkl.tmp")
    with tmp.open("xb") as stream:
        pickle.dump(model, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    with path.open("rb") as stream:
        reloaded = pickle.load(stream)
    require([rank(reloaded, scoring_state(g, index["entries"][g["state_id"]])) for g in index["groups"]] ==
            [r["aggregated_h32"] for r in rows], "serialized prediction mismatch")
    result = sealed(dict(binding=plan["binding"], held=held, train_ids=m["train_ids"], train_maps=m["train_maps"],
          train_group_ids=m["group_ids"], train_directed_rows=len(m["x"]), weight_sum=math.fsum(m["weights"]),
          training_matrix_fingerprint=json_fingerprint(m), rows=rows, model_sha256=sha256_file(path),
          frozen_model_sha256=old_receipt["model_sha256"], feature_names=index["feature_names"], model_params=MODEL_PARAMS,
          seconds=time.monotonic()-start, sklearn=sklearn.__version__, python=sys.version, numerical_threads=1,
          index_sha256=sha256_file(out / "training_index.json"), deployment_allowed=False))
    once(path.with_suffix(".json"), result)
    return dict(held=held, rows=len(rows), directed_training_rows=len(m["x"]), seconds=result["seconds"])


def verify_fold(plan, index, out, held):
    row = prediction.seal_json(out / "folds" / (held + ".json"), plan["binding"])
    require(row["held"] == held and row["model_params"] == MODEL_PARAMS and
            row["feature_names"] == index["feature_names"] and row["index_sha256"] == sha256_file(out / "training_index.json"),
            "fold config mismatch")
    require(row["train_ids"] == sorted({g["state_id"] for g in index["groups"] if g["map_id"] != held}) and
            row["train_maps"] == sorted(set(plan["maps"]) - {held}) and
            row["train_group_ids"] == sorted(g["group_id"] for g in index["groups"] if g["map_id"] != held), "fold leakage")
    require(row["training_matrix_fingerprint"] == json_fingerprint(matrix(index["groups"], index["feature_names"], held)), "training matrix drift")
    require(sha256_file(out / "folds" / (held + ".pkl")) == row["model_sha256"], "model changed")
    expected = {g["group_id"]: g for g in index["groups"]}
    require(len(row["rows"]) == len(expected) and {r["group_id"] for r in row["rows"]} == set(expected), "prediction group coverage")
    for p in row["rows"]:
        g = expected[p["group_id"]]
        require((p["state_id"], p["map_id"], p["kind"], p["held"]) ==
                (g["state_id"], g["map_id"], g["kind"], g["map_id"] == held), "prediction identity")
        ids = {c["candidate_id"] for c in g["candidates"]}
        for method in ("aggregated_h32", "frozen_h32", "frozen_pool"):
            require(p[method]["selected"] in ids and set(p[method]["scores"]) == ids and
                    all(math.isfinite(v) for v in p[method]["scores"].values()), "invalid prediction")
    return row


def fit():
    plan, out = verify()
    index = load_index(plan, out)
    with _CollectionRunLock(out, plan["binding"], "aggregation-fit"):
        pending = []
        for held in plan["maps"]:
            if (out / "folds" / (held + ".json")).exists():
                verify_fold(plan, index, out, held)
            else:
                pending.append((plan, index, held, str(out)))
        phase_status(out, plan["binding"], "fit", status="running", remaining=len(pending))
        try:
            if pending:
                with ProcessPoolExecutor(max_workers=min(20, len(pending))) as pool:
                    futures = [pool.submit(fit_job, job) for job in pending]
                    for count, future in enumerate(as_completed(futures), 1):
                        row = future.result()
                        print("FIT", count, "/", len(pending), json.dumps(row), flush=True)
                        phase_status(out, plan["binding"], "fit", status="running", remaining=len(pending)-count)
            for held in plan["maps"]:
                verify_fold(plan, index, out, held)
            files = {"folds/" + m + ext: sha256_file(out / "folds" / (m + ext))
                     for m in plan["maps"] for ext in (".json", ".pkl")}
            once(out / "fit_receipt.json", sealed(dict(binding=plan["binding"], files=files)))
            check_inputs(plan["inputs"])
            phase_status(out, plan["binding"], "fit", status="completed")
            return dict(folds=8, new_fits=8, solver_calls=0)
        except BaseException as exc:
            phase_status(out, plan["binding"], "fit", status="error", error=repr(exc))
            raise


def evaluate(groups, predictions, samples=5000, seed=202609183):
    require(len({g["group_id"] for g in groups}) == len(groups) == len(predictions) and
            {p["group_id"] for p in predictions} == {g["group_id"] for g in groups}, "evaluation coverage")
    lookup = {g["group_id"]: g for g in groups}
    rows = []
    for p in predictions:
        g = lookup[p["group_id"]]
        require(p["held"] is True and (p["state_id"], p["map_id"], p["kind"]) ==
                (g["state_id"], g["map_id"], g["kind"]), "evaluation is not held-map prediction")
        require(all(len(c["values"]) == 8 and all(type(v) is bool for v in c["values"]) for c in g["candidates"]), "incomplete evaluation labels")
        rates = {c["candidate_id"]: mean(c["values"]) for c in g["candidates"]}
        row = dict(state_id=g["state_id"], map_id=g["map_id"], group_id=g["group_id"], kind=g["kind"],
                   family=g["family"], uniform=mean(rates.values()), empirical_best=max(rates.values()), rates=rates,
                   selected={m: p[m]["selected"] for m in ("aggregated_h32", "frozen_h32", "frozen_pool")})
        for method in row["selected"]:
            require(row["selected"][method] in rates, "selected unknown candidate")
            row[method] = rates[row["selected"][method]]
        for baseline in ("uniform", "frozen_pool", "frozen_h32"):
            row["gain_vs_" + baseline] = row["aggregated_h32"] - row[baseline]
        row["oracle_gap"] = row["empirical_best"] - row["aggregated_h32"]
        rows.append(row)
    sections = {}
    for kind in ("matched_pair", "old_grid"):
        selected = [r for r in rows if r["kind"] == kind]
        require(selected and len({r["state_id"] for r in selected}) == len(selected), "duplicate root in estimand")
        maps = sorted({r["map_id"] for r in selected})
        rng = random.Random(seed)
        draws = [[rng.randrange(len(maps)) for _ in maps] for _ in range(samples)]
        cols = ("uniform", "frozen_pool", "frozen_h32", "aggregated_h32", "empirical_best",
                "gain_vs_uniform", "gain_vs_frozen_pool", "gain_vs_frozen_h32", "oracle_gap")
        sections[kind] = dict(states=len(selected), maps=len(maps),
            metrics={k: prediction.aggregate(selected, k, draws) for k in cols},
            changed_vs_frozen_h32=sum(r["selected"]["aggregated_h32"] != r["selected"]["frozen_h32"] for r in selected),
            empirical_equal_rates=sum(len(set(r["rates"].values())) == 1 for r in selected))
    return dict(rows=rows, sections=sections)


def analyze():
    plan, out = verify()
    index = load_index(plan, out)
    receipt = prediction.seal_json(out / "fit_receipt.json", plan["binding"])
    expected = {"folds/" + m + ext for m in plan["maps"] for ext in (".json", ".pkl")}
    require(set(receipt["files"]) == expected, "fit receipt coverage")
    for name, digest in receipt["files"].items():
        require(sha256_file(out / name) == digest, "fit output drift")
    held = []
    for m in plan["maps"]:
        f = verify_fold(plan, index, out, m)
        held.extend(r for r in f["rows"] if r["held"])
    result = evaluate(index["groups"], held, plan["config"]["bootstrap"], plan["config"]["bootstrap_seed"])
    # Training predictions are descriptive; never mix with held-map estimates.
    training = []
    groups = {g["group_id"]: g for g in index["groups"]}
    for m in plan["maps"]:
        f = prediction.seal_json(out / "folds" / (m + ".json"), plan["binding"])
        values = []
        for p in f["rows"]:
            if p["held"]:
                continue
            g = groups[p["group_id"]]
            rates = {c["candidate_id"]: mean(c["values"]) for c in g["candidates"]}
            values.append(dict(state_id=g["state_id"], map_id=g["map_id"], kind=g["kind"],
                gain= rates[p["aggregated_h32"]["selected"]] - mean(rates.values())))
        training.append(dict(held=m, map_equal_gain={kind: mean(mean(r["gain"] for r in values if r["kind"] == kind and r["map_id"] == n)
                for n in plan["maps"] if n != m) for kind in ("matched_pair", "old_grid")}))
    metrics = result["sections"]["matched_pair"]["metrics"]
    positive = all(metrics["gain_vs_" + b]["mean"] > 0 for b in ("uniform", "frozen_pool", "frozen_h32"))
    report = sealed(result | dict(schema="lns2.sa.matched_aggregation_report.v1", binding=plan["binding"],
         fit_receipt_sha256=sha256_file(out / "fit_receipt.json"), index_sha256=sha256_file(out / "training_index.json"),
         training_descriptive=training, features=127, roots=47, maps=8, groups=94, new_fits=8, new_labels=0,
         solver_calls=0, automatic_promotion=False, independent_confirmation=False, thresholds_applied=False,
         decision="positive_development_mean_requires_independent_validation" if positive else "no_consistent_development_mean_advantage",
         model_target="H32_completion_difference_not_one_step_drop"))
    once(out / "report.json", report)
    lines = ["# H32 grouped aggregation development validation", "", "No solver calls, tuning, formal TTF, or automatic promotion.",
             "", "| Group | Method | Map-equal H32 rate |", "|---|---|---:|"]
    for kind, section in report["sections"].items():
        for method in plan["config"]["methods"]:
            lines.append(f"| {kind} | {method} | {section['metrics'][method]['mean']:.6%} |")
    lines += ["", "Primary paired contrasts (percentage points):"]
    for b in ("uniform", "frozen_pool", "frozen_h32"):
        v = metrics["gain_vs_" + b]
        lines.append(f"- vs {b}: {100*v['mean']:.6f}; map bootstrap 95% [{100*v['ci95'][0]:.6f}, {100*v['ci95'][1]:.6f}]")
    lines += ["", report["decision"], "", "All maps were previously viewed. H32 completion is not end-to-end TTF or a live-policy success rate."]
    text = "\n".join(lines) + "\n"
    path = out / "report.md"
    if path.exists():
        require(path.read_text(encoding="utf-8") == text, "report text changed")
    else:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
    once(out / "complete.json", sealed(dict(binding=plan["binding"], files={n: sha256_file(out / n)
         for n in ("report.json", "report.md", "fit_receipt.json", "index_receipt.json")})))
    phase_status(out, plan["binding"], "analyze", status="completed")
    return dict(decision=report["decision"], sections=report["sections"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "features", "fit", "analyze", "verify"))
    args = parser.parse_args()
    if args.phase == "verify":
        plan, out = verify()
        if (out / "index_receipt.json").exists():
            index = load_index(plan, out)
            if (out / "fit_receipt.json").exists():
                for m in plan["maps"]:
                    verify_fold(plan, index, out, m)
        if (out / "complete.json").exists():
            complete = prediction.seal_json(out / "complete.json", plan["binding"])
            for name, digest in complete["files"].items():
                require(sha256_file(out / name) == digest, "completed output changed")
        result = dict(binding=plan["binding"], registered_inputs=len(plan["inputs"]))
    else:
        result = globals()[args.phase]()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
