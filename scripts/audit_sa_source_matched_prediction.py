"""Frozen input-only scoring on matched pairs; no fit, reset, or repair calls."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments._common import json_fingerprint, read_json, sha256_file
from experiments.sa_history_information import profile_features
from experiments.sa_paired_completion import MODEL_PARAMS, pair_vector
from scripts.audit_sa_history_information import atomic, require
from scripts.collect_sa_source_matched import merge_pinned, NATIVE, NATIVE_SHA
from scripts.run_sa_paired_closed_loop import once

CONFIG = ROOT / "configs/sa_source_matched_prediction.json"
HISTORICAL = "historical_16_state_lomo_gbdt"
POOL = "frozen_dual16_pool_score"


def check_folds(folds, training, maps):
    require(len(folds) == len(maps) and {f["map_id"] for f in folds} == set(maps), "fold map coverage")
    require(len({s["state_id"] for s in training}) == len(training) == 16, "historical training cohort")
    for fold in folds:
        expected = sorted(s["state_id"] for s in training if s["map_id"] != fold["map_id"])
        expected_maps = sorted({s["map_id"] for s in training if s["map_id"] != fold["map_id"]})
        require(fold["train_ids"] == expected and fold["train_maps"] == expected_maps,
                "historical fold train/held leakage")


def score_coverage(roots, predictions):
    rows = []
    for root in roots:
        scores = predictions[root["state_id"]]["scores"]
        pair = root["pair_ids"]
        rows.append(dict(state_id=root["state_id"], pair_ids=pair,
                         available_pair_ids=[c for c in pair if c in scores],
                         missing_pair_ids=[c for c in pair if c not in scores],
                         available_union_ids=[c["candidate_id"] for c in root["candidates"] if c["candidate_id"] in scores]))
    return dict(rows=rows, pair_candidate_scores=sum(len(r["available_pair_ids"]) for r in rows),
                union_candidate_scores=sum(len(r["available_union_ids"]) for r in rows),
                complete_pairs=sum(not r["missing_pair_ids"] for r in rows))


def build_plan():
    cfg = read_json(CONFIG)
    require(cfg["solver_calls"] == 0 and cfg["no_ttf"] and not cfg["training_allowed"], "audit boundary changed")
    require((ROOT/cfg["output"]).resolve().is_relative_to((ROOT/"build").resolve()), "unsafe output")
    inputs = {}
    def load(rel, digest=None):
        actual = sha256_file(ROOT / rel)
        require(digest is None or actual == digest, "registered file changed: " + rel)
        inputs[rel] = actual
        return read_json(ROOT / rel)
    evidence = load(cfg["recovery_evidence"], cfg["recovery_evidence_sha256"])
    merge_pinned(inputs, evidence["files"])
    source = load("build/sa-source-matched-collection-v1/plan.json")
    merge_pinned(inputs, source["inputs"])
    require(source["binding"] == json_fingerprint({k:v for k,v in source.items() if k != "binding"}), "source binding")
    prep_cfg = load(cfg["preparation_config"])
    data = load(prep_cfg["index"], prep_cfg["index_sha256"])
    receipt = load(prep_cfg["prediction_receipt"], prep_cfg["prediction_receipt_sha256"])
    base = Path(prep_cfg["prediction_receipt"]).parent
    predictions = {}
    for name, digest in receipt["files"].items():
        if not name.startswith("fits/gbdt-"):
            continue
        fit = load((base / name).as_posix(), digest)
        require(fit["binding"] == receipt["binding"] and fit["integrity"] ==
                json_fingerprint({k:v for k,v in fit.items() if k != "integrity"}), "recent fit identity")
        expected = sorted(s["state_id"] for s in data["states"] if s["map_id"] != fit["held"])
        require(fit["train_ids"] == expected, "recent model map leakage")
        held_ids = {s["state_id"] for s in data["states"] if s["map_id"] == fit["held"]}
        for row in fit["rows"]:
            if row["held"]:
                require(row["state_id"] in held_ids and row["state_id"] not in predictions, "recent prediction identity")
                predictions[row["state_id"]] = row
    require(set(predictions) == {s["state_id"] for s in data["states"]}, "recent held prediction missing")
    availability = score_coverage(source["roots"], predictions)
    availability["registered_weight_files"] = [n for n in receipt["files"] if n.endswith((".pkl", ".pt", ".pth")) or "bundle" in n]
    availability["status"] = "not_evaluable_missing_pair_scores_and_weights"
    require(availability["complete_pairs"] == 0 and not availability["registered_weight_files"], "review changed availability")
    fold_plan = load(cfg["historical_fold_plan"], cfg["historical_fold_plan_sha256"])
    require(fold_plan["binding"] == json_fingerprint({k:v for k,v in fold_plan.items() if k != "binding"}), "historical plan binding")
    fold_base = Path(cfg["historical_fold_plan"]).parent
    evaluation_receipt = load((fold_base / "evaluation_receipt.json").as_posix())
    golden_report = load((fold_base / "model_report.json").as_posix(), evaluation_receipt["report_sha256"])
    require(evaluation_receipt["binding"] == golden_report["binding"] == fold_plan["binding"], "historical evaluation identity")
    golden_predictions = {p["state_id"]:p["old"]["paired"] for f in golden_report["folds"] for p in f["predictions"]}
    old_index = fold_plan["old_output"] + "/training_index.json"
    training = load(old_index, fold_plan["inputs"][old_index])
    maps = sorted({r["map_id"] for r in source["roots"]})
    check_folds(fold_plan["frozen_folds"], training["states"], maps)
    require(training["feature_names"] == data["feature_names"] and len(training["feature_names"]) == 127, "feature schema changed")
    merge_pinned(inputs, {f["file"]:f["sha256"] for f in fold_plan["frozen_folds"]})
    for name in ("experiments/sa_paired_completion.py", "experiments/sa_unbalanced_coverage.py"):
        merge_pinned(inputs, {name: fold_plan["inputs"][name]})
    for root in source["roots"]:
        load(root["source_root"], source["inputs"][root["source_root"]])
    for rel in (CONFIG.relative_to(ROOT).as_posix(), Path(__file__).relative_to(ROOT).as_posix(),
                "tests/evaluation/test_sa_source_matched_prediction.py", "docs/SA_SOURCE_MATCHED_PREDICTION_ZH.md"):
        inputs[rel] = sha256_file(ROOT / rel)
    require(inputs[NATIVE] == NATIVE_SHA, "native changed")
    plan = dict(config=cfg, inputs=inputs, roots=source["roots"], feature_names=training["feature_names"],
                folds=fold_plan["frozen_folds"], old_training_index=old_index,
                golden_predictions={r["state_id"]:golden_predictions[r["state_id"]] for r in source["roots"] if r["state_id"] in golden_predictions},
                recent_model_availability=availability, methods=[HISTORICAL, POOL],
                model_params=fold_plan["model"], native_sha256=NATIVE_SHA,
                label_records="build/sa-source-matched-resource-recovery-v1/analysis_records.json",
                commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    require(plan["model_params"] == MODEL_PARAMS, "historical model contract drift")
    plan["binding"] = json_fingerprint(plan)
    return plan, ROOT / cfg["output"]


def verify():
    cfg = read_json(CONFIG)
    out = ROOT / cfg["output"]
    require(out.resolve().is_relative_to((ROOT / "build").resolve()), "unsafe output")
    plan = read_json(out / "plan.json")
    require(plan["config"] == cfg and plan["binding"] == json_fingerprint({k:v for k,v in plan.items() if k != "binding"}), "plan identity")
    for rel, digest in plan["inputs"].items():
        require(sha256_file(ROOT / rel) == digest, "input drift: " + rel)
    return plan, out


def feature_state(root, candidates, names, engine_factory):
    # Only current observation and pre-action proposal data enter the extractor.
    from experiments.repair_collection import state_fingerprint
    require(state_fingerprint(root["state"]) == root["state_fingerprint"], "root observation changed")
    rows, _ = engine_factory(root["state"], backend="native").realized_rows(candidates, state_hash=root["state_fingerprint"])
    expected_ids = [c["candidate_id"] for c in candidates]
    require([r["candidate_id"] for r in rows] == expected_ids, "feature candidate order changed")
    by_id = {}
    for candidate, row in zip(candidates, rows):
        features = row["features"]["realized_dynamic"] | {
            "source.score": candidate["score"], "sa.log_decision": math.log1p(root["source"]["decision"]),
            "sa.log_temperature": math.log1p(root["control_event"]["temperature"])}
        require(sorted(features) == names and all(math.isfinite(v) for v in features.values()), "invalid feature vector")
        by_id[candidate["candidate_id"]] = dict(candidate_id=candidate["candidate_id"], agents=list(candidate["agents"]), features=features)
    require(len(root["candidates"]) == len(root["feature_rows"]), "golden feature shape")
    for candidate, saved in zip(root["candidates"], root["feature_rows"]):
        require(by_id[candidate["candidate_id"]]["features"] == profile_features(dict(base=saved), "dynamic"),
                "golden native feature mismatch: " + candidate["candidate_id"])
    return by_id


def feature_job(job):
    from experiments.online_feature_engine import OnlineFeatureEngine
    import lns2_env
    entry, plan = job["entry"], job["plan"]
    require(sha256_file(Path(lns2_env.__file__)) == plan["native_sha256"], "wrong feature native loaded")
    root = read_json(ROOT / entry["source_root"])
    lookup = {c["candidate_id"]:c for c in root["control_event"]["pool"]}
    wanted = {c["candidate_id"] for c in root["candidates"]} | {c["candidate_id"] for c in entry["candidates"]}
    candidates = [lookup[c] for c in sorted(wanted)]
    features = feature_state(root, candidates, plan["feature_names"], OnlineFeatureEngine)
    payload = dict(binding=plan["binding"], state_id=entry["state_id"], map_id=entry["map_id"],
                   anchor_id=entry["anchor_id"], agent_ids=[a["id"] for a in root["state"]["agents"]],
                   pair_ids=entry["pair_ids"], candidates=[features[c["candidate_id"]] for c in entry["candidates"]],
                   control_candidates=[features[c["candidate_id"]] for c in root["candidates"]],
                   original_pool_scores={c["candidate_id"]:c["score"] for c in candidates},
                   golden_feature_candidates=len(root["candidates"]), solver_calls=0)
    payload["integrity"] = json_fingerprint(payload)
    return payload


def seal_json(path, binding):
    row = read_json(path)
    require(row["binding"] == binding and row["integrity"] == json_fingerprint({k:v for k,v in row.items() if k != "integrity"}), "output integrity")
    return row


def feature_phase():
    from experiments.repair_collection import _CollectionRunLock
    plan, out = verify()
    with _CollectionRunLock(out, plan["binding"], "frozen-prediction-features"):
        pending = []
        for entry in plan["roots"]:
            path = out / "features" / (entry["state_id"] + ".json")
            if path.exists():
                seal_json(path, plan["binding"])
            else:
                pending.append(dict(entry=entry, plan=plan))
        atomic(out / "run_status.json", dict(binding=plan["binding"], phase="features", status="running"))
        try:
            with ProcessPoolExecutor(max_workers=min(plan["config"]["feature_workers"], max(1,len(pending)))) as pool:
                for row in pool.map(feature_job, pending):
                    once(out / "features" / (row["state_id"] + ".json"), row)
                    print("FEATURES", row["state_id"], flush=True)
            files = {("features/"+r["state_id"]+".json"):sha256_file(out / "features" / (r["state_id"]+".json")) for r in plan["roots"]}
            once(out / "feature_receipt.json", dict(binding=plan["binding"], files=files))
            verify()
            atomic(out / "run_status.json", dict(binding=plan["binding"], phase="features", status="completed"))
            return dict(states=len(files), max_workers=min(len(plan["roots"]),plan["config"]["feature_workers"]), solver_calls=0)
        except BaseException as exc:
            atomic(out / "run_status.json", dict(binding=plan["binding"], status="error", error=repr(exc)))
            raise


def choose(scores, anchor, rounded=False):
    require(scores and all(math.isfinite(v) for v in scores.values()), "invalid scores")
    return min(scores, key=lambda c: (-(round(scores[c],12) if rounded else scores[c]), c != anchor if not rounded else False, c))


def paired_preference(model, state, ids):
    a,b = [next(c for c in state["candidates"] if c["candidate_id"] == cid) for cid in sorted(ids)]
    f,r = model.estimator.predict([pair_vector(a,b,model.feature_names), pair_vector(b,a,model.feature_names)])
    require(math.isfinite(f) and math.isfinite(r), "nonfinite pair margin")
    margin = max(-1.,min(1.,float(f)/2-float(r)/2))
    scores = {a["candidate_id"]:margin,b["candidate_id"]:-margin}
    return dict(selected=choose(scores,state["anchor_id"]), scores=scores, forward=float(f), reverse=float(r))


def verify_features(plan,out):
    rc=read_json(out/"feature_receipt.json")
    require(rc["binding"]==plan["binding"],"feature receipt binding")
    expected={"features/"+r["state_id"]+".json" for r in plan["roots"]}
    require(set(rc["files"])==expected,"feature receipt coverage")
    for name,digest in rc["files"].items():
        require(sha256_file(out/name)==digest,"features changed")
    return rc


def predict_phase():
    import sklearn
    require(sklearn.__version__ == "1.5.0", "registered sklearn required")
    plan,out = verify()
    verify_features(plan,out)
    folds = {}
    for f in plan["folds"]:
        require(sha256_file(ROOT/f["file"]) == f["sha256"], "pickle bytes changed before load")
        with (ROOT/f["file"]).open("rb") as stream:
            model = pickle.load(stream)
        require(model["held"] == f["map_id"] and model["train_ids"] == f["train_ids"] and
                model["train_maps"] == f["train_maps"] and model["names"] == plan["feature_names"] and
                model["paired"].feature_names == plan["feature_names"], "serialized model identity")
        require(all(model["paired"].estimator.get_params()[k] == v for k,v in plan["model_params"].items()), "model parameters changed")
        folds[f["map_id"]] = model["paired"]
    rows = []
    golden_matches = 0
    for entry in plan["roots"]:
        state = seal_json(out / "features" / (entry["state_id"]+".json"),plan["binding"])
        require(state["state_id"] not in next(f for f in plan["folds"] if f["map_id"] == state["map_id"])["train_ids"], "evaluation state leaked")
        model = folds[state["map_id"]]
        if state["state_id"] in plan["golden_predictions"]:
            control_state = dict(state, candidates=state["control_candidates"])
            require(model.rank(control_state) == plan["golden_predictions"][state["state_id"]], "historical prediction mismatch")
            golden_matches += 1
        primary = paired_preference(model,state,state["pair_ids"])
        union = model.rank(state)
        pool_scores = {c:state["original_pool_scores"][c] for c in state["pair_ids"]}
        rows.append(dict(state_id=state["state_id"], map_id=state["map_id"], anchor_id=state["anchor_id"],
                         pair_ids=state["pair_ids"], historical_model=primary, historical_union=union,
                         frozen_pool=dict(selected=choose(pool_scores,state["anchor_id"],rounded=True),scores=pool_scores)))
    result = dict(binding=plan["binding"], rows=rows, feature_receipt_sha256=sha256_file(out/"feature_receipt.json"),
                  model_version=HISTORICAL, training_calls=0, label_values_used_for_scoring=False,
                  golden_prediction_matches=golden_matches)
    result["integrity"] = json_fingerprint(result)
    once(out/"predictions.json",result)
    verify()
    return dict(states=len(rows), model_version=HISTORICAL, training_calls=0)


def aggregate(rows, key, draws):
    from experiments.sa_source_matched_analysis import _quantile
    maps=sorted({r["map_id"] for r in rows})
    per_map={m:sum(r[key] for r in rows if r["map_id"]==m)/sum(r["map_id"]==m for r in rows) for m in maps}
    values=[per_map[m] for m in maps]
    samples=sorted(sum(values[i] for i in draw)/len(maps) for draw in draws)
    return dict(mean=sum(values)/len(values),ci95=[_quantile(samples,.025),_quantile(samples,.975)],per_map=per_map,
                positive_states=sum(r[key]>0 for r in rows),negative_states=sum(r[key]<0 for r in rows),zero_states=sum(r[key]==0 for r in rows))


def evaluate(records, predictions, samples=5000, seed=202609183):
    require({r["state_id"] for r in records} == {r["state_id"] for r in predictions}, "prediction/label cohort changed")
    require(len({r["state_id"] for r in predictions}) == len(predictions) == len(records), "duplicate state")
    labels={r["state_id"]:r for r in records}
    rows=[]
    for p in predictions:
        r=labels[p["state_id"]]
        require(r["map_id"] == p["map_id"] and r["pair_ids"] == p["pair_ids"] and r["anchor_id"] == p["anchor_id"], "prediction identity changed")
        require(all(len(v)==8 and all(type(x) is bool for x in v) for v in r["values"].values()), "incomplete labels")
        rates={c:sum(v)/8 for c,v in r["values"].items()}
        uniform=sum(rates[c] for c in r["pair_ids"])/2
        row=dict(state_id=r["state_id"],map_id=r["map_id"],rates=rates,pair_uniform=uniform,anchor=rates[r["anchor_id"]],
                 empirical_best_pair=max(rates[c] for c in r["pair_ids"]))
        for method,column in ((HISTORICAL,"historical_model"),(POOL,"frozen_pool")):
            cid=p[column]["selected"]
            require(cid in r["pair_ids"], "selection outside fixed pair")
            row[method]=rates[cid]
            row[method+"_minus_uniform"]=rates[cid]-uniform
            row[method+"_minus_anchor"]=rates[cid]-row["anchor"]
            row[method+"_selected"]=cid
        cid=p["historical_union"]["selected"]
        require(cid in rates, "union selection outside registered candidates")
        row["historical_union"]=rates[cid]
        row["historical_union_minus_anchor"]=rates[cid]-row["anchor"]
        rows.append(row)
    rows.sort(key=lambda r:(r["map_id"],r["state_id"]))
    maps=sorted({r["map_id"] for r in rows})
    rng=random.Random(seed)
    draws=[tuple(rng.randrange(len(maps)) for _ in maps) for _ in range(samples)]
    keys=["pair_uniform","anchor","empirical_best_pair",HISTORICAL,POOL,"historical_union",
          HISTORICAL+"_minus_uniform",POOL+"_minus_uniform",HISTORICAL+"_minus_anchor",POOL+"_minus_anchor","historical_union_minus_anchor"]
    return dict(rows=rows,metrics={k:aggregate(rows,k,draws) for k in keys},states=len(rows),maps=len(maps),
                trials_are_independent_states=False,bootstrap_samples=samples,bootstrap_seed=seed,
                thresholds_applied=False,promotion_allowed=False,independent_confirmation=False,
                primary_contrast="within_fixed_pair",union_comparison="secondary_descriptive")


def analyze():
    plan,out=verify()
    verify_features(plan,out)
    p=seal_json(out/"predictions.json",plan["binding"])
    require(p["feature_receipt_sha256"]==sha256_file(out/"feature_receipt.json"),"prediction features changed")
    records=read_json(ROOT/plan["label_records"])
    report=evaluate(records,p["rows"],plan["config"]["bootstrap"],plan["config"]["bootstrap_seed"])
    report.update(binding=plan["binding"],role=plan["config"]["role"],recent_model_availability=plan["recent_model_availability"],
                  historical_model_version=HISTORICAL,model_replacement=False,solver_calls=0,training_calls=0,no_ttf=True,
                  prediction_sha256=sha256_file(out/"predictions.json"),label_sha256=sha256_file(ROOT/plan["label_records"]))
    once(out/"report.json",report)
    return {k:v for k,v in report.items() if k not in ("rows","recent_model_availability")}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("prepare","verify","features","predict","analyze"))
    args=parser.parse_args()
    if args.phase=="prepare":
        plan,out=build_plan()
        once(out/"plan.json",plan)
        result=dict(binding=plan["binding"],states=len(plan["roots"]),recent_complete_pairs=plan["recent_model_availability"]["complete_pairs"])
    elif args.phase=="verify":
        plan,out=verify()
        result=dict(binding=plan["binding"],inputs=len(plan["inputs"]))
    elif args.phase=="features": result=feature_phase()
    elif args.phase=="predict": result=predict_phase()
    else: result=analyze()
    print(json.dumps(result,indent=2,sort_keys=True,ensure_ascii=False))


if __name__=="__main__":
    main()
