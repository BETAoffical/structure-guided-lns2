"""Rebuild only the original 47-state GBDT folds; require exact old predictions."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
import time

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments._common import contained_file, json_fingerprint, read_json, sha256_file
from experiments.sa_paired_completion import MODEL_PARAMS, PairedCompletionModel, require, training_matrix
from experiments.sa_unbalanced_coverage import state_weights
from scripts import audit_sa_source_matched_prediction as previous
from scripts.audit_sa_history_information import atomic
from scripts.run_sa_paired_closed_loop import once

CONFIG = ROOT / "configs/sa_recent_model_reconstruction.json"
RECENT = "reconstructed_47_state_lomo_gbdt"


def sealed(value):
    return value | {"integrity": json_fingerprint(value)}


def check_contract(cfg):
    require(not cfg["new_labels_allowed_in_fit"] and not cfg["hyperparameter_changes_allowed"] and
            not cfg["formal_ttf_allowed"] and not cfg["promotion_allowed"] and cfg["solver_calls"] == 0,
            "reconstruction boundary changed")
    require(1 <= cfg["workers"] <= 20, "worker bound")


def prepare():
    cfg = read_json(CONFIG)
    check_contract(cfg)
    out = ROOT / cfg["output"]
    require(out.resolve().is_relative_to((ROOT / "build").resolve()) and not out.exists(),
            "existing or unsafe output; use verify/resume")
    from scripts import run_sa_spatiotemporal_model_probe as source
    old, old_out = source.verify()
    require(sha256_file(ROOT / cfg["source_plan"]) == cfg["source_plan_sha256"], "old plan changed")
    require((old_out / "plan.json").resolve() == (ROOT / cfg["source_plan"]).resolve(), "old plan path")
    require(sha256_file(old_out / "complete.json") == cfg["source_receipt_sha256"], "old receipt changed")
    receipt = read_json(old_out / "complete.json")
    require(receipt["binding"] == old["binding"], "old receipt binding")
    index = old_out / "development_index.json"
    require(sha256_file(index) == cfg["training_index_sha256"], "old training data changed")
    data = read_json(index)
    require(json_fingerprint(data) == old["data_fingerprint"], "old dataset fingerprint")
    require(len(data["states"]) == cfg["expected_states"] == 47 and
            len(data["feature_names"]) == cfg["expected_features"] == 127, "old schema changed")
    require(data["horizon"] == 32 and data["trial_count"] == 8 and
            all(len(s["candidates"]) == 4 for s in data["states"]), "old label or candidate budget changed")
    maps = sorted({s["map_id"] for s in data["states"]})
    require(maps == old["maps"] and len(maps) == cfg["expected_maps"] == 8, "old maps changed")
    prev, prev_out = previous.verify()
    previous.verify_features(prev, prev_out)
    require(sha256_file(prev_out / "plan.json") == cfg["previous_prediction_plan_sha256"], "previous plan changed")
    require(sha256_file(prev_out / "report.json") == cfg["previous_report_sha256"], "previous report changed")
    require(data["feature_names"] == prev["feature_names"], "candidate feature schema drift")
    inputs = dict(old["inputs"])
    for name, digest in prev["inputs"].items():
        require(name not in inputs or inputs[name] == digest, "conflicting input pin")
        inputs[name] = digest
    folds = []
    for m in maps:
        path = source.result_file(old_out, "gbdt", m, MODEL_PARAMS["random_state"])
        name = path.relative_to(old_out).as_posix()
        require(sha256_file(path) == receipt["files"][name], "old fold changed")
        golden = source.sealed_fit(path, old["binding"])
        check_golden_identity(data, m, golden)
        folds.append(dict(held=m, golden=path.relative_to(ROOT).as_posix()))
        inputs[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    paths = [CONFIG, Path(__file__), ROOT / "tests/evaluation/test_sa_recent_model_reconstruction.py",
             ROOT / "docs/SA_RECENT_MODEL_RECONSTRUCTION_PROTOCOL_ZH.md", ROOT / cfg["source_plan"],
             old_out / "complete.json", prev_out / "plan.json", prev_out / "report.json",
             prev_out / "feature_receipt.json", prev_out / "predictions.json",
             ROOT / "scripts/run_sa_paired_closed_loop.py", ROOT / "scripts/audit_sa_history_information.py"]
    paths += [prev_out / f for f in read_json(prev_out / "feature_receipt.json")["files"]]
    for path in paths:
        inputs[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    for name, digest in inputs.items():
        require(sha256_file(contained_file(ROOT, name, field="reconstruction input")) == digest, "input drift: " + name)
    plan = dict(config=cfg, inputs=inputs, folds=folds, model_params=MODEL_PARAMS,
                training_index=index.relative_to(ROOT).as_posix(), training_binding=old["binding"],
                feature_names=data["feature_names"], previous_output=prev_out.relative_to(ROOT).as_posix(),
                previous_binding=prev["binding"], label_records=prev["label_records"], roots=prev["roots"],
                commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    plan["binding"] = json_fingerprint(plan)
    once(out / "plan.json", plan)
    return dict(binding=plan["binding"], fits=8, training_states=47, new_label_training=False)


def verify():
    cfg = read_json(CONFIG)
    check_contract(cfg)
    out = ROOT / cfg["output"]
    require(out.resolve().is_relative_to((ROOT / "build").resolve()), "unsafe output")
    plan = read_json(out / "plan.json")
    require(plan["config"] == cfg and plan["binding"] == json_fingerprint({k:v for k,v in plan.items() if k != "binding"}),
            "plan identity")
    require(plan["model_params"] == MODEL_PARAMS, "parameter drift")
    for name, digest in plan["inputs"].items():
        require(sha256_file(contained_file(ROOT, name, field="reconstruction input")) == digest, "input drift: " + name)
    return plan, out


def check_golden_identity(data, held, golden):
    require((golden["method"], golden["held"], golden["seed"], golden["sklearn"]) ==
            ("gbdt", held, MODEL_PARAMS["random_state"], "1.5.0"), "golden fold identity")
    require(golden["train_ids"] == sorted(s["state_id"] for s in data["states"] if s["map_id"] != held),
            "golden train-map leakage")
    require(len(golden["rows"]) == len(data["states"]) and
            {r["state_id"] for r in golden["rows"]} == {s["state_id"] for s in data["states"]}, "golden coverage")
    states = {s["state_id"]:s for s in data["states"]}
    for row in golden["rows"]:
        state = states[row["state_id"]]
        require(row["held"] == (state["map_id"] == held), "golden held flag")
        require(set(row["scores"]) == {c["candidate_id"] for c in state["candidates"]} and
                row["selected"] in row["scores"], "golden candidate coverage")


def old_predictions(model, data, held):
    rows = []
    for state in data["states"]:
        rank = model.rank(state)
        rows.append(dict(state_id=state["state_id"], selected=rank["selected"], scores=rank["scores"],
                         held=state["map_id"] == held))
    return rows


def fit_exact(data, held, golden, factory=None):
    check_golden_identity(data, held, golden)
    matrix = training_matrix(data, {held})
    weights = state_weights(data["states"], held)
    if factory is None:
        import sklearn
        from sklearn.ensemble import HistGradientBoostingRegressor
        require(sklearn.__version__ == "1.5.0", "registered sklearn required")
        factory = HistGradientBoostingRegressor
    # This is the original gbdt_job fit expression, without new data or tuning.
    estimator = factory(loss="squared_error", **MODEL_PARAMS).fit(matrix["x"], matrix["y"],
        sample_weight=[w*weights[s] for w,s in zip(matrix["weights"],matrix["state_ids"])])
    model = PairedCompletionModel(data["feature_names"], estimator)
    rows = old_predictions(model, data, held)
    require(rows == golden["rows"], "reconstruction prediction mismatch; no new-candidate evaluation")
    return model, rows


def load_fold(out, fold, plan):
    stem = "folds/" + fold["held"]
    receipt = previous.seal_json(out / (stem + ".json"), plan["binding"])
    require(receipt["held"] == fold["held"] and receipt["golden_sha256"] == plan["inputs"][fold["golden"]],
            "reconstructed fold identity")
    path = out / (stem + ".pkl")
    require(sha256_file(path) == receipt["model_sha256"], "model bytes changed before load")
    with path.open("rb") as stream:
        model = pickle.load(stream)
    require(type(model) is PairedCompletionModel and model.feature_names == plan["feature_names"] and
            all(model.estimator.get_params()[k] == v for k,v in plan["model_params"].items()), "model contract drift")
    data, golden = read_json(ROOT / plan["training_index"]), read_json(ROOT / fold["golden"])
    check_golden_identity(data, fold["held"], golden)
    require(old_predictions(model, data, fold["held"]) == golden["rows"] == receipt["rows"], "loaded model mismatch")
    return model, receipt


def reconstruct_job(job):
    plan, fold, directory = job
    out = Path(directory)
    path = out / "folds" / (fold["held"] + ".pkl")
    require(not path.exists() and not path.with_suffix(".pkl.tmp").exists(), "interrupted fold; inspect before retry")
    import numpy
    import scipy
    import sklearn
    data, golden = read_json(ROOT / plan["training_index"]), read_json(ROOT / fold["golden"])
    start = time.monotonic()
    model, rows = fit_exact(data, fold["held"], golden)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pkl.tmp")
    with tmp.open("xb") as stream:
        pickle.dump(model, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    receipt = sealed(dict(binding=plan["binding"], held=fold["held"], train_ids=golden["train_ids"],
                          rows=rows, exact_old_predictions=True, model_sha256=sha256_file(path),
                          golden_sha256=plan["inputs"][fold["golden"]], seconds=time.monotonic()-start,
                          training_index_sha256=plan["inputs"][plan["training_index"]],
                          new_label_values_used=False, model_version=RECENT,
                          environment=dict(python=sys.version, sklearn=sklearn.__version__,
                                           numpy=numpy.__version__, scipy=scipy.__version__,
                                           numerical_library_threads=1)))
    once(path.with_suffix(".json"), receipt)
    load_fold(out, fold, plan)
    return dict(held=fold["held"], rows=len(rows), scores=sum(len(r["scores"]) for r in rows))


def reconstruct():
    from experiments.repair_collection import _CollectionRunLock
    import sklearn
    require(sklearn.__version__ == "1.5.0", "registered sklearn required")
    plan,out = verify()
    with _CollectionRunLock(out, plan["binding"], "recent-model-reconstruction"):
        pending = []
        for fold in plan["folds"]:
            if (out / "folds" / (fold["held"] + ".json")).exists():
                load_fold(out, fold, plan)
            else:
                pending.append((plan, fold, str(out)))
        atomic(out / "run_status.json", dict(binding=plan["binding"], phase="reconstruct", status="running"))
        try:
            with ProcessPoolExecutor(max_workers=min(plan["config"]["workers"], max(1,len(pending)))) as pool:
                for row in pool.map(reconstruct_job, pending):
                    print("EXACT_FOLD", row, flush=True)
            files = {}
            row_count = score_count = 0
            for fold in plan["folds"]:
                _, receipt = load_fold(out, fold, plan)
                row_count += len(receipt["rows"])
                score_count += sum(len(r["scores"]) for r in receipt["rows"])
                for ext in (".json", ".pkl"):
                    name = "folds/" + fold["held"] + ext
                    files[name] = sha256_file(out / name)
            verify()
            require((len(plan["folds"]), row_count, score_count) == (8,376,1504), "reconstruction coverage")
            once(out / "model_receipt.json", sealed(dict(binding=plan["binding"], files=files, folds=len(plan["folds"]),
                 exact_prediction_rows=row_count, exact_candidate_scores=score_count, model_version=RECENT,
                 original_weights_recovered=False, original_procedure_reconstructed=True)))
            atomic(out / "run_status.json", dict(binding=plan["binding"], phase="reconstruct", status="completed"))
        except BaseException as exc:
            atomic(out / "run_status.json", dict(binding=plan["binding"], phase="reconstruct", status="error", error=repr(exc)))
            raise
    return dict(folds=8, exact_prediction_rows=376, exact_candidate_scores=1504)


def verify_models(plan, out):
    rc = previous.seal_json(out / "model_receipt.json", plan["binding"])
    expected = {"folds/" + f["held"] + ext for f in plan["folds"] for ext in (".json", ".pkl")}
    require(set(rc["files"]) == expected, "model receipt coverage")
    for name,digest in rc["files"].items():
        require(sha256_file(out / name) == digest, "reconstructed model output changed")
    return rc


def predict():
    plan,out = verify()
    verify_models(plan,out)
    prev_out = ROOT / plan["previous_output"]
    models = {f["held"]:load_fold(out,f,plan)[0] for f in plan["folds"]}
    rows = []
    for entry in plan["roots"]:
        state = previous.seal_json(prev_out / "features" / (entry["state_id"] + ".json"), plan["previous_binding"])
        require(state["map_id"] == entry["map_id"] and state["state_id"] == entry["state_id"] and
                state["pair_ids"] == entry["pair_ids"] and state["anchor_id"] == entry["anchor_id"], "feature identity")
        model = models[state["map_id"]]
        fold = next(f for f in plan["folds"] if f["held"] == state["map_id"])
        require(state["state_id"] not in read_json(ROOT / fold["golden"])["train_ids"], "prediction map leakage")
        scores = {c:state["original_pool_scores"][c] for c in state["pair_ids"]}
        rows.append(dict(state_id=state["state_id"], map_id=state["map_id"], pair_ids=state["pair_ids"],
                         anchor_id=state["anchor_id"], recent_model=previous.paired_preference(model,state,state["pair_ids"]),
                         recent_union=model.rank(state), frozen_pool=dict(scores=scores,
                         selected=previous.choose(scores,state["anchor_id"],rounded=True))))
    once(out / "predictions.json", sealed(dict(binding=plan["binding"], rows=rows, model_version=RECENT,
         model_receipt_sha256=sha256_file(out / "model_receipt.json"), new_label_values_used=False,
         historical_feature_receipt_sha256=sha256_file(prev_out / "feature_receipt.json"))))
    verify()
    return dict(states=len(rows), complete_pairs=len(rows), new_label_values_used=False)


def evaluate_recent(records, predictions, old_report, samples, seed):
    compatible = [dict(p, historical_model=p["recent_model"], historical_union=p["recent_union"]) for p in predictions]
    result = previous.evaluate(records, compatible, samples, seed)
    def name(key):
        return key.replace(previous.HISTORICAL, RECENT).replace("historical_union", "recent_union")
    result["metrics"] = {name(k):v for k,v in result["metrics"].items()}
    result["rows"] = [{name(k):v for k,v in row.items()} for row in result["rows"]]
    old = {r["state_id"]:r for r in old_report["rows"]}
    require(len(old) == len(old_report["rows"]) == len(result["rows"]) and
            set(old) == {r["state_id"] for r in result["rows"]}, "historical comparison coverage")
    for row in result["rows"]:
        other = old[row["state_id"]]
        require(other["map_id"] == row["map_id"] and other["rates"] == row["rates"], "historical comparison drift")
        row["recent_minus_historical"] = row[RECENT] - other[previous.HISTORICAL]
    maps = sorted({r["map_id"] for r in result["rows"]})
    rng = random.Random(seed)
    draws = [tuple(rng.randrange(len(maps)) for _ in maps) for _ in range(samples)]
    result["metrics"]["recent_minus_historical"] = previous.aggregate(result["rows"], "recent_minus_historical", draws)
    return result


def analyze():
    plan,out = verify()
    verify_models(plan,out)
    predictions = previous.seal_json(out / "predictions.json", plan["binding"])
    require(predictions["model_receipt_sha256"] == sha256_file(out / "model_receipt.json"), "prediction model receipt drift")
    records = read_json(ROOT / plan["label_records"])
    old = read_json(ROOT / plan["previous_output"] / "report.json")
    result = evaluate_recent(records, predictions["rows"], old, plan["config"]["bootstrap"], plan["config"]["bootstrap_seed"])
    result.update(binding=plan["binding"], model_version=RECENT, role=plan["config"]["role"],
                  original_procedure_reconstructed=True, original_weights_recovered=False,
                  new_label_training=False, reconstruction_fits=8, solver_calls=0, no_ttf=True,
                  prediction_sha256=sha256_file(out / "predictions.json"),
                  label_sha256=sha256_file(ROOT / plan["label_records"]),
                  historical_report_sha256=sha256_file(ROOT / plan["previous_output"] / "report.json"))
    once(out / "report.json", sealed(result))
    verify()
    return {k:v for k,v in result.items() if k != "rows"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "reconstruct", "predict", "analyze", "verify"))
    args = parser.parse_args()
    if args.phase == "verify":
        plan,out = verify()
        result = dict(binding=plan["binding"], inputs=len(plan["inputs"]))
        if (out / "model_receipt.json").exists():
            result["models"] = verify_models(plan,out)["folds"]
        for name in ("predictions.json", "report.json"):
            if (out / name).exists():
                previous.seal_json(out / name,plan["binding"])
    else:
        result = globals()[args.phase]()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
