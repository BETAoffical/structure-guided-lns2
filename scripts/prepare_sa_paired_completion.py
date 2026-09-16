"""Inspect existing labels under the new model contract; never fit or collect."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments._common import json_fingerprint, read_json, sha256_file, write_json
from experiments.sa_history_information import profile_features
from experiments.sa_paired_completion import SCHEMA, inspect_dataset, require
from scripts.collect_sa_history_candidate_bridge import check_root_receipt, verify as verify_bridge

CONFIG = ROOT / "configs/sa_paired_completion_model.json"


def bridge_dataset(plan, source, preflight, report):
    cfg = plan["config"]
    require(cfg["horizon"] == 32 and cfg["trials"] == 8, "historical budget changed")
    outcomes = {s["id"]: s for s in report["states"]}
    sampling = read_json(ROOT / cfg["sampling_config"])
    states, names = [], None
    for root in preflight["roots"]:
        target = root["target"]
        check_root_receipt(source / "branches" / target["id"], root, plan)
        root_path = ROOT / sampling["output"] / "states" / target["id"] / "root.json"
        require(sha256_file(root_path) == root["root_sha"], "root identity changed")
        native_root = read_json(root_path)
        require(root["state_fingerprint"] == native_root["state_fingerprint"], "root fingerprint mismatch")
        actual = outcomes[target["id"]]
        require(not actual["censored"], "historical bridge censoring changed")
        require(set(actual["labels"]) == set(root["selected"]), "source candidate coverage")
        rows = {r["candidate_id"]: r for r in root["rows"]}
        candidates = []
        for cid in sorted(root["selected"]):
            row = rows[cid]
            values = profile_features(row, "dynamic")
            if names is None:
                names = sorted(values)
            require(sorted(values) == names, "feature columns differ between roots")
            trials = []
            for t in range(cfg["trials"]):
                branch = read_json(source / "branches" / target["id"] / f"{cid}-t{t}.json")
                require(branch["status"] == "ok" and branch["root_id"] == target["id"] and
                        branch["candidate_id"] == cid and branch["trial"] == t and
                        branch["root_fingerprint"] == root["state_fingerprint"], "branch identity mismatch")
                completed = branch["stop"] == "feasible" if branch["stop"] in ("feasible", "horizon") else None
                require(completed == actual["labels"][cid][t]["completion"], "source completion label mismatch")
                trials.append(dict(trial=t, randomization_key=json_fingerprint(
                    [plan["binding"], target["id"], t, target["decision"], cfg["seed"]]),
                    stop=branch["stop"], steps=len(branch["events"]),
                    final_conflicts=branch["final_conflicts"], completed=completed))
            candidates.append(dict(candidate_id=cid, agents=row["agents"], features=values, trials=trials))
        states.append(dict(state_id=target["id"], map_id=target["map_id"], episode=target["item"]["job_id"],
                           decision=target["decision"], anchor_id=root["old_selected_id"],
                           state_fingerprint=root["state_fingerprint"],
                           history_fingerprint=json_fingerprint([r["records"] for r in root["rows"]]),
                           agent_ids=[a["id"] for a in native_root["state"]["agents"]],
                           source_pool_count=len(native_root["control_event"]["pool"]),
                           diagnostic_pool_count=len(root["rows"]), candidates=candidates))
    return dict(schema=SCHEMA, role="diagnostic_only", sampling="model_selected",
                source_kind="historical_candidate_bridge", continuation_binding=plan["binding"],
                horizon=cfg["horizon"], trial_count=cfg["trials"], feature_names=names, states=states)


def inspect_bridge():
    cfg = read_json(CONFIG)
    require(cfg["source_role"] == "diagnostic_only" and cfg["new_solver_calls"] == 0 and
            cfg["fit_historical_bridge"] is False and cfg["runtime_integration_allowed"] is False,
            "prototype boundary changed")
    require(cfg["feature_profile"] == "dynamic" and cfg["horizon"] == 32 and cfg["trials"] == 8,
            "registered profile or budget changed")
    plan, source = verify_bridge()
    require(sha256_file(source / "report.json") == cfg["source_report_sha256"], "historical report changed")
    paths = [CONFIG, Path(__file__), ROOT / "experiments/sa_paired_completion.py",
             ROOT / "tests/evaluation/test_sa_paired_completion.py", ROOT / "docs/SA_PAIRED_COMPLETION_MODEL_ZH.md",
             source / "preflight.json", source / "report.json", source / "collection_plan.json"]
    inputs = {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in paths}
    data = bridge_dataset(plan, source, read_json(source / "preflight.json"), read_json(source / "report.json"))
    report = inspect_dataset(data)
    require((report["state_count"], report["candidate_count"], report["trial_count"]) == (8, 24, 192), "source size changed")
    report.update(schema=cfg["schema"], inputs=inputs, source_binding=plan["binding"],
                  feature_count=len(data["feature_names"]), paired_feature_count=2*len(data["feature_names"]),
                  index_fingerprint=json_fingerprint(data), new_solver_calls=0, real_data_fits=0,
                  decision="prototype_ready_historical_data_not_admitted_for_training")
    verify_bridge()
    require(all(sha256_file(ROOT / p) == h for p, h in inputs.items()), "input changed while reading")
    out = ROOT / cfg["output"]
    require(out.resolve().is_relative_to((ROOT / "build").resolve()), "output must stay under build")
    out.mkdir(parents=True, exist_ok=True)
    # Failed writes deliberately leave a lock for inspection, never auto-resume.
    with (out / "run.lock").open("x", encoding="utf8"):
        for name, value in (("diagnostic_index.json", data), ("report.json", report)):
            path = out / name
            if path.exists():
                require(read_json(path) == value, "existing output differs; preserve old run")
            else:
                write_json(path, value)
        receipt = dict(inputs=inputs, files={n: sha256_file(out/n) for n in ("diagnostic_index.json", "report.json")})
        if (out / "complete.json").exists():
            require(read_json(out / "complete.json") == receipt, "completion receipt mismatch")
        else:
            write_json(out / "complete.json", receipt)
    (out / "run.lock").unlink()
    return {k: v for k, v in report.items() if k not in ("states", "inputs")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("inspect-bridge",))
    parser.parse_args()
    print(json.dumps(inspect_bridge(), indent=2))


if __name__ == "__main__":
    main()
