"""Prepare pre-action spatiotemporal inputs without fitting or solver calls."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments._common import contained_file, json_fingerprint, read_json, sha256_file, write_json, write_jsonl
from experiments.repair_collection import state_fingerprint
from experiments.sa_history_information import profile_features
from experiments.sa_paired_completion import require, validate_dataset
from experiments.sa_spatiotemporal_input import candidate_diagnostics, candidate_input, encode_state

CONFIG = "configs/sa_spatiotemporal_input_audit.json"
IMPLEMENTATION = ["scripts/audit_sa_spatiotemporal_input.py", "experiments/sa_spatiotemporal_input.py",
                  "experiments/state_analysis.py", "experiments/repair_collection.py",
                  "experiments/sa_paired_completion.py", "experiments/sa_history_information.py",
                  "experiments/_common.py", "tests/evaluation/test_sa_spatiotemporal_input.py"]


def checked_path(name):
    return contained_file(ROOT, name, field="registered input")


def frozen_read(name, digest):
    path = checked_path(name)
    require(sha256_file(path) == digest, "input SHA mismatch: " + name)
    return read_json(path)


def prepare_jobs(cfg):
    require(cfg["schema"] == "sa-spatiotemporal-input-audit-v1" and cfg["new_solver_calls"] == 0 and
            cfg["model_fits"] == 0 and cfg["runtime_integration_allowed"] is False and
            cfg["source_role"] == "previously_viewed_development", "audit boundary changed")
    inputs, jobs, seen, occurrences = {}, [], set(), set()
    for source in cfg["sources"]:
        folder = source["directory"]
        require(source["name"] in {"old16", "new31"}, "unregistered source")
        index_name, plan_name = folder + "/training_index.json", folder + "/plan.json"
        data = frozen_read(index_name, source["index_sha256"])
        plan = frozen_read(plan_name, source["plan_sha256"])
        validate_dataset(data)
        require(plan["binding"] == json_fingerprint({k:v for k,v in plan.items() if k != "binding"}) and
                data["continuation_binding"] == plan["binding"], "source plan binding mismatch")
        require(data["horizon"] == 32 and data["trial_count"] == 8 and
                len(data["states"]) == source["states"], "source size or label contract changed")
        inputs.update({index_name: source["index_sha256"], plan_name: source["plan_sha256"]})
        targets = {r["id"]: r for r in plan["roots"]}
        require(set(targets) == {s["state_id"] for s in data["states"]}, "source root coverage")
        for state in data["states"]:
            sid = state["state_id"]
            require(sid not in seen and (state["episode"], state["decision"]) not in occurrences,
                    "duplicate source state occurrence")
            seen.add(sid)
            occurrences.add((state["episode"], state["decision"]))
            target = targets[sid]
            require(state["map_id"] == target["map_id"] and state["decision"] == target["decision"] and
                    state["episode"] == target["item"]["job_id"], "source provenance mismatch")
            root_name, receipt_name = folder+"/roots/"+sid+"/root.json", folder+"/roots/"+sid+"/receipt.json"
            receipt = read_json(checked_path(receipt_name))
            require(receipt["binding"] == plan["binding"], "source receipt binding mismatch")
            inputs[root_name] = receipt["files"]["root.json"]
            inputs[receipt_name] = sha256_file(checked_path(receipt_name))
            jobs.append(dict(root=root_name, digest=inputs[root_name], state=state, target=target,
                             binding=plan["binding"], source=source["name"], index=index_name))
    require(len(jobs) == cfg["expected_states"] and
            len({j["state"]["map_id"] for j in jobs}) == cfg["expected_maps"], "registered cohort changed")
    return sorted(jobs, key=lambda j: j["state"]["state_id"]), inputs


def write_pack(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".pack-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
                zipped.write(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                        ensure_ascii=True, allow_nan=False).encode("ascii"))
        with gzip.open(temporary, "rt", encoding="ascii") as stream:
            require(json.load(stream) == value, "compressed input roundtrip")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def process_root(job):
    root = frozen_read(job["root"], job["digest"])
    source = job["state"]
    require(root["binding"] == job["binding"] and root["source"] == job["target"], "root provenance")
    require(root["state_fingerprint"] == state_fingerprint(root["state"]) == source["state_fingerprint"],
            "pre-action root fingerprint mismatch")
    require(root["old_selected_id"] == source["anchor_id"], "anchor mismatch")
    raw_candidates = {c["candidate_id"]: c for c in root["candidates"]}
    bases = {c["candidate_id"]: base for c, base in zip(root["candidates"], root["feature_rows"], strict=True)}
    require(set(raw_candidates) == {c["candidate_id"] for c in source["candidates"]}, "candidate coverage mismatch")
    require({a["id"] for a in root["state"]["agents"]} == set(source["agent_ids"]), "source agent IDs changed")
    # The extractor sees only the saved pre-action state. Neither control_event
    # metrics nor H32 branches/labels are accessible through this interface.
    encoded = encode_state(root["state"])
    relative = "states/" + source["state_id"] + ".json.gz"
    path = Path(job["output"]) / relative
    write_pack(path, encoded)
    candidates, summaries = [], []
    for c in sorted(source["candidates"], key=lambda c: c["candidate_id"]):
        raw = raw_candidates[c["candidate_id"]]
        require(raw["agents"] == c["agents"], "candidate membership mismatch")
        require(profile_features(dict(base=bases[c["candidate_id"]]), "dynamic") == c["features"],
                "baseline feature index mismatch")
        candidates.append(dict(state_id=source["state_id"], candidate_id=c["candidate_id"], shared_input=relative,
                               model_input=candidate_input(encoded, c["agents"]),
                               label_reference=dict(index=job["index"], state_id=source["state_id"],
                                                    candidate_id=c["candidate_id"])))
        summaries.append(dict(candidate_id=c["candidate_id"], **candidate_diagnostics(encoded, c["agents"])))
    return dict(state_id=source["state_id"], map_id=source["map_id"], source=job["source"],
                phase=source["phase"], decision=source["decision"],
                input_file=relative, input_sha256=sha256_file(path), bytes=path.stat().st_size,
                semantic_fingerprint=json_fingerprint(encoded),
                path_points=sum(a["path_cost"]+1 for a in encoded["agents"]),
                occupancy_intervals=sum(len(a["occupancy"]) for a in encoded["agents"]),
                agents=len(encoded["agents"]), max_path_cost=encoded["horizon"],
                conflict_pairs=root["state"]["num_of_colliding_pairs"], conflict_events=len(encoded["events"]),
                candidates=candidates, candidate_diagnostics=summaries)


def summarize(results):
    candidates = [c for r in results for c in r["candidate_diagnostics"]]
    count = lambda predicate: sum(bool(predicate(c)) for c in candidates)
    return dict(states=len(results), maps=len({r["map_id"] for r in results}), candidates=len(candidates),
                existing_label_trials=8*len(candidates),
                by_source=dict(Counter(r["source"] for r in results)),
                maps_by_source={s: sorted({r["map_id"] for r in results if r["source"]==s})
                                for s in sorted({r["source"] for r in results})},
                state_counts_by_map=dict(sorted(Counter(r["map_id"] for r in results).items())),
                conflict_pairs=sum(r["conflict_pairs"] for r in results),
                conflict_events=sum(r["conflict_events"] for r in results),
                candidates_with_boundary_events=count(lambda c:c["boundary_events"]),
                candidates_with_outsider_goal_tail_events=count(lambda c:c["outsider_goal_tail_events"]),
                candidates_with_multiple_incident_times=count(lambda c:len(c["incident_times"])>1),
                candidates_with_repeated_incident_pairs=count(lambda c:c["incident_events"]>c["incident_pairs"]),
                candidates_with_nearby_noncolliding_outsiders=count(lambda c:c["nearby_noncolliding_outsiders"]),
                path_points=sum(r["path_points"] for r in results),
                occupancy_intervals=sum(r["occupancy_intervals"] for r in results),
                shared_input_bytes=sum(r["bytes"] for r in results),
                maximum_path_cost=max(r["max_path_cost"] for r in results),
                new_solver_calls=0, model_fits=0, input_integrity_passed=True,
                learnability_established=False, closed_loop_benefit_established=False,
                automatic_training_allowed=False, runtime_integration_allowed=False,
                decision="inputs_ready_representation_and_label_protocol_not_yet_validated")


def run(config_name):
    cfg = read_json(checked_path(config_name))
    jobs, inputs = prepare_jobs(cfg)
    for name in [config_name] + IMPLEMENTATION:
        inputs[name] = sha256_file(checked_path(name))
    output = Path(cfg["output"])
    require(not output.is_absolute() and ".." not in output.parts and output.parts[0] == "build", "output outside build")
    out = ROOT / output
    require(not out.exists() and not out.is_symlink() and
            out.resolve().is_relative_to((ROOT/"build").resolve()), "existing or unsafe output; preserve prior audit")
    out.mkdir(parents=True, exist_ok=False)
    binding = json_fingerprint(dict(config=cfg, inputs=inputs))
    workers = min(cfg["workers"], len(jobs))
    require(type(workers) is int and 1 <= workers <= 20, "invalid workers")
    identity = dict(binding=binding, inputs=inputs, config=cfg,
                    source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    write_json(out/"run_config.json", identity)
    lock = out/"run.lock"
    with lock.open("x", encoding="ascii"):
        try:
            write_json(out/"run_status.json", dict(status="running", binding=binding))
            for job in jobs:
                job["output"] = str(out)
            with ProcessPoolExecutor(max_workers=workers) as pool:
                results = []
                for result in pool.map(process_root, jobs):
                    results.append(result)
                    print("INPUT", len(results), "/", len(jobs), result["state_id"], flush=True)
            for name, digest in inputs.items():
                require(sha256_file(checked_path(name)) == digest, "input changed during audit: " + name)
            report = dict(schema=cfg["schema"], binding=binding, summary=summarize(results),
                          states=[{k:v for k,v in r.items() if k != "candidates"} for r in results])
            write_json(out/"report.json", report)
            write_jsonl(out/"candidate_index.jsonl", (c for r in results for c in r["candidates"]))
            files = {r["input_file"]:r["input_sha256"] for r in results}
            for name in ("report.json", "candidate_index.jsonl", "run_config.json"):
                files[name] = sha256_file(out/name)
            write_json(out/"complete.json", dict(binding=binding, inputs=inputs, files=files))
            write_json(out/"run_status.json", dict(status="completed", binding=binding))
        except BaseException as error:
            write_json(out/"run_status.json", dict(status="error", binding=binding, error=repr(error)))
            raise
    lock.unlink()
    return report["summary"]


def verify(config_name):
    cfg = read_json(checked_path(config_name))
    # Resolve through contained_file before reading any generated path.
    receipt_path = checked_path(cfg["output"] + "/complete.json")
    out = receipt_path.parent
    receipt = read_json(receipt_path)
    for name, digest in receipt["inputs"].items():
        require(sha256_file(checked_path(name)) == digest, "registered input changed: " + name)
    require(receipt["inputs"][config_name] == sha256_file(checked_path(config_name)), "configuration mismatch")
    for name, digest in receipt["files"].items():
        require(sha256_file(contained_file(out,name,field="audit output")) == digest, "output changed: " + name)
    identity = read_json(out/"run_config.json")
    require(identity["config"] == cfg and identity["inputs"] == receipt["inputs"] and
            receipt["binding"] == identity["binding"] == json_fingerprint(dict(config=cfg,inputs=receipt["inputs"])),
            "audit binding mismatch")
    require(not (out/"run.lock").exists() and read_json(out/"run_status.json") ==
            dict(status="completed", binding=receipt["binding"]), "audit not completed")
    return dict(status="verified", files=len(receipt["files"]), inputs=len(receipt["inputs"]), binding=receipt["binding"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run", "verify"))
    parser.add_argument("--config", default=CONFIG)
    args = parser.parse_args()
    print(json.dumps((run if args.stage == "run" else verify)(args.config), indent=2))


if __name__ == "__main__":
    main()
