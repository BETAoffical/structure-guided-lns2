"""Checkpoint validation and opt-in serial execution for the frozen cohort."""
import argparse
import json
import platform
import random
import signal
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments._common import mean, quantile, read_json, read_jsonl, sha256_file
from experiments.repair_collection import _CollectionRunLock, _fingerprint, _make_environment, _plain, _run_jobs, state_fingerprint
from experiments.closed_loop_trace_storage import read_state_blob
from experiments.warehouse_disruption_checkpoints import compute_checkpoint_identity_sha256, inject_global_hotspot_delays
from experiments.stride_warehouse_disruption_recovery_ttf import TTF_OVERRIDE_SCHEMA
from lns2_selector.evaluation.path_quality_execution import (
    PathJournal, _episode_child, controller_job, read_artifact, spec_fingerprint, supervise_episode,
)
from lns2_selector.evaluation.trace_validation import validate_closed_loop_trace
from scripts.run_warehouse_repair_confirmation import (
    BASE, CONFIG, checkpoint_report, environment_config, install_native, task_plan,
    verify_inputs, write_json, write_jsonl,
)
from scripts.prepare_warehouse_repair_confirmation import schedule_slots

TEMPLATE = "configs/stride_warehouse_fixed16_development_runtime_v2.json"
TEMPLATE_SHA = "4a7376f428338abedf8a7defb645588b59fdda081f575918ea2a7abb278f673a"


def checkpoints(config):
    root = BASE / "checkpoints"
    results = [read_json(root / f"result-{index:02d}.json") for index in range(16)]
    identity = read_json(root / "preparation_identity.json")
    if any(r.get("run_fingerprint") != _fingerprint(identity) for r in results):
        raise ValueError("checkpoint preparation identity mismatch")
    if identity["config"] != sha256_file(ROOT / CONFIG) or identity["plan"] != task_plan(config):
        raise ValueError("checkpoint plan mismatch")
    report = checkpoint_report(results)
    if report != read_json(root / "checkpoint_report.json") or not report["passed"]:
        raise ValueError("checkpoint supply gate failed or report changed")
    if read_jsonl(root / "checkpoint_manifest.jsonl") != report["checkpoints"]:
        raise ValueError("checkpoint manifest changed")
    planned = {s["checkpoint_slot"]: (item, s) for item in task_plan(config) for s in item["slots"]}
    for cp in report["checkpoints"]:
        item, slot = planned[cp["key_index"]]
        row = item["row"]
        split = ROOT / config["dataset"]["output"] / "confirmation"
        fields = {"checkpoint_id": f"repair-confirm-{slot['checkpoint_slot']:02d}",
                  "map_id": row["map_id"], "task_id": row["task_id"], "agent_count": row["agent_count"],
                  "screen_solver_seed": slot["solver_seed"], "restore_seed": slot["selection_seed"],
                  "replica": slot["replica"], "split": "confirmation", "task_variant": row["task_variant"],
                  "map_sha256": sha256_file(split / row["map_file"]), "task_sha256": sha256_file(split / row["task_file"])}
        if any(cp.get(key) != value for key, value in fields.items()):
            raise ValueError("checkpoint registered identity changed")
        if cp["checkpoint_identity_sha256"] != compute_checkpoint_identity_sha256(cp):
            raise ValueError("checkpoint content SHA mismatch")
        for filename, digest in [(cp["state_blob"], cp["state_blob_sha256"]),
                                 (cp["incumbent"]["file"], cp["incumbent"]["file_sha256"])]:
            path = (root / filename).resolve()
            if not path.is_relative_to(root.resolve()) or sha256_file(path) != digest:
                raise ValueError("checkpoint artifact changed")
        if cp["incumbent"]["solver_seed"] != slot["incumbent_seed"]:
            raise ValueError("incumbent seed changed")
    for task_id in {c["task_id"] for c in report["checkpoints"]}:
        pair = [c for c in report["checkpoints"] if c["task_id"] == task_id]
        if len({c["incumbent"]["file_sha256"] for c in pair}) != 1:
            raise ValueError("replicas do not share a frozen incumbent")
    return report


def case_for(row, dataset):
    files = {key: (dataset / row["split"] / row[key]).relative_to(ROOT).as_posix()
             for key in ("map_file", "scenario_file", "map_metadata_file", "task_file")}
    return {"task_id": row["task_id"], "map_id": row["map_id"], "family": "warehouse",
            "status": "static_ready_runtime_unverified", "files": files,
            "static_audit": {"agent_count": row["agent_count"]}}


def paired_restore_worker(job):
    config, cp, row = job["config"], job["checkpoint"], job["row"]
    install_native(config)
    source = read_state_blob(BASE / "checkpoints" / cp["state_blob"])
    if state_fingerprint(source) != cp["expected_fingerprint"]:
        raise ValueError("saved checkpoint state fingerprint mismatch")
    incumbent = read_json(BASE / "checkpoints" / cp["incumbent"]["file"])
    dataset = ROOT / config["dataset"]["output"]
    metadata = read_json(dataset / row["split"] / row["map_metadata_file"])["metadata"]
    disturbance = inject_global_hotspot_delays(incumbent["paths"],
        semantic_cell_types=metadata["semantic_cell_types"], station_zones=metadata["station_zones"],
        global_time=cp["disturbance"]["global_time"], delay_ticks=4,
        delayed_agent_fraction=0.15, selection_seed=cp["restore_seed"], neighborhood_radius=1)
    paths = [a["path"] for a in sorted(source["agents"], key=lambda a: a["id"])]
    if paths != disturbance["paths"] or disturbance["qualification"] != cp["qualification"]:
        raise ValueError("disturbance replay differs")
    env_config = {**environment_config(config), "time_limit": 60}
    fingerprints = {}
    case = case_for(row, dataset)
    for controller in config["controllers"]:
        env = _make_environment(str(dataset), row, env_config, "Adaptive")
        state = _plain(env.reset_paths(paths, seed=cp["restore_seed"]))
        fp = state_fingerprint(state)
        if fp != cp["expected_fingerprint"]:
            raise ValueError("paired restored fingerprint mismatch")
        journal = PathJournal(Path(job["output"]), "validation-only", ROOT / case["files"]["map_file"],
                              ROOT / case["files"]["scenario_file"])
        journal.quality(state)
        fingerprints[controller] = fp
    return {"status": "ok", "checkpoint_id": cp["checkpoint_id"], "fingerprints": fingerprints,
            "disturbance_reproduced": True, "paths_validated": True, "controller_steps": 0}


def bound_schedule(config, report):
    by_index = {c["key_index"]: c for c in report["checkpoints"] if c["qualification"]["passed"]}
    schedule = []
    for slot in schedule_slots(config):
        if slot["checkpoint_slot"] not in by_index:
            continue
        cp = by_index[slot["checkpoint_slot"]]
        item = {**slot, "schedule_index": len(schedule), "checkpoint_id": cp["checkpoint_id"],
                "checkpoint_identity_sha256": cp["checkpoint_identity_sha256"],
                "task_id": cp["task_id"], "map_id": cp["map_id"], "controller": slot["controller"],
                "solver_seed": cp["screen_solver_seed"], "family": "warehouse",
                "protocol": "first_feasible", "budget_seconds": 60.0, "bound_to_checkpoint": True}
        item["job_id"] = _fingerprint(item)[:24]
        schedule.append(item)
    return schedule


def make_spec(config, cp, row, item, input_sha, output, template=None):
    template = read_json(ROOT / TEMPLATE) if template is None else template
    case = case_for(row, ROOT / config["dataset"]["output"])
    worker = controller_job(ROOT, case, item, template, output, _fingerprint({"item": item, "inputs": input_sha}))
    worker["episode_override"] = {"schema": TTF_OVERRIDE_SCHEMA, "state_id": cp["checkpoint_id"],
                                 "initial_restore": {**cp, "collection_root": str(BASE / "checkpoints")}}
    spec = {"root": str(ROOT), "output": str(output), "item": item, "case": case,
            "input_sha256": input_sha, "native_sha256": config["frozen"]["native_sha256"],
            "process_timeout_seconds": 90.0, "expected_initial_fingerprint": cp["expected_fingerprint"],
            "worker_job": worker}
    spec["binding"] = spec_fingerprint(spec)
    return spec


def runtime_identity(config):
    if sha256_file(ROOT / TEMPLATE) != TEMPLATE_SHA:
        raise ValueError("runtime template changed")
    inputs = dict(read_json(BASE / "preparation/preflight.json")["input_sha256"])
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    for name in tracked:
        if name.endswith((".py", ".cpp", ".h", ".hpp")):
            inputs[name] = sha256_file(ROOT / name)
    for name in ("scripts/warehouse_repair_confirmation_runtime.py", "scripts/run_warehouse_repair_confirmation.py", TEMPLATE):
        inputs[name] = sha256_file(ROOT / name)
    inputs.update(read_json(BASE / "dataset-audit/dataset_validation.json")["files"])
    for path in (BASE / "checkpoints").rglob("*"):
        if path.is_file() and path.name not in {"collection_progress.json", ".collection.lock"}:
            inputs[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    return dict(sorted(inputs.items()))


def validate_preparation():
    config = verify_inputs()
    install_native(config)
    report = checkpoints(config)
    output = BASE / "readiness"
    if (output / "preparation_validation.json").exists():
        raise ValueError("readiness validation exists; use verify, not repeated construction")
    rows = {item["row"]["task_id"]: item["row"] for item in task_plan(config)}
    jobs = [{"job_id": c["checkpoint_id"], "checkpoint": c, "config": config,
             "row": rows[c["task_id"]], "output": str(output)} for c in report["checkpoints"]]
    with _CollectionRunLock(output, _fingerprint(jobs), "confirmation-restore-validation"):
        results = _run_jobs(paired_restore_worker, jobs, 20, phase="confirmation-restore-validation",
                            output_root=output, timeout_seconds=60)
        if len(results) != len(jobs) or any(r.get("status") != "ok" for r in results):
            write_json(output / "restore_validation_errors.json", {"results": results})
            raise ValueError("paired checkpoint validation failed")
        inputs = runtime_identity(config)
        schedule = bound_schedule(config, report)
        by_slot = {c["key_index"]: c for c in report["checkpoints"]}
        specs = [make_spec(config, by_slot[item["checkpoint_slot"]], rows[item["task_id"]],
                           item, inputs, BASE / "timed" / item["job_id"]) for item in schedule]
        for spec in specs:
            if spec["worker_job"]["max_decisions"] != 0 or spec["worker_job"]["environment"]["max_repair_iterations"] != 0:
                raise ValueError("repair iteration limit reintroduced")
        write_jsonl(output / "paired_restore_validation.jsonl", results)
        write_jsonl(output / "schedule.jsonl", schedule)
        write_jsonl(output / "episode_specs.jsonl", specs)
        write_json(output / "preparation_validation.json", {
            "schema": "lns2.warehouse_confirmation_preparation_validation.v1", "input_sha256": inputs,
            "checkpoint_count": len(jobs), "paired_restore_count": len(jobs) * 3, "episode_count": len(specs),
            "specs_sha256": sha256_file(output / "episode_specs.jsonl"),
            "schedule_sha256": sha256_file(output / "schedule.jsonl"),
            "restore_sha256": sha256_file(output / "paired_restore_validation.jsonl"),
            "formal_controller_steps": 0, "timed_execution_authorized": False,
            "status": "awaiting_functional_tests_and_final_seal"})
    return {"restores_validated": len(jobs) * 3, "checkpoints": len(jobs), "scheduled_episodes": len(specs), "timed_episodes": 0}


def verify_prepared(require_ready=False):
    config = verify_inputs()
    install_native(config)
    checkpoints(config)
    output = BASE / "readiness"
    report = read_json(output / "preparation_validation.json")
    for name, sha in report["input_sha256"].items():
        if sha256_file(ROOT / name) != sha:
            raise ValueError("prepared source or input changed: " + name)
    for filename, key in [("episode_specs.jsonl", "specs_sha256"), ("schedule.jsonl", "schedule_sha256"),
                          ("paired_restore_validation.jsonl", "restore_sha256")]:
        if sha256_file(output / filename) != report[key]:
            raise ValueError("prepared execution artifact changed")
    if require_ready:
        seal = read_json(ROOT / "artifacts/warehouse-repair-confirmation-v1/readiness.json")
        if seal["status"] != "ready_awaiting_timing_authorization" or seal["preparation_sha256"] != sha256_file(output / "preparation_validation.json"):
            raise ValueError("readiness seal mismatch")
        for name, sha in seal["validation_evidence"].items():
            if sha256_file(ROOT / name) != sha:
                raise ValueError("validation evidence changed")
    specs = read_jsonl(output / "episode_specs.jsonl")
    for spec in specs:
        if spec["binding"] != spec_fingerprint(spec):
            raise ValueError("episode spec changed")
    return config, specs


def validated_episode(spec):
    out = Path(spec["output"])
    if not (out / "supervisor.json").exists():
        return None
    binding = spec["binding"]
    summary = read_artifact(out / "supervisor.json", binding)
    if summary["error"]:
        raise ValueError("unexplained episode error; inspect before resume")
    if (out / "first_phase_result.json").exists():
        result = read_artifact(out / "first_phase_result.json", binding)
        if result.get("status") != "ok":
            raise ValueError("first-phase worker error")
        job = spec["worker_job"]
        trace = Path(job["output_root"]) / result["trace_file"]
        if sha256_file(trace) != result["trace_sha256"]:
            raise ValueError("trace SHA mismatch")
        checked = validate_closed_loop_trace(trace, job["run_fingerprint"],
            expected_episode_id=result["episode_id"], expected_policy=job["policy"],
            expected_solver_seed=job["solver_seed"], metric_iteration_budget=None,
            collection_root=Path(job["output_root"]))
        if checked["summary"] != result["summary"] or result["summary"]["initial_fingerprint"] != spec["expected_initial_fingerprint"]:
            raise ValueError("trace/summary/checkpoint mismatch")
        initial = read_artifact(out / "initial.json", binding)
        terminal = read_artifact(out / "terminal.json", binding)
        journal = PathJournal(out, binding, ROOT / spec["case"]["files"]["map_file"], ROOT / spec["case"]["files"]["scenario_file"])
        for artifact in (initial, terminal):
            if journal.quality(artifact["observation"]) != artifact["quality"]:
                raise ValueError("saved path quality mismatch")
    elif summary["status"] != "external_timeout":
        raise ValueError("completed episode lacks worker result")
    return summary


def guarded_episode_child(spec):
    # Ctrl-C requests a parent-side safe stop; the explicit process fuse still
    # uses SIGTERM and can terminate an over-budget child.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _episode_child(spec)


def execute_episode(spec, *, authorized, resume):
    return supervise_episode(spec, authorized=authorized, resume=resume, child_entry=guarded_episode_child)


def run_serial(specs, output, *, resume, stop_requested, execute=execute_episode, validate_result=validated_episode):
    progress = []
    for spec in specs:
        if stop_requested():
            break
        previous = validate_result(spec)
        if previous is not None:
            if not resume:
                raise ValueError("episode exists; explicit resume required")
            summary = previous
        else:
            summary = execute(spec, authorized=True, resume=resume)
            if summary.get("error"):
                raise ValueError("episode error; collection stopped")
            validated = validate_result(spec)
            if validated != summary:
                raise ValueError("saved episode validation differs")
        progress.append({"item": spec["item"], "binding": spec["binding"], "summary": summary})
        write_json(output / "collection_progress.json", {"completed": len(progress), "total": len(specs), "rows": progress})
        print(f"Completed {len(progress)}/{len(specs)}: {spec['item']['controller']}", flush=True)
    status = "complete" if len(progress) == len(specs) else "paused_after_episode"
    write_json(output / "collection_status.json", {"status": status, "completed": len(progress), "total": len(specs)})
    return {"status": status, "completed": len(progress), "total": len(specs)}


def collect(*, authorized=False, resume=False):
    if authorized is not True:
        raise PermissionError("separate timing authorization required; no episode started")
    _config, specs = verify_prepared(require_ready=True)
    output = BASE / "timed"
    stopped = [False]
    def stop(_signum, _frame):
        stopped[0] = True
    old = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        with _CollectionRunLock(output, _fingerprint(specs), "warehouse-confirmation-timing"):
            write_json(output / "execution_environment.json", {"platform": platform.platform(), "python": sys.version})
            try:
                return run_serial(specs, output, resume=resume,
                                  stop_requested=lambda: stopped[0] or (BASE / "STOP_AFTER_EPISODE").exists())
            except Exception as error:
                write_json(output / "collection_status.json", {"status": "error_requires_inspection", "error": str(error)})
                raise
    finally:
        for sig, handler in old.items():
            signal.signal(sig, handler)


def comparison(left, right, gate):
    if not left or set(left) != set(right):
        return {"status": "incomplete", "passed": False}
    keys = sorted(left)
    maps = sorted({left[k]["map_id"] for k in keys})
    if any(left[k]["map_id"] != right[k]["map_id"] for k in keys):
        raise ValueError("unpaired maps")
    def improvement(selected):
        base = sum(left[k]["capped_delivery_seconds"] for k in selected)
        return (base - sum(right[k]["capped_delivery_seconds"] for k in selected)) / base if base else None
    observed = improvement(keys)
    rng = random.Random(2026102500)
    groups = {m: [k for k in keys if left[k]["map_id"] == m] for m in maps}
    samples = [improvement([k for _ in maps for k in groups[rng.choice(maps)]]) for _ in range(5000)]
    ci = [quantile(samples, p) for p in (0.025, 0.975)] if all(s is not None for s in samples) else None
    def throughput(rows):
        elapsed = sum(r["process_wall_seconds"] for r in rows.values())
        return sum(r["success"] for r in rows.values()) * 3600 / elapsed if elapsed else None
    base_rate, new_rate = throughput(left), throughput(right)
    rate_gain = new_rate / base_rate - 1 if base_rate and new_rate is not None else None
    wins = sum(right[k]["capped_delivery_seconds"] < left[k]["capped_delivery_seconds"] for k in keys)
    per_map = {m: improvement(group) for m, group in groups.items()}
    checks = {
        "success_count": sum(r["success"] for r in right.values()) >= sum(r["success"] for r in left.values()),
        "no_additional_censor": sum(r["censored"] for r in right.values()) <= sum(r["censored"] for r in left.values()),
        "paired_win_rate": wins / len(keys) >= gate["minimum_paired_win_rate"],
        "capped_delivery": observed is not None and observed >= gate["minimum_mean_capped_ttf_improvement"],
        "throughput": rate_gain is not None and rate_gain >= gate["minimum_successes_per_hour_improvement"],
        "nonworse_maps": sum(v is not None and v >= 0 for v in per_map.values()) >= gate["minimum_nonworse_maps"],
        "bootstrap": ci is not None and ci[0] >= 0,
    }
    return {"paired_count": len(keys), "map_count": len(maps), "wins": wins, "win_rate": wins / len(keys),
            "delivery_improvement": observed, "throughput_improvement": rate_gain,
            "map_bootstrap_95ci": ci, "per_map_improvement": per_map, "checks": checks,
            "passed": all(checks.values()), "bootstrap_unit": "map", "bootstrap_replicates": 5000}


def analyze():
    config, specs = verify_prepared(require_ready=True)
    groups = {c: {} for c in config["controllers"]}
    missing = []
    rows = []
    for spec in specs:
        supervisor = validated_episode(spec)
        item = spec["item"]
        if supervisor is None:
            missing.append(item["job_id"])
            continue
        out, binding = Path(spec["output"]), spec["binding"]
        delivered = read_artifact(out / "result.json", binding) if (out / "result.json").exists() else {}
        first = read_artifact(out / "first_phase_result.json", binding) if (out / "first_phase_result.json").exists() else {}
        paths = read_artifact(out / "first_feasible.json", binding) if (out / "first_feasible.json").exists() else {}
        success = bool(supervisor["success_by_deadline"])
        dispatch = delivered.get("dispatch_wall_seconds") if supervisor["status"] == "completed" else None
        row = {"checkpoint_id": item["checkpoint_id"], "map_id": item["map_id"], "controller": item["controller"],
               "status": supervisor["status"], "success": success,
               "censored": not success or not bool(supervisor.get("delivered_within_budget")),
               "algorithm_ttf_seconds": delivered.get("first_feasible_elapsed_seconds"),
               "validated_delivery_seconds": dispatch,
               "capped_delivery_seconds": min(dispatch, 60.0) if success and dispatch is not None else 60.0,
               "process_wall_seconds": supervisor["process_observed_wall_seconds"],
               "quality": paths.get("quality"), "solver_summary": first.get("summary"),
               "cold_start_seconds": delivered.get("startup_before_reset_seconds"),
               "binding": binding}
        groups[item["controller"]][item["checkpoint_id"]] = row
        rows.append(row)
    summaries = {name: {"count": len(values), "success_count": sum(r["success"] for r in values.values()),
                       "mean_capped_delivery_seconds": mean(r["capped_delivery_seconds"] for r in values.values()) if values else None,
                       "median_capped_delivery_seconds": statistics.median(r["capped_delivery_seconds"] for r in values.values()) if values else None}
                 for name, values in groups.items()}
    comparisons = {} if missing else {f"{right}_vs_{left}": comparison(groups[left], groups[right], config["gate"])
                   for left, right in [("official_adaptive", "dual16"), ("official_adaptive", "v2-full"), ("v2-full", "dual16")]}
    report = {"schema": "lns2.warehouse_confirmation_results.v1", "missing": missing, "summaries": summaries,
              "comparisons": comparisons, "rows": rows, "primary": "dual16_vs_official_adaptive",
              "primary_time": "reset_to_validated_paths_delivery", "algorithm_ttf_reported_separately": True,
              "status": "incomplete" if missing else "complete",
              "decision": "pending" if missing else "pass" if comparisons["dual16_vs_official_adaptive"]["passed"] else "fail"}
    write_json(BASE / "timed/confirmation_report.json", report)
    return {k: v for k, v in report.items() if k != "rows"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["validate", "verify", "collect", "analyze", "stop", "clear-stop"])
    parser.add_argument("--authorize-timing", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.phase == "validate":
        result = validate_preparation()
    elif args.phase == "verify":
        _, specs = verify_prepared(require_ready=True)
        result = {"status": "ready_awaiting_timing_authorization", "episodes": len(specs)}
    elif args.phase == "collect":
        result = collect(authorized=args.authorize_timing, resume=args.resume)
    elif args.phase == "analyze":
        result = analyze()
    elif args.phase == "stop":
        write_json(BASE / "STOP_AFTER_EPISODE", {"stop_after_current_episode": True})
        result = {"stop_requested": True}
    else:
        (BASE / "STOP_AFTER_EPISODE").unlink(missing_ok=True)
        result = {"stop_request_cleared": True}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
