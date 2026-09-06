"""Frozen whole-cohort orchestration; validation executes resets, never repairs."""
from __future__ import annotations

import importlib.metadata
import multiprocessing
import os
import platform
import subprocess
import time
from pathlib import Path

from experiments._common import json_fingerprint, read_json, sha256_file, write_json, write_jsonl
from experiments.repair_collection import _CollectionRunLock, _make_environment, _plain, state_fingerprint
from lns2_selector.evaluation.path_quality_execution import (
    PathJournal, controller_job, prepare_episode_spec, read_artifact, spec_fingerprint, supervise_episode,
)
from lns2_selector.evaluation.path_quality_preflight import contained, prepare, publish
from lns2_selector.evaluation.path_quality_analysis import inspect_episode, publish_analysis, summarize
from lns2_selector.solver.native import load_native_module, native_identity


def git_read(root: Path, *args) -> str:
    return subprocess.check_output(["git", "-c", f"safe.directory={root}", *args], cwd=root, text=True).strip()


def load_inputs(root: Path, config_path: Path):
    config = read_json(config_path)
    if (config.get("schema") != "lns2.path_quality_evaluation.v1" or config.get("workers") != 1
        or config.get("process_fuse_extra_seconds") != 120 or config.get("bootstrap_samples") != 5000
        or config.get("primary_clock") != "reset_to_saved_path_delivery"
        or config.get("step_seconds_sensitivity") != [0.5, 1.0, 2.0]
        or config.get("timing_requires_separate_authorization") is not True):
        raise ValueError("evaluation contract changed")
    preflight_config = contained(root, config["preflight_config"])
    output = contained(root, config["output"])
    output.relative_to((root / "build").resolve())
    return config, preflight_config, output


def static_prepare(root: Path, config_path: Path) -> dict:
    config, preflight_config, output = load_inputs(root, config_path)
    report = prepare(root, preflight_config)
    publish(root, report, read_json(preflight_config)["output"])
    if report["eligible_tasks"] != 14 or report["quarantined_tasks"] != 2 or report["proposed_total_episode_count"] != 252:
        raise ValueError("frozen case count changed")
    paths = {**report["input_sha256"], config_path.relative_to(root).as_posix(): sha256_file(config_path)}
    for relative in ("lns2_selector/evaluation/path_quality_cohort.py", "lns2_selector/evaluation/path_quality_analysis.py",
                     "scripts/run_path_quality_evaluation.py"):
        paths[relative] = sha256_file(contained(root, relative))
    for directory in ("src", "include", "third_party/mapf_lns2/inc", "third_party/mapf_lns2/src"):
        for path in (root / directory).rglob("*"):
            if path.is_file() and path.suffix in {".cpp", ".h", ".hpp"}:
                paths[path.relative_to(root).as_posix()] = sha256_file(path)
    registration = {"schema": "lns2.path_quality_registration.v1", "inputs": dict(sorted(paths.items())),
                    "native_file": config["native_file"], "native_sha256": config["native_sha256"],
                    "preflight": report["fingerprint"], "timed_episodes_started": 0}
    registration["fingerprint"] = json_fingerprint(registration)
    path = output / "registration.json"
    if path.exists() and read_json(path) != registration:
        raise ValueError("registered inputs changed; create a new run directory/config revision")
    if not path.exists():
        write_json(path, registration)
        write_json(output / "preflight.json", report)
        write_jsonl(output / "execution_schedule.jsonl", report["execution_schedule"])
    return registration


def verified_registration(root: Path, config_path: Path, *, require_native: bool = False):
    config, _, output = load_inputs(root, config_path)
    registration = read_json(output / "registration.json")
    if json_fingerprint({k:v for k,v in registration.items() if k != "fingerprint"}) != registration["fingerprint"]:
        raise ValueError("registration fingerprint mismatch")
    if registration["inputs"].get(config_path.relative_to(root).as_posix()) != sha256_file(config_path):
        raise ValueError("unregistered evaluation config")
    for relative, expected in registration["inputs"].items():
        if sha256_file(contained(root, relative)) != expected:
            raise ValueError(f"registered input changed: {relative}")
    report = read_json(output / "preflight.json")
    if report["fingerprint"] != registration["preflight"] or json_fingerprint({k:v for k,v in report.items() if k != "fingerprint"}) != report["fingerprint"]:
        raise ValueError("preflight fingerprint mismatch")
    from experiments._common import read_jsonl
    if read_jsonl(output / "execution_schedule.jsonl") != report["execution_schedule"]:
        raise ValueError("schedule changed")
    if sha256_file(contained(root, config["native_file"])) != config["native_sha256"]:
        raise ValueError("registered native binary changed")
    if require_native:
        module = load_native_module()
        identity = native_identity(module)
        if Path(identity["path"]).resolve() != contained(root, config["native_file"]) or identity["sha256"] != config["native_sha256"]:
            raise ValueError("wrong native module loaded")
        if getattr(module, "anytime_handoff_schema", None) != "lns2.anytime_handoff.v1":
            raise ValueError("native handoff schema absent")
    return config, output, registration, report


def admission_key(item: dict) -> str:
    return json_fingerprint({k:item[k] for k in ("task_id", "solver_seed", "budget_seconds")})[:24]


def _reset_child(job: dict) -> None:
    root, output = Path(job["root"]), Path(job["output"])
    case, item = job["case"], job["item"]
    try:
        identity = native_identity()
        if identity["sha256"] != job["native_sha256"]:
            raise ValueError("reset loaded native mismatch")
        worker = controller_job(root, case, item, job["template"], output, job["binding"])
        environment = _make_environment(worker["dataset_root"], worker["row"], worker["environment"], "Adaptive")
        state = _plain(environment.reset(seed=item["solver_seed"]))
        quality = PathJournal(output, job["binding"], contained(root, case["files"]["map_file"]),
                              contained(root, case["files"]["scenario_file"])).quality(state)
        result = {"status": "ok", "state_fingerprint": state_fingerprint(state), "initial_quality": quality,
                  "paths": [a["path"] for a in sorted(state["agents"], key=lambda a:a["id"])], "repairs_executed": 0}
    except Exception as error:
        result = {"status": "error", "error_type": type(error).__name__, "error": str(error), "repairs_executed": 0}
    result.update(binding=job["binding"], key=admission_key(item))
    write_json(output / "reset.json", result)


def runtime_environment() -> dict:
    versions = {}
    for name in ("numpy", "scikit-learn", "pytest"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "logical_cpus": os.cpu_count(),
            "dependency_versions": versions, "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
            "windows_power_and_gpu_load": "not_measured_operator_attestation_required"}


def validate_resets(root: Path, config_path: Path, *, resume=False) -> dict:
    config, output, registration, report = verified_registration(root, config_path, require_native=True)
    cases = {c["task_id"]:c for c in report["cases"]}
    template = read_json(contained(root, report["runtime_template"]["manifest"]))
    unique = {}
    for item in report["execution_schedule"]:
        if item["controller"] == "official_adaptive":
            unique.setdefault(admission_key(item), item)
    rows = []
    with _CollectionRunLock(output, registration["fingerprint"], "path-quality-reset-admission"):
        for key, item in unique.items():
            folder = output / "admission" / key
            path = folder / "reset.json"
            if path.exists():
                if not resume:
                    raise ValueError("reset admission exists; request resume")
                row = read_json(path)
                if row["binding"] != registration["fingerprint"] or row["key"] != key:
                    raise ValueError("reset admission identity changed")
            else:
                process = multiprocessing.get_context("spawn").Process(target=_reset_child, args=({"root": str(root),
                    "output": str(folder), "case": cases[item["task_id"]], "item": item, "template": template,
                    "binding": registration["fingerprint"], "native_sha256": config["native_sha256"]},))
                try:
                    process.start()
                    process.join(item["budget_seconds"] + 120)
                    if process.is_alive():
                        process.terminate()
                        process.join(5)
                        if process.is_alive():
                            process.kill()
                            process.join()
                    if process.exitcode != 0 or not path.exists():
                        write_json(path, {"binding": registration["fingerprint"], "key": key, "status": "error",
                                          "error": "reset process did not return normally", "repairs_executed": 0})
                finally:
                    if process.is_alive():
                        process.terminate()
                        process.join()
                    process.close()
                row = read_json(path)
            rows.append({**row, "reset_file": path.relative_to(output).as_posix(), "reset_sha256": sha256_file(path),
                         "task_id": item["task_id"], "solver_seed": item["solver_seed"], "budget_seconds": item["budget_seconds"]})
            write_jsonl(output / "admission_manifest.jsonl", rows)
            print(f"reset admission {len(rows)}/{len(unique)}: {item['task_id']} seed={item['solver_seed']} budget={item['budget_seconds']:g} {row['status']}", flush=True)
            if row["status"] != "ok":
                break
        result = {"schema": "lns2.path_quality_admission.v1", "binding": registration["fingerprint"],
                  "expected": len(unique), "completed": len(rows), "passed": len(rows) == len(unique) and all(r["status"] == "ok" for r in rows),
                  "reset_count_note": "one per task/seed/budget; three controllers share identical reset configuration",
                  "timed_episodes_started": 0, "repairs_executed": 0, "environment": runtime_environment(),
                  "manifest_sha256": sha256_file(output / "admission_manifest.jsonl")}
        write_json(output / "admission.json", result)
        return result


def load_admission(output: Path, registration: dict) -> dict:
    from experiments._common import read_jsonl
    result = read_json(output / "admission.json")
    if not result["passed"] or result["binding"] != registration["fingerprint"] or result["manifest_sha256"] != sha256_file(output / "admission_manifest.jsonl"):
        raise ValueError("reset admission failed or changed")
    rows = read_jsonl(output / "admission_manifest.jsonl")
    if len(rows) != result["expected"] or len({r["key"] for r in rows}) != len(rows):
        raise ValueError("reset admission incomplete")
    for row in rows:
        if sha256_file(contained(output, row["reset_file"])) != row["reset_sha256"] or row["status"] != "ok":
            raise ValueError("reset anchor changed")
    return {r["key"]: r for r in rows}


def bound_spec(root, config, output, registration, report, item, anchors):
    template = read_json(contained(root, report["runtime_template"]["manifest"]))
    spec = prepare_episode_spec(root, report, item, output / "episodes" / item["job_id"], template,
                                config["native_sha256"], item["budget_seconds"] + 120)
    spec["expected_initial_fingerprint"] = anchors[admission_key(item)]["state_fingerprint"]
    spec["cohort_registration"] = registration["fingerprint"]
    spec["binding"] = spec_fingerprint(spec)
    return spec


def analyze(root: Path, config_path: Path, *, partial=False) -> dict:
    config, output, registration, report = verified_registration(root, config_path)
    anchors = load_admission(output, registration)
    cases = {c["task_id"]:c for c in report["cases"]}
    rows = []
    for item in report["execution_schedule"]:
        spec = bound_spec(root, config, output, registration, report, item, anchors)
        rows.append(inspect_episode(root, cases[item["task_id"]], item, Path(spec["output"]), spec["binding"], anchors[admission_key(item)]))
    complete = not any(r["status"] in {"pending", "interrupted"} for r in rows)
    if not complete and not partial:
        raise ValueError("cohort incomplete; partial reports require --partial")
    result = {"schema": "lns2.path_quality_report.v1", "binding": registration["fingerprint"],
              "complete": complete, "scheduled": len(rows), "scientific_status": report["scientific_status"],
              "promotion_allowed": False, "statistics": summarize(rows, samples=config["bootstrap_samples"]), "episodes": rows}
    publish_analysis(output / ("partial_report" if partial else "analysis"), rows, result)
    return result


def collect(root: Path, config_path: Path, *, authorize=False, quiet_machine=False, on_ac=False, resume=False) -> dict:
    if not authorize or not quiet_machine or not on_ac:
        raise PermissionError("timing requires explicit authorization, quiet-machine and AC-power confirmation")
    config, output, registration, report = verified_registration(root, config_path, require_native=True)
    anchors = load_admission(output, registration)
    if git_read(root, "status", "--porcelain") or git_read(root, "branch", "--show-current") != config["branch"]:
        raise ValueError("timing requires the clean preregistered branch")
    head = git_read(root, "rev-parse", "HEAD")
    remote = git_read(root, "ls-remote", config["remote"], f"refs/heads/{config['branch']}")
    if not remote or remote.split()[0] != head:
        raise ValueError("preregistration has not been pushed")
    receipt = {"binding": registration["fingerprint"], "commit": head, "branch": config["branch"],
               "admission_sha256": sha256_file(output / "admission.json"), "authorized": True,
               "operator_attestation": {"quiet_machine": quiet_machine, "on_ac": on_ac}}
    receipt_path = output / "timing_authorization.json"
    if receipt_path.exists() and read_json(receipt_path) != receipt:
        raise ValueError("authorization receipt changed; do not mix runs")
    write_json(receipt_path, receipt)
    cases = {c["task_id"]:c for c in report["cases"]}
    results = []
    with _CollectionRunLock(output, registration["fingerprint"], "path-quality-timing"):
        for item in report["execution_schedule"]:
            host = runtime_environment()
            busy = host["load_average"] is not None and host["load_average"][0] > max(2, (host["logical_cpus"] or 1) / 2)
            if (output / "pause.request").exists() or busy:
                write_json(output / "run_status.json", {"status": "paused", "next_job": item["job_id"], "environment": host})
                return {"status": "paused", "completed": len(results)}
            verified_registration(root, config_path, require_native=True)
            spec = bound_spec(root, config, output, registration, report, item, anchors)
            summary = supervise_episode(spec, authorized=True, resume=resume)
            inspected = inspect_episode(root, cases[item["task_id"]], item, Path(spec["output"]), spec["binding"], anchors[admission_key(item)])
            result = {"job_id": item["job_id"], "summary": summary, "status": inspected["status"], "environment": host}
            results.append(result)
            write_jsonl(output / "collection_manifest.jsonl", results)
            write_json(output / "run_status.json", {"status": "running", "completed": len(results), "total": len(report["execution_schedule"])})
            print(f"episode {len(results)}/252 {item['controller']} {item['task_id']}: {inspected['status']}", flush=True)
            if summary["error"]:
                write_json(output / "run_status.json", {"status": "blocked_on_error", "job_id": item["job_id"], "completed": len(results)})
                return {"status": "blocked_on_error", "completed": len(results)}
    result = analyze(root, config_path)
    write_json(output / "run_status.json", {"status": "completed", "completed": 252, "report_complete": result["complete"]})
    return {"status": "completed", "completed": 252}
