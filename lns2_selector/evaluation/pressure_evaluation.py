"""Pressure-cohort adapter around the unchanged path-quality episode engine."""

from __future__ import annotations

import copy
import shutil
from pathlib import Path

from experiments._common import json_fingerprint, read_json, read_jsonl, sha256_file, write_json, write_jsonl
from experiments.repair_collection import _CollectionRunLock
from lns2_selector.evaluation import path_quality_cohort as cohort
from lns2_selector.evaluation.path_quality_analysis import publish_analysis, summarize
from lns2_selector.evaluation.path_quality_preflight import checked_input, contained, prepare, publish
from lns2_selector.evaluation.path_quality_pressure import prepare_pressure


def write_once(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"frozen generated input changed: {path.name}")
    else:
        write_json(path, value)


def copy_once(source, destination):
    if destination.exists():
        if sha256_file(source) != sha256_file(destination):
            raise ValueError("frozen input copy differs")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)


def load_design(root, design_path):
    design = read_json(design_path)
    if (design["schema"] != "lns2.pressure_evaluation_design.v1"
            or design["timing_requires_separate_authorization"] is not True
            or design["controllers"] != ["official_adaptive", "v2-full", "dual16"]
            or design["solver_seeds"] != [61, 62]
            or (design["expected_maps"], design["expected_tasks"], design["expected_episodes"]) != (8, 48, 864)):
        raise ValueError("frozen pressure design changed")
    output = contained(root, design["output"])
    if not output.is_relative_to((root / "build").resolve()):
        raise ValueError("output must be inside build")
    return design, output


def prepare_evaluation(root, design_path):
    design, output = load_design(root, design_path)
    pressure = prepare_pressure(root, contained(root, design["preparation_config"]), verify=True)
    prepared_path = checked_input(root, design["preparation_report"])
    if pressure != read_json(prepared_path) or pressure["generation_errors"]:
        raise ValueError("pressure preparation changed or failed")
    reset_path = checked_input(root, design["reset_report"])
    reset = read_json(reset_path)
    reset_registration = reset_path.parent / "registration.json"
    if (reset["status"] != "complete_reset_only" or reset["completed"] != 96
            or json_fingerprint(read_json(reset_registration)) != reset["binding"]):
        raise ValueError("pressure reset provenance incomplete")
    if sha256_file(reset_path.parent / "manifest.jsonl") != reset["manifest_sha256"]:
        raise ValueError("pressure reset manifest changed")
    expected_keys = {(c["case"]["task_id"], s) for c in pressure["cases"] for s in design["solver_seeds"]}
    keys = [(r["case"]["task_id"], r["solver_seed"]) for r in reset["rows"]]
    if len(keys) != len(expected_keys) or set(keys) != expected_keys or any(r["status"] != "ok" for r in reset["rows"]):
        raise ValueError("pressure reset roster mismatch")
    inputs = {p.relative_to(root).as_posix(): sha256_file(p) for p in (
        design_path, prepared_path, reset_path, reset_registration, reset_path.parent / "manifest.jsonl",
        contained(root, design["preparation_config"]), checked_input(root, design["protocol_template"]))}
    for row in reset["rows"]:
        path = contained(root, row["reset_file"])
        if sha256_file(path) != row["sha256"]:
            raise ValueError("pressure reset state SHA changed")
        inputs[row["reset_file"]] = row["sha256"]
    source = output / "inputs"
    normalized, by_task = [], {}
    for row in pressure["cases"]:
        case = row["case"]
        files = {}
        for field in ("map_file", "scenario_file", "task_file"):
            original = contained(root, row["files"][field])
            destination = source / ("maps" if field == "map_file" else "instances") / original.name
            copy_once(original, destination)
            files[field] = destination.relative_to(source).as_posix()
            inputs[row["files"][field]] = sha256_file(original)
        normalized.append({"map_id": case["map_id"], "task_id": case["task_id"],
                           "agent_count": row["quality"]["agent_count"], **files})
        by_task[case["task_id"]] = case
    manifest = source / "manifest.jsonl"
    if manifest.exists() and read_jsonl(manifest) != normalized:
        raise ValueError("adapted manifest changed")
    write_jsonl(manifest, normalized)
    empty = source / "no_checkpoints.jsonl"
    if empty.exists() and empty.read_bytes():
        raise ValueError("fresh-planning cohort cannot contain checkpoint starts")
    write_jsonl(empty, [])
    preflight_config = copy.deepcopy(read_json(checked_input(root, design["protocol_template"])))
    preflight_config.update(scientific_status=design["scientific_status"], preparation_revision=1,
        output=(output / "static_preflight").relative_to(root).as_posix(), solver_seeds=design["solver_seeds"],
        quarantined_maps={}, sources=[{"name": "warehouse", "manifest": manifest.relative_to(root).as_posix(),
                                     "sha256": sha256_file(manifest), "all_tasks": True, "expected_tasks": 48}],
        supplementary_checkpoints={"manifest": empty.relative_to(root).as_posix(), "sha256": sha256_file(empty),
                                   "expected_count": 0, "role": "not_used_fresh_planning_only"})
    preflight_path = source / "preflight_config.json"
    write_once(preflight_path, preflight_config)
    report = prepare(root, preflight_path)
    if (report["map_count"], report["eligible_tasks"], report["quarantined_tasks"], report["proposed_total_episode_count"]) != (8, 48, 0, 864):
        raise ValueError("pressure cohort dimensions changed")
    for case in report["cases"]:
        case["pressure_design"] = by_task[case["task_id"]]
    report["blocking_items"] = ["Separate explicit timing authorization and quiet-machine/AC attestation required.",
        "Fresh budget-specific reset admission required; the exploratory 120s reset report is not reused as 60s admission.",
        "Reused Pilot development evidence, not independent confirmation or real-robot execution."]
    report["fingerprint"] = json_fingerprint({k:v for k,v in report.items() if k != "fingerprint"})
    publish(root, report, preflight_config["output"])
    evaluation_config = {"schema": "lns2.path_quality_evaluation.v1",
        "preflight_config": preflight_path.relative_to(root).as_posix(), "output": design["output"],
        "native_file": design["native_file"], "native_sha256": design["native_sha256"],
        "process_fuse_extra_seconds": 120, "bootstrap_samples": 5000, "bootstrap_seed": 20260907,
        "primary_step_seconds": 1.0, "step_seconds_sensitivity": [0.5, 1.0, 2.0],
        "primary_clock": "reset_to_saved_path_delivery", "workers": 1,
        "timing_requires_separate_authorization": True, "branch": design["branch"], "remote": design["remote"]}
    evaluation_path = source / "evaluation_config.json"
    write_once(evaluation_path, evaluation_config)
    paths = {**report["input_sha256"], **inputs,
             evaluation_path.relative_to(root).as_posix(): sha256_file(evaluation_path),
             preflight_path.relative_to(root).as_posix(): sha256_file(preflight_path)}
    for relative in ("lns2_selector/evaluation/pressure_evaluation.py", "scripts/run_path_quality_pressure_evaluation.py",
                     "lns2_selector/evaluation/path_quality_cohort.py", "lns2_selector/evaluation/path_quality_analysis.py"):
        paths[relative] = sha256_file(contained(root, relative))
    for directory in ("src", "include", "third_party/mapf_lns2/inc", "third_party/mapf_lns2/src"):
        for path in (root / directory).rglob("*"):
            if path.is_file() and path.suffix in {".cpp", ".h", ".hpp"}:
                paths[path.relative_to(root).as_posix()] = sha256_file(path)
    registration = {"schema": "lns2.path_quality_registration.v1", "inputs": dict(sorted(paths.items())),
                    "native_file": design["native_file"], "native_sha256": design["native_sha256"],
                    "preflight": report["fingerprint"], "timed_episodes_started": 0}
    registration["fingerprint"] = json_fingerprint(registration)
    write_once(output / "registration.json", registration)
    write_once(output / "preflight.json", report)
    schedule_path = output / "execution_schedule.jsonl"
    if schedule_path.exists() and read_jsonl(schedule_path) != report["execution_schedule"]:
        raise ValueError("frozen pressure schedule changed")
    write_jsonl(schedule_path, report["execution_schedule"])
    return {"status": "prepared_timing_blocked", "scheduled": 864, "reset_admissions_required": 192,
            "registration": registration["fingerprint"], "evaluation_config": evaluation_path.relative_to(root).as_posix()}


def evaluation_path(root, design_path):
    _, output = load_design(root, design_path)
    path = output / "inputs/evaluation_config.json"
    _, _, registration, _ = cohort.verified_registration(root, path)
    if registration["inputs"].get(design_path.relative_to(root).as_posix()) != sha256_file(design_path):
        raise ValueError("unregistered pressure design")
    return path


def analyze_pressure(root, design_path, *, partial=False, samples=5000):
    config_path = evaluation_path(root, design_path)
    report = cohort.analyze(root, config_path, partial=partial)
    _, output, _, preflight = cohort.verified_registration(root, config_path)
    designs = {c["task_id"]: c["pressure_design"] for c in preflight["cases"]}
    for row in report["episodes"]:
        design = designs[row["task_id"]]
        row.update(design_density=design["density"], od_mode=design["mode"]["name"])
    report["pressure_strata"] = {}
    for field in ("design_density", "od_mode"):
        for value in sorted({r[field] for r in report["episodes"]}):
            report["pressure_strata"][f"{field}:{value}"] = summarize(
                [r for r in report["episodes"] if r[field] == value], samples=samples)
    publish_analysis(output / ("pressure_partial_analysis" if partial else "pressure_analysis"), report["episodes"], report)
    return report


def collect_pressure(root, design_path, *, authorize=False, quiet_machine=False, on_ac=False, resume=False):
    if not authorize or not quiet_machine or not on_ac:
        raise PermissionError("separate timing authorization, quiet-machine and AC confirmation required")
    config_path = evaluation_path(root, design_path)
    config, output, registration, report = cohort.verified_registration(root, config_path, require_native=True)
    anchors = cohort.load_admission(output, registration)
    if cohort.git_read(root, "status", "--porcelain") or cohort.git_read(root, "branch", "--show-current") != config["branch"]:
        raise ValueError("clean preregistered branch required")
    head = cohort.git_read(root, "rev-parse", "HEAD")
    remote = cohort.git_read(root, "ls-remote", config["remote"], f"refs/heads/{config['branch']}")
    if not remote or remote.split()[0] != head:
        raise ValueError("preregistration is not pushed")
    write_once(output / "timing_authorization.json", {"binding": registration["fingerprint"], "commit": head,
        "branch": config["branch"], "admission_sha256": sha256_file(output / "admission.json"), "authorized": True,
        "operator_attestation": {"quiet_machine": quiet_machine, "on_ac": on_ac}})
    cases = {c["task_id"]: c for c in report["cases"]}
    results, total = [], len(report["execution_schedule"])
    with _CollectionRunLock(output, registration["fingerprint"], "pressure-path-quality-timing"):
        for item in report["execution_schedule"]:
            host = cohort.runtime_environment()
            busy = host["load_average"] is not None and host["load_average"][0] > max(2, (host["logical_cpus"] or 1) / 2)
            if (output / "pause.request").exists() or busy:
                write_json(output / "run_status.json", {"status": "paused", "next_job": item["job_id"], "completed": len(results), "total": total})
                return {"status": "paused", "completed": len(results), "total": total}
            cohort.verified_registration(root, config_path, require_native=True)
            spec = cohort.bound_spec(root, config, output, registration, report, item, anchors)
            summary = cohort.supervise_episode(spec, authorized=True, resume=resume)
            inspected = cohort.inspect_episode(root, cases[item["task_id"]], item, Path(spec["output"]), spec["binding"], anchors[cohort.admission_key(item)])
            results.append({"job_id": item["job_id"], "summary": summary, "status": inspected["status"], "environment": host})
            write_jsonl(output / "collection_manifest.jsonl", results)
            write_json(output / "run_status.json", {"status": "running", "completed": len(results), "total": total})
            print(f"episode {len(results)}/{total} {item['controller']} {item['task_id']}: {inspected['status']}", flush=True)
            if summary["error"]:
                write_json(output / "run_status.json", {"status": "blocked_on_error", "completed": len(results), "total": total})
                return {"status": "blocked_on_error", "completed": len(results), "total": total}
    write_json(output / "run_status.json", {"status": "completed", "completed": total, "total": total})
    return {"status": "completed", "completed": total, "analysis_pending": True}
