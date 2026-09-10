"""Prepare the frozen warehouse confirmation; timed collection is opt-in."""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments._common import read_json, read_jsonl, sha256_file
from experiments.repair_collection import (
    _CollectionRunLock, _fingerprint, _load_dataset_rows, _make_environment, _plain,
    _run_jobs, _write_json as write_json, _write_jsonl as write_jsonl, state_fingerprint,
)
from experiments.closed_loop_trace_storage import write_state_blob
from experiments.state_analysis import summarize_initial_state_complexity
from experiments.stride_warehouse_disruption_recovery import _select_global_time
from experiments.warehouse_disruption_checkpoints import (
    checkpoint_manifest_fields, compute_checkpoint_identity_sha256,
    inject_global_hotspot_delays, warehouse_checkpoint_gate, warehouse_hotspot_definition,
)
from lns2_selector.runtime.fingerprints import repair_structure_fingerprint
from scripts.prepare_warehouse_repair_confirmation import CONFIG, schedule_slots, validate

BASE = ROOT / "build/warehouse-repair-confirmation-v1"


def verify_inputs():
    config = read_json(ROOT / CONFIG)
    validate(config)
    preflight = read_json(BASE / "preparation/preflight.json")
    for name, digest in preflight["input_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            raise ValueError("frozen input changed: " + name)
    registration = read_json(ROOT / "artifacts/warehouse-repair-confirmation-v1/dataset_validation.json")
    report_path = ROOT / registration["report_path"]
    if sha256_file(report_path) != registration["report_sha256"]:
        raise ValueError("registered dataset validation changed")
    for name, digest in read_json(report_path)["files"].items():
        if sha256_file(ROOT / name) != digest:
            raise ValueError("dataset changed: " + name)
    return config


def install_native(config):
    path = ROOT / config["frozen"]["native_path"]
    sys.path.insert(0, str(path.parent))
    os.environ["PYTHONPATH"] = str(path.parent) + os.pathsep + str(ROOT)
    import lns2_env
    if Path(lns2_env.__file__).resolve() != path.resolve() or sha256_file(path) != config["frozen"]["native_sha256"]:
        raise ValueError("loaded native is not the registered binary")
    return lns2_env


def environment_config(config):
    return dict(time_limit=config["checkpoint"]["incumbent_time_limit_seconds"],
                max_repair_iterations=0, neighborhood_size=8, replan_algorithm="PP", use_sipp=True)


def task_plan(config):
    rows = sorted(_load_dataset_rows(ROOT / config["dataset"]["output"], ["confirmation"]), key=lambda r: r["task_id"])
    slots = schedule_slots(config)
    return [{"task_index": index, "row": row,
             "slots": [s for s in slots if s["task_slot"] == index and s["controller"] == "official_adaptive"]}
            for index, row in enumerate(rows)]


def unavailable(item, reason):
    return [{"key_index": slot["checkpoint_slot"], "task_id": item["row"]["task_id"],
             "map_id": item["row"]["map_id"], "status": "state_supply_unavailable", "reason": reason}
            for slot in item["slots"]]


def checkpoint_worker(job):
    config, item = job["config"], job["item"]
    install_native(config)
    row = item["row"]
    dataset = ROOT / config["dataset"]["output"]
    out = BASE / "checkpoints"
    task_dir = out / f"task-{item['task_index']:02d}"
    task_dir.mkdir(parents=True, exist_ok=True)
    env_config = environment_config(config)
    env = _make_environment(str(dataset), row, env_config, "Adaptive")
    state = _plain(env.reset(seed=item["slots"][0]["incumbent_seed"]))
    decisions = 0
    while not state["done"]:
        state = _plain(env.step({"mode": "official"}))["observation"]
        decisions += 1
    if not state["feasible"] or state["num_of_colliding_pairs"]:
        return {"task_index": item["task_index"], "status": "ok", "checkpoints": [],
                "failures": unavailable(item, "Official incumbent budget exhausted"),
                "incumbent_feasible": False, "incumbent_decisions": decisions}
    paths = [list(a["path"]) for a in sorted(state["agents"], key=lambda a: a["id"])]
    incumbent = {"paths": paths, "state_fingerprint": state_fingerprint(state),
                 "solver_seed": item["slots"][0]["incumbent_seed"], "decisions": decisions,
                 "map_id": row["map_id"], "task_id": row["task_id"], "feasible": True}
    write_json(task_dir / "incumbent.json", incumbent)
    split = dataset / row["split"]
    metadata = read_json(split / row["map_metadata_file"])["metadata"]
    generation = config["checkpoint"]
    hotspot = warehouse_hotspot_definition(metadata["station_zones"], metadata["semantic_cell_types"],
                                           neighborhood_radius=generation["hotspot_neighborhood_radius"])
    try:
        timing = _select_global_time(paths, hotspot, generation["delayed_agent_fraction"])
    except ValueError as error:
        return {"task_index": item["task_index"], "status": "ok", "checkpoints": [],
                "failures": unavailable(item, str(error)), "incumbent_feasible": True}
    checkpoints = []
    for slot in item["slots"]:
        disturbance = inject_global_hotspot_delays(paths, semantic_cell_types=metadata["semantic_cell_types"],
            station_zones=metadata["station_zones"], global_time=timing["global_time"],
            delay_ticks=generation["delay_ticks"], delayed_agent_fraction=generation["delayed_agent_fraction"],
            selection_seed=slot["selection_seed"], neighborhood_radius=generation["hotspot_neighborhood_radius"])
        cp = checkpoint_manifest_fields(map_id=row["map_id"], task_id=row["task_id"],
            map_sha256=sha256_file(split / row["map_file"]), task_sha256=sha256_file(split / row["task_file"]),
            source_paths=paths, disturbance=disturbance, checkpoint_id=f"repair-confirm-{slot['checkpoint_slot']:02d}")
        restored = _make_environment(str(dataset), row, env_config, "Adaptive")
        observation = _plain(restored.reset_paths(disturbance["paths"], seed=slot["selection_seed"]))
        complexity = summarize_initial_state_complexity(observation)
        for key in ("conflict_pair_count", "active_conflict_agent_count", "largest_conflict_component_size"):
            if complexity[key] != disturbance["conflicts"][key]:
                raise ValueError("native conflict mismatch: " + key)
        relative, blob = write_state_blob(out, observation)
        cp.update(schema="lns2.warehouse_disruption_checkpoint.native.v1", source_kind="checkpoint_blob_v1",
            key_index=slot["checkpoint_slot"], key_id=f"{row['task_id']}#repair-{slot['replica']}",
            split=row["split"], task_variant=row["task_variant"], replica=slot["replica"],
            screen_solver_seed=slot["solver_seed"], restore_seed=slot["selection_seed"],
            incumbent={"planner": "official_adaptive", "solver_seed": incumbent["solver_seed"],
                       "repair_decision_count": decisions, "state_fingerprint": incumbent["state_fingerprint"],
                       "paths_sha256": disturbance["source_paths_sha256"], "feasible": True, "conflict_pair_count": 0,
                       "file": (task_dir / "incumbent.json").relative_to(out).as_posix(),
                       "file_sha256": sha256_file(task_dir / "incumbent.json")},
            global_time_selection={"rule": generation["global_time_rule"], **timing},
            qualification=warehouse_checkpoint_gate(complexity), native_complexity=complexity,
            state_blob=relative, state_blob_sha256=sha256_file(blob),
            expected_fingerprint=state_fingerprint(observation), expected_conflicts=observation["num_of_colliding_pairs"],
            repair_structure_fingerprint=repair_structure_fingerprint(observation),
            native_solver_or_controller_invoked=True, checkpoint_selection_controller_outcomes_consulted=False)
        cp["checkpoint_identity_sha256"] = compute_checkpoint_identity_sha256(cp)
        checkpoints.append(cp)
    return {"task_index": item["task_index"], "status": "ok", "checkpoints": checkpoints,
            "failures": [], "incumbent_feasible": True}


def failure_result(job, status, error):
    return {"task_index": job["item"]["task_index"], "status": status, "error": error,
            "checkpoints": [], "failures": [{**r, "status": status} for r in unavailable(job["item"], error)]}


def checkpoint_report(results):
    rows = sorted([c for r in results for c in r["checkpoints"]], key=lambda c: c["key_index"])
    failures = [f for r in results for f in r["failures"]]
    keys = [r["key_index"] for r in rows + failures]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate checkpoint positions")
    qualified = [c for c in rows if c["qualification"]["passed"]]
    maps = {c["map_id"] for c in qualified}
    errors = [f for f in failures if f["status"] != "state_supply_unavailable"]
    return {"schema": "lns2.warehouse_confirmation_checkpoints.v1", "attempted": len(keys),
            "completed": len(rows), "qualified": len(qualified), "qualified_maps": len(maps),
            "errors": errors, "failures": failures, "qualified_ids": [c["checkpoint_id"] for c in qualified],
            "passed": set(keys) == set(range(32)) and len(qualified) >= 24 and len(maps) == 8 and not errors,
            "timed_episodes": 0, "replacement_permitted": False, "checkpoints": rows}


def prepare(resume=False):
    config = verify_inputs()
    install_native(config)
    plan = task_plan(config)
    root = BASE / "checkpoints"
    identity = {"config": sha256_file(ROOT / CONFIG), "script": sha256_file(Path(__file__)), "plan": plan}
    fingerprint = _fingerprint(identity)
    with _CollectionRunLock(root, fingerprint, "confirmation-checkpoints"):
        registered = root / "preparation_identity.json"
        if registered.exists() and (not resume or read_json(registered) != identity):
            raise ValueError("preparation exists or identity changed; refusing automatic rerun")
        write_json(registered, identity)
        jobs, existing = [], []
        for item in plan:
            path = root / f"result-{item['task_index']:02d}.json"
            if path.exists():
                result = read_json(path)
                if result["run_fingerprint"] != fingerprint:
                    raise ValueError("resume identity mismatch")
                existing.append(result)
            else:
                if (root / f"task-{item['task_index']:02d}").exists():
                    raise ValueError("partial task exists without result; audit required, not rerun")
                jobs.append({"job_id": str(item["task_index"]), "item": item, "config": config})
        def save(result):
            result["run_fingerprint"] = fingerprint
            write_json(root / f"result-{result['task_index']:02d}.json", result)
            print(f"Checkpoint task {result['task_index'] + 1}/16: {result['status']}", flush=True)
        results = existing + _run_jobs(checkpoint_worker, jobs, 20, phase="confirmation-preparation",
            output_root=root, run_fingerprint=fingerprint, timeout_seconds=300,
            on_result=save, failure_result=failure_result, stop_on_failure=False)
        report = checkpoint_report(results)
        write_json(root / "checkpoint_report.json", report)
        write_jsonl(root / "checkpoint_manifest.jsonl", report["checkpoints"])
        print(json.dumps({k: v for k, v in report.items() if k != "checkpoints"}, indent=2))
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare"])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    prepare(args.resume)


if __name__ == "__main__":
    main()
