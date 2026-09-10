"""Freeze a report-oriented confirmation plan; never generate or solve tasks."""
import argparse
import copy
import itertools
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments._common import read_json, sha256_file, write_json, write_jsonl

CONFIG = "configs/warehouse_repair_confirmation_v1.json"


def dataset_config(config, base):
    result = copy.deepcopy(base)
    settings = config["dataset"]
    template = copy.deepcopy(next(v for v in base["task_variants"] if v["name"] == "balanced_od_d15"))
    result.update(master_seed=settings["master_seed"], map_id_prefix="repair_confirm_v1",
                  output_dir=settings["output"], tasks_per_map=settings["tasks_per_map"],
                  splits={settings["split"]: {"layout_counts": {"station_centric": settings["map_count"]}}})
    result["task_variants"] = []
    for index in range(settings["tasks_per_map"]):
        variant = copy.deepcopy(template)
        variant["name"] = f"balanced_od_d15_{index}"
        variant["task"]["agent_density"] = settings["agent_density"]
        result["task_variants"].append(variant)
    return result


def schedule_slots(config):
    orders = list(itertools.permutations(config["controllers"]))
    rows = []
    generation = config["checkpoint"]
    for map_index in range(config["dataset"]["map_count"]):
        for task in range(config["dataset"]["tasks_per_map"]):
            task_index = map_index * config["dataset"]["tasks_per_map"] + task
            for replica in range(generation["replicas_per_task"]):
                key = task_index * generation["replicas_per_task"] + replica
                for controller in orders[key % len(orders)]:
                    rows.append(dict(slot=len(rows), checkpoint_slot=key, map_slot=map_index,
                                     task_slot=task_index, replica=replica, controller=controller,
                                     incumbent_seed=generation["incumbent_solver_seed_base"] + task_index,
                                     selection_seed=generation["selection_seed_base"] + key,
                                     solver_seed=generation["screen_solver_seed_base"] + key,
                                     bound_to_checkpoint=False))
    return rows


def validate(config):
    if config["schema"] != "lns2.warehouse_repair_confirmation_preparation.v1":
        raise ValueError("unexpected preparation schema")
    if config["timed_execution_authorized"] is not False:
        raise ValueError("this preparation tool cannot authorize timing")
    if config["controllers"] != ["official_adaptive", "v2-full", "dual16"]:
        raise ValueError("frozen controller set changed")
    if config["runtime"]["timed_workers"] != 1 or config["runtime"]["max_repair_iterations"] != 0:
        raise ValueError("serial, uncapped repair protocol changed")
    d, c = config["dataset"], config["checkpoint"]
    if (d["map_count"], d["tasks_per_map"], d["agent_density"], c["replicas_per_task"]) != (8, 2, 0.15, 2):
        raise ValueError("fixed cohort product changed")
    if c["replace_failed_candidates"] is not False:
        raise ValueError("candidate replacement is forbidden")
    if config["no_model_selection_from_confirmation"] is not True:
        raise ValueError("confirmation cannot select models")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT / "build"):
        raise ValueError("output must be inside build")
    config = read_json(ROOT / CONFIG)
    validate(config)
    frozen = config["frozen"]
    for name, expected in [(config["base_dataset_config"], config["base_dataset_sha256"]),
                           (frozen["native_path"], frozen["native_sha256"]),
                           (frozen["model_manifest"], frozen["model_manifest_sha256"])]:
        if sha256_file(ROOT / name) != expected:
            raise ValueError("frozen input changed: " + name)
    expanded = dataset_config(config, read_json(ROOT / config["base_dataset_config"]))
    # Snapshot code, not results. Future execution must verify checkpoint and
    # dataset identities separately before this unbound schedule becomes usable.
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    names = {name for name in tracked if Path(name).suffix in {".py", ".cpp", ".h", ".hpp"}}
    names.update([CONFIG, config["base_dataset_config"], frozen["native_path"],
                  "scripts/prepare_warehouse_repair_confirmation.py"])
    names.update(p.relative_to(ROOT).as_posix() for p in (ROOT / Path(frozen["model_manifest"]).parent).glob("*.json"))
    names = sorted(names)
    with ThreadPoolExecutor(max_workers=20) as pool:
        hashes = dict(zip(names, pool.map(lambda name: sha256_file(ROOT / name), names)))
    runtime_roots = ["experiments", "lns2_selector", "generators", "src", "include", "third_party"]
    subprocess.run(["git", "diff", "--exit-code", frozen["runtime_commit"], "--", *runtime_roots], cwd=ROOT, check=True)
    untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "--", *runtime_roots], cwd=ROOT)
    if untracked.strip():
        raise ValueError("unregistered runtime source files exist")
    rows = schedule_slots(config)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "expanded_dataset_config.json", expanded)
    write_jsonl(output / "unbound_schedule.jsonl", rows)
    report = dict(schema="lns2.warehouse_repair_confirmation_preflight.v1",
                  preparation_status="protocol_frozen_not_ready_for_timing",
                  native_invoked=False, maps_generated=0, solver_calls=0,
                  map_count=8, task_count=16, checkpoint_candidate_count=32,
                  maximum_timed_episodes=len(rows), input_sha256=hashes,
                  expanded_dataset_sha256=sha256_file(output / "expanded_dataset_config.json"),
                  schedule_sha256=sha256_file(output / "unbound_schedule.jsonl"),
                  remaining=["historical_seed_and_map_inventory", "generate_and_verify_new_dataset",
                             "construct_and_validate_checkpoints", "three_controller_runner_and_clock_tests",
                             "register_final_checkpoint_bound_schedule", "separate_timing_authorization"])
    write_json(output / "preflight.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "input_sha256"}, indent=2))


if __name__ == "__main__":
    main()
