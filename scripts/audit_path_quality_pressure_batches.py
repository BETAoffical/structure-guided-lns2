"""Audit two prepared pressure batches without invoking a solver or collector."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments._common import json_fingerprint, read_json, sha256_file, write_json
from experiments.repair_collection import state_fingerprint
from lns2_selector.evaluation import path_quality_cohort as cohort
from lns2_selector.evaluation.path_quality_preflight import checked_input, contained
from lns2_selector.evaluation.pressure_evaluation import evaluation_path


DESIGNS = ("configs/path_quality_pressure_evaluation_v1.json",
           "configs/path_quality_pressure_evaluation_replica2.json")


def check_rosters(reports):
    if len(reports) != 2:
        raise ValueError("exactly two pressure batches required")
    cells, tasks, seeds, jobs = [], set(), set(), set()
    for report in reports:
        cases, schedule = report["cases"], report["execution_schedule"]
        if len(cases) != 48 or len(schedule) != 864:
            raise ValueError("incomplete batch roster")
        batch_cells = set()
        for case in cases:
            design = case["pressure_design"]
            cell = (case["map_id"], design["density"], design["mode"]["name"])
            if (cell in batch_cells or case["task_id"] in tasks
                    or design["task_seed"] in seeds or case["solver_seeds"] != [61, 62]):
                raise ValueError("duplicate task, seed or design cell")
            batch_cells.add(cell)
            tasks.add(case["task_id"])
            seeds.add(design["task_seed"])
        if len({cell[0] for cell in batch_cells}) != 8:
            raise ValueError("expected eight maps per batch")
        counts = Counter()
        for item in schedule:
            if item["job_id"] in jobs or item["execution_authorized"] or item["repair_iteration_cap"] is not None:
                raise ValueError("duplicate or authorized/capped job")
            jobs.add(item["job_id"])
            counts[(item["protocol"], item["budget_seconds"], item["controller"])] += 1
        expected = {(p, b, c): 96 for p, b in (
            ("first_feasible", 120), ("fixed_budget", 60), ("fixed_budget", 120))
            for c in ("official_adaptive", "v2-full", "dual16")}
        if counts != expected:
            raise ValueError("unbalanced protocol roster")
        cells.append(batch_cells)
    if cells[0] != cells[1]:
        raise ValueError("replicated map/density/OD cells differ")
    return {"independent_maps": 8, "tasks": len(tasks), "episodes": len(jobs)}


def audit(root):
    reports, batches, reset_rows, hashes, reference = [], [], [], {}, None
    od_signatures, map_hashes = set(), {}
    for relative in DESIGNS:
        design_path = root / relative
        design = read_json(design_path)
        config_path = evaluation_path(root, design_path)
        config, output, registry, report = cohort.verified_registration(root, config_path)
        anchors = cohort.load_admission(output, registry)
        expected_keys = {cohort.admission_key(item) for item in report["execution_schedule"]}
        if set(anchors) != expected_keys or len(anchors) != 192:
            raise ValueError("budget admission roster mismatch")
        identity = (config["native_sha256"], report["runtime_template"])
        if reference is not None and identity != reference:
            raise ValueError("batches use different native or runtime/model template")
        reference = identity
        if (output / "timing_authorization.json").exists() or (output / "episodes").exists():
            raise ValueError("this is a pre-timing audit; timing artifacts already exist")
        reset_path = checked_input(root, design["reset_report"])
        reset_report = read_json(reset_path)
        source = {}
        for row in reset_report["rows"]:
            path = contained(root, row["reset_file"])
            if sha256_file(path) != row["sha256"]:
                raise ValueError("source reset SHA mismatch")
            saved = read_json(path)
            if (saved["status"] != "ok" or saved["repairs_executed"] != 0
                    or state_fingerprint(saved["observation"]) != saved["state_fingerprint"]):
                raise ValueError("invalid source reset")
            key = (row["case"]["task_id"], row["solver_seed"])
            if key in source:
                raise ValueError("duplicate source reset")
            source[key] = saved["state_fingerprint"]
            reset_rows.append(row)
        required_source = {(c["task_id"], s) for c in report["cases"] for s in (61, 62)}
        if set(source) != required_source:
            raise ValueError("source reset roster mismatch")
        for anchor in anchors.values():
            if anchor["state_fingerprint"] != source[(anchor["task_id"], anchor["solver_seed"])]:
                raise ValueError("budget-specific initial fingerprint mismatch")
        for case in report["cases"]:
            task = read_json(contained(root, case["files"]["task_file"]))
            signature = json_fingerprint({"map_id": case["map_id"], "starts": task["starts"], "goals": task["goals"]})
            if signature in od_signatures:
                raise ValueError("duplicate actual OD task")
            od_signatures.add(signature)
            digest = sha256_file(contained(root, case["files"]["map_file"]))
            if map_hashes.setdefault(case["map_id"], digest) != digest:
                raise ValueError("same map ID has different grid files")
        for path in (design_path, reset_path, output / "registration.json", output / "admission.json",
                     output / "admission_manifest.jsonl", output / "execution_schedule.jsonl"):
            hashes[path.relative_to(root).as_posix()] = sha256_file(path)
        batches.append({"design": relative, "output": output.relative_to(root).as_posix(),
                        "registration": registry["fingerprint"], "admissions": len(anchors),
                        "episodes": len(report["execution_schedule"])})
        reports.append(report)
    counts = check_rosters(reports)
    strata = []
    for density in (0.15, 0.20, 0.25):
        rows = [r for r in reset_rows if r["case"]["density"] == density]
        values = [r["quality"]["colliding_pairs"] for r in rows]
        strata.append({"density": density, "conditions": len(rows), "min_conflicts": min(values),
                       "mean_conflicts": mean(values), "max_conflicts": max(values),
                       "zero_conflict": values.count(0)})
    return {"schema": "lns2.pressure_batches_admission.v1", "status": "ready_pending_separate_timing_authorization",
            **counts, "task_seed_conditions": len(reset_rows), "budget_admissions": sum(b["admissions"] for b in batches),
            "fingerprint_mismatches": 0, "timed_episodes_started": 0, "timing_authorized": False,
            "batches": batches, "conflict_strata": strata, "input_sha256": hashes,
            "implementation_sha256": sha256_file(Path(__file__))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="build/path-quality-pressure-batches-v1/admission_audit.json")
    args = parser.parse_args()
    output = contained(ROOT, args.output)
    if not output.is_relative_to((ROOT / "build").resolve()):
        parser.error("audit output must be inside build")
    report = audit(ROOT)
    write_json(output, report)
    print(json.dumps({k: v for k, v in report.items() if k not in {"input_sha256", "batches"}}, indent=2))


if __name__ == "__main__":
    main()
