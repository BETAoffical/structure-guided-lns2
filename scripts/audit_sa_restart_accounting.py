"""Read-only fixed two-seed budget accounting, not a restart experiment."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments._common import (atomic_write_text, read_json,
                                 sha256_file, write_json, write_jsonl)


def contained(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("expected contained relative path")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("path escaped root")
    return path


def checked(root, relative, digest):
    path = contained(root, relative)
    if sha256_file(path) != digest:
        raise ValueError(f"source SHA mismatch: {relative}")
    return path


def validate_row(row, budget):
    if row["protocol"] != "first_feasible" or row["budget_seconds"] != budget:
        raise ValueError("incompatible clock or budget")
    if row["timed_workers"] != 1 or row["repair_iteration_cap"] is not None:
        raise ValueError("incompatible worker or iteration budget")
    if type(row["success"]) is not bool or row["success"] != row["found_within_budget"]:
        raise ValueError("inconsistent deadline success")
    expected = "completed" if row["success"] else "no_feasible_solution"
    if row["status"] != expected:
        raise ValueError("unknown/error result is not ordinary failure")
    ttf = row["ttf_seconds"]
    if row["success"]:
        if type(ttf) not in (float, int) or not math.isfinite(ttf) or not 0 <= ttf <= budget:
            raise ValueError("invalid successful TTF")
        if row.get("observed_first_feasible_seconds", ttf) != ttf:
            raise ValueError("clock mismatch")
    elif ttf is not None:
        raise ValueError("failed raw TTF must be null")


def load_sources(config, root=ROOT):
    rows, receipts, identities, case_identity = [], [], {}, {}
    expected_controllers = set()
    for source in config["sources"]:
        base = contained(root, source["root"])
        for name, digest in source["pins"].items():
            checked(base, name, digest)
        report = read_json(base / "analysis/report.json")
        registration = read_json(base / "registration.json")
        manifest = read_json(base / "manifest.json")
        if manifest["binding"] != registration["fingerprint"]:
            raise ValueError("manifest registration mismatch")
        if registration["native_sha256"] != config["native_sha256"]:
            raise ValueError("native identity mismatch")
        if set(registration["controllers"]) != set(source["controllers"]):
            raise ValueError("unexpected registered controller")
        if expected_controllers.intersection(source["controllers"]):
            raise ValueError("duplicate controller source")
        expected_controllers.update(source["controllers"])
        cases = registration["cases"]
        if len({case["task_id"] for case in cases}) != config["expected_tasks"]:
            raise ValueError("case count or identity mismatch")
        for case in cases:
            pins = {}
            for name, relative in case["files"].items():
                digest = registration["inputs"][relative]
                checked(root, relative, digest)
                pins[name] = digest
            identity = dict(map_id=case["map_id"], files=pins,
                            static_audit=case["static_audit"],
                            pressure_design=case["pressure_design"])
            task = case["task_id"]
            if task in case_identity and case_identity[task] != identity:
                raise ValueError("cross-batch task identity mismatch")
            case_identity[task] = identity
        schedule = {item["job_id"]: item for item in registration["schedule"]}
        selected = [row for row in report["episodes"] if row["controller"] in source["controllers"]]
        if len(selected) != len(schedule) or set(manifest["jobs"]) != set(schedule):
            raise ValueError("incomplete source schedule")
        if {r["job_id"] for r in selected} != set(schedule):
            raise ValueError("missing or repeated report job")
        for row in selected:
            validate_row(row, config["budget_seconds"])
            job = row["job_id"]
            if any(row.get(key) != value for key, value in schedule[job].items()):
                raise ValueError("report schedule mismatch")
            entry = manifest["jobs"][job]
            if entry["success"] != row["success"] or entry["status"] != row["status"]:
                raise ValueError("manifest outcome mismatch")
            directory = contained(base, f"episodes/{job}")
            objects = {}
            for name in ("binding.json", "result.json", "initial.json", "supervisor.json"):
                objects[name] = read_json(checked(directory, name, entry["files"][name]))
            binding = objects["binding.json"]
            if binding["item"] != schedule[job] or binding["native_sha256"] != config["native_sha256"]:
                raise ValueError("episode binding mismatch")
            if any(obj["binding"] != binding["binding"] for obj in objects.values()):
                raise ValueError("artifact binding mismatch")
            result = objects["result.json"]["payload"]
            if result["status"] != row["status"] or result["success_by_deadline"] != row["success"]:
                raise ValueError("raw outcome differs from summary")
            if result.get("first_feasible_elapsed_seconds") != row["ttf_seconds"]:
                raise ValueError("raw TTF differs from summary")
            supervisor = objects["supervisor.json"]["payload"]
            if supervisor.get("error") or supervisor["status"] != row["status"]:
                raise ValueError("supervisor error")
            fingerprint = objects["initial.json"]["payload"]["state_fingerprint"]
            if row["initial_fingerprint"] != fingerprint:
                raise ValueError("initial summary mismatch")
            key = (row["task_id"], row["solver_seed"])
            if key in identities and identities[key] != fingerprint:
                raise ValueError("cross-controller initial mismatch")
            identities[key] = fingerprint
            if row["map_id"] != case_identity[row["task_id"]]["map_id"]:
                raise ValueError("task map mismatch")
            rows.append(row)
        receipts.append(dict(root=source["root"], pins=source["pins"],
                             episodes=len(selected), summary_files_verified=4 * len(selected)))
    return rows, receipts


def account(first, second, cutoff, budget):
    """Conditional prefix replay: never select an order using its outcome."""
    if not 0 < cutoff < budget:
        raise ValueError("invalid cutoff")
    for row in (first, second):
        validate_row(row, budget)
    if first["task_id"] != second["task_id"] or first["map_id"] != second["map_id"]:
        raise ValueError("restart must keep task unchanged")
    if first["controller"] != second["controller"] or first["solver_seed"] == second["solver_seed"]:
        raise ValueError("restart must change seed, not controller")
    t1, t2 = first["ttf_seconds"], second["ttf_seconds"]
    switches = t1 is None or t1 > cutoff
    if not switches:
        simulated = t1
    elif t2 is not None and t2 <= budget - cutoff:
        simulated = cutoff + t2
    else:
        simulated = None
    before, after = t1 is not None, simulated is not None
    return dict(task_id=first["task_id"], map_id=first["map_id"], controller=first["controller"],
                order=[first["solver_seed"], second["solver_seed"]],
                initial_fingerprints=[first["initial_fingerprint"], second["initial_fingerprint"]],
                source_jobs=[first["job_id"], second["job_id"]],
                baseline_success=before, restart_success=after,
                baseline_ttf=t1, restart_ttf=simulated,
                baseline_capped=t1 if before else budget,
                restart_capped=simulated if after else budget,
                switched=switches, gained=after and not before, lost=before and not after,
                sacrificed_late_first_success=t1 is not None and t1 > cutoff,
                unused_second_late_success=t2 is not None and t2 > budget - cutoff)


def summarize(rows):
    common = [r for r in rows if r["baseline_success"] and r["restart_success"]]
    return dict(ordered_cases=len(rows), distinct_tasks=len({r["task_id"] for r in rows}),
                baseline_success=sum(r["baseline_success"] for r in rows),
                restart_success=sum(r["restart_success"] for r in rows),
                gained=sum(r["gained"] for r in rows), lost=sum(r["lost"] for r in rows),
                switched=sum(r["switched"] for r in rows),
                sacrificed_late_first_success=sum(r["sacrificed_late_first_success"] for r in rows),
                baseline_capped_mean=statistics.fmean(r["baseline_capped"] for r in rows),
                restart_capped_mean=statistics.fmean(r["restart_capped"] for r in rows),
                common_success=len(common),
                common_baseline_ttf=statistics.fmean(r["baseline_ttf"] for r in common) if common else None,
                common_restart_ttf=statistics.fmean(r["restart_ttf"] for r in common) if common else None)


def map_bootstrap(rows, replicates, seed):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["map_id"]].append(row)
    # Both seed orders stay inside each map; they are not independent samples.
    totals = []
    for key in sorted(grouped):
        group = grouped[key]
        totals.append((sum(int(r["restart_success"]) - int(r["baseline_success"]) for r in group),
                       sum(r["baseline_capped"] - r["restart_capped"] for r in group), len(group)))
    rng = random.Random(seed)
    success, seconds = [], []
    for _ in range(replicates):
        sampled = [rng.choice(totals) for _ in totals]
        count = sum(x[2] for x in sampled)
        success.append(100 * sum(x[0] for x in sampled) / count)
        seconds.append(sum(x[1] for x in sampled) / count)
    def interval(values):
        values.sort()
        return [values[int(.025 * (len(values) - 1))], values[int(.975 * (len(values) - 1))]]
    return dict(success_gain_pp=interval(success), capped_seconds_saved=interval(seconds),
                maps=len(totals), replicates=replicates, role="exploratory_not_confirmation")


def analyze(rows, config):
    controllers = sorted(c for s in config["sources"] for c in s["controllers"])
    lookup = {(r["controller"], r["task_id"], r["solver_seed"]): r for r in rows}
    if len(lookup) != len(rows):
        raise ValueError("duplicate episode")
    tasks = sorted({r["task_id"] for r in rows})
    seeds = set(config["seed_orders"][0])
    if config["seed_orders"] != [[61, 62], [62, 61]] or config["cutoff_seconds"] != 60 or config["budget_seconds"] != 120:
        raise ValueError("unregistered schedule; no cutoff search")
    expected = {(c, t, s) for c in controllers for t in tasks for s in seeds}
    if set(lookup) != expected or len(tasks) != config["expected_tasks"]:
        raise ValueError("incomplete task/seed/controller Cartesian product")
    if len({r["map_id"] for r in rows}) != config["expected_maps"]:
        raise ValueError("map count mismatch")
    comparisons, summaries = [], {}
    for controller in controllers:
        values = [account(lookup[controller, task, order[0]], lookup[controller, task, order[1]],
                          config["cutoff_seconds"], config["budget_seconds"])
                  for task in tasks for order in config["seed_orders"]]
        combined = summarize(values)
        combined["by_order"] = {f"{order[0]}-{order[1]}": summarize([r for r in values if r["order"] == order])
                                for order in config["seed_orders"]}
        combined["by_map"] = {m: summarize([r for r in values if r["map_id"] == m])
                              for m in sorted({r["map_id"] for r in values})}
        combined["bootstrap"] = map_bootstrap(values, config["bootstrap_replicates"], config["bootstrap_seed"])
        opportunity = combined["gained"] > combined["lost"] and combined["restart_capped_mean"] < combined["baseline_capped_mean"]
        combined["decision"] = "runtime_equivalence_check_candidate_only" if opportunity else "stop_fixed_schedule_no_joint_net_opportunity"
        summaries[controller] = combined
        comparisons.extend(values)
    return comparisons, summaries


def markdown(report):
    lines = ["# Fixed 60+60 Restart: Conditional Log Accounting", "",
             "NOT actual restart timing, an independent confirmation, or a physical upper bound.",
             "96 ordered cases per controller = 48 tasks x two fixed, dependent orders; eight maps.",
             "TTF includes reset, but excludes cold startup and final delivery. Extra switch overhead assumed zero.",
             "Official+SA is a different source batch. No causal cross-controller speed ranking.", "",
             "|Controller|Continuous success|Accounted success|Gain/loss|Capped seconds before/after|Decision|",
             "|---|---:|---:|---:|---:|---|"]
    for name, row in report["controllers"].items():
        lines.append(f"|{name}|{row['baseline_success']}/96|{row['restart_success']}/96|{row['gained']}/{row['lost']}|"
                     f"{row['baseline_capped_mean']:.6f}/{row['restart_capped_mean']:.6f}|{row['decision']}|")
    lines += ["", "The JSON contains per-order, per-map, common-success and map-bootstrap details.",
              "Failed raw TTF is null; capped time is a budget statistic, not measured failed TTF.",
              "No solver/native calls, model training, cutoff tuning or controller changes.",
              "Shorter native deadlines, interruption semantics and switching costs remain unvalidated.", ""]
    return "\n".join(lines)


def run(config_path, root=ROOT):
    config = read_json(config_path)
    rows, receipts = load_sources(config, root)
    comparisons, summaries = analyze(rows, config)
    report = dict(schema="lns2.sa_restart_accounting_result.v1", promotion_allowed=False,
                  evidence_role=config["evidence_role"], config_sha256=sha256_file(config_path),
                  implementation_sha256=sha256_file(Path(__file__)), source_receipts=receipts,
                  controllers=summaries, episode_count=len(rows), solver_calls=0,
                  verification_scope="pinned_reports_registrations_manifests_inputs_and_episode_summaries_not_full_traces")
    output = contained(root, config["output"])
    report_path = output / "report.json"
    if report_path.exists():
        old = read_json(report_path)
        pair_sha = old.pop("paired_cases_sha256")
        if old != report:
            raise ValueError("refusing to replace a different accounting result")
        checked(output, "paired_cases.jsonl", pair_sha)
    write_jsonl(output / "paired_cases.jsonl", comparisons)
    report["paired_cases_sha256"] = sha256_file(output / "paired_cases.jsonl")
    # All scientific values are deterministic; no elapsed analysis time enters results.
    write_json(report_path, report)
    atomic_write_text(output / "REPORT.md", markdown(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/sa_restart_accounting.json")
    args = parser.parse_args()
    result = run(args.config)
    print(json.dumps(dict(episodes=result["episode_count"], solver_calls=0,
                         report="build/sa-restart-accounting-v1/report.json",
                         controllers={c: {k: v for k, v in summary.items() if k not in {"by_order", "by_map"}}
                                      for c, summary in result["controllers"].items()}), indent=2))


if __name__ == "__main__":
    main()
