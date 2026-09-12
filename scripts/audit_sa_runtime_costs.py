"""Read frozen serial traces; disjoint timing accounting, never a solver run."""

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments._common import read_json, sha256_file, write_json
from scripts.run_sa_wall_clock import digest

OUT = ROOT / "build/sa-runtime-cost-audit-v1"
MANIFEST = "build/sa-single-check-runtime-v1/timed_report.json"
MANIFEST_SHA = "154ca77c8ccd5702afc73f406528ddad5399dce09ee8a067a03ef6caa3d60902"
REGISTRATION = "build/sa-single-check-runtime-v1/registration.json"
REGISTRATION_SHA = "2c2f8d9fa945d55a15cc4cfb193c91c37cee28a49323af569e699951edeacb0c"
VARIANTS = ("reference_double_check", "single_full_check")
PARTS = ("reset", "selection", "pp", "native_other", "solver_wrapper", "binding_export", "python_loop_other")


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("invalid nonnegative timer")
    return value


def difference(total, *parts):
    value = number(total) - sum(number(p) for p in parts)
    if value < -1e-7:
        raise ValueError("nested timers exceed enclosing interval")
    return max(0., value)


def stage(conflicts):
    if type(conflicts) is not int or conflicts < 1:
        raise ValueError("repair must start in nonzero conflict state")
    return "low_1_10" if conflicts <= 10 else "medium_11_100" if conflicts <= 100 else "high_over_100"


def event_parts(event, interval):
    m = event["metrics"]
    pp = number(m["native_replan_seconds"])
    native = number(m["native_step_seconds"])
    solver = number(m["binding_solver_call_seconds"])
    binding = number(m["binding_total_seconds"])
    selection = number(event["selection_seconds"])
    result = dict(selection=selection, pp=pp, native_other=difference(native, pp),
                  solver_wrapper=difference(solver, native), binding_export=difference(binding, solver),
                  python_loop_other=difference(interval, selection, binding))
    for total, children in (
        (native, (pp, m["native_neighborhood_generation_seconds"], m["native_repair_bookkeeping_seconds"],
                  m["native_state_snapshot_seconds"], m["native_residual_seconds"])),
        (binding, (solver, m["binding_state_snapshot_seconds"], m["state_to_python_seconds"],
                   m["metrics_to_python_seconds"], m["binding_residual_seconds"]))):
        if not math.isclose(number(total), sum(number(v) for v in children), rel_tol=1e-6, abs_tol=1e-7):
            raise ValueError("timer children do not reconstruct parent")
    return result


def summarize_episode(row):
    if row["status"] != "ok" or row["arm"] != "dual16_sa" or row["phase"] != "timed":
        raise ValueError("unexpected runtime episode")
    if row["runtime_variant"] not in VARIANTS or row["plan_sha256"] != REGISTRATION_SHA:
        raise ValueError("runtime registration mismatch")
    if row["integrity_sha256"] != digest({k:v for k,v in row.items() if k != "integrity_sha256"}):
        raise ValueError("raw integrity mismatch")
    if row["success_within_budget"] is not True or row["final_state"]["feasible"] is not True:
        raise ValueError("this fixed collection must be entirely successful")
    ttf, reset = number(row["ttf_seconds"]), number(row["reset_seconds"])
    previous = reset
    count = row["initial_state"]["num_of_colliding_pairs"]
    totals = Counter(reset=reset)
    groups = defaultdict(Counter)
    diagnostics = Counter()
    for event in row["events"]:
        m = event["metrics"]
        if not m["action_valid"] or not m["step_applied"] or m["conflicts_before"] != count:
            raise ValueError("invalid or discontinuous repair")
        interval = difference(event["elapsed_seconds"], previous)
        parts = event_parts(event, interval)
        totals.update(parts)
        group = groups[stage(count)]
        group.update(parts)
        group.update(steps=1, interval=interval, attempted_agents=m["pp_attempted_agent_count"],
                     selected_agents=len(m["neighborhood"]))
        for key in ("native_state_snapshot_seconds", "binding_state_snapshot_seconds", "state_to_python_seconds",
                    "metrics_to_python_seconds", "native_neighborhood_generation_seconds", "native_repair_bookkeeping_seconds"):
            diagnostics[key] += number(m[key])
        diagnostics["detailed_pp_steps"] += bool(m["requested_collect_pp_diagnostics"]) or bool(m["pp_agent_diagnostics"])
        diagnostics["rejected_pp_seconds"] += parts["pp"] if m["pp_failure_reason"] == "acceptance_rejected" else 0.
        previous, count = event["elapsed_seconds"], m["conflicts_after"]
    if count != 0 or ttf != previous:
        raise ValueError("TTF must end at first feasible native return")
    if not math.isclose(totals["pp"], row["pp_seconds"], abs_tol=1e-8) or not math.isclose(totals["selection"], row["selection_seconds"], abs_tol=1e-8):
        raise ValueError("episode timer summary mismatch")
    if not math.isclose(sum(totals.values()), ttf, abs_tol=1e-6):
        raise ValueError("disjoint TTF reconstruction mismatch")
    return dict(job_id=row["job_id"], case_id=row["case_id"], map_id=row["map_id"],
                runtime_variant=row["runtime_variant"], repeat=row["repeat"], ttf=ttf,
                steps=len(row["events"]), parts=dict(totals), stages={k:dict(v) for k,v in groups.items()},
                diagnostics=dict(diagnostics), post_ttf_loop_seconds=difference(row["loop_end_seconds"], ttf))


def aggregate(rows):
    totals = sum((Counter(r["parts"]) for r in rows), Counter())
    ttf = sum(r["ttf"] for r in rows)
    groups = defaultdict(Counter)
    for row in rows:
        for key, value in row["stages"].items():
            groups[key].update(value)
    return dict(episodes=len(rows), ttf_seconds=ttf, steps=sum(r["steps"] for r in rows),
                parts_seconds=dict(totals), parts_percent={k:100*totals[k]/ttf for k in PARTS},
                stages={k:dict(v, percent_of_ttf=100*v["interval"]/ttf) for k,v in groups.items()},
                diagnostics=dict(sum((Counter(r["diagnostics"]) for r in rows), Counter())),
                counterfactual_cost_bounds={k:dict(
                    eliminating_all_fraction_percent=100*totals[k]/ttf,
                    halving_cost_ttf_reduction_percent=50*totals[k]/ttf) for k in ("pp", "selection", "binding_export", "native_other")})


def run():
    if sha256_file(ROOT/MANIFEST) != MANIFEST_SHA or sha256_file(ROOT/REGISTRATION) != REGISTRATION_SHA:
        raise ValueError("frozen source identity changed")
    manifest = read_json(ROOT/MANIFEST)
    if manifest["complete"] is not True or manifest["jobs"] != 32 or manifest["registration_sha256"] != REGISTRATION_SHA:
        raise ValueError("incomplete source")
    raw, receipts = {}, {}
    for name, h in manifest["files"].items():
        if sha256_file(ROOT/name) != h:
            raise ValueError("source SHA mismatch: " + name)
        row = read_json(ROOT/name)
        target = raw if "/raw/" in name else receipts if "/audits/" in name else None
        if target is None or row["job_id"] in target:
            raise ValueError("unexpected or duplicate source")
        target[row["job_id"]] = (row, h)
    if set(raw) != set(receipts) or len(raw) != 32:
        raise ValueError("missing paired audit")
    rows = []
    for key, (row, h) in sorted(raw.items()):
        receipt = receipts[key][0]
        if receipt["status"] != "ok" or receipt["full_audit"] is not True or receipt["raw_sha256"] != h or receipt["registration_sha256"] != REGISTRATION_SHA or receipt["events"] != len(row["events"]):
            raise ValueError("unbound audit receipt")
        rows.append(summarize_episode(row))
    pairs = defaultdict(set)
    for r in rows:
        pair = pairs[r["case_id"], r["repeat"]]
        if r["runtime_variant"] in pair:
            raise ValueError("duplicate runtime pair")
        pair.add(r["runtime_variant"])
    if len(pairs) != 16 or any(v != set(VARIANTS) for v in pairs.values()):
        raise ValueError("incomplete runtime pairs")
    result = dict(schema="lns2.sa_runtime_cost_audit.v1", posthoc_descriptive=True,
        new_solver_runs=0, no_speedup_claim=True, no_default_change=True,
        source_manifest=MANIFEST, source_sha256=MANIFEST_SHA, registration_sha256=REGISTRATION_SHA,
        script_sha256=sha256_file(Path(__file__)),
        summary={v:aggregate([r for r in rows if r["runtime_variant"] == v]) for v in VARIANTS},
        cases={c:{v:aggregate([r for r in rows if r["case_id"] == c and r["runtime_variant"] == v])
                  for v in VARIANTS} for c in sorted({r["case_id"] for r in rows})}, episodes=rows,
        absent_measurements=["selection proposal/feature/inference sub-timers", "PP findPath/path-table/conflict-scan/rollback sub-timers"],
        interpretation="cost bounds assume identical actions, trajectories and all other costs; not predicted or measured speedups")
    write_json(OUT/"report.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(run()["summary"], indent=2))
