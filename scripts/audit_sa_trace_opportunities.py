"""Read frozen traces only; no solver, profiling, policy fitting or new labels."""

from collections import Counter
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments._common import read_json, sha256_file, write_json
from scripts.run_sa_wall_clock import digest

OUT = ROOT / "build/sa-trace-opportunity-audit-v1"
SOURCE = "build/sa-independent-confirmation-supervisor-v2/timed_report.json"
SOURCE_SHA = "926a72001abead832c225f8e80c766298a61bd8a864da5791cec627138ff0fd5"
PLAN_SHA = "2efbfa268fb62b2de7ccfcee7f41a2982ca86bafae31046f81a9ca053e443de7"
ARMS = ("official", "official_sa", "dual16", "dual16_sa")
HORIZONS = (4, 16, 64)
TIMERS = ("native_replan_seconds", "native_repair_bookkeeping_seconds",
          "native_neighborhood_generation_seconds", "native_state_snapshot_seconds",
          "native_residual_seconds", "native_step_seconds", "binding_solver_call_seconds",
          "binding_state_snapshot_seconds", "state_to_python_seconds", "metrics_to_python_seconds",
          "binding_residual_seconds", "binding_total_seconds")


def recovery_window(future, baseline, horizon, terminal_feasible):
    observed = future[:horizon]
    if any(c < baseline for c in observed):
        return "below_pre_increase"
    if len(observed) >= horizon or terminal_feasible:
        return "not_below_in_observed_window"
    return "right_censored"


def episode(row):
    if row["status"] != "ok" or row["arm"] not in ARMS or row["plan_sha256"] != PLAN_SHA:
        raise ValueError("unexpected episode identity")
    if row["integrity_sha256"] != digest({k:v for k,v in row.items() if k != "integrity_sha256"}):
        raise ValueError("episode integrity mismatch")
    events = row["events"]
    counts = [int(row["initial_state"]["num_of_colliding_pairs"])]
    timers = Counter()
    categories = Counter()
    increases, worse_attempts = [], []
    diagnostics = 0
    last_time = row["reset_seconds"]
    for d, e in enumerate(events):
        m = e["metrics"]
        if not m["action_valid"] or not m["step_applied"] or m["conflicts_before"] != counts[-1]:
            raise ValueError("invalid action or discontinuous conflict series")
        if e["elapsed_seconds"] < last_time:
            raise ValueError("time went backwards")
        for k in TIMERS:
            value = m[k]
            if not math.isfinite(value) or value < 0:
                raise ValueError("invalid timer: " + k)
            timers[k] += value
        diagnostics += bool(m["requested_collect_pp_diagnostics"]) or bool(m["pp_agent_diagnostics"])
        before, after = counts[-1], int(m["conflicts_after"])
        delta = m["pp_attempt_conflict_pair_count"] - m["pp_old_conflict_pair_count"]
        accepted = m["replan_success"]
        if accepted:
            if after - before != delta or m["pp_rolled_back"]:
                raise ValueError("accepted delta mismatch")
            categories["increase" if delta > 0 else "decrease" if delta < 0 else "equal"] += 1
        else:
            if after != before or not m["pp_rolled_back"]:
                raise ValueError("rejected attempt did not roll back conflicts")
            categories["rejected_or_incomplete"] += 1
        if row["arm"].endswith("_sa") and m["acceptance_evaluated"]:
            t, u = m["acceptance_temperature"], m["acceptance_uniform"]
            p = 1. if delta <= 0 else math.exp(-delta/t) if t > 0 else 0.
            if not math.isclose(p, m["acceptance_probability"], rel_tol=1e-12, abs_tol=1e-14):
                raise ValueError("acceptance probability mismatch")
            if accepted != (delta <= 0 or u < p):
                raise ValueError("acceptance decision mismatch")
            if delta > 0:
                attempt = dict(decision=d, delta=delta, temperature=t, probability=p, accepted=accepted,
                    before=before, after=after, remaining_seconds_before_try=max(0., 60-last_time),
                    neighborhood_size=len(m["neighborhood"]), relative_increase=delta/max(1,before))
                worse_attempts.append(attempt)
                if accepted:
                    increases.append(attempt)
        counts.append(after)
        last_time = e["elapsed_seconds"]
    feasible = bool(row["final_state"]["feasible"])
    if counts[-1] != row["final_state"]["num_of_colliding_pairs"] or feasible != (counts[-1] == 0):
        raise ValueError("final conflicts mismatch")
    if row["ttf_seconds"] != (last_time if feasible else None):
        raise ValueError("TTF anchor mismatch")
    if row["success_within_budget"] != (feasible and last_time <= 60):
        raise ValueError("budget status mismatch")
    for item in increases:
        d = item["decision"]
        future = counts[d+2:]
        item["windows"] = {str(h):recovery_window(future,item["before"],h,feasible) for h in HORIZONS}
        hit = next((i for i,c in enumerate(future,1) if c < item["before"]), None)
        item["first_below_steps"] = hit
        item["first_below_seconds"] = None if hit is None else events[d+hit]["elapsed_seconds"]-events[d]["elapsed_seconds"]
        item["eventually_below_observed"] = hit is not None
    summary = dict(case_id=row["case_id"], map_id=row["map_id"], arm=row["arm"],
        success=row["success_within_budget"], initial_conflicts=counts[0], final_conflicts=counts[-1],
        ttf=row["ttf_seconds"], steps=len(events), categories=dict(categories),
        accepted_increases=len(increases), worse_attempts=len(worse_attempts), detailed_diagnostic_steps=diagnostics,
        timers=dict(timers), loop_seconds=row["loop_end_seconds"], reset_seconds=row["reset_seconds"],
        selection_seconds=row["selection_seconds"], pp_seconds=row["pp_seconds"])
    if not math.isclose(summary["pp_seconds"],timers["native_replan_seconds"],abs_tol=1e-8):
        raise ValueError("PP summary mismatch")
    return summary, increases, worse_attempts


def summarize(episodes, increases, attempts):
    result = {}
    for arm in ARMS:
        rows = [r for r in episodes if r["arm"]==arm]
        if len(rows)!=32 or len({r["case_id"] for r in rows})!=32:
            raise ValueError("incomplete paired cases")
        a = [r for r in attempts if r["arm"]==arm]
        inc = [r for r in increases if r["arm"]==arm]
        totals = {k:sum(r[k] for r in rows) for k in ("steps","loop_seconds","reset_seconds","selection_seconds","pp_seconds","detailed_diagnostic_steps")}
        totals["other_loop_seconds"] = totals["loop_seconds"]-sum(totals[k] for k in ("reset_seconds","selection_seconds","pp_seconds"))
        result[arm] = dict(episodes=32,successes=sum(r["success"] for r in rows),totals=totals,
            native_timers={k:sum(r["timers"].get(k,0) for r in rows) for k in TIMERS},
            categories=dict(sum((Counter(r["categories"]) for r in rows),Counter())),
            worse_attempts=len(a), accepted_increases=len(inc),
            probability_min=min((r["probability"] for r in a),default=None),
            probability_mean=statistics.mean(r["probability"] for r in a) if a else None,
            windows={str(h):dict(Counter(r["windows"][str(h)] for r in inc)) for h in HORIZONS},
            eventually_below_observed=sum(r["eventually_below_observed"] for r in inc),
            episodes_with_increases=sum(r["accepted_increases"]>0 for r in rows),
            successes_without_increases=sum(r["success"] and not r["accepted_increases"] for r in rows))
    keys = {r["case_id"] for r in episodes if r["arm"]=="dual16"}
    if any({r["case_id"] for r in episodes if r["arm"]==a}!=keys for a in ARMS):
        raise ValueError("arms have different case sets")
    base={r["case_id"]:r for r in episodes if r["arm"]=="dual16"}
    sa={r["case_id"]:r for r in episodes if r["arm"]=="dual16_sa"}
    result["dual16_recoveries"]=[dict(case_id=k,map_id=sa[k]["map_id"],ttf=sa[k]["ttf"],
        steps=sa[k]["steps"],accepted_increases=sa[k]["accepted_increases"])
        for k in sorted(keys) if sa[k]["success"] and not base[k]["success"]]
    return result


def run():
    if (OUT/"report.json").exists():
        raise ValueError("report exists; preserve it")
    if sha256_file(ROOT/SOURCE)!=SOURCE_SHA:
        raise ValueError("frozen manifest changed")
    manifest=read_json(ROOT/SOURCE)
    if not manifest["complete"] or manifest["jobs"]!=128 or len(manifest["files"])!=128:
        raise ValueError("source collection incomplete")
    for name,h in manifest["audits"].items():
        if sha256_file(ROOT/name)!=h:
            raise ValueError("audit attestation changed")
        receipt=read_json(ROOT/name)
        if receipt.get("status")!="ok" or not receipt.get("full_audit"):
            raise ValueError("invalid audit attestation")
    rows,increases,attempts=[],[],[]
    for i,(name,h) in enumerate(sorted(manifest["files"].items())):
        if sha256_file(ROOT/name)!=h:
            raise ValueError("frozen trace changed: "+name)
        row,inc,attempt=episode(read_json(ROOT/name))
        rows.append(row)
        for target,values in ((increases,inc),(attempts,attempt)):
            target.extend(dict(v,arm=row["arm"],case_id=row["case_id"],map_id=row["map_id"]) for v in values)
        if (i+1)%16==0:
            print(f"read-only audit {i+1}/128",flush=True)
    payload=dict(schema="lns2.sa_trace_opportunities.v1",source_manifest=SOURCE,source_sha256=SOURCE_SHA,
        script_sha256=sha256_file(Path(__file__)),complete=True,posthoc_descriptive=True,
        new_solver_runs=0,new_model_fits=0,overlapping_windows=True,no_causal_or_online_gate_claim=True,
        summary=summarize(rows,increases,attempts),episodes=rows,increases=increases,attempts=attempts)
    write_json(OUT/"report.json",payload)
    print(json.dumps(payload["summary"],indent=2))


if __name__=="__main__":
    run()
