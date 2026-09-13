"""Post-hoc scheduling of immutable independent prefixes; never call a solver."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments._common import read_json, sha256_file, write_json

SOURCE = ROOT / "build/sa-independent-lanes-v1"
LANES = ("rank_sa", "uniform_greedy")
HORIZON = 256
FROZEN = {
    "plan.json": "42249ea15aa9135bab4d23835a9ba6a1d2c6369cb2076d298a84ccc9e0f4071a",
    "round1_report.json": "c76dfb668bba0333b2b4d9623bd9c89605a7553cdfc3740b74e0f8ce8460dbfe",
    "round2_report.json": "fcd53bc916fcca8d88411cceb4b451b30c39087b275df28744339dec0cc9d6de",
    "analysis.json": "ad68f9f328f1bbb1df1412348e9a62167e64fb30aec0e4d896822e5a245e8d1c",
}
EVENT_FIELDS = ("action", "selected_id", "rank_best_id", "before_fingerprint",
                "after_fingerprint", "generated_delta", "temperature", "uniform")


def checked(condition, message):
    if not condition:
        raise ValueError(message)


def next_lane(spent, used, mode):
    checked(mode in ("nodes", "alternating"), "unknown scheduling mode")
    if mode == "alternating":
        return sum(used) % 2
    return min(range(2), key=lambda i: (spent[i], used[i], i))


def simulate(lanes, mode, budget=HORIZON):
    checked(type(budget) is int and budget >= 0, "invalid budget")
    checked(len(lanes) == 2, "two lanes required")
    checked(mode in ("nodes", "alternating"), "unknown scheduling mode")
    checked(lanes[0]["initial_fingerprint"] == lanes[1]["initial_fingerprint"], "unpaired roots")
    checked(not any(l["censored"] for l in lanes), "censored source")
    used, spent, schedule = [0, 0], [0, 0], []
    initial = any(l["conflicts"][0] == 0 for l in lanes)
    checked((lanes[0]["conflicts"][0] == 0) == (lanes[1]["conflicts"][0] == 0), "root feasibility mismatch")
    result = dict(feasible=initial, unknown=False, winner=None)
    for _ in range(0 if initial else budget):
        # The selector sees only costs of completed calls, not the next event.
        i = next_lane(spent, used, mode)
        lane = lanes[i]
        if used[i] >= len(lane["transitions"]):
            result["unknown"] = True
            break
        e = lane["transitions"][used[i]]
        cost = e["generated_delta"]
        checked(type(cost) is int and cost >= 0, "invalid node increment")
        schedule.append(dict(lane=LANES[i], local_decision=used[i], spent_before=list(spent), generated=cost))
        used[i] += 1
        spent[i] += cost
        if lane["conflicts"][used[i]] == 0:
            result.update(feasible=True, winner=LANES[i])
            break
    return dict(result, used=used, spent=spent, generated=sum(spent), repair_calls=sum(used), schedule=schedule)


def validate_row(row, case, arm):
    checked(row["status"] == "ok" and not row["censored"], "failed source")
    checked(row["arm"] == arm and row["case_id"] == case["case_id"] and row["map_id"] == case["map_id"], "source identity")
    checked(row["horizon"] == HORIZON and row["plan_sha256"] == FROZEN["plan.json"], "source protocol")
    expected = set(LANES) if arm == "portfolio" else {arm}
    checked(set(row["lanes"]) == expected, "lane identities")
    for name, lane in row["lanes"].items():
        events, conflicts = lane["transitions"], lane["conflicts"]
        checked(not lane["censored"] and len(conflicts) == len(events) + 1, "incomplete lane")
        previous = lane["initial_fingerprint"]
        for d, e in enumerate(events):
            checked(type(e["generated_delta"]) is int and e["generated_delta"] >= 0, "invalid nodes")
            checked(e["before_fingerprint"] == previous, "broken fingerprint chain")
            checked([e["metrics"]["conflicts_before"], e["metrics"]["conflicts_after"]] == conflicts[d:d+2], "conflict chain")
            checked(conflicts[d] > 0, "continued after feasible")
            previous = e["after_fingerprint"]
        checked(lane["generated"] == sum(e["generated_delta"] for e in events), "lane node accounting")
        checked(lane["feasible"] == (conflicts[-1] == 0), "terminal feasibility")
        checked(row["schedule"].count(name) == len(events), "schedule count")
    checked(row["repair_calls"] == len(row["schedule"]) <= HORIZON, "call accounting")
    checked(row["generated"] == sum(l["generated"] for l in row["lanes"].values()), "total node accounting")
    checked(row["feasible"] == any(l["feasible"] for l in row["lanes"].values()), "total feasibility")


def totals(rows, key):
    return {"episodes": len(rows), "success": sum(r[key]["feasible"] for r in rows),
            "generated": sum(r[key]["generated"] for r in rows),
            "repair_calls": sum(r[key]["repair_calls"] for r in rows)}


def summarize(rows):
    summary = {k: totals(rows, k) for k in (*LANES, "alternating", "nodes")}
    proposed, uniform = summary["nodes"], summary["uniform_greedy"]
    gates = dict(no_unknown=not any(r["nodes"]["unknown"] for r in rows),
        no_success_losses=all(r["nodes"]["feasible"] or not any(r[k]["feasible"] for k in LANES) for r in rows),
        calls_reduce_5pct_vs_uniform=proposed["repair_calls"] * 100 <= uniform["repair_calls"] * 95,
        nodes_within_110pct_uniform=proposed["generated"] * 100 <= uniform["generated"] * 110)
    return dict(arms=summary, gates=gates,
        recoveries=sum(r["nodes"]["feasible"] and not r["rank_sa"]["feasible"] for r in rows))


def run(output):
    output = output.resolve()
    checked(output.is_relative_to((ROOT / "build").resolve()) and output != (ROOT / "build").resolve(), "output outside build")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "status.json", {"status": "running", "no_solver": True})
    try:
        for name, sha in FROZEN.items():
            checked(sha256_file(SOURCE / name) == sha, "frozen file changed: " + name)
        plan = read_json(SOURCE / "plan.json")
        checked(len(plan["cases"]) == 16 and len({c["map_id"] for c in plan["cases"]}) == 8, "source coverage")
        identities, rows = {}, []
        for case in plan["cases"]:
            phase, cid = case["phase"], case["case_id"]
            source_report = read_json(SOURCE / (phase + "_report.json"))
            saved = {}
            for arm in (*LANES, "portfolio"):
                name = cid + "-" + arm + ".json"
                path = SOURCE / phase / name
                sha = sha256_file(path)
                checked(sha == source_report["outcome_sha256"][name], "source changed: " + name)
                identities[path.relative_to(ROOT).as_posix()] = sha
                row = read_json(path)
                validate_row(row, case, arm)
                saved[arm] = row
            lanes = [saved[k]["lanes"][k] for k in LANES]
            alternating = simulate(lanes, "alternating")
            for key in ("feasible", "repair_calls", "generated"):
                checked(alternating[key] == saved["portfolio"][key], "alternating reproduction mismatch: " + key)
            for i, name in enumerate(LANES):
                recorded = saved["portfolio"]["lanes"][name]
                checked(recorded["initial_fingerprint"] == lanes[i]["initial_fingerprint"], "portfolio root")
                checked(len(recorded["transitions"]) == alternating["used"][i], "portfolio prefix length")
                for a, b in zip(recorded["transitions"], lanes[i]["transitions"]):
                    checked(all(a[k] == b[k] for k in EVENT_FIELDS), "recorded scientific prefix changed")
            rows.append(dict(case_id=cid, map_id=case["map_id"], phase=phase,
                **{k: {f: saved[k][f] for f in ("feasible", "repair_calls", "generated")} for k in LANES},
                alternating=alternating, nodes=simulate(lanes, "nodes")))
        summary = summarize(rows)
        checked(summary["arms"]["alternating"] == dict(episodes=16, success=16, generated=9213999, repair_calls=337), "old aggregate mismatch")
        phases = {p: summarize([r for r in rows if r["phase"] == p]) for p in ("round1", "round2")}
        eligible = (all(summary["gates"].values()) and summary["recoveries"] > 0
                    and all(all(s["gates"].values()) for s in phases.values()))
        report = dict(schema="lns2.sa_node_allocation_audit.v1", posthoc=True, no_solver=True,
            no_ttf_or_promotion=True, method="least_spent_nodes_then_calls_then_rank_sa",
            horizon=HORIZON, source_files=identities, source_registry=FROZEN,
            implementation={p: sha256_file(ROOT / p) for p in (
                "scripts/audit_sa_node_allocation.py", "docs/SA_NODE_ALLOCATION_PROTOCOL_ZH.md")},
            summary=summary, phases=phases, per_case=rows,
            decision="eligible_for_separate_native_protocol" if eligible else "stop_this_node_balancing_rule")
        write_json(output / "report.json", report)
        write_json(output / "status.json", dict(status="complete", report_sha256=sha256_file(output / "report.json")))
        return dict(summary=summary, decision=report["decision"], report_sha256=sha256_file(output / "report.json"))
    except BaseException as exc:
        write_json(output / "status.json", dict(status="failed", error=type(exc).__name__ + ": " + str(exc)))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(run(args.output))
