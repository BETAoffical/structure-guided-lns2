"""Read-only post-failure audit; preserves the failed preregistered reports and gates."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments._common import sha256_file, write_json
from scripts import diagnose_cplns_observation_divergence as divergence
from scripts import verify_cplns_observation as bridge


def first_bad_state(events, grid, scenario):
    lines = grid.splitlines()
    width = int(lines[2].split()[1])
    expected = {i: (int(v[5]) * width + int(v[4]), int(v[7]) * width + int(v[6]))
                for i, v in enumerate(line.split() for line in scenario.splitlines()[1:])}
    for index, event in enumerate(events):
        if event["event"] == "pp_order" or event.get("conflicts", -1) < 0:
            continue
        actual, cost, _ = bridge.path_metrics(event["agents"], width, lines[4:], expected)
        if actual != event["conflicts"] or cost != event["cost"]:
            return {"event_index": index, "event": event["event"], "restart": event["restart"],
                    "iteration": event["iteration"], "actual_conflicts": actual,
                    "reported_conflicts": event["conflicts"], "actual_soc": cost,
                    "reported_soc": event["cost"], "selected": event["selected"],
                    "one_vertex_paths": [{"id": a["id"], "location": a["path"][0]} for a in event["agents"] if len(a["path"]) == 1]}
    return None


def audit():
    plan = divergence.verify()
    groups, rows, bindings = {}, [], {}
    for job in plan["jobs"]:
        root = divergence.OUT / "jobs"
        file = root / (job["job_id"] + ".json")
        row = json.loads(file.read_text())
        bridge.verify_job(row, job)
        bindings[file.relative_to(ROOT).as_posix()] = sha256_file(file)
        text = (root / (job["job_id"] + ".log")).read_text(encoding="utf-8", errors="replace")
        parsed = bridge.parse_output(text, row["returncode"], row["timed_out"])
        item = {"job_id": job["job_id"], "process_valid": parsed["valid"], "path_check": "not_exported"}
        if job["lane"] == "observer_on":
            events = [json.loads(line) for line in (root / (job["job_id"] + ".events.jsonl")).read_text().splitlines()]
            grid, scen, _ = bridge.fixture(job["fixture"], 917)
            try:
                checked = bridge.validate_events(events, grid, scen)
                if checked["final_conflicts"] != parsed["conflicts"]:
                    raise ValueError("final log/path mismatch")
                item.update(path_check="passed", metrics=checked)
            except ValueError as exc:
                item.update(path_check="failed", error=str(exc), first_bad_state=first_bad_state(events, grid, scen))
        rows.append(item)
        groups.setdefault((job["fixture"], job["rule"]), {})[job["lane"]] = bridge.scientific_log(text)
    comparisons = []
    for (fixture, rule), lanes in groups.items():
        for left, right in (("upstream", "upstream_repeat"), ("upstream", "observer_off"),
                            ("upstream", "observer_on"), ("observer_off", "observer_on")):
            comparisons.append({"fixture": fixture, "rule": rule, "left_lane": left, "right_lane": right,
                                **divergence.first_difference(lanes[left], lanes[right])})
    result = {"schema": "lns2.cplns_observer_failure_audit.v1", "post_hoc_localization": True,
              "decision": "blocked_on_target_parity_and_collision_accounting", "no_ttf_or_promotion": True,
              "real_case_collection_allowed": False, "rows": rows, "comparisons": comparisons,
              "job_record_bindings": bindings, "input_bindings": {
                  Path(__file__).relative_to(ROOT).as_posix(): sha256_file(Path(__file__)),
                  (divergence.OUT / "plan.json").relative_to(ROOT).as_posix(): sha256_file(divergence.OUT / "plan.json")}}
    output = ROOT / "build/cplns-observer-failure-audit-v1/report.json"
    if output.exists() and json.loads(output.read_text()) != result:
        raise ValueError("existing failure audit differs; preserve prior evidence")
    write_json(output, result)
    return {"jobs": len(rows), "process_valid": sum(r["process_valid"] for r in rows),
            "path_checks_passed": sum(r["path_check"] == "passed" for r in rows),
            "path_checks_failed": sum(r["path_check"] == "failed" for r in rows),
            "decision": result["decision"], "report_sha256": sha256_file(output)}


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
