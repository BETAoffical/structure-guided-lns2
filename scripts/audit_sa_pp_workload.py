"""Read retained paired traces; quantify PP work, never estimate unmeasured speedups."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments._common import read_json, sha256_file, write_json
from scripts import confirm_sa_native_shadow_ttf as source

OUT = ROOT / "build/sa-pp-workload-audit-v1"
MANIFEST_SHA = "bb3261ee52c3575ae0b847bded4a1a4264efec2cdea2f19acf00941c30ab7fdb"


def count(value):
    if type(value) is not int or value < 0:
        raise ValueError("invalid workload counter")
    return value


def workload(row):
    initial, final = row["initial_state"], row["final_state"]
    if not row["success_within_budget"] or not final["feasible"]:
        raise ValueError("this workload audit requires complete successful traces")
    result = {k: count(final["low_level"][k]) - count(initial["low_level"][k])
              for k in ("runs", "expanded", "generated")}
    if any(v < 0 for v in result.values()):
        raise ValueError("decreasing low-level counter")
    result["pp_attempted"] = sum(count(e["metrics"]["pp_attempted_agent_count"]) for e in row["events"])
    result["pp_inserted"] = sum(count(e["metrics"]["pp_inserted_agent_count"]) for e in row["events"])
    if result["pp_attempted"] != result["runs"]:
        raise ValueError("PP and single-agent search counts disagree")
    if any(e["metrics"]["requested_collect_pp_diagnostics"] for e in row["events"]):
        raise ValueError("detailed diagnostics unexpectedly enabled")
    cells = count(initial["rows"]) * count(initial["cols"])
    if not cells or len(initial["obstacles"]) != cells:
        raise ValueError("invalid static grid")
    result["sit_slot_initializations_from_source"] = cells * result["runs"]
    return result


def main():
    registration = source.verify()
    manifest_path = source.OUT / "timed_report.json"
    if sha256_file(manifest_path) != MANIFEST_SHA:
        raise ValueError("timed manifest changed")
    manifest = read_json(manifest_path)
    for name, h in manifest["files"].items():
        if sha256_file(ROOT / name) != h:
            raise ValueError("retained trace or audit changed")
    cases = {}
    for job in source.schedule(registration):
        row = source.load_validated(job)
        key = (row["case_id"], row["repeat"])
        cases.setdefault(key, {})[row["runtime_variant"]] = workload(row)
    if len(cases) != 16:
        raise ValueError("paired coverage changed")
    rows = []
    for (case, repeat), values in sorted(cases.items()):
        if set(values) != set(source.VARIANTS) or values[source.VARIANTS[0]] != values[source.VARIANTS[1]]:
            raise ValueError("paired search workload differs")
        rows.append(dict(case_id=case, repeat=repeat, **values[source.VARIANTS[1]]))
    fields = list(cases[next(iter(cases))][source.VARIANTS[1]])
    files = ["third_party/mapf_lns2/src/Instance.cpp", "third_party/mapf_lns2/src/SIPP.cpp",
             "third_party/mapf_lns2/inc/ReservationTable.h", "third_party/mapf_lns2/src/ReservationTable.cpp",
             "third_party/mapf_lns2/inc/SingleAgentSolver.h", "third_party/mapf_lns2/src/InitLNS.cpp",
             "build/linux/sa-wall-clock-v1/CMakeFiles/mapf_lns2_core.dir/flags.make"]
    # Source identities accompany counts; no claim that a count is a measured timer.
    inputs = {name: sha256_file(ROOT / name) for name in files}
    flags = (ROOT / files[-1]).read_text(encoding="utf-8")
    if "-O3" not in flags or "-DNDEBUG" not in flags:
        raise ValueError("frozen Release flags changed")
    result = dict(schema="lns2.sa_pp_workload.v1", no_solver_run=True, no_new_timing=True,
                  manifest_sha256=MANIFEST_SHA, inputs=inputs, paired_episodes=16, maps=4,
                  paired_workload_equal=True, cases=rows,
                  totals_one_variant={k: sum(r[k] for r in rows) for k in fields},
                  decision="profile_order_preserving_neighbor_enumeration_before_native_change",
                  unmeasured=["getNeighbors calls and allocator time", "SIT touched cells and constructor time",
                              "interval-result allocation time", "SIPP node allocation count"])
    write_json(OUT / "report.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("cases", "inputs")}, indent=2))


if __name__ == "__main__":
    main()
