"""Read-only reaggregation of the frozen SA factorial; no solver or model fitting."""

from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments._common import read_json, sha256_file, write_json
from scripts import diagnose_sa_selector_rounds as rounds


def gates(reports):
    result = {}
    for arm in rounds.ARMS[1:]:
        coverage, conditions = set(), []
        for report in reports:
            c = report["comparisons"][arm + "_vs_standard"]
            coverage.update(k.split("-native-")[0] for k in c["gains"])
            conditions.append(bool(c["gains"]) and not c["losses"] and not any(a["censored"] for a in report["arms"].values()))
        ratio = sum(r["arms"][arm]["generated"] for r in reports) / max(1, sum(r["arms"]["standard"]["generated"] for r in reports))
        result[arm] = {"each_round_success_gate": conditions, "gain_source_tasks": sorted(coverage),
                       "generated_ratio": ratio, "passed": all(conditions) and len(coverage) >= 2 and ratio <= 1.25}
    return result


def analyze():
    plan = rounds.verify()
    reports, rows = [], []
    for phase in ("round2", "round3"):
        supplied = read_json(rounds.OUT / (phase + "_report.json"))
        cases = [read_json(rounds.OUT / "admission" / (c["case_id"] + ".json"))["case"] for c in plan["cases"]]
        schedule = rounds.jobs(plan, cases, phase)
        observed = []
        if set(supplied["outcome_sha256"]) != {j["job_id"] for j in schedule}:
            raise ValueError("phase schedule incomplete")
        for job in schedule:
            path = rounds.OUT / phase / (job["job_id"] + ".json")
            if sha256_file(path) != supplied["outcome_sha256"][job["job_id"]]:
                raise ValueError("outcome changed")
            observed.append(rounds.load_result(path, job))
        recomputed = rounds.summarize(observed)
        if any(supplied[k] != value for k, value in recomputed.items()):
            raise ValueError("phase summary mismatch")
        reports.append(recomputed)
        rows.extend(observed)
    summary = rounds.summarize(rows)
    summary.update(schema="lns2.sa_selector_rounds.analysis.v1", preregistered_gates=gates(reports),
                   distinct_development_maps=5, initialization_states=20, episodes=len(rows),
                   analysis_source_sha256=sha256_file(Path(__file__)),
                   source_report_sha256={p.name: sha256_file(p) for p in rounds.OUT.glob("round*_report.json")})
    checked = {}
    for archive in rounds.OUT.glob("*backup.zip"):
        with zipfile.ZipFile(archive) as z:
            count = 0
            for name in z.namelist():
                if name.endswith("/"):
                    continue
                path = rounds.OUT / name
                if not path.is_file() or z.read(name) != path.read_bytes():
                    raise ValueError("backup mismatch: " + name)
                count += 1
        checked[archive.name] = {"files_verified": count, "sha256": sha256_file(archive)}
    summary["backups"] = checked
    write_json(rounds.OUT / "analysis.json", summary)
    return {"arms": summary["arms"], "gates": summary["preregistered_gates"], "backups": checked}


if __name__ == "__main__":
    import json
    print(json.dumps(analyze(), indent=2))
