"""Canonicalize only the sealed report's integer-key serialization; retain raw evidence."""
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import probe_sa_hard_official_reference as reference

run = reference.run
RAW_SHA = "49daa0a6c04395c8cae5bdddad6c73ccfd12dd40925c0a1555f196130b59ab6d"
FINAL = "report-canonical-v2.json"
CODE = ("scripts/repair_sa_hard_reference_report.py", "tests/evaluation/test_sa_hard_reference_report.py")


def canonical_report(value):
    restored = copy.deepcopy(value)
    for b in restored["behavior"]:
        sizes = b["sizes"]
        run.require(all(isinstance(k,str) and k == str(int(k)) and int(k) >= 0 for k in sizes), "size key format")
        b["sizes"] = {int(k):v for k,v in sizes.items()}
    # Reconstruct the exact in-memory object originally sealed before JSON converted its keys.
    run.check_seal(restored)
    result = run.sealed({k:v for k,v in value.items() if k != "integrity"})
    run.check_seal(json.loads(json.dumps(result)))
    run.require({k:v for k,v in result.items() if k != "integrity"} ==
                {k:v for k,v in value.items() if k != "integrity"}, "scientific fields changed")
    return result


def repair():
    reg,source,out,_,_ = reference.verify()
    proof = reference.completed(reg,source,out)
    aud = run.check_seal(run.read_json(out/"audit.json"))
    run.require(aud["binding"] == reg["binding"] and len(aud["results"]) == 16 and
                all(r["status"] == "ok" for r in aud["results"]) and
                {r["job_id"]:r["result_sha256"] for r in aud["results"]} == proof["files"], "audited collection")
    run.require(run.sha256_file(out/"report.json") == RAW_SHA, "only the registered raw report may be repaired")
    raw = run.read_json(out/"report.json")
    run.require(raw["binding"] == reg["binding"] and raw["audit_sha256"] == run.sha256_file(out/"audit.json"),
                "raw evidence identity")
    fixed = canonical_report(raw)
    with reference.recovery.strict_lock(out,reg["binding"],"canonical-report-repair"):
        run.once(out/FINAL,fixed)
        run.check_seal(run.read_json(out/FINAL))
        receipt = run.sealed(dict(schema="lns2.sa.report_serialization_repair.v1",binding=reg["binding"],
            raw_sha256=RAW_SHA,canonical_path=FINAL,canonical_sha256=run.sha256_file(out/FINAL),
            scientific_fields_equal=True,no_solver_rerun=True,no_model_change=True,
            source_commit=run.subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
            code={p:run.sha256_file(ROOT/p) for p in CODE},
            reason="Counter integer keys sorted differently after JSON converted them to strings"))
        run.once(out/"report-repair.json",receipt)
    return receipt


if __name__ == "__main__":
    print(json.dumps(repair(),indent=2),flush=True)
