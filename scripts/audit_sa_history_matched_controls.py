"""Post-result, fixed matched controls; not an independent confirmation."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments._common import read_json,sha256_file,json_fingerprint
from experiments.sa_history_information import TEMPORAL,bag_features,ordered_features,metric
from scripts.audit_sa_history_information import verify,receipt_result,save_result,atomic,comparison,fit_fold,progress

EXPECTED="1ffdca0ced13d71ce575cccfa692906afc97854c5041b4e27e2b217ce58cbd69"
PROFILES=["temporal_bag"]+[f"temporal_shuffled_{i}" for i in range(5)]


def matched_features(row,profile):
    if profile=="temporal_bag":
        extra={k:v for k,v in bag_features(row["records"]).items() if k.split(".",1)[1] in TEMPORAL}
    elif profile in PROFILES:
        seed=int(json_fingerprint([row["id"],profile,20260917])[:12],16)
        extra=ordered_features(row["records"],False,seed)
    else:
        raise ValueError("unknown matched profile")
    return row["base"] | extra


def worker(job):
    converted=[r | {"base":matched_features(r,job["profile"])} for r in job["rows"]]
    result=fit_fold(job | {"profile":"aggregate","rows":converted})
    return result | {"profile":job["profile"]}


def main():
    for n in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
        os.environ[n]="1"
    plan,source=verify()
    if sha256_file(source/"report.json")!=EXPECTED:
        raise ValueError("first-round report changed")
    cfg=plan["config"]
    out=ROOT/"build/sa-history-information-matched-v1"
    out.mkdir(parents=True,exist_ok=True)
    identity={"source_binding":plan["binding"],"source_report_sha":EXPECTED,
              "index_sha":sha256_file(source/"index.json"),"profiles":PROFILES,
              "implementation":sha256_file(Path(__file__)),
              "protocol":sha256_file(ROOT/"docs/SA_HISTORY_INFORMATION_MATCHED_CONTROL_ZH.md")}
    binding=json_fingerprint(identity)
    if (out/"plan.json").exists():
        if read_json(out/"plan.json")["identity"]!=identity:
            raise ValueError("matched control identity changed")
    else:
        atomic(out/"plan.json",dict(identity=identity,binding=binding,commit=subprocess.check_output(
            ["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()))
    lock=out/"run.lock"
    with lock.open("x",encoding="utf8") as f:
        f.write(str(os.getpid()))
    try:
        atomic(out/"run_status.json",dict(status="running",binding=binding))
        rows=[r for r in receipt_result(source/"index.json",plan["binding"])["rows"] if r["labels"] is not None]
        folds=[]
        for path in sorted((source/"folds").glob("*.json")):
            if not path.name.endswith(".receipt.json"):
                folds.append(receipt_result(path,plan["binding"]))
        jobs=[dict(rows=rows,map_id=m,profile=p,target=t,model=cfg["model"])
              for m in sorted({r["map_id"] for r in rows}) for p in PROFILES for t in cfg["targets"]]
        with ProcessPoolExecutor(max_workers=cfg["workers"]) as pool:
            pending={}
            completed=0
            for job in jobs:
                path=out/"folds"/(json_fingerprint([job["map_id"],job["profile"],job["target"]])[:20]+".json")
                if path.exists():
                    folds.append(receipt_result(path,binding))
                    completed+=1
                else:
                    pending[pool.submit(worker,job)]=path
            for future in as_completed(pending):
                result=future.result()
                save_result(pending[future],binding,result)
                folds.append(result)
                completed+=1
                progress(out,"matched_fit",complete=completed,total=len(jobs))
        predictions={}
        for f in sorted(folds,key=lambda f:(f["profile"],f["target"],f["map_id"])):
            predictions.setdefault((f["profile"],f["target"]),{}).update({r["id"]:r["prediction"] for r in f["rows"]})
        names=PROFILES+["ordered","dynamic","aggregate"]
        metrics={t:{p:metric(rows,[predictions[p,t][r["id"]] for r in rows],t) for p in names} for t in cfg["targets"]}
        pairs=[("ordered",p) for p in PROFILES+["dynamic"]]+[("temporal_bag","aggregate")]
        contrasts={t:{a+" vs "+b:comparison(rows,predictions[a,t],predictions[b,t],t,cfg["bootstrap"],cfg["seed"])
                      for a,b in pairs} for t in cfg["targets"]}
        checks=[contrasts["sustained_progress"]["ordered vs "+p] for p in PROFILES]
        signal=all(c["delta"]<0 and c["ci95"][1]<0 for c in checks)
        report=dict(binding=binding,metrics=metrics,comparisons=contrasts,stable_matched_order_signal=signal,
                    decision="observational_order_signal_only" if signal else "order_increment_not_established",
                    no_policy_promotion=True,post_result_exploratory=True)
        verify()
        if (out/"report.json").exists():
            if read_json(out/"report.json")!=report:
                raise ValueError("matched result changed")
        else:
            atomic(out/"report.json",report)
        atomic(out/"run_status.json",dict(status="completed",binding=binding))
        print(json.dumps(dict(decision=report["decision"],sha256=sha256_file(out/"report.json"))))
    except BaseException as exc:
        atomic(out/"run_status.json",dict(status="failed",binding=binding,error=repr(exc)))
        raise
    finally:
        lock.unlink()


if __name__=="__main__":
    main()
