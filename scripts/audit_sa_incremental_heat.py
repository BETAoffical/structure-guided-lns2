"""Paired selection-component test on frozen trajectories, never free rollout."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import time
from types import FunctionType, MethodType

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments._common import read_json, sha256_file, write_json
from experiments.sa_incremental_heat_runtime import IncrementalHeatPool
from experiments.sa_single_check_runtime import SingleFullCheckPool
from scripts import confirm_sa_single_check_runtime as prior

pilot=prior.pilot
OUT=ROOT/"build/sa-incremental-heat-audit-v1"


def capture_features(pool):
    original=pool.select.__func__
    bindings=dict(original.__globals__)
    scorer=bindings["score_online_candidates"]
    def capture(rows,model):
        pool.captured_features=rows
        return scorer(rows,model)
    bindings["score_online_candidates"]=capture
    pool.select=MethodType(FunctionType(original.__code__,bindings,original.__name__,
                                       original.__defaults__,original.__closure__),pool)
    return pool


def prepare():
    if (OUT/"plan.json").exists(): raise ValueError("plan exists")
    previous=prior.verify()
    files=["experiments/sa_incremental_heat_runtime.py","scripts/audit_sa_incremental_heat.py",
           "tests/evaluation/test_sa_incremental_heat.py","docs/SA_INCREMENTAL_HEAT_PROTOCOL_ZH.md"]
    plan=dict(schema="lns2.sa_incremental_heat.v1",cases=previous["cases"],records=previous["records"],
        config=previous["config"],source_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        previous_registration_sha256=sha256_file(prior.OUT/"registration.json"),
        inputs={n:sha256_file(ROOT/n) for n in files},workers=1,job_fuse_seconds=600,
        replay_pp_seconds=300.,shadow_interval=20,no_ttf=True,no_default_change=True,
        gate=dict(minimum_selection_reduction_percent=5.,minimum_faster_cases=6))
    write_json(OUT/"plan.json",plan)
    return dict(cases=8,paired_selector_calls=792,prefix_repairs=792,workers=1)


def verify():
    previous=prior.verify()
    p=read_json(OUT/"plan.json")
    for name,h in p["inputs"].items():
        if sha256_file(ROOT/name)!=h: raise ValueError("registered source changed: "+name)
    if p["previous_registration_sha256"]!=sha256_file(prior.OUT/"registration.json") or p["cases"]!=previous["cases"] or p["records"]!=previous["records"] or p["config"]!=previous["config"]:
        raise ValueError("historical inputs changed")
    if (p["workers"],p["shadow_interval"],p["gate"])!=(1,20,dict(minimum_selection_reduction_percent=5.,minimum_faster_cases=6)):
        raise ValueError("profiling protocol changed")
    return p


def worker(job):
    p,record=job["plan"],job["record"]
    if sha256_file(ROOT/record["path"])!=record["sha256"]: raise ValueError("historical trace changed")
    saved=read_json(ROOT/record["path"])
    env=pilot.make_env(dict(job,budget=3000.))
    state=pilot._plain(env.reset(seed=job["case"]["solver_seed"]))
    expected=saved["initial_state"]
    reference=capture_features(SingleFullCheckPool(job["case"]))
    incremental=capture_features(IncrementalHeatPool(job["case"]))
    if pilot.state_fingerprint(state)!=pilot.state_fingerprint(expected): raise ValueError("initial mismatch")
    rows=[]
    for d,event in enumerate(saved["events"]):
        pools={"reference":reference,"incremental":incremental}
        order=("reference","incremental") if (d+job["case_index"])%2==0 else ("incremental","reference")
        times,choices={},{}
        for name in order:
            copied=deepcopy(state)
            started=time.perf_counter()
            choices[name]=pools[name].select(env,copied,d)
            times[name]=time.perf_counter()-started
        if choices["reference"]!=choices["incremental"] or choices["reference"]!=(event["selected_index"],event["pool"]):
            raise ValueError(f"candidate/score/action mismatch at {d}")
        if reference.captured_features!=incremental.captured_features or reference.topology.analysis!=incremental.topology.analysis:
            raise ValueError(f"feature/topology mismatch at {d}")
        if reference.topology.visit_heat!=incremental.topology.visit_heat or reference.topology.agent_heat!=incremental.topology.agent_heat:
            raise ValueError("heat counter mismatch")
        if incremental.topology.shadow_interval!=p["shadow_interval"] or incremental.topology.last_shadow_validation!=(d>0 and d%20==0):
            raise ValueError("periodic shadow missing")
        if pilot.state_fingerprint(env.get_state())!=pilot.state_fingerprint(state): raise ValueError("selection mutated environment")
        rows.append(dict(decision=d,reference_seconds=times["reference"],incremental_seconds=times["incremental"],
            initial=d==0,shadow=incremental.topology.last_shadow_validation,
            changed_agents=incremental.topology.heat_changed_agents,conflicts=state["num_of_colliding_pairs"],
            incremental_first=order[0]=="incremental",feature_sha256=pilot.digest(reference.captured_features),
            selection_sha256=pilot.digest(choices["reference"])))
        step=pilot._plain(env.step_experimental_pp(event["action"],p["replay_pp_seconds"],"annealed",event["temperature"],event["uniform"]))
        state=step["observation"]
        expected=pilot.apply_state_delta(expected,event["delta"])
        if pilot.state_fingerprint(state)!=pilot.state_fingerprint(expected): raise ValueError(f"prefix mismatch at {d}")
        for key in ("repair_order","neighborhood","replan_success","pp_failure_reason"):
            if step["metrics"][key]!=event["metrics"][key]: raise ValueError("PP outcome mismatch")
    pilot.validate_final(state)
    return dict(status="ok",job_id=job["job_id"],plan_sha256=job["plan_sha256"],map_id=job["case"]["map_id"],
                samples=rows,replay_steps=len(rows),full_prefix_equal=True,no_ttf=True)


def summarize(samples):
    a=sum(s["reference_seconds"] for s in samples)
    b=sum(s["incremental_seconds"] for s in samples)
    return dict(calls=len(samples),reference_seconds=a,incremental_seconds=b,
                reduction_percent=100*(a-b)/a if a else None,
                faster_calls=sum(s["incremental_seconds"]<s["reference_seconds"] for s in samples))


def collect(resume=False):
    p=verify()
    identity=sha256_file(OUT/"plan.json")
    with pilot._CollectionRunLock(OUT,identity,"serial-selector-components"):
        if resume: (OUT/"STOP_AFTER_CASE.json").unlink(missing_ok=True)
        results=[]
        for i,case in enumerate(p["cases"]):
            job=dict(job_id=case["case_id"],case=case,config=p["config"],case_index=i,plan=p,
                     record=p["records"][case["case_id"]],plan_sha256=identity)
            path=OUT/"cases"/(job["job_id"]+".json")
            if not path.exists():
                if (OUT/"STOP_AFTER_CASE.json").exists(): return dict(paused=True,completed=i)
                marker=OUT/"attempts"/(job["job_id"]+".json")
                if marker.exists(): raise ValueError("interrupted attempt: inspect before retry")
                write_json(marker,dict(job_id=job["job_id"],plan_sha256=identity,started=True))
                print(f"case {i+1}/8: paired selection and full-prefix replay",flush=True)
                def failure(j,status,error): return dict(status=status,job_id=j["job_id"],plan_sha256=identity,error=str(error))
                def save(row):
                    row["integrity_sha256"]=pilot.digest(row)
                    write_json(path,row)
                pilot._run_jobs(worker,[job],workers=1,phase="paired-selector",output_root=OUT,
                    run_fingerprint=identity,timeout_seconds=p["job_fuse_seconds"],failure_result=failure,
                    on_result=save,stop_on_failure=True)
            row=read_json(path)
            if row.get("status")!="ok" or row.get("job_id")!=job["job_id"] or row.get("plan_sha256")!=identity or row.get("integrity_sha256")!=pilot.digest({k:v for k,v in row.items() if k!="integrity_sha256"}):
                raise ValueError("invalid prior result; no automatic retry")
            results.append(row)
            write_json(OUT/"progress.json",dict(completed=i+1,total=8))
        all_samples=[s for r in results for s in r["samples"]]
        cases={r["job_id"]:summarize(r["samples"]) for r in results}
        aggregate=summarize(all_samples)
        faster_cases=sum(v["reduction_percent"]>0 for v in cases.values())
        retained=aggregate["reduction_percent"]>=p["gate"]["minimum_selection_reduction_percent"] and faster_cases>=p["gate"]["minimum_faster_cases"]
        report=dict(complete=True,plan_sha256=identity,no_ttf=True,no_default_change=True,
            cases=cases,total=aggregate,faster_cases=faster_cases,
            noninitial=summarize([s for s in all_samples if not s["initial"]]),
            shadow=summarize([s for s in all_samples if s["shadow"]]),
            ordinary_updates=summarize([s for s in all_samples if not s["initial"] and not s["shadow"]]),
            unchanged_paths=summarize([s for s in all_samples if s["changed_agents"]==[]]),
            replay_steps=sum(r["replay_steps"] for r in results),
            files={r["job_id"]+".json":sha256_file(OUT/"cases"/(r["job_id"]+".json")) for r in results},
            decision="retain_for_independent_ttf_confirmation" if retained else "do_not_promote_incremental_heat")
        write_json(OUT/"report.json",report)
        return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("prepare","verify","collect","resume","stop"))
    phase=parser.parse_args().phase
    if phase=="stop":
        write_json(OUT/"STOP_AFTER_CASE.json",dict(requested=True)); result=dict(stop_after_case=True)
    elif phase=="resume": result=collect(True)
    elif phase=="verify": result=dict(verified=bool(verify()))
    else: result=globals()[phase]()
    print(json.dumps(result,indent=2))


if __name__=="__main__": main()
