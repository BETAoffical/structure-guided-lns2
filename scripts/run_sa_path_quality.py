"""Frozen-native Official/Dual16/Dual16-SA path-quality comparison."""

import argparse
from copy import deepcopy
import itertools
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments._common import read_json, write_json, sha256_file, json_fingerprint, read_jsonl
from experiments.closed_loop_trace_storage import encode_state_delta, apply_state_delta
from experiments.nonmonotonic_repair import temperature, acceptance_draw, validate_transition
from experiments.repair_collection import _CollectionRunLock, _make_environment, _plain, _run_jobs, state_fingerprint
from experiments.sa_single_check_runtime import SingleFullCheckPool
from lns2_selector.evaluation import path_quality_execution as execution
from lns2_selector.evaluation.path_quality_analysis import inspect_episode, summarize, publish_analysis
from lns2_selector.evaluation.path_quality_cohort import admission_key, runtime_environment
from lns2_selector.evaluation.path_quality_preflight import audit_task
from lns2_selector.runtime.online_selection import ProposalDeadlineExceeded
from lns2_selector.solver.native import native_identity, load_native_module
from scripts.run_sa_wall_clock import action_for
from scripts.diagnose_nonmonotonic_repair import seed

OUT = ROOT / "build/sa-path-quality-evaluation-v1"
NATIVE = "build/linux/sa-wall-clock-v1/lns2_env.cpython-310-x86_64-linux-gnu.so"
NATIVE_SHA = "5b1b2af1a8a388e76567c3f1e874674ef8c2701775c3706d0a1e4d9678d1925d"
SOURCE = "build/path-quality-evaluation-v1/preflight.json"
CONTROLLERS = ("official_adaptive", "dual16", "dual16_sa")
COMPARISONS = tuple(itertools.combinations(CONTROLLERS, 2))
PROTOCOL = dict(first_feasible_budget_seconds=120, total_planning_budgets_seconds=[60,120])


def schedule(cases):
    rows = execution.execution_schedule(cases, PROTOCOL)
    mapping = dict(zip(execution.CONTROLLERS, CONTROLLERS))
    for row in rows:
        row["controller"] = mapping[row["controller"]]
        row["job_id"] = json_fingerprint({k:row[k] for k in
            ("task_id", "solver_seed", "controller", "protocol", "budget_seconds")})[:24]
    return rows


def prepare():
    if (OUT / "registration.json").exists():
        raise ValueError("registration exists; never overwrite")
    old = read_json(ROOT / SOURCE)
    if json_fingerprint({k:v for k,v in old.items() if k != "fingerprint"}) != old["fingerprint"]:
        raise ValueError("historical case manifest corrupted")
    cases = [c for c in old["cases"] if c["status"] != "quarantined"]
    if len(cases) != 14 or len({c["map_id"] for c in cases}) != 11 or any(c["solver_seeds"] != [51,52] for c in cases):
        raise ValueError("historical cohort changed")
    template_path = old["runtime_template"]["manifest"]
    if sha256_file(ROOT / template_path) != old["runtime_template"]["sha256"]:
        raise ValueError("frozen runtime template changed")
    if sha256_file(ROOT / NATIVE) != NATIVE_SHA:
        raise ValueError("not the original frozen native")
    inputs = {SOURCE:sha256_file(ROOT / SOURCE), template_path:sha256_file(ROOT / template_path), NATIVE:NATIVE_SHA}
    for c in cases:
        for relative in c["files"].values():
            if sha256_file(ROOT / relative) != old["input_sha256"][relative]:
                raise ValueError("historical task bytes changed")
            inputs[relative] = sha256_file(ROOT / relative)
        checked = audit_task(ROOT/c["files"]["map_file"], ROOT/c["files"]["scenario_file"],
                             ROOT/c["files"]["task_file"], c["static_audit"]["agent_count"])
        if checked != c["static_audit"]:
            raise ValueError("static case audit changed")
    for directory in ("experiments", "lns2_selector", "scripts", "src", "include", "third_party/mapf_lns2",
                      "artifacts/initlns-closed-loop-controller-v2", "artifacts/initlns-closed-loop-policy-v1"):
        for path in (ROOT/directory).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix in {".py",".cpp",".h",".hpp",".json"}:
                inputs[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    for name in ("docs/SA_PATH_QUALITY_PROTOCOL_ZH.md", "tests/evaluation/test_sa_path_quality.py"):
        inputs[name] = sha256_file(ROOT/name)
    r = dict(schema="lns2.sa_path_quality.v1", cases=cases, template=read_json(ROOT/template_path),
             schedule=schedule(cases), inputs=inputs, native_file=NATIVE, native_sha256=NATIVE_SHA,
             controllers=list(CONTROLLERS), protocol=PROTOCOL, workers=1, fuse_extra_seconds=120,
             sa=dict(initial_temperature=1000.0,cooling=.99), engineering_variant_used=False,
             source_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip())
    r["fingerprint"] = json_fingerprint(r)
    write_json(OUT/"registration.json",r)
    return dict(prepared=True,tasks=14,maps=11,timed_jobs=252,reset_jobs=56,native_sha256=NATIVE_SHA)


def verify(native=False):
    r = read_json(OUT/"registration.json")
    if json_fingerprint({k:v for k,v in r.items() if k != "fingerprint"}) != r["fingerprint"]:
        raise ValueError("registration fingerprint mismatch")
    if (r["controllers"],r["protocol"],r["workers"],r["native_file"],r["native_sha256"],r["engineering_variant_used"]) != (
            list(CONTROLLERS),PROTOCOL,1,NATIVE,NATIVE_SHA,False):
        raise ValueError("registered protocol/native changed")
    if r["schedule"] != schedule(r["cases"]):
        raise ValueError("schedule changed")
    for name,h in r["inputs"].items():
        if sha256_file(ROOT/name) != h:
            raise ValueError("registered file changed: " + name)
    if native:
        mod = load_native_module()
        if native_identity(mod)["sha256"] != NATIVE_SHA or getattr(mod,"anytime_handoff_schema",None) != "lns2.anytime_handoff.v1":
            raise ValueError("wrong native or missing handoff support")
        if not hasattr(mod.LNS2RepairEnv,"step_experimental_pp"):
            raise ValueError("frozen native lacks SA API")
    return r


def worker_job(case,item,template,folder,binding):
    mapped = dict(item,controller="official_adaptive" if item["controller"] in {"official_adaptive", "official_sa"} else "dual16")
    job = execution.controller_job(ROOT,case,mapped,template,folder,binding)
    job.update(sa_controller=item["controller"],sa_proposal=deepcopy(template["proposal"]),
               sa_case_id=f"{case['task_id']}-seed{item['solver_seed']}")
    return job


def first_phase(job, *, path_observer):
    setup = time.perf_counter()
    env = _make_environment(job["dataset_root"],job["row"],job["environment"],"Adaptive")
    env_seconds = time.perf_counter()-setup
    case = dict(case_id=job["sa_case_id"],task_id=job["row"]["task_id"],solver_seed=job["solver_seed"],proposal=job["sa_proposal"])
    arm = "official" if job["sa_controller"] == "official_adaptive" else job["sa_controller"]
    selector = SingleFullCheckPool(case) if arm.startswith("dual16") else None
    folder = Path(job["output_root"])
    folder.mkdir(parents=True,exist_ok=True)
    trace_path = folder/"trace.jsonl"
    iterations, pp_seconds, selection_seconds = 0,0.,0.
    start = time.perf_counter()
    state = _plain(env.reset(seed=job["solver_seed"]))
    reset_done = time.perf_counter()
    if not state["initial_solution_complete"]:
        raise ValueError("incomplete initial PP")
    path_observer("initial",state,start,reset_done,state_fingerprint(state))
    completed, stop = reset_done, "feasible" if state["feasible"] else "deadline"
    with trace_path.open("x",encoding="utf-8") as stream:
        while not state["feasible"]:
            remaining = job["wall_time_budget_seconds"]-(time.perf_counter()-start)
            if remaining <= 0: break
            select_start = time.perf_counter()
            try:
                index,pool = selector.select(env,state,iterations) if selector else (None,[])
            except ProposalDeadlineExceeded:
                if state_fingerprint(env.get_state()) != state_fingerprint(state):
                    raise ValueError("deadline proposal mutated state")
                stop = "proposal_deadline"
                break
            select_duration = time.perf_counter()-select_start
            selection_seconds += select_duration
            remaining = job["wall_time_budget_seconds"]-(time.perf_counter()-start)
            if remaining <= 0:
                stop = "selection_deadline"
                break
            action = action_for(arm,case,iterations,pool,index)
            temp = temperature(iterations)
            draw = acceptance_draw(seed(case["case_id"],0,iterations,"accept"))
            before = state
            raw = (env.step_experimental_pp(action,remaining,"annealed",temp,draw)
                   if arm in {"dual16_sa", "official_sa"} else env.step_with_time_limit(action,remaining))
            step = _plain(raw)
            completed = time.perf_counter()
            state,m = step["observation"],step["metrics"]
            if not m["action_valid"] or not m["step_applied"]:
                raise ValueError("invalid/unapplied repair action")
            pp_seconds += m["native_replan_seconds"]
            event = dict(decision=iterations,action=action,metrics=m,pool=pool,selected_index=index,
                         temperature=temp,uniform=draw,elapsed_seconds=completed-start,
                         selection_seconds=select_duration,delta=encode_state_delta(before,state))
            stream.write(json.dumps(event,separators=(",",":"),allow_nan=False)+"\n")
            iterations += 1
            if state["feasible"]:
                stop = "feasible"
                break
            if m["pp_failure_reason"] == "time_limit" or step["truncated"]: break
        stream.flush()
    path_observer("terminal",state,start,completed if state["feasible"] else time.perf_counter(),state_fingerprint(state))
    return dict(status="ok",trace_file="trace.jsonl",trace_sha256=sha256_file(trace_path),stop_reason=stop,
        summary=dict(repair_iterations=iterations,environment_construct_seconds=env_seconds,
                     selection_seconds=selection_seconds,pp_seconds=pp_seconds))


def child(spec):
    execution._episode_child(spec, first_phase_worker=first_phase)


def reset_worker(job):
    if native_identity()["sha256"] != NATIVE_SHA:
        raise ValueError("reset native mismatch")
    j = worker_job(job["case"],job["item"],job["template"],OUT/"admission",job["binding"])
    env = _make_environment(j["dataset_root"],j["row"],j["environment"],"Adaptive")
    s = _plain(env.reset(seed=j["solver_seed"]))
    if not s["initial_solution_complete"]: raise ValueError("incomplete reset")
    journal = execution.PathJournal(OUT/"admission",job["binding"],ROOT/job["case"]["files"]["map_file"],
                                    ROOT/job["case"]["files"]["scenario_file"])
    quality = journal.quality(s)
    return dict(status="ok",job_id=job["job_id"],key=job["job_id"],state_fingerprint=state_fingerprint(s),
                initial_quality=quality,paths=[a["path"] for a in sorted(s["agents"],key=lambda a:a["id"])],binding=job["binding"])


def validate():
    r = verify(True)
    with _CollectionRunLock(OUT,r["fingerprint"],"sa-quality-admission"):
        if (OUT/"admission_started.json").exists(): raise ValueError("admission attempt exists")
        unique = {admission_key(i):i for i in r["schedule"] if i["controller"] == "official_adaptive"}
        cases = {c["task_id"]:c for c in r["cases"]}
        jobs = [dict(job_id=k,item=i,case=cases[i["task_id"]],template=r["template"],binding=r["fingerprint"]) for k,i in unique.items()]
        write_json(OUT/"admission_started.json",dict(jobs=len(jobs)))
        def failed(j,status,error): return dict(status=status,job_id=j["job_id"],error=str(error))
        rows = _run_jobs(reset_worker,jobs,workers=4,phase="sa-quality-admission",output_root=OUT/"admission-progress",
            run_fingerprint=r["fingerprint"],timeout_seconds=240,failure_result=failed,stop_on_failure=True,
            on_result=lambda row:write_json(OUT/"admission"/(row["job_id"]+".json"),row))
        passed = len(rows)==56 and all(x["status"]=="ok" for x in rows)
        report = dict(passed=passed,completed=len(rows),expected=56,binding=r["fingerprint"],
                      files={x["job_id"]:sha256_file(OUT/"admission"/(x["job_id"]+".json")) for x in rows})
        write_json(OUT/"admission.json",report)
        return report


def anchors(r):
    a = read_json(OUT/"admission.json")
    if not a["passed"] or a["binding"]!=r["fingerprint"] or len(a["files"])!=56:
        raise ValueError("admission incomplete")
    result = {}
    for k,h in a["files"].items():
        path = OUT/"admission"/(k+".json")
        if sha256_file(path)!=h: raise ValueError("reset artifact changed")
        result[k]=read_json(path)
    return result


def spec_for(r,item,anchor,folder=None):
    c = next(c for c in r["cases"] if c["task_id"]==item["task_id"])
    folder = folder or OUT/"episodes"/item["job_id"]
    spec = dict(root=str(ROOT),output=str(folder),item=item,case=c,native_sha256=NATIVE_SHA,
                process_timeout_seconds=item["budget_seconds"]+120,input_sha256=r["inputs"],
                expected_initial_fingerprint=anchor["state_fingerprint"],cohort_registration=r["fingerprint"],
                worker_job=worker_job(c,item,r["template"],folder,r["fingerprint"]))
    spec["binding"]=execution.spec_fingerprint(spec)
    return spec


def audit_trace(spec):
    folder=Path(spec["output"])
    if not (folder/"first_phase_result.json").exists(): return
    phase=execution.read_artifact(folder/"first_phase_result.json",spec["binding"])
    path=folder/"first_phase"/phase["trace_file"]
    if sha256_file(path)!=phase["trace_sha256"]: raise ValueError("trace changed")
    state=execution.read_artifact(folder/"initial.json",spec["binding"])["observation"]
    case=dict(case_id=spec["worker_job"]["sa_case_id"])
    arm="official" if spec["item"]["controller"]=="official_adaptive" else spec["item"]["controller"]
    count=0
    for d,e in enumerate(read_jsonl(path)):
        if e["decision"]!=d or state["feasible"]: raise ValueError("trace sequence invalid")
        after=apply_state_delta(state,e["delta"])
        if e["action"]!=action_for(arm,case,d,e["pool"],e["selected_index"]): raise ValueError("action identity changed")
        if (e["temperature"],e["uniform"])!=(temperature(d),acceptance_draw(seed(case["case_id"],0,d,"accept"))):
            raise ValueError("SA draw changed")
        validate_transition(state,after,e["metrics"],e["metrics"]["neighborhood"],
            "annealed" if arm in {"dual16_sa", "official_sa"} else "standard",e["temperature"],e["uniform"])
        state=after
        count+=1
    terminal=execution.read_artifact(folder/"terminal.json",spec["binding"])
    if state_fingerprint(state)!=terminal["state_fingerprint"] or count!=phase["summary"]["repair_iterations"]:
        raise ValueError("terminal trace mismatch")


def smoke():
    r=verify(True)
    a=anchors(r)
    if (OUT/"smoke_started.json").exists(): raise ValueError("smoke attempt exists")
    write_json(OUT/"smoke_started.json",dict(binding=r["fingerprint"]))
    case=next(c for c in r["cases"] if c["family"]=="maze")
    results=[]
    for mode in ("first_feasible","fixed_budget"):
        for controller in CONTROLLERS:
            original=next(i for i in r["schedule"] if i["task_id"]==case["task_id"] and
                          i["solver_seed"]==51 and i["protocol"]==mode and i["controller"]==controller)
            item=dict(original,budget_seconds=2.0,job_id=f"smoke-{mode}-{controller}")
            anchor=a[admission_key(original)]
            spec=spec_for(r,item,anchor,OUT/"smoke"/item["job_id"])
            result=execution.supervise_episode(spec,authorized=True,child_entry=child)
            if result["error"] or result["status"]=="external_timeout":
                raise ValueError("smoke failed; inspect saved result")
            audit_trace(spec)
            row=inspect_episode(ROOT,case,item,Path(spec["output"]),spec["binding"],anchor)
            results.append(dict(job_id=item["job_id"],status=row["status"]))
    report=dict(passed=True,binding=r["fingerprint"],jobs=results,not_performance_evidence=True)
    write_json(OUT/"smoke.json",report)
    return report


def collect(resume=False):
    r=verify(True)
    a=anchors(r)
    preflight=read_json(OUT/"smoke.json")
    if preflight.get("passed") is not True or preflight.get("binding")!=r["fingerprint"]:
        raise ValueError("smoke not passed")
    with _CollectionRunLock(OUT,r["fingerprint"],"sa-quality-timed"):
        manifest_path=OUT/"manifest.json"
        if manifest_path.exists() and not resume: raise ValueError("run exists; resume explicitly")
        manifest=read_json(manifest_path) if manifest_path.exists() else dict(binding=r["fingerprint"],jobs={})
        if manifest["binding"]!=r["fingerprint"]: raise ValueError("manifest identity changed")
        write_json(OUT/"environment.json",runtime_environment())
        try:
            for item in r["schedule"]:
                spec=spec_for(r,item,a[admission_key(item)])
                folder=Path(spec["output"])
                old=manifest["jobs"].get(item["job_id"])
                if old:
                    for name,h in old["files"].items():
                        if sha256_file(folder/name)!=h: raise ValueError("completed job changed")
                    continue
                if (OUT/"STOP_AFTER_EPISODE.json").exists():
                    write_json(OUT/"status.json",dict(status="stopped",completed=len(manifest["jobs"]),total=252))
                    return dict(stopped=True,completed=len(manifest["jobs"]))
                write_json(OUT/"status.json",dict(status="running",completed=len(manifest["jobs"]),total=252,current=item))
                print(f"START {item['schedule_index']+1}/252 {item['controller']} {item['protocol']} {item['budget_seconds']} {item['task_id']}",flush=True)
                result=execution.supervise_episode(spec,authorized=True,resume=False,child_entry=child)
                if result["error"]: raise ValueError("episode error; inspect preserved artifacts")
                row=inspect_episode(ROOT,spec["case"],item,folder,spec["binding"],a[admission_key(item)])
                audit_trace(spec)
                files={p.relative_to(folder).as_posix():sha256_file(p) for p in folder.rglob("*.json*") if p.is_file()}
                manifest["jobs"][item["job_id"]]=dict(status=row["status"],success=row["success"],files=files)
                write_json(manifest_path,manifest)
                print(f"COMPLETE {len(manifest['jobs'])}/252 {row['status']} ttf={row['ttf_seconds']}",flush=True)
            write_json(OUT/"status.json",dict(status="complete",completed=252,total=252))
            return dict(complete=True,jobs=252)
        except BaseException:
            write_json(OUT/"status.json",dict(status="failed_or_interrupted",completed=len(manifest["jobs"]),total=252))
            raise


def analyze():
    r=verify()
    a=anchors(r)
    manifest=read_json(OUT/"manifest.json")
    if manifest["binding"]!=r["fingerprint"] or set(manifest["jobs"])!={i["job_id"] for i in r["schedule"]}:
        raise ValueError("incomplete manifest")
    rows=[]
    for item in r["schedule"]:
        spec=spec_for(r,item,a[admission_key(item)])
        for name,h in manifest["jobs"][item["job_id"]]["files"].items():
            if sha256_file(Path(spec["output"])/name)!=h: raise ValueError("result changed")
        rows.append(inspect_episode(ROOT,spec["case"],item,Path(spec["output"]),spec["binding"],a[admission_key(item)]))
    report=dict(schema="lns2.sa_path_quality_report.v1",complete=True,scheduled=252,binding=r["fingerprint"],
                promotion_allowed=False,engineering_variant_used=False,native_sha256=NATIVE_SHA,
                statistics=summarize(rows,controllers=CONTROLLERS,comparisons=COMPARISONS),episodes=rows)
    publish_analysis(OUT/"analysis",rows,report)
    return dict(complete=True,scheduled=252,successes=sum(x["success"] for x in rows))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("prepare","verify","validate","smoke","collect","resume","stop","analyze","run"))
    args=parser.parse_args()
    if args.phase=="stop":
        write_json(OUT/"STOP_AFTER_EPISODE.json",dict(requested=True)); result=dict(stop_after_episode=True)
    elif args.phase=="resume": result=collect(True)
    elif args.phase=="verify": result=dict(verified=bool(verify(True)))
    elif args.phase=="run":
        result=collect()
        if result.get("complete"): result=analyze()
    else: result=globals()[args.phase]()
    print(json.dumps(result,indent=2))
