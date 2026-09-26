"""Frozen old-16 paired selector on new warehouse tasks; no formal timing."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments._common import read_json, read_jsonl, sha256_file, json_fingerprint
from experiments.sa_paired_completion import MODEL_PARAMS, require
from experiments.sa_paired_closed_loop import fit_complete, portable_payload, portable_model, choose, stop_reason, contrast, gate, subset
from experiments.sa_history_information import profile_features
from scripts.audit_sa_history_information import atomic

CONFIG = ROOT / "configs/sa_paired_closed_loop.json"


def once(path, value):
    if path.exists():
        require(read_json(path) == value, "existing output differs: " + path.name)
    else:
        atomic(path, value)


def sealed(value):
    return value | {"integrity":json_fingerprint(value)}


def check_seal(value):
    require(isinstance(value, dict), "artifact must be a JSON object")
    require(value["integrity"]==json_fingerprint({k:v for k,v in value.items() if k!="integrity"}),"artifact integrity")
    return value


def verify():
    cfg = read_json(CONFIG)
    out = ROOT / cfg["output"]
    p = read_json(out / "plan.json")
    require(p["config"] == cfg and p["binding"] == json_fingerprint({k:v for k,v in p.items() if k != "binding"}), "plan identity")
    for name,digest in p["inputs"].items():
        require(sha256_file(ROOT/name) == digest, "registered input changed: " + name)
    return p,out


def prepare():
    from scripts.run_sa_independent_confirmation import historical_inventory
    cfg = read_json(CONFIG)
    out = ROOT/cfg["output"]
    require(not out.exists(), "output exists; do not overwrite")
    source = read_json(ROOT/cfg["source_plan"])
    require(source["binding"] == cfg["source_binding"] == json_fingerprint({k:v for k,v in source.items() if k != "binding"}), "source binding")
    require(sha256_file(ROOT/cfg["training_index"]) == cfg["training_index_sha256"], "training data changed")
    require(sha256_file(ROOT/cfg["dataset_base"]) == cfg["dataset_base_sha256"], "map template changed")
    inputs = dict(source["inputs"])
    additions = [CONFIG, Path(__file__), ROOT/cfg["training_index"], ROOT/cfg["dataset_base"],
        ROOT/cfg["runtime_registration"], ROOT/cfg["source_plan"],
        ROOT/"experiments/sa_paired_closed_loop.py", ROOT/"tests/evaluation/test_sa_paired_closed_loop.py",
        ROOT/"docs/SA_PAIRED_CLOSED_LOOP_PROTOCOL_ZH.md",
        ROOT/"lns2_selector/training/tree_utils.py", ROOT/"lns2_selector/runtime/portable_scalar.py",
        ROOT/"scripts/run_sa_independent_confirmation.py"]
    additions += list((ROOT/"generators").rglob("*.py"))
    for path in additions:
        inputs[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    for name,digest in inputs.items():
        require(sha256_file(ROOT/name) == digest, "source changed: " + name)
    history = historical_inventory(exclude=out)
    rng = random.Random(cfg["master_seed"])
    masters = [rng.randrange(1,2**31) for _ in range(cfg["maps"])]
    require(len(set(masters)) == cfg["maps"] and not set(masters)&set(history["seeds"]), "seed overlap")
    plan = dict(config=cfg, inputs=inputs, history=history, map_masters=masters, model=MODEL_PARAMS,
        native_file=source["native_file"], native_sha256=source["native_sha256"],
        commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        role="new_map_bounded_closed_loop_not_ttf", no_ttf=True)
    plan["binding"] = json_fingerprint(plan)
    once(out/"plan.json",plan)
    return dict(binding=plan["binding"], jobs=96, maximum_repairs=96*cfg["max_decisions"], workers=cfg["workers"],
                missing_historical_maps=len(history["missing_map_files"]))


def model_files(out, plan):
    r = read_json(out/"model/receipt.json")
    require(r["binding"] == plan["binding"], "model binding")
    for name,digest in r["files"].items():
        require(sha256_file(out/"model"/name) == digest, "model artifact changed")
    return read_json(out/"model/bundle.json")


def freeze_model():
    plan,out = verify()
    if (out/"model/receipt.json").exists():
        model_files(out,plan)
        return dict(resumed=True)
    require(not (out/"model").exists(), "partial model publication needs audit")
    data = read_json(ROOT/plan["config"]["training_index"])
    model = fit_complete(data)
    payload = portable_payload(model)
    portable = portable_model(payload, data["feature_names"], native=False)
    fixtures = []
    for s in data["states"]:
        expected, actual = model.rank(s), portable.rank(s)
        require(expected == actual, "sklearn/Python portable mismatch")
        # Only pre-action inputs and predictions, never trial outcomes.
        fixture = {k:s[k] for k in ("state_id","anchor_id","agent_ids")}
        fixture["candidates"] = [{k:c[k] for k in ("candidate_id","agents","features")} for c in s["candidates"]]
        fixtures.append(dict(state=fixture, prediction=expected))
    ranges = {n:[min(c["features"][n] for s in data["states"] for c in s["candidates"]),
                 max(c["features"][n] for s in data["states"] for c in s["candidates"])] for n in data["feature_names"]}
    folder = out/"model"
    folder.mkdir()
    with (folder/"model.pkl").open("xb") as f:
        pickle.dump(model,f,protocol=4)
    once(folder/"bundle.json",payload)
    once(folder/"fixtures.json",fixtures)
    once(folder/"metadata.json",dict(feature_names=data["feature_names"], ranges=ranges,
        train_state_ids=[s["state_id"] for s in data["states"]], train_map_ids=sorted({s["map_id"] for s in data["states"]}),
        training_sha256=sha256_file(ROOT/plan["config"]["training_index"]), params=MODEL_PARAMS))
    once(folder/"receipt.json",dict(binding=plan["binding"],files={p.name:sha256_file(p) for p in sorted(folder.iterdir())}))
    return dict(fitted_states=16, fixtures=len(fixtures), bundle_sha256=sha256_file(folder/"bundle.json"))


def verify_model():
    plan,out = verify()
    payload = model_files(out,plan)
    meta = read_json(out/"model/metadata.json")
    model = portable_model(payload,meta["feature_names"])
    fixtures = read_json(out/"model/fixtures.json")
    for row in fixtures:
        require(model.rank(row["state"]) == row["prediction"], "portable/native scores or selection changed")
    backend = model.estimator.model.inference_backend
    once(out/f"model-verification-{backend}.json",dict(binding=plan["binding"], fixtures=len(fixtures),backend=backend,exact=True))
    return dict(backend=backend,fixtures=len(fixtures),exact=True)


def generate_one(job):
    from generators.dataset import generate_dataset
    cfg,plan = job["config"],job["plan"]
    out = ROOT/cfg["output"]
    folder = out/f"dataset/shards/{job['index']:02d}"
    receipt = folder/"receipt.json"
    if receipt.exists():
        r = read_json(receipt)
        require(r["binding"] == plan["binding"], "generated shard binding")
        for n,h in r["files"].items():
            require(sha256_file(ROOT/n)==h, "generated shard changed")
        return r
    require(not folder.exists(), "partial dataset shard needs audit")
    base = read_json(ROOT/cfg["dataset_base"])
    base.update(master_seed=plan["map_masters"][job["index"]], map_id_prefix=f"sa_pairloop_v1_m{job['index']:02d}",
                tasks_per_map=2, splits={"confirmation":{"layout_counts":{"station_centric":1}}})
    task_template = deepcopy(base["task_variants"][0])
    base["task_variants"] = []
    for density in cfg["densities"]:
        v = deepcopy(task_template)
        v["name"] = f"bottleneck_d{int(density*100)}"
        v["task"].update(agent_density=density,required_bottleneck_crossing_ratio=cfg["bottleneck_ratio"])
        base["task_variants"].append(v)
    generate_dataset(base,output_override=folder.relative_to(ROOT))
    rows = read_jsonl(folder/"confirmation/manifest.jsonl")
    for row in rows:
        for key in ("map_file","map_metadata_file","task_file","scenario_file","instance_file","legacy_instance_file"):
            row[key] = (folder/"confirmation"/row[key]).relative_to(ROOT).as_posix()
    files = {p.relative_to(ROOT).as_posix():sha256_file(p) for p in sorted(folder.rglob("*")) if p.is_file()}
    r = dict(status="ok",binding=plan["binding"],rows=rows,files=files)
    once(receipt,r)
    return r


def generate():
    from lns2_selector.evaluation.path_quality_preflight import audit_task
    plan,out = verify()
    cfg = plan["config"]
    model_files(out,plan)
    require((out/"model-verification-native-portable-tree.json").exists(), "native parity required before new cases")
    once(out/"generation_registration.json",dict(binding=plan["binding"],model_receipt_sha256=sha256_file(out/"model/receipt.json")))
    jobs = [dict(config=cfg,plan=plan,index=i) for i in range(cfg["maps"])]
    with ProcessPoolExecutor(max_workers=min(cfg["workers"],cfg["maps"])) as pool:
        shards = list(pool.map(generate_one,jobs))
    rows = [r for shard in shards for r in shard["rows"]]
    seeds = {r["map_seed"] for r in rows}|{r["task_seed"] for r in rows}
    require(len(rows)==16 and len({r["map_seed"] for r in rows})==8 and len({r["task_seed"] for r in rows})==16, "dataset counts")
    require(len(seeds)==24 and not seeds&set(plan["history"]["seeds"]), "historical seed overlap")
    hashes = {r["map_id"]:sha256_file(ROOT/r["map_file"]) for r in rows}
    require(len(set(hashes.values()))==8 and not set(hashes.values())&set(plan["history"]["map_hashes"]), "map geometry overlap")
    cases=[]
    for r in rows:
        audit = audit_task(ROOT/r["map_file"],ROOT/r["scenario_file"],ROOT/r["task_file"],r["agent_count"])
        cases.append(dict(task_id=r["task_id"],map_id=r["map_id"],family="warehouse",solver_seeds=cfg["solver_seeds"],
            status="static_ready_runtime_unverified",static_audit=audit,files={k:r[k] for k in ("map_file","scenario_file","task_file")},
            density=r.get("agent_density"), task_variant=r["task_variant"]))
    result = dict(binding=plan["binding"],cases=cases,files={k:v for s in shards for k,v in s["files"].items()},
        map_sha256=hashes, map_seeds=sorted({r["map_seed"] for r in rows}),task_seeds=sorted({r["task_seed"] for r in rows}))
    once(out/"cases.json",sealed(result))
    return dict(tasks=len(cases),maps=len(hashes),geometry_overlap=0,seed_overlap=0)


def cases_verified(plan,out):
    r=check_seal(read_json(out/"cases.json"))
    freeze=read_json(out/"generation_registration.json")
    require(freeze["binding"]==plan["binding"] and freeze["model_receipt_sha256"]==sha256_file(out/"model/receipt.json"),"model changed after generation registration")
    require(r["binding"]==plan["binding"],"case binding")
    for name,h in r["files"].items():
        require(sha256_file(ROOT/name)==h,"case file changed")
    return r["cases"]


def reset_case(case, solver_seed, plan):
    from scripts import run_sa_path_quality as q
    require(q.native_identity()["sha256"]==plan["native_sha256"],"wrong native")
    template=read_json(ROOT/plan["config"]["runtime_registration"])["template"]
    item=dict(task_id=case["task_id"],solver_seed=solver_seed,controller="dual16_sa",budget_seconds=100000.)
    job=q.worker_job(case,item,template,ROOT/plan["config"]["output"] / "unused",plan["binding"])
    env=q._make_environment(job["dataset_root"],job["row"],job["environment"],"Adaptive")
    state=q._plain(env.reset(seed=solver_seed))
    require(state["initial_solution_complete"], "incomplete initial PP")
    return env,state,dict(case_id=job["sa_case_id"],task_id=case["task_id"],solver_seed=solver_seed,proposal=job["sa_proposal"])


def reset_worker(job):
    from scripts.run_feedback_exploration_diagnostics import validate_final
    from experiments.repair_collection import state_fingerprint
    _,state,_=reset_case(job["case"],job["solver_seed"],job["plan"])
    validate_final(state)
    return dict(status="ok",job_id=job["job_id"],map_id=job["case"]["map_id"],initial=state_fingerprint(state),
                conflicts=state["num_of_colliding_pairs"])


def paired_tasks(cases,cfg):
    return [(c,s,f"{c['task_id']}-s{s}") for c in cases for s in cfg["solver_seeds"]]


def qualify():
    from experiments.repair_collection import _run_jobs,_CollectionRunLock
    plan,out=verify()
    cfg=plan["config"]
    cases=cases_verified(plan,out)
    if (out/"qualification.json").exists():
        return check_seal(read_json(out/"qualification.json"))
    jobs=[dict(job_id=k,case=c,solver_seed=s,plan=plan) for c,s,k in paired_tasks(cases,cfg)]
    with _CollectionRunLock(out,plan["binding"],"paired-loop-reset"):
        results=_run_jobs(reset_worker,jobs,cfg["workers"],phase="reset",output_root=out/"reset-progress",
            run_fingerprint=plan["binding"],timeout_seconds=cfg["fuse_seconds"],stop_on_failure=True)
        active=[r for r in results if r.get("conflicts",0)>0]
        passed=len(results)==32 and all(r["status"]=="ok" for r in results) and len(active)>=cfg["minimum_active_episodes"] and len({r["map_id"] for r in active})>=cfg["minimum_active_maps"]
        result=dict(binding=plan["binding"],passed=passed,results=results,active=len(active),active_maps=len({r["map_id"] for r in active}))
        once(out/"qualification.json",sealed(result))
    return result


def receipt_valid(folder,plan):
    r=read_json(folder/"receipt.json")
    require(r["binding"]==plan["binding"] and set(r["files"])=={"initial.json","trace.jsonl","result.json"},"episode receipt identity")
    for name,h in r["files"].items():
        require(sha256_file(folder/name)==h,"episode artifact changed")
    return read_json(folder/"result.json")


def features_for(state,candidates,engine,history,temp,fp):
    engine.prepare(state)
    rows,_=engine.realized_rows(candidates,state_hash=fp)
    return [profile_features(dict(base=r["features"]["realized_dynamic"]|{"source.score":c["score"]}|
            history.features(state,c,temp)),"dynamic") for r,c in zip(rows,candidates,strict=True)]


def episode_worker(job):
    from scripts import run_sa_path_quality as q
    from scripts.run_feedback_exploration_diagnostics import validate_final
    from scripts.train_sa_history_selector import die_with_parent
    from experiments.online_feature_engine import OnlineFeatureEngine
    from experiments.sa_history_selector import History
    die_with_parent(job["parent_pid"])
    plan,case,arm=job["plan"],job["case"],job["arm"]
    cfg=plan["config"]
    out=ROOT/cfg["output"]
    folder=out/"episodes"/job["job_id"]
    if (folder/"receipt.json").exists():
        r=receipt_valid(folder,plan)
        return dict(status="ok",job_id=job["job_id"],stop=r["stop"],resumed=True)
    require(not folder.exists(),"partial episode requires audit; not automatic retry")
    payload=model_files(out,plan)
    meta=read_json(out/"model/metadata.json")
    model=portable_model(payload,meta["feature_names"])
    require(model.estimator.model.inference_backend=="native-portable-tree","native portable required")
    env,state,source=reset_case(case,job["solver_seed"],plan)
    initial_fp=q.state_fingerprint(state)
    require(initial_fp==job["expected_initial"],"paired initial mismatch")
    validate_final(state)
    folder.mkdir(parents=True)
    once(folder/"initial.json",dict(binding=plan["binding"],state=state))
    initial_nodes=state["low_level"]["generated"]
    history=History(state)
    selector=q.SingleFullCheckPool(source)
    engine=OnlineFeatureEngine(state,backend="native")
    began=time.monotonic()
    decisions=0
    signatures={}
    repeated=0
    trajectory=[state["num_of_colliding_pairs"]]
    with (folder/"trace.jsonl").open("x",encoding="utf8") as stream:
        while True:
            generated=state["low_level"]["generated"]-initial_nodes
            stop=stop_reason(state,decisions,generated,time.monotonic()-began,cfg)
            if stop:
                break
            fp=q.state_fingerprint(state)
            anchor_index,pool=selector.select(env,state,decisions)
            require(q.state_fingerprint(env.get_state())==fp,"proposal changed state")
            anchor=pool[anchor_index]["candidate_id"]
            occurrence=f"{job['pair_id']}-d{decisions:04d}-{fp}"
            ids=subset(pool,anchor,occurrence,cfg["candidate_seed"],cfg["candidate_limit"])
            candidates=[next(c for c in pool if c["candidate_id"]==i) for i in ids]
            temp=q.temperature(decisions)
            features=features_for(state,candidates,engine,history,temp,fp)
            ranking=choose(arm,candidates,anchor,model,features,[a["id"] for a in state["agents"]],occurrence,cfg["candidate_seed"])
            index=next(i for i,c in enumerate(pool) if c["candidate_id"]==ranking["selected"])
            # Production stream convention, shared across all arms.
            pp=q.seed(source["case_id"],0,decisions,"pp")
            draw=q.acceptance_draw(q.seed(source["case_id"],0,decisions,"accept"))
            action=dict(mode="explicit_neighborhood",agents=pool[index]["agents"],random_seed=pp)
            remaining=cfg["episode_seconds"]-(time.monotonic()-began)
            if remaining<=0:
                stop="wall_safety"
                break
            raw=q._plain(env.step_experimental_pp(action,min(cfg["pp_seconds"],remaining),"annealed",temp,draw))
            after,m=raw["observation"],raw["metrics"]
            q.validate_transition(state,after,m,action["agents"],"annealed",temp,draw)
            physical=json_fingerprint([(a["id"],a["path"]) for a in state["agents"]])
            repeated+=physical in signatures
            signatures[physical]=signatures.get(physical,0)+1
            outside=sum(v<meta["ranges"][n][0] or v>meta["ranges"][n][1] for f in features for n,v in f.items())
            event=dict(decision=decisions,before=fp,action=action,temperature=temp,uniform=draw,metrics=m,
                pool=pool,subset=ids,anchor_id=anchor,selected_index=index,ranking=ranking,
                features=features,feature_outside_fraction=outside/(len(features)*len(meta["feature_names"])),
                physical=physical,delta=q.encode_state_delta(state,after))
            stream.write(json.dumps(event,separators=(",",":"),allow_nan=False)+"\n")
            stream.flush()
            history.observe(state,event,after)
            state=after
            decisions+=1
            trajectory.append(state["num_of_colliding_pairs"])
            if m["pp_failure_reason"]=="time_limit" or not m["acceptance_evaluated"]:
                stop="incomplete_pp"
                break
    validate_final(state)
    r=dict(binding=plan["binding"],job_id=job["job_id"],pair_id=job["pair_id"],map_id=case["map_id"],
        arm=arm,solver_seed=job["solver_seed"],initial_fingerprint=initial_fp,final_fingerprint=q.state_fingerprint(state),
        stop=stop,success=state["feasible"],decisions=decisions,generated=state["low_level"]["generated"]-initial_nodes,
        node_overshoot=max(0,state["low_level"]["generated"]-initial_nodes-cfg["node_budget"]),
        conflicts=trajectory,physical_revisits=repeated,final_soc=state["sum_of_costs"],
        final_makespan=max(len(a["path"])-1 for a in state["agents"]),
        diagnostic_seconds=time.monotonic()-began,no_ttf=True)
    once(folder/"result.json",r)
    once(folder/"receipt.json",dict(binding=plan["binding"],files={n:sha256_file(folder/n) for n in ("initial.json","trace.jsonl","result.json")}))
    return dict(status="ok",job_id=job["job_id"],stop=stop,success=r["success"],decisions=decisions)


def collect(resume=False,limit=None):
    from experiments.repair_collection import _run_jobs,_CollectionRunLock
    plan,out=verify()
    cfg=plan["config"]
    q=check_seal(read_json(out/"qualification.json"))
    require(q["binding"]==plan["binding"] and q["passed"],"qualification gate failed")
    require(resume or not (out/"episodes").exists(),"existing results require resume")
    anchors={r["job_id"]:r["initial"] for r in q["results"]}
    once(out/"execution_registration.json",dict(binding=plan["binding"],qualification_sha256=sha256_file(out/"qualification.json"),
         cases_sha256=sha256_file(out/"cases.json"),model_receipt_sha256=sha256_file(out/"model/receipt.json")))
    jobs=[]
    for case,seed,key in paired_tasks(cases_verified(plan,out),cfg):
        for arm in cfg["arms"]:
            jobs.append(dict(job_id=f"{key}-{arm}",pair_id=key,case=case,solver_seed=seed,arm=arm,
                expected_initial=anchors[key],plan=plan,parent_pid=os.getpid()))
    if limit is not None:
        require(0<limit<=len(jobs),"invalid limit")
        jobs=jobs[:limit]
    with _CollectionRunLock(out,plan["binding"],"paired-closed-loop"):
        atomic(out/"run_status.json",dict(status="running",binding=plan["binding"]))
        try:
            for offset in range(0,len(jobs),cfg["workers"]):
                if (out/"STOP_AFTER_BATCH").exists():
                    break
                batch=jobs[offset:offset+cfg["workers"]]
                def record(row):
                    with (out/"progress.jsonl").open("a",encoding="utf8") as f:
                        f.write(json.dumps(row)+"\n")
                    print("EPISODE",row,flush=True)
                result=_run_jobs(episode_worker,batch,cfg["workers"],phase=f"batch{offset}",output_root=out/f"progress/{offset}",
                    run_fingerprint=plan["binding"],timeout_seconds=cfg["fuse_seconds"],stop_on_failure=True,on_result=record)
                require(len(result)==len(batch) and all(r["status"]=="ok" for r in result),"episode batch failed")
            count=sum((out/"episodes"/j["job_id"]/"receipt.json").exists() for j in jobs)
            atomic(out/"run_status.json",dict(status="completed" if count==96 else "paused",completed=count,binding=plan["binding"]))
        except BaseException as exc:
            atomic(out/"run_status.json",dict(status="error",error=repr(exc),binding=plan["binding"]))
            raise
    return dict(completed=count,total=96,no_ttf=True)


def audit_episode(job):
    from scripts import run_sa_path_quality as q
    from scripts.run_feedback_exploration_diagnostics import validate_final
    from experiments.sa_history_selector import History
    from experiments.online_feature_engine import OnlineFeatureEngine
    plan,folder=job["plan"],Path(job["folder"])
    cfg=plan["config"]
    result=receipt_valid(folder,plan)
    state=read_json(folder/"initial.json")["state"]
    require(q.state_fingerprint(state)==result["initial_fingerprint"],"initial hash")
    meta=read_json(ROOT/cfg["output"]/"model/metadata.json")
    model=portable_model(read_json(ROOT/cfg["output"]/"model/bundle.json"),meta["feature_names"],native=False)
    history=History(state)
    engine=OnlineFeatureEngine(state,backend="native")
    nodes=state["low_level"]["generated"]
    conflicts=[state["num_of_colliding_pairs"]]
    count=0
    changed=rollbacks=0
    outside=[]
    seen=set()
    repeats=0
    for e in read_jsonl(folder/"trace.jsonl"):
        require(e["decision"]==count and e["before"]==q.state_fingerprint(state),"trace sequence")
        key=f"{result['pair_id']}-d{count:04d}-{e['before']}"
        ids=subset(e["pool"],e["anchor_id"],key,cfg["candidate_seed"],cfg["candidate_limit"])
        require(ids==e["subset"],"subset changed")
        candidates=[next(c for c in e["pool"] if c["candidate_id"]==i) for i in ids]
        actual_features=features_for(state,candidates,engine,history,e["temperature"],e["before"])
        require(actual_features==e["features"],"online/pre-action feature mismatch")
        predicted=choose(result["arm"],candidates,e["anchor_id"],model,e["features"],[a["id"] for a in state["agents"]],key,cfg["candidate_seed"])
        require(predicted==e["ranking"] and e["pool"][e["selected_index"]]["candidate_id"]==predicted["selected"],"inference mismatch")
        case_id=result["pair_id"].rsplit("-s",1)[0]+f"-seed{result['solver_seed']}"
        require(e["action"]["random_seed"]==q.seed(case_id,0,count,"pp") and
                e["uniform"]==q.acceptance_draw(q.seed(case_id,0,count,"accept")) and e["temperature"]==q.temperature(count),"random/SA identity")
        after=q.apply_state_delta(state,e["delta"])
        q.validate_transition(state,after,e["metrics"],e["action"]["agents"],"annealed",e["temperature"],e["uniform"])
        require(e["action"]["agents"]==e["pool"][e["selected_index"]]["agents"],"action altered")
        physical=json_fingerprint([(a["id"],a["path"]) for a in state["agents"]])
        require(e["physical"]==physical,"physical identity")
        repeats+=physical in seen
        seen.add(physical)
        changed+=predicted["selected"]!=e["anchor_id"]
        rollbacks+=bool(e["metrics"]["pp_rolled_back"])
        outside.append(e["feature_outside_fraction"])
        history.observe(state,e,after)
        state=after
        count+=1
        conflicts.append(state["num_of_colliding_pairs"])
    validate_final(state)
    require(result["final_fingerprint"]==q.state_fingerprint(state) and result["conflicts"]==conflicts and
            result["decisions"]==count and result["generated"]==state["low_level"]["generated"]-nodes and
            result["physical_revisits"]==repeats and result["success"]==state["feasible"],"result mismatch")
    require(result["final_soc"]==state["sum_of_costs"] and result["final_makespan"]==max(len(a["path"])-1 for a in state["agents"]),"path quality mismatch")
    if result["stop"] not in {"wall_safety","incomplete_pp"}:
        require(stop_reason(state,count,result["generated"],0,cfg)==result["stop"],"incorrect termination")
    return result|dict(changed_from_anchor=changed,rollbacks=rollbacks,mean_feature_outside=sum(outside)/max(1,len(outside)))


def analyze():
    plan,out=verify()
    cfg=plan["config"]
    cases=cases_verified(plan,out)
    jobs=[dict(folder=str(out/"episodes"/f"{k}-{arm}"),plan=plan) for _,_,k in paired_tasks(cases,cfg) for arm in cfg["arms"]]
    require(all((Path(j["folder"])/"receipt.json").exists() for j in jobs),"incomplete cohort")
    with ProcessPoolExecutor(max_workers=cfg["workers"]) as pool:
        episodes=list(pool.map(audit_episode,jobs))
    rows=[]
    for c,seed,key in paired_tasks(cases,cfg):
        matched={r["arm"]:r for r in episodes if r["pair_id"]==key}
        require(len(matched)==3 and len({r["initial_fingerprint"] for r in matched.values()})==1,"paired arms mismatch")
        rows.append(dict(pair_id=key,map_id=c["map_id"],success={a:int(r["success"]) for a,r in matched.items()},
                         generated={a:r["generated"] for a,r in matched.items()}))
    contrasts={a:contrast(rows,a,cfg["bootstrap"],cfg["master_seed"]) for a in ("frozen","uniform")}
    common=[r for r in rows if r["success"]["paired"] and r["success"]["frozen"]]
    den=sum(r["generated"]["frozen"] for r in common)
    ratio=sum(r["generated"]["paired"] for r in common)/den if den else None
    censored=sum(r["stop"] in {"wall_safety","incomplete_pp","external_timeout"} for r in episodes)
    summary={a:dict(success=sum(r["success"][a] for r in rows),episodes=len(rows),
        generated=sum(r["generated"][a] for r in rows),decisions=sum(r["decisions"] for r in episodes if r["arm"]==a),
        physical_revisits=sum(r["physical_revisits"] for r in episodes if r["arm"]==a),
        rollbacks=sum(r["rollbacks"] for r in episodes if r["arm"]==a)) for a in cfg["arms"]}
    report=dict(binding=plan["binding"],summary=summary,contrasts=contrasts,common_success_count=len(common),
        common_success_generated_ratio=ratio,censored=censored,gate=gate(contrasts,ratio,censored,cfg),
        paired_cases=rows,episodes=episodes,no_ttf=True,production_changed=False)
    verify()
    once(out/"report.json",report)
    return {k:v for k,v in report.items() if k not in ("paired_cases","episodes")}


def main():
    for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
        os.environ[key]="1"
    os.chdir(ROOT)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase",choices=("prepare","freeze","verify-model","generate","qualify","collect","analyze","request-stop"))
    p.add_argument("--resume",action="store_true")
    p.add_argument("--limit",type=int)
    a=p.parse_args()
    if a.phase=="collect":
        r=collect(a.resume,a.limit)
    elif a.phase=="request-stop":
        plan,out=verify()
        atomic(out/"STOP_AFTER_BATCH",dict(binding=plan["binding"],requested=True))
        r=dict(stop_after_current_batch=True)
    else:
        r={"prepare":prepare,"freeze":freeze_model,"verify-model":verify_model,"generate":generate,
           "qualify":qualify,"analyze":analyze}[a.phase]()
    print(json.dumps(r,indent=2),flush=True)


if __name__=="__main__":
    main()
