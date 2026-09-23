"""One fixed curriculum fork; original-task evaluation never updates the model."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import shutil
import sys

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts import train_sa_parent_update as old
from scripts import probe_sa_terminal_transition as transition
from scripts.compare_sa_fresh_stream import registry_streams, stream_key
from scripts.register_sa_crossfit_report_fix import first_difference

run, batch, compare, recovery = old.run, old.collection, transition.compare, old.recovery
CONFIG = "configs/sa_curriculum_update.json"
REGISTRATION = "training_registration.json"
EVALUATION_REGISTRATION = "evaluation_registration.json"
ARMS = ("uncapped_condition", "second_condition", "curriculum_condition")
CODE = (CONFIG, "scripts/run_sa_curriculum_update.py", "tests/evaluation/test_sa_curriculum_update.py",
        "docs/SA_CURRICULUM_UPDATE_PROTOCOL_ZH.md", "scripts/train_sa_parent_update.py",
        "experiments/sa_parent_update.py", "experiments/sa_uncapped_training_contract.py",
        "scripts/run_sa_crossfit_update.py", "experiments/sa_onpolicy_actor.py")


def config():
    cfg = run.read_json(ROOT/CONFIG)
    required = dict(expected_episodes=144, parent_iteration=1, output_iteration=2, output_arm=ARMS[-1],
        maximum_updates=1, workers=20, max_decisions=None, decision_feature_reference=256, node_budget=25000000,
        pp_safety_seconds=20., episode_safety_seconds=900., process_fuse_seconds=960.,
        evaluation_phase="train-1-new-conditions", evaluation_replicas=[4, 5], evaluation_jobs=72,
        formal_ttf=False, automatic_promotion=False)
    run.require(all(cfg[k] == v for k,v in required.items()), "fixed curriculum scope")
    run.require(cfg["update"] == run.read_json(ROOT/old.CONFIG)["update"], "same update as old actor-2")
    return cfg


def checked_entries(entries, policy, train_maps):
    run.require(len(entries) == len({e["job_id"] for e in entries}) == 144, "complete unique training episodes")
    run.require(Counter(e["origin"] for e in entries) == dict(original=96, transition=48), "all data, not successes only")
    for e in entries:
        r = e["episode"]
        run.require(r["split"] == "train" and r["map_id"] in train_maps and r["policy_sha256"] == policy,
                    "Train/current-policy only")
        run.require(old.credit.terminal_return(r,max_decisions=None,node_budget=25000000) is not None,
                    "no censored training labels")
    groups = {e["episode"]["pair_id"]:e["episode"]["map_id"] for e in entries}
    run.require(len(groups) == 36 and set(groups.values()) == set(train_maps), "six maps and 36 conditions")
    run.require(Counter(groups.values()) == {m:6 for m in train_maps}, "map-equal curriculum coverage")
    return groups


def evidence():
    cfg = config()
    ocfg, oreg, source, oreport, actor, original = old.evidence()
    treg, tsource, tout, _ = transition.verify()
    run.require(source["binding"] == tsource["binding"], "same scientific source")
    run.require(run.sha256_file(tout/"report.json") == cfg["transition_report_sha256"], "transition report")
    tr = run.check_seal(run.read_json(tout/"report.json"))
    jobs, proof = transition.completed(treg, source, tout)
    audit = run.check_seal(run.read_json(tout/"audit.json"))
    run.require(audit["binding"] == tr["binding"] == treg["binding"] and len(audit["results"]) == 48 and
        all(r["status"] == "ok" for r in audit["results"]) and
        {r["job_id"]:r["result_sha256"] for r in audit["results"]} == proof["files"] and
        tr["audit_sha256"] == run.sha256_file(tout/"audit.json"), "audited transition data")
    entries = [dict(e,origin="original") for e in original]
    for j in jobs:
        folder = compare.prior.folder_for(treg["config"],j)
        row = compare.read_result(treg,source,j)
        entries.append(dict(job_id=j["job_id"],folder=folder.relative_to(ROOT).as_posix(),
            result_sha256=proof["files"][j["job_id"]],episode=row,origin="transition"))
    checked_entries(entries,run.validate_bundle(actor),source["split"]["train_maps"])
    return cfg,source,actor,sorted(entries,key=lambda e:e["job_id"]),oreg,treg,ocfg


def merged_credits(entries,original,transition_rows):
    sources={"original":{r["episode_id"]:r for r in original},
             "transition":{r["episode_id"]:r for r in transition_rows}}
    run.require(len(sources["original"])==96 and len(sources["transition"])==48,"source credit counts")
    result=[]
    for e in entries:
        c=sources[e["origin"]][e["job_id"]]
        run.require(c["return_value"]==float(e["episode"]["success"]) and
            c["episode_weight"]==1/len(sources[e["origin"]]),"source credit alignment")
        result.append(dict(c,episode_weight=1/144,coefficient=(c["return_value"]-c["baseline"])/144))
    return sorted(result,key=lambda c:c["episode_id"])


def prepare():
    cfg,source,parent,entries,oreg,treg,ocfg = evidence()
    out = ROOT/cfg["output"]
    run.require(not out.exists(), "existing training; inspect, do not overwrite")
    inputs = {n:run.sha256_file(ROOT/n) for n in CODE}
    inputs.update({s["path"]:s["file_sha256"] for s in oreg["config"]["models"].values()})
    for folder,names in ((ROOT/oreg["config"]["output"],("registration.json","report.json","audit.json","collection.complete.json")),
        (ROOT/treg["config"]["output"],(transition.REGISTRATION,"report.json","audit.json","collection.complete.json")),
        (ROOT/ocfg["output"],("training_registration.json","update.json","parity.json","models/actor-2.json"))):
        for n in names:
            p=folder/n
            inputs[p.relative_to(ROOT).as_posix()] = run.sha256_file(p)
    for e in entries:
        for n in ("result.json",*e["episode"]["files"]):
            p=ROOT/e["folder"]/n
            inputs[p.relative_to(ROOT).as_posix()] = run.sha256_file(p)
    original=run.read_json(ROOT/oreg["config"]["output"]/"report.json")["training_information"]["coefficients"]
    transitioned=run.read_json(ROOT/treg["config"]["output"]/"report.json")["terminal_information"]["coefficients"]
    body = dict(schema=cfg["schema"],config=cfg,inputs=inputs,entries=entries,credits=merged_credits(entries,original,transitioned),
        parent_policy=run.validate_bundle(parent),
        scientific_binding=source["binding"],source_bindings=[oreg["binding"],treg["binding"]],
        source_commit=run.subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        no_heldout=True,no_ttf=True,original_task_evaluation_not_training=True)
    body["binding"] = run.json_fingerprint(body)
    with recovery.strict_lock(out,body["binding"],"curriculum-prepare"):
        run.publish_actor(out,source,parent,dict(copied_not_trained=True,registration_binding=body["binding"]))
        run.once(out/REGISTRATION,run.sealed(body))
    return dict(registered=True,episodes=144,binding=body["binding"],maximum_updates=1)


def pins(reg):
    for n,h in reg["inputs"].items():
        run.require(run.sha256_file(run.contained_file(ROOT,n,field="curriculum input")) == h,"changed input: "+n)


def verify():
    cfg,source,parent,entries,oreg,treg,_ = evidence()
    out=ROOT/cfg["output"]
    reg=run.check_seal(run.read_json(out/REGISTRATION))
    run.require(reg["binding"] == run.json_fingerprint({k:v for k,v in reg.items() if k not in ("binding","integrity")}),"registration hash")
    run.require(reg["config"] == cfg and reg["entries"] == entries and reg["parent_policy"] == run.validate_bundle(parent) and
        reg["source_bindings"] == [oreg["binding"],treg["binding"]] and reg["scientific_binding"] == source["binding"],"registration lineage")
    pins(reg)
    run.require(run.actor_load(out,source,1) == parent,"parent copy")
    return reg,source,out,parent


def extract():
    reg,source,out,parent=verify()
    with recovery.strict_lock(out,reg["binding"],"curriculum-extract"):
        run.require(not (out/"cache").exists(),"partial/existing cache needs inspection")
        jobs=[dict(e,plan=source,actor=parent,output=reg["config"]["output"]) for e in reg["entries"]]
        rows=[]
        with ProcessPoolExecutor(max_workers=20) as pool:
            for row in pool.map(old.cache_tools.extract_worker,jobs):
                rows.append(row)
                if len(rows)%12 == 0:print(f"Extracted {len(rows)}/144",flush=True)
        groups=checked_entries(reg["entries"],reg["parent_policy"],source["split"]["train_maps"])
        credits=old.credit.gradient_coefficients([dict(r["episode"],steps=r["steps"]) for r in rows],
            policy_sha256=reg["parent_policy"],expected_groups=groups,replicas=4,max_decisions=None,node_budget=25000000)
        credits.sort(key=lambda r:r["episode_id"])
        run.require(len(credits)==len(reg["credits"]) and all(
            {k:v for k,v in c.items() if k!="coefficient"}=={k:v for k,v in expected.items() if k!="coefficient"} and
            abs(c["coefficient"]-expected["coefficient"])<=1e-15
            for c,expected in zip(credits,reg["credits"])),"independent source credit recomputation")
        by_id={r["episode_id"]:r for r in credits}
        for r in rows:
            r.pop("steps")
            r["credit"]=by_id[r["episode"]["episode_id"]]
        run.once(out/"cache.json",run.sealed(dict(binding=reg["binding"],rows=rows,credits=credits)))
    return dict(extracted=144,nonzero_episodes=sum(r["coefficient"]!=0 for r in credits),no_solver=True)


def cached(reg,out):
    cache=run.check_seal(run.read_json(out/"cache.json"))
    entries={e["job_id"]:e for e in reg["entries"]}
    run.require(cache["binding"] == reg["binding"] and len(cache["rows"]) == 144 and
        {r["episode"]["job_id"] for r in cache["rows"]} == set(entries),"complete cache")
    credits={c["episode_id"]:c for c in cache["credits"]}
    run.require(len(credits)==144,"complete coefficients")
    for c,expected in zip(cache["credits"],reg["credits"]):
        run.require({k:v for k,v in c.items() if k!="coefficient"}=={k:v for k,v in expected.items() if k!="coefficient"} and
            abs(c["coefficient"]-expected["coefficient"])<=1e-15,"registered credit")
    for r in cache["rows"]:
        e=entries[r["episode"]["job_id"]]
        run.require(r["episode"] == e["episode"] and r["source_sha256"] == e["result_sha256"] and
            r["credit"] == credits[e["job_id"]] and run.sha256_file(ROOT/r["cache"]) == r["sha256"],"cache identity")
        with np.load(ROOT/r["cache"],allow_pickle=False) as data:d=dict(data)
        run.require(len(d["lengths"]) == e["episode"]["decisions"],"no truncated trajectory")
        yield r,d,e["origin"]


def torch_environment(parent):
    import torch
    torch.set_num_threads(1)
    run.require(torch.__version__ == parent["training_library"],"frozen Torch version")
    return torch


def train():
    reg,source,out,parent=verify()
    torch=torch_environment(parent)
    with recovery.strict_lock(out,reg["binding"],"curriculum-single-update"):
        run.require(not (out/"update.json").exists() and not run.actor_file(out,2).exists(),"one update only")
        model=old.torch_actor(parent)
        gradients={k:[torch.zeros_like(p) for p in model.parameters()] for k in ("original","transition")}
        packs=[]
        max_error=0.
        for i,(r,data,origin) in enumerate(cached(reg,out)):
            padded,prior=old.padded_with_prior(data,parent)
            logp=old.log_distribution(model,parent,padded,prior)
            error=float(np.max(np.abs(logp.exp().detach().numpy()-padded[1])))
            max_error=max(max_error,error)
            run.require(error<=1e-12,"all parent probabilities reproduce")
            selected=logp[torch.arange(len(data["lengths"])),torch.as_tensor(data["selected"],dtype=torch.long)]
            raw=torch.autograd.grad(-r["credit"]["coefficient"]*selected.sum(),tuple(model.parameters()))
            for total,value in zip(gradients[origin],raw):total+=value
            packs.append(dict(padded=padded,prior=prior,weight=r["credit"]["episode_weight"],
                coefficient=r["credit"]["coefficient"],draws=data["draws"],lengths=data["lengths"],selected=data["selected"]))
            if (i+1)%12 == 0:print(f"Gradient {i+1}/144",flush=True)
        gradient=[a+b for a,b in zip(gradients["original"],gradients["transition"])]
        # Verify the complete accumulated derivative, not only a single sampled state.
        def loss():
            value=0.
            for p in packs:
                logp=old.log_distribution(model,parent,p["padded"],p["prior"])
                value+=float((-p["coefficient"]*logp[torch.arange(len(p["lengths"])),
                    torch.as_tensor(p["selected"],dtype=torch.long)].sum()).detach())
            return value
        differences=[]
        for parameter,g in zip(model.parameters(),gradient):
            index=int(g.abs().argmax()); flat=parameter.view(-1); original=float(flat[index].detach())
            try:
                with torch.no_grad():flat[index]=original+1e-6
                plus=loss()
                with torch.no_grad():flat[index]=original-1e-6
                minus=loss()
            finally:
                with torch.no_grad():flat[index]=original
            actual=(plus-minus)/2e-6; expected=float(g.flatten()[index])
            run.require(abs(actual-expected)<=1e-6*max(1.,abs(expected)),"complete gradient finite difference")
            differences.append(dict(index=index,analytic=expected,numerical=actual,error=abs(actual-expected)))
        a,b=(torch.cat([g.flatten() for g in gradients[k]]) for k in ("original","transition"))
        norms={k:float(v.norm()) for k,v in (("original",a),("transition",b))}
        cosine=float(torch.dot(a,b)/(a.norm()*b.norm())) if all(norms.values()) else None
        norm=sum(float((g*g).sum()) for g in gradient)**.5
        bundle,diag=(None,dict(decision="zero_gradient_no_update")) if norm==0 else old.guarded_parent_update(
            parent,gradient,packs,reg["config"]["update"],reg["binding"],ARMS[-1])
        body=dict(binding=reg["binding"],parent_policy=reg["parent_policy"],updated=bundle is not None,
            diagnostic=diag,gradient_by_source=norms,gradient_cosine=cosine,finite_differences=differences,
            replay_probability_max_error=max_error,cache_sha256=run.sha256_file(out/"cache.json"),
            no_ttf=True,no_heldout=True,automatic_promotion=False)
        if bundle is not None:
            run.require(bundle["iteration"]==2 and bundle["parent_policy"]==reg["parent_policy"],"sibling lineage")
            run.publish_actor(out,source,bundle,body)
            body.update(model_sha256=run.sha256_file(run.actor_file(out,2)),policy_sha256=run.validate_bundle(bundle))
        run.once(out/"update.json",run.sealed(body))
    return body


def parity(torch_side=False):
    reg,source,out,parent=verify()
    update=run.check_seal(run.read_json(out/"update.json"))
    run.require(update["binding"]==reg["binding"] and update["updated"] and
        update["cache_sha256"]==run.sha256_file(out/"cache.json"),"verified update/cache")
    bundle=run.actor_load(out,source,2)
    run.require(run.sha256_file(run.actor_file(out,2))==update["model_sha256"] and
        run.validate_bundle(bundle)==update["policy_sha256"] and bundle["parent_policy"]==reg["parent_policy"],"updated identity")
    actor=old.NumpyActor(bundle)
    fixtures=[]
    if torch_side:
        torch=torch_environment(parent); model=old.torch_actor(bundle)
        for e in reg["entries"]:
            for s in run.trace_read(ROOT/e["folder"]):
                if s["decision"] not in {0,32,128,255,256,e["episode"]["decisions"]-1}:continue
                ids=s["candidate_ids"]
                with torch.no_grad():probs=old.torch_distribution(model,bundle,ids,s["anchor_id"],s["features"]).probs.numpy().tolist()
                fixtures.append(dict(job_id=e["job_id"],decision=s["decision"],candidate_ids=ids,
                    anchor_id=s["anchor_id"],features=s["features"],probabilities=dict(zip(ids,probs))))
    else:
        run.native_runtime(source)
        ref=run.check_seal(run.read_json(out/"torch-parity.json"))
        run.require(ref["binding"]==reg["binding"] and ref["update_sha256"]==run.sha256_file(out/"update.json") and
            ref["policy_sha256"]==actor.sha,"Torch reference identity")
        fixtures=ref["fixtures"]
    error=0.
    for f in fixtures:
        actual=actor.probabilities(f["candidate_ids"],f["anchor_id"],f["features"])
        error=max(error,max(abs(actual[k]-f["probabilities"][k]) for k in actual))
        run.require(error<=1e-12,"portable probabilities")
        for i in range(16):
            draw=run.random.Random(i).random()
            run.require(run.select_with_draw(actual,draw)==run.select_with_draw(f["probabilities"],draw),"portable selections")
    run.require({f["job_id"] for f in fixtures}=={e["job_id"] for e in reg["entries"]},"all episodes parity")
    body=dict(binding=reg["binding"],policy_sha256=actor.sha,update_sha256=run.sha256_file(out/"update.json"),
        fixture_count=len(fixtures),max_error=error,choices_checked=16*len(fixtures))
    with recovery.strict_lock(out,reg["binding"],"curriculum-parity"):
        run.once(out/("torch-parity.json" if torch_side else "parity.json"),run.sealed(dict(body,fixtures=fixtures) if torch_side else body))
    return body


def schedule(roots,cfg):
    run.require(len(roots)==12 and len({r["case"]["map_id"] for r in roots})==6,"all original high-load conditions")
    run.require(all(r["case"]["task_variant"]=="bottleneck_d25" and
        r["case"]["task_id"]==r["case"]["map_id"]+"__task_0001" and r["solver_seed"] in (233,239) for r in roots),"unreduced tasks")
    jobs=[dict(r,split="train_diagnostic_evaluation",phase=cfg["evaluation_phase"],replica=k,
        comparison_arm=a,arm="trained_actor",iteration=1 if a==ARMS[0] else 2,
        job_id=run.json_fingerprint([cfg["evaluation_phase"],r["pair_id"],k,a])[:24])
        for r in roots for k in cfg["evaluation_replicas"] for a in ARMS]
    run.require(len(jobs)==len({j["job_id"] for j in jobs})==72,"full schedule")
    return jobs


def inventory(jobs,out):
    keys={stream_key(j) for j in jobs}; inputs={}
    run.require(len(keys)==24,"24 paired streams")
    for p in sorted((ROOT/"build").glob("sa-*/*registration.json")):
        if p.parent==out:continue
        old_keys=set(registry_streams(run.read_json(p)))
        run.require(not keys & old_keys,"already used streams: "+p.parent.name)
        if old_keys:inputs[p.relative_to(ROOT).as_posix()]=run.sha256_file(p)
    return inputs


def prepare_evaluation():
    treg,source,tout,parent=verify()
    proof=run.check_seal(run.read_json(tout/"parity.json"))
    update=run.check_seal(run.read_json(tout/"update.json"))
    run.require(proof["binding"]==update["binding"]==treg["binding"] and update["updated"] and
        proof["update_sha256"]==run.sha256_file(tout/"update.json") and proof["max_error"]<=1e-12,"verified new model")
    oreg,_,src=batch.verify()
    unique={j["pair_id"]:j for j in batch.ready_jobs(oreg,source,src) if j["case"]["task_variant"]=="bottleneck_d25"}
    roots=[{k:j[k] for k in ("case","pair_id","solver_seed","expected_initial")} for _,j in sorted(unique.items())]
    cfg=dict(treg["config"],output=treg["config"]["evaluation_output"],arms=list(ARMS),models={})
    out=ROOT/cfg["output"]
    run.require(not out.exists(),"existing evaluation")
    parent_out=ROOT/run.read_json(ROOT/old.CONFIG)["output"]
    old_update=run.check_seal(run.read_json(parent_out/"update.json"))
    old_parity=run.check_seal(run.read_json(parent_out/"parity.json"))
    run.require(old_update["updated"] and old_parity["update_sha256"]==run.sha256_file(parent_out/"update.json") and
        old_update["parent_policy"]==run.validate_bundle(parent),"old actor-2 ablation")
    paths={ARMS[0]:run.actor_file(tout,1),ARMS[1]:run.actor_file(parent_out,2),ARMS[2]:run.actor_file(tout,2)}
    bundles={a:run.read_json(p) for a,p in paths.items()}
    for a,b in bundles.items():
        cfg["models"][a]=dict(path=paths[a].relative_to(ROOT).as_posix(),file_sha256=run.sha256_file(paths[a]),
            policy_sha256=run.validate_bundle(b),iteration=b["iteration"])
    run.require(cfg["models"][ARMS[2]]["policy_sha256"]==proof["policy_sha256"]==update["policy_sha256"] and
        cfg["models"][ARMS[2]]["file_sha256"]==update["model_sha256"] and
        cfg["models"][ARMS[1]]["file_sha256"]==old_update["model_sha256"],"both child identities")
    jobs=schedule(roots,cfg)
    inputs=dict(treg["inputs"])
    inputs.update(inventory(jobs,out))
    for p in (tout/REGISTRATION,tout/"update.json",tout/"parity.json",*paths.values()):
        inputs[p.relative_to(ROOT).as_posix()]=run.sha256_file(p)
    body=dict(config=cfg,inputs=inputs,jobs=jobs,training_binding=treg["binding"],scientific_binding=source["binding"],
        no_training=True,no_ttf=True,role="unreduced_seen_train_tasks_new_streams_not_generalization")
    body["binding"]=run.json_fingerprint(body)
    with recovery.strict_lock(out,body["binding"],"curriculum-eval-prepare"):
        for a,b in bundles.items():
            p=compare.runtime_plan(source,cfg,a); target=run.actor_file(ROOT/p["config"]["output"],b["iteration"])
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(paths[a],target)
            run.require(run.sha256_file(target)==cfg["models"][a]["file_sha256"],"model copy")
            run.once(target.parent/f"receipt-{b['iteration']}.json",run.sealed(dict(binding=source["binding"],
                policy_sha256=run.validate_bundle(b),file_sha256=run.sha256_file(target),metadata=dict(copied_not_trained=True))))
        run.once(out/EVALUATION_REGISTRATION,run.sealed(body))
    return dict(episodes=72,paired_streams=24,workers=20,max_decisions=None,binding=body["binding"])


def verify_evaluation():
    treg,source,_,_=verify()
    out=ROOT/treg["config"]["evaluation_output"]
    reg=run.check_seal(run.read_json(out/EVALUATION_REGISTRATION))
    run.require(reg["binding"]==run.json_fingerprint({k:v for k,v in reg.items() if k not in ("binding","integrity")}) and
        reg["training_binding"]==treg["binding"] and reg["scientific_binding"]==source["binding"],"evaluation registration")
    pins(reg);inventory(reg["jobs"],out)
    for a,s in reg["config"]["models"].items():
        p=compare.runtime_plan(source,reg["config"],a)
        run.require(run.actor_load(ROOT/p["config"]["output"],p,s["iteration"])==
            compare.prior.check_model(reg["config"],a,source["binding"]),"evaluation copy identity")
    return reg,source,out


def interpretation(comparisons,hard_gains):
    gains=[c["overall"]["net_success_bounds"][0] for c in comparisons.values()]
    if all(v>0 for v in gains):return "local_curriculum_net_gain_needs_independent_confirmation"
    if any(v>0 for v in hard_gains):return "localized_hard_task_tradeoff_do_not_promote"
    return "no_consistent_curriculum_completion_gain_do_not_promote"


def evaluate(resume=False):
    from experiments.repair_collection import _run_jobs
    reg,source,out=verify_evaluation()
    jobs=reg["jobs"];cfg=reg["config"]
    with recovery.strict_lock(out,reg["binding"],"curriculum-evaluation"):
        done=batch.execute_batches(reg,out,[compare.augmented(reg,source,j) for j in jobs],compare.worker,
            lambda j:compare.prior.folder_for(cfg,j),lambda j:compare.read_result(reg,source,j),"collection",resume,960.)
        if not done:return dict(status="paused_or_needs_inspection")
        complete=run.check_seal(run.read_json(out/"collection.complete.json"))
        audit=_run_jobs(transition.runtime.audit_worker,[compare.augmented(reg,source,j) for j in jobs],20,
            phase="curriculum-eval-audit",output_root=out/"audit-progress",run_fingerprint=reg["binding"],timeout_seconds=960.)
        run.require(len(audit)==72 and all(r["status"]=="ok" for r in audit) and
            {r["job_id"]:r["result_sha256"] for r in audit}==complete["files"],"full evaluation audit")
        run.once(out/"audit.json",run.sealed(dict(binding=reg["binding"],results=audit)))
        data={a:{} for a in ARMS}; folders={a:{} for a in ARMS}
        for j in jobs:
            r=compare.read_result(reg,source,j);key=(r["pair_id"],r["replica"])
            data[j["comparison_arm"]][key]=r;folders[j["comparison_arm"]][key]=compare.prior.folder_for(cfg,j)
        run.require(all(len(v)==24 and set(v)==set(data[ARMS[0]]) for v in data.values()),"paired denominator")
        for key in data[ARMS[0]]:
            for field in ("initial_fingerprint","rng_stream_id","map_id"):
                run.require(len({v[key][field] for v in data.values()})==1,"paired "+field)
        maps=sorted({r["map_id"] for r in data[ARMS[0]].values()})
        summaries={a:dict(success=sum(r["success"] for r in rows.values()),episodes=24,
            by_map={m:sum(r["success"] for r in rows.values() if r["map_id"]==m) for m in maps},
            stops=dict(Counter(r["stop"] for r in rows.values()))) for a,rows in data.items()}
        comparisons={a:dict(overall=compare.prior.paired_comparison(data[ARMS[-1]],data[a]),by_map={m:
            compare.prior.paired_comparison({k:r for k,r in data[ARMS[-1]].items() if r["map_id"]==m},
                {k:r for k,r in data[a].items() if r["map_id"]==m}) for m in maps}) for a in ARMS[:-1]}
        behavior=[]
        for key in sorted(data[ARMS[0]]):
            count,first=first_difference(run.trace_read(folders[ARMS[-1]][key]),run.trace_read(folders[ARMS[0]][key]))
            if first is None:run.require(data[ARMS[-1]][key]["final_fingerprint"]==data[ARMS[0]][key]["final_fingerprint"],"unexplained divergence")
            behavior.append(dict(pair_id=key[0],replica=key[1],common_prefix=count,first_difference=first))
        hard="sa_linear_v1_m04_station_centric_0000"
        gains=[c["by_map"][hard]["net_success_bounds"][0] for c in comparisons.values()]
        body=dict(binding=reg["binding"],summaries=summaries,comparisons=comparisons,behavior=behavior,
            episodes=[dict(comparison_arm=a,**r) for a,rows in data.items() for r in rows.values()],
            decision=interpretation(comparisons,gains),audit_sha256=run.sha256_file(out/"audit.json"),
            no_training=True,no_ttf=True,independent_generalization=False,automatic_promotion=False)
        artifact=run.sealed(body);run.check_seal(json.loads(json.dumps(artifact)))
        run.once(out/"report.json",artifact)
        run.write_json(out/"run_status.json",dict(status="completed",report_sha256=run.sha256_file(out/"report.json")))
    return dict(summaries=summaries,comparisons={a:c["overall"] for a,c in comparisons.items()},decision=body["decision"])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase",choices=("prepare","verify","extract","train","parity","prepare-evaluation","dry-run","evaluate","stop"))
    p.add_argument("--torch",action="store_true");p.add_argument("--resume",action="store_true")
    args=p.parse_args()
    if args.phase=="parity":result=parity(args.torch)
    elif args.phase=="prepare-evaluation":result=prepare_evaluation()
    elif args.phase=="dry-run":
        reg,_,_=verify_evaluation();result=dict(jobs=len(reg["jobs"]),waves=4,workers=20,max_decisions=None,safety_seconds=3840,no_ttf=True)
    elif args.phase=="evaluate":result=evaluate(args.resume)
    elif args.phase=="verify":result=dict(verified=True,binding=verify()[0]["binding"])
    elif args.phase=="stop":
        run.write_json(ROOT/config()["evaluation_output"]/"STOP_AFTER_BATCH",dict(requested=True));result=dict(stop_after_current_batch=True)
    else:result=globals()[args.phase]()
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":main()
