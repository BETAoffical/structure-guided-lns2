"""Bounded episodic on-policy prototype. Never a formal TTF runner."""

import argparse
import gzip
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments._common import contained_file, json_fingerprint, read_json, sha256_file, write_json
from experiments.sa_paired_completion import require
from experiments.sa_onpolicy_actor import NumpyActor, initial_bundle, select_with_draw, update_once, validate_bundle
from experiments.sa_onpolicy_contract import gradient_coefficients, terminal_return
from experiments.sa_policy_aligned_update import budget_features, work_stop
from scripts.check_sa_onpolicy_contract import inspect as design_check
from scripts.run_sa_paired_closed_loop import once, sealed, check_seal, features_for

CONFIG = "configs/sa_onpolicy_execution.json"


def prepare():
    design_check()
    cfg = read_json(ROOT / CONFIG)
    require(cfg["formal_ttf"] is False and cfg["automatic_promotion"] is False, "scope")
    design = read_json(ROOT / cfg["design"])
    fixed = design["proposal"]
    require((fixed["maximum_updates"], fixed["learning_rate"], fixed["gradient_norm_clip"],
             fixed["optimizer_steps_per_batch"], fixed["max_decisions"], fixed["node_budget"]) ==
            (2, .001, 1., 1, 256, 25000000), "registered training budget changed")
    out = ROOT / cfg["output"]
    require(not out.exists(), "existing run: use verify/resume, do not overwrite")
    old = read_json(ROOT / design["source_plan"])
    require(sha256_file(ROOT / cfg["source_cases"]) == cfg["source_cases_sha256"], "source cases changed")
    source_cases = check_seal(read_json(ROOT / cfg["source_cases"]))
    cases = old["cases"]
    require({c["task_id"]: c["files"] for c in cases} ==
            {c["task_id"]: c["files"] for c in source_cases["cases"]}, "case identity")
    native = old["source_plan"]["native_file"]
    require(sha256_file(ROOT / native) == cfg["native_sha256"], "frozen native changed")
    template_path = old["source_plan"]["config"]["runtime_registration"]
    template = read_json(ROOT / template_path)["template"]
    require(template["environment"]["max_repair_iterations"] == 0, "hidden repair cap")
    inputs = {CONFIG: sha256_file(ROOT / CONFIG), cfg["design"]: sha256_file(ROOT / cfg["design"]),
              design["source_plan"]: design["source_plan_sha256"],
              cfg["source_cases"]: cfg["source_cases_sha256"], native: cfg["native_sha256"],
              template_path: sha256_file(ROOT / template_path)}
    for c in cases:
        for name in c["files"].values():
            path = contained_file(ROOT, name, field="MAPF input")
            require(sha256_file(path) == source_cases["files"][name], "MAPF input changed")
            inputs[name] = sha256_file(path)
    for folder in ("experiments", "scripts", "lns2_selector"):
        for path in (ROOT / folder).rglob("*.py"):
            inputs[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    for path in (ROOT / "artifacts/initlns-closed-loop-controller-v2").glob("*.json"):
        inputs[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    for name in ("docs/SA_ONPOLICY_EXECUTION_ZH.md", "tests/evaluation/test_sa_onpolicy_runtime.py"):
        inputs[name] = sha256_file(ROOT / name)
    plan = dict(schema="lns2.sa.onpolicy_plan.v1", config=cfg, proposal=design["proposal"],
                inputs=inputs, cases=cases, split=old["split"], native_file=native, template=template,
                source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                role="viewed_development_only", no_ttf=True)
    plan["binding"] = json_fingerprint(plan)
    once(out / "plan.json", plan)
    return dict(prepared=True, qualification=32, maximum_train_episodes=192, evaluation=64, no_ttf=True)


def verify():
    cfg = read_json(ROOT / CONFIG)
    out = ROOT / cfg["output"]
    plan = read_json(out / "plan.json")
    require(plan["config"] == cfg and plan["binding"] == json_fingerprint({k: v for k, v in plan.items() if k != "binding"}), "plan binding")
    for name, digest in plan["inputs"].items():
        require(sha256_file(contained_file(ROOT, name, field="registered input")) == digest, "changed input: " + name)
    return plan, out


def native_runtime(plan):
    sys.path.insert(0, str((ROOT / plan["native_file"]).parent))
    from scripts import run_sa_path_quality as q
    require(q.native_identity()["sha256"] == plan["config"]["native_sha256"], "loaded wrong native")
    return q


def reset(job):
    q = native_runtime(job["plan"])
    case, plan = job["case"], job["plan"]
    item = dict(task_id=case["task_id"], solver_seed=job["solver_seed"], controller="dual16_sa", budget_seconds=100000.)
    mapped = q.worker_job(case, item, plan["template"], ROOT / plan["config"]["output"] / "unused", plan["binding"])
    require(mapped["environment"]["max_repair_iterations"] == 0, "hidden native step limit")
    env = q._make_environment(mapped["dataset_root"], mapped["row"], mapped["environment"], "Adaptive")
    state = q._plain(env.reset(seed=job["solver_seed"]))
    require(state["initial_solution_complete"], "incomplete initial PP")
    ctx = dict(case_id=mapped["sa_case_id"], task_id=case["task_id"],
               solver_seed=job["solver_seed"], proposal=mapped["sa_proposal"])
    return q, env, state, ctx


def paired_conditions(plan, split=None):
    result = []
    for case in plan["cases"]:
        role = "train" if case["map_id"] in plan["split"]["train_maps"] else "development_holdout"
        if split is not None and role != split:
            continue
        for seed in case["solver_seeds"]:
            result.append(dict(case=case, solver_seed=seed, split=role, pair_id=f"{case['task_id']}-s{seed}"))
    return sorted(result, key=lambda j: j["pair_id"])


def stream_draw(plan, phase, pair, replica, decision, purpose):
    key = json_fingerprint([plan["config"]["stream_seed"], phase, pair, replica, decision, purpose])
    seed = int(key[:16], 16)
    return seed & 0x7fffffff if purpose == "pp" else random.Random(seed).random()


def actor_file(out, iteration):
    return out / "models" / f"actor-{iteration}.json"


def actor_load(out, plan, iteration):
    receipt = check_seal(read_json(out / "models" / f"receipt-{iteration}.json"))
    require(receipt["binding"] == plan["binding"], "model receipt")
    path = actor_file(out, iteration)
    require(sha256_file(path) == receipt["file_sha256"], "model bytes changed")
    bundle = read_json(path)
    require(bundle["binding"] == plan["binding"] and bundle["iteration"] == iteration, "model iteration/binding")
    require(validate_bundle(bundle) == receipt["policy_sha256"], "model semantics changed")
    return bundle


def publish_actor(out, plan, bundle, metadata):
    i = bundle["iteration"]
    path = actor_file(out, i)
    once(path, bundle)
    once(out / "models" / f"receipt-{i}.json", sealed(dict(binding=plan["binding"], policy_sha256=validate_bundle(bundle),
         file_sha256=sha256_file(path), metadata=metadata)))


def folder_for(out, phase, job):
    return out / phase / job["job_id"]


def result_read(folder, plan):
    row = check_seal(read_json(folder / "result.json"))
    require(row["binding"] == plan["binding"], "result run identity")
    for name, digest in row["files"].items():
        require(sha256_file(contained_file(folder, name, field="episode artifact")) == digest, "changed episode artifact")
    return row


def qualify_worker(job):
    from scripts.train_sa_history_selector import die_with_parent
    from scripts.run_feedback_exploration_diagnostics import validate_final
    from experiments.online_feature_engine import OnlineFeatureEngine
    from experiments.sa_history_selector import History
    die_with_parent(job["parent_pid"])
    q, env, state, ctx = reset(job)
    validate_final(state)
    pool, features, anchor = [], [], None
    if job["split"] == "train" and not state["feasible"]:
        index, pool = q.SingleFullCheckPool(ctx).select(env, state, 0)
        anchor = pool[index]["candidate_id"]
        pool = sorted(pool, key=lambda c: c["candidate_id"])
        base = features_for(state, pool, OnlineFeatureEngine(state, backend="native"), History(state),
                            q.temperature(0), q.state_fingerprint(state))
        features = [budget_features(f, 0, 0, job["plan"]["proposal"]) for f in base]
        require(all(len(f) == 129 for f in features), "initial feature schema")
    folder = folder_for(ROOT / job["plan"]["config"]["output"], job["phase"], job)
    once(folder / "initial.json", state)
    once(folder / "features.json", dict(pool=pool, anchor_id=anchor, features=features))
    row = dict(binding=job["plan"]["binding"], status="ok", job_id=job["job_id"], pair_id=job["pair_id"],
               split=job["split"], initial_fingerprint=q.state_fingerprint(state), initial_conflicts=state["num_of_colliding_pairs"],
               files={n: sha256_file(folder / n) for n in ("initial.json", "features.json")})
    once(folder / "result.json", sealed(row))
    return dict(status="ok", job_id=job["job_id"], conflicts=row["initial_conflicts"])


def selection(env, state, ctx, history, engine, selector, plan, actor):
    from scripts import run_sa_path_quality as q
    index, pool = selector.select(env, state, history.decision)
    anchor = pool[index]["candidate_id"]
    proposal_order = [c["candidate_id"] for c in pool]
    pool = sorted(pool, key=lambda c: c["candidate_id"])
    ids = [c["candidate_id"] for c in pool]
    require(len(ids) == len(set(ids)), "duplicate candidate ID")
    known = {a["id"] for a in state["agents"]}
    require(all(c["agents"] and len(c["agents"]) == len(set(c["agents"])) and set(c["agents"]) <= known for c in pool), "candidate agents")
    generated = state["low_level"]["generated"] - ctx["initial_nodes"]
    fs = features_for(state, pool, engine, history, q.temperature(history.decision), q.state_fingerprint(state))
    features = [budget_features(f, history.decision, generated, plan["proposal"]) for f in fs]
    probabilities = actor.probabilities(ids, anchor, features) if actor else {c: float(c == anchor) for c in ids}
    return dict(pool=pool, proposal_order=proposal_order, candidate_ids=ids, anchor_id=anchor,
                features=features, probabilities=probabilities,
                out_of_range_fraction=actor.out_of_range_fraction(features) if actor else None)


def episode_loop(job, q, env, state, ctx, folder, bundle):
    from scripts.run_feedback_exploration_diagnostics import validate_final
    from experiments.online_feature_engine import OnlineFeatureEngine
    from experiments.sa_history_selector import History
    plan, cfg = job["plan"], job["plan"]["proposal"]
    initial = state
    ctx = dict(ctx, initial_nodes=state["low_level"]["generated"])
    history = History(state)
    actor = NumpyActor(bundle) if job["arm"] in {"trained_actor", "untrained_exploration"} else None
    selector = q.SingleFullCheckPool(ctx) if job["arm"] != "official_sa" else None
    engine = OnlineFeatureEngine(state, backend="native") if selector else None
    once(folder / "initial.json", state)
    began = time.monotonic()
    with gzip.open(folder / "trace.jsonl.gz", "xt", encoding="utf8", compresslevel=3) as stream:
        while True:
            d = history.decision
            used = state["low_level"]["generated"] - ctx["initial_nodes"]
            stop = work_stop(state["feasible"], d, used, cfg)
            if stop:
                break
            if time.monotonic() - began >= cfg["episode_safety_seconds"]:
                stop = "wall_safety"
                break
            event = selection(env, state, ctx, history, engine, selector, plan, actor) if selector else {}
            draw = stream_draw(plan, job["phase"], job["pair_id"], job["replica"], d, "select")
            if selector:
                cid = select_with_draw(event["probabilities"], draw) if actor else event["anchor_id"]
                chosen = next(c for c in event["pool"] if c["candidate_id"] == cid)
                action = dict(mode="explicit_neighborhood", agents=chosen["agents"],
                              random_seed=stream_draw(plan, job["phase"], job["pair_id"], job["replica"], d, "pp"))
                event.update(selected_id=cid, behavior_log_probability=math.log(event["probabilities"][cid]), selection_draw=draw)
            else:
                action = dict(mode="official")
            remaining = cfg["episode_safety_seconds"] - (time.monotonic() - began)
            if remaining <= 0:
                stop = "wall_safety"
                break
            temp = q.temperature(d)
            uniform = stream_draw(plan, job["phase"], job["pair_id"], job["replica"], d, "accept")
            raw = q._plain(env.step_experimental_pp(action, min(cfg["pp_safety_seconds"], remaining), "annealed", temp, uniform))
            after, metrics = raw["observation"], raw["metrics"]
            selected = action.get("agents", metrics["neighborhood"])
            q.validate_transition(state, after, metrics, selected, "annealed", temp, uniform)
            event.update(decision=d, policy_sha256=actor.sha if actor else job["arm"],
                         before=q.state_fingerprint(state), action=action, temperature=temp, uniform=uniform,
                         metrics=metrics, delta=q.encode_state_delta(state, after))
            stream.write(json.dumps(event, separators=(",", ":"), allow_nan=False) + "\n")
            stream.flush()
            history.observe(state, event, after)
            state = after
            if metrics["pp_failure_reason"] == "time_limit" or not metrics["acceptance_evaluated"]:
                stop = "incomplete_pp"
                break
    validate_final(state)
    paths = [a["path"] for a in state["agents"]]
    require(sum(len(p) - 1 for p in paths) == state["sum_of_costs"], "SOC reconstruction")
    once(folder / "final.json", state)
    status = "censored" if stop in {"wall_safety", "incomplete_pp"} else "ok"
    row = dict(binding=plan["binding"], status=status, job_id=job["job_id"], episode_id=job["job_id"],
               pair_id=job["pair_id"], map_id=job["case"]["map_id"], split=job["split"], replica=job["replica"],
               arm=job["arm"], policy_sha256=validate_bundle(bundle) if bundle else job["arm"],
               rng_stream_id=json_fingerprint([plan["config"]["stream_seed"], job["phase"], job["pair_id"], job["replica"]]),
               initial_fingerprint=q.state_fingerprint(initial), final_fingerprint=q.state_fingerprint(state),
               stop=stop, success=state["feasible"], final_conflicts=state["num_of_colliding_pairs"],
               decisions=history.decision, generated=state["low_level"]["generated"] - ctx["initial_nodes"],
               soc=state["sum_of_costs"], makespan=max(len(p) - 1 for p in paths),
               wait_steps=sum(a == b for p in paths for a, b in zip(p, p[1:])),
               seconds_diagnostic=time.monotonic() - began, no_ttf=True,
               files={n: sha256_file(folder / n) for n in ("initial.json", "final.json", "trace.jsonl.gz")})
    terminal_return(row, max_decisions=cfg["max_decisions"], node_budget=cfg["node_budget"])
    once(folder / "result.json", sealed(row))
    return row


def episode_worker(job):
    from scripts.train_sa_history_selector import die_with_parent
    die_with_parent(job["parent_pid"])
    plan = job["plan"]
    out = ROOT / plan["config"]["output"]
    folder = folder_for(out, job["phase"], job)
    require(not folder.exists(), "partial episode requires inspection, never silent retry")
    q, env, state, ctx = reset(job)
    require(q.state_fingerprint(state) == job["expected_initial"], "paired initial mismatch")
    bundle = actor_load(out, plan, job["iteration"]) if job["arm"] in {"trained_actor", "untrained_exploration"} else None
    row = episode_loop(job, q, env, state, ctx, folder, bundle)
    return {k: row[k] for k in ("status", "job_id", "stop", "success", "decisions")}


def jobs_for(plan, phase, iteration=0):
    result = []
    if phase == "qualify":
        return [dict(j, phase=phase, job_id=json_fingerprint([phase, j["pair_id"]])[:24]) for j in paired_conditions(plan)]
    training = phase.startswith("train-")
    if training:
        require(0 <= iteration < plan["proposal"]["maximum_updates"] and phase == f"train-{iteration}", "training round limit")
    else:
        require(phase == "evaluation" and iteration == plan["proposal"]["maximum_updates"], "evaluate final checkpoint only")
    for condition in paired_conditions(plan, "train" if training else "development_holdout"):
        replicas = plan["proposal"]["train_replicas_per_condition" if training else "evaluation_replicas_per_condition"]
        arms = ["trained_actor"] if training else plan["proposal"]["evaluation_arms"]
        for replica in range(replicas):
            for arm in arms:
                result.append(dict(condition, phase=phase, replica=replica, arm=arm,
                                   iteration=0 if arm == "untrained_exploration" else iteration,
                                   job_id=json_fingerprint([phase, condition["pair_id"], replica, arm])[:24]))
    return result


def collect(phase, iteration=0, resume=False):
    from experiments.repair_collection import _run_jobs, _CollectionRunLock
    plan, out = verify()
    if phase != "qualify":
        admission = check_seal(read_json(out / "qualify.complete.json"))
        require(admission["binding"] == plan["binding"], "qualification missing")
        actor_load(out, plan, iteration)
        require(check_seal(read_json(out / f"parity-{iteration}.json"))["binding"] == plan["binding"], "native parity missing")
        require(check_seal(read_json(out / "micro.json"))["binding"] == plan["binding"], "micro validation missing")
    jobs = jobs_for(plan, phase, iteration)
    initials = {}
    if phase != "qualify":
        for j in jobs_for(plan, "qualify"):
            r = result_read(folder_for(out, "qualify", j), plan)
            initials[r["pair_id"]] = r["initial_fingerprint"]
    with _CollectionRunLock(out, plan["binding"], "onpolicy-" + phase):
        require(resume or not (out / phase).exists(), "resume required")
        stop_file = out / "STOP_AFTER_BATCH"
        if resume and stop_file.exists():
            stop_file.unlink()
        pending = []
        for j in jobs:
            folder = folder_for(out, phase, j)
            if (folder / "result.json").exists():
                require(result_read(folder, plan)["status"] == "ok", "censored episode needs audit before continuation")
            else:
                require(not (out / "failures" / (j["job_id"] + ".json")).exists(), "recorded failure needs explicit recovery")
                require(not folder.exists(), "interrupted partial output: audit before retry")
                pending.append(dict(j, plan=plan, parent_pid=os.getpid(), expected_initial=initials.get(j["pair_id"])))
        write_json(out / "run_status.json", dict(status="running", phase=phase, pending=len(pending)))
        try:
            for offset in range(0, len(pending), plan["config"]["workers"]):
                if stop_file.exists():
                    break
                batch = pending[offset:offset + plan["config"]["workers"]]
                def progress(row):
                    with (out / "progress.jsonl").open("a", encoding="utf8") as f:
                        f.write(json.dumps(dict(phase=phase, **row)) + "\n")
                    print(phase, row, flush=True)
                def failure(job, status, error):
                    value = dict(status="censored" if status == "timeout" else "error", job_id=job["job_id"],
                                 stop="external_timeout" if status == "timeout" else "exception", error=error)
                    once(out / "failures" / (job["job_id"] + ".json"), sealed(dict(binding=plan["binding"], **value)))
                    return value
                rows = _run_jobs(qualify_worker if phase == "qualify" else episode_worker, batch,
                                 plan["config"]["workers"], phase=phase, output_root=out / "progress" / f"{phase}-{offset}",
                                 run_fingerprint=plan["binding"], timeout_seconds=plan["proposal"]["process_fuse_seconds"],
                                 on_result=progress, failure_result=failure, stop_on_failure=True)
                if len(rows) != len(batch) or any(r["status"] != "ok" for r in rows):
                    write_json(out / "run_status.json", dict(status="needs_audit", phase=phase, no_update=True))
                    return dict(status="needs_audit", phase=phase, no_update=True)
            complete = all((folder_for(out, phase, j) / "result.json").exists() for j in jobs)
            if complete:
                records = [result_read(folder_for(out, phase, j), plan) for j in jobs]
                require(all(r["status"] == "ok" for r in records), "censored result: update prohibited")
                once(out / f"{phase}.complete.json", sealed(dict(binding=plan["binding"], jobs=len(jobs),
                     files={j["job_id"]: sha256_file(folder_for(out, phase, j) / "result.json") for j in jobs})))
            write_json(out / "run_status.json", dict(status="completed" if complete else "paused", phase=phase))
        except BaseException as exc:
            write_json(out / "run_status.json", dict(status="error", phase=phase, error=repr(exc)))
            raise
    return dict(status="completed" if complete else "paused", phase=phase)


def initialize():
    plan, out = verify()
    require(check_seal(read_json(out / "qualify.complete.json"))["binding"] == plan["binding"], "qualification")
    rows, fixtures = [], []
    for job in jobs_for(plan, "qualify"):
        result_read(folder_for(out, "qualify", job), plan)
        if job["split"] != "train":
            continue
        data = read_json(folder_for(out, "qualify", job) / "features.json")
        rows.extend(data["features"])
        if data["features"]:
            fixtures.append(dict(candidate_ids=[c["candidate_id"] for c in data["pool"]],
                                 anchor_id=data["anchor_id"], features=data["features"]))
    require(not actor_file(out, 0).exists(), "initial actor already exists")
    bundle = initial_bundle(rows, plan["config"]["network_seed"], plan["binding"])
    publish_actor(out, plan, bundle, dict(initial_candidates=len(rows), normalization_split="train"))
    once(out / "fixtures.json", fixtures)
    return dict(policy_sha256=validate_bundle(bundle), parameters=4193, candidate_rows=len(rows))


def trace_read(folder):
    with gzip.open(folder / "trace.jsonl.gz", "rt", encoding="utf8") as f:
        for line in f:
            yield json.loads(line)


def parity(iteration, torch_side=False):
    plan, out = verify()
    bundle = actor_load(out, plan, iteration)
    fixtures = read_json(out / "fixtures.json")
    if torch_side:
        from experiments.sa_onpolicy_actor import torch_actor, torch_distribution
        model = torch_actor(bundle)
        fixture_rows = []
        for row in fixtures:
            order = sorted(range(len(row["candidate_ids"])), key=lambda i: row["candidate_ids"][i])
            ids, fs = [row["candidate_ids"][i] for i in order], [row["features"][i] for i in order]
            p = torch_distribution(model, bundle, ids, row["anchor_id"], fs).probs.detach().numpy()
            fixture_rows.append(dict(candidate_ids=ids, anchor_id=row["anchor_id"], features=fs,
                                     probabilities=dict(zip(ids, p.tolist()))))
        once(out / f"torch-parity-{iteration}.json", sealed(dict(binding=plan["binding"],
             policy_sha256=validate_bundle(bundle), rows=fixture_rows)))
        return dict(torch_fixtures=len(fixtures))
    native_runtime(plan)
    reference = check_seal(read_json(out / f"torch-parity-{iteration}.json"))
    require(reference["binding"] == plan["binding"] and reference["policy_sha256"] == validate_bundle(bundle), "parity model")
    actor = NumpyActor(bundle)
    error = 0.
    for row in reference["rows"]:
        actual = actor.probabilities(row["candidate_ids"], row["anchor_id"], row["features"])
        error = max(error, max(abs(actual[c] - row["probabilities"][c]) for c in actual))
        require(error <= 1e-12, "cross-platform score mismatch")
        for i in range(16):
            draw = random.Random(i).random()
            require(select_with_draw(actual, draw) == select_with_draw(row["probabilities"], draw), "cross-platform selection mismatch")
    once(out / f"parity-{iteration}.json", sealed(dict(binding=plan["binding"], policy_sha256=actor.sha,
         fixtures=len(fixtures), max_probability_error=error)))
    return dict(fixtures=len(fixtures), max_probability_error=error)


def audit_worker(job):
    from experiments.online_feature_engine import OnlineFeatureEngine
    from experiments.sa_history_selector import History
    from scripts.run_feedback_exploration_diagnostics import validate_final
    p = job["plan"]
    out = ROOT / p["config"]["output"]
    q = native_runtime(p)
    folder = folder_for(out, job["phase"], job)
    result = result_read(folder, p)
    state = read_json(folder / "initial.json")
    history = History(state)
    initial_nodes = state["low_level"]["generated"]
    actor = NumpyActor(actor_load(out, p, job["iteration"])) if job["arm"] in {"trained_actor", "untrained_exploration"} else None
    engine = OnlineFeatureEngine(state, backend="native") if job["arm"] != "official_sa" else None
    if engine:
        from experiments.compact_controller_model import load_controller_bundle
        from lns2_selector.runtime.online_selection import score_online_candidates
        anchor_model = load_controller_bundle(ROOT / "artifacts/initlns-closed-loop-controller-v2").main_models["realized_dynamic"]
    require(q.state_fingerprint(state) == result["initial_fingerprint"], "audit initial")
    for event in trace_read(folder):
        d = history.decision
        used = state["low_level"]["generated"] - initial_nodes
        require(work_stop(state["feasible"], d, used, p["proposal"]) is None, "action after terminal")
        require(event["decision"] == d and event["before"] == q.state_fingerprint(state), "audit trace order")
        if job["arm"] != "official_sa":
            pool = event["pool"]
            ids = [c["candidate_id"] for c in pool]
            require(ids == sorted(set(ids)) == event["candidate_ids"], "audit pool IDs")
            fs = features_for(state, pool, engine, history, q.temperature(d), event["before"])
            require(set(event["proposal_order"]) == set(ids) and len(event["proposal_order"]) == len(ids), "proposal order")
            lookup = {c["candidate_id"]: c for c in pool}
            original = [lookup[cid] for cid in event["proposal_order"]]
            score_rows, _ = engine.realized_rows(original, state_hash=event["before"])
            anchor_index, scores, _ = score_online_candidates(score_rows, anchor_model)
            require(original[anchor_index]["candidate_id"] == event["anchor_id"] and
                    all(abs(c["score"] - s) <= 1e-12 for c, s in zip(original, scores)), "frozen anchor score mismatch")
            supplied = [budget_features(f, d, used, p["proposal"]) for f in fs]
            require(supplied == event["features"], "online/offline feature mismatch")
            probabilities = actor.probabilities(ids, event["anchor_id"], supplied) if actor else {c: float(c == event["anchor_id"]) for c in ids}
            require(probabilities == event["probabilities"], "audit actor probabilities")
            draw = stream_draw(p, job["phase"], job["pair_id"], job["replica"], d, "select")
            cid = select_with_draw(probabilities, draw) if actor else event["anchor_id"]
            selected = next(c["agents"] for c in pool if c["candidate_id"] == cid)
            require(cid == event["selected_id"] and selected == event["action"]["agents"], "audit selection")
            require(event["action"]["random_seed"] == stream_draw(p, job["phase"], job["pair_id"], job["replica"], d, "pp"), "PP RNG")
        else:
            require(event["action"] == {"mode": "official"}, "official RNG changed")
            selected = event["metrics"]["neighborhood"]
        require(event["temperature"] == q.temperature(d) and event["uniform"] ==
                stream_draw(p, job["phase"], job["pair_id"], job["replica"], d, "accept"), "SA stream")
        after = q.apply_state_delta(state, event["delta"])
        q.validate_transition(state, after, event["metrics"], selected, "annealed", event["temperature"], event["uniform"])
        history.observe(state, event, after)
        state = after
    validate_final(state)
    require(q.state_fingerprint(state) == result["final_fingerprint"] and state == read_json(folder / "final.json"), "audit final paths")
    require(history.decision == result["decisions"] and state["low_level"]["generated"] - initial_nodes == result["generated"], "audit work")
    require(state["num_of_colliding_pairs"] == result["final_conflicts"] and state["feasible"] == result["success"], "audit outcome")
    terminal_return(result, max_decisions=p["proposal"]["max_decisions"], node_budget=p["proposal"]["node_budget"])
    return dict(status="ok", job_id=job["job_id"], result_sha256=sha256_file(folder / "result.json"))


def audit(phase, iteration):
    from experiments.repair_collection import _run_jobs, _CollectionRunLock
    plan, out = verify()
    require((out / f"{phase}.complete.json").exists(), "incomplete collection")
    jobs = [dict(j, plan=plan) for j in jobs_for(plan, phase, iteration)]
    with _CollectionRunLock(out, plan["binding"], "onpolicy-audit"):
        results = _run_jobs(audit_worker, jobs, plan["config"]["workers"], phase="audit-" + phase,
                            timeout_seconds=plan["proposal"]["process_fuse_seconds"], stop_on_failure=True)
        require(len(results) == len(jobs) and all(r["status"] == "ok" for r in results), "trace audit failed")
        once(out / f"{phase}.audit.json", sealed(dict(binding=plan["binding"], results=results)))
    return dict(audited=len(results))


def update(iteration):
    plan, out = verify()
    require(0 <= iteration < plan["proposal"]["maximum_updates"], "update budget")
    phase = f"train-{iteration}"
    audit_report = check_seal(read_json(out / f"{phase}.audit.json"))
    require(audit_report["binding"] == plan["binding"], "audit binding")
    bundle = actor_load(out, plan, iteration)
    require(not actor_file(out, iteration + 1).exists(), "update already published; no repeated fitting")
    jobs = jobs_for(plan, phase, iteration)
    require(len(audit_report["results"]) == len(jobs), "audit coverage")
    audit_hashes = {r["job_id"]: r["result_sha256"] for r in audit_report["results"]}
    folders = {}
    rows = []
    for job in jobs:
        folder = folder_for(out, phase, job)
        row = result_read(folder, plan)
        require(sha256_file(folder / "result.json") == audit_hashes[job["job_id"]], "changed audited result")
        row["steps"] = [{k: e[k] for k in ("decision", "policy_sha256", "probabilities", "selected_id", "behavior_log_probability")}
                        for e in trace_read(folder)]
        rows.append(row)
        folders[row["episode_id"]] = folder
    coefficients = gradient_coefficients(rows, policy_sha256=validate_bundle(bundle),
        expected_groups={j["pair_id"]: j["case"]["map_id"] for j in jobs},
        replicas=plan["proposal"]["train_replicas_per_condition"],
        max_decisions=plan["proposal"]["max_decisions"], node_budget=plan["proposal"]["node_budget"])
    if not any(r["coefficient"] != 0 for r in coefficients):
        once(out / f"update-{iteration}.json", sealed(dict(binding=plan["binding"], status="no_signal_stop", no_update=True)))
        return dict(status="no_signal_stop", no_update=True)
    model, diagnostics = update_once(bundle, coefficients, lambda key: trace_read(folders[key]),
                                    sha256_file(out / f"{phase}.audit.json"))
    publish_actor(out, plan, model, diagnostics)
    once(out / f"update-{iteration}.json", sealed(dict(binding=plan["binding"], status="updated", **diagnostics)))
    return dict(status="updated", iteration=iteration + 1, **diagnostics)


def micro():
    """Real frozen-native integration on tiny paths, never evidence for speed."""
    plan, out = verify()
    q = native_runtime(plan)
    import lns2_env
    bundle = actor_load(out, plan, 0)
    folder = out / "micro"
    require(not folder.exists(), "micro output already exists")
    folder.mkdir()
    (folder / "tiny.map").write_text("type octile\nheight 3\nwidth 4\nmap\n....\n....\n....\n", encoding="utf8")
    (folder / "tiny.scen").write_text("version 1\n0\ttiny.map\t4\t3\t0\t0\t3\t0\t3\n0\ttiny.map\t4\t3\t3\t0\t0\t0\t3\n", encoding="utf8")
    rows = []
    for i, arm in enumerate([*plan["proposal"]["evaluation_arms"], "trained_actor", "trained_actor"]):
        p = dict(plan, proposal=dict(plan["proposal"], max_decisions=8,
                                    episode_safety_seconds=0 if i == 5 else 30.))
        env = lns2_env.LNS2RepairEnv(str(folder / "tiny.map"), str(folder / "tiny.scen"), 2)
        state = q._plain(env.reset_paths([[0, 1, 2, 3], [3, 2, 1, 0]], seed=11))
        job = dict(plan=p, arm=arm, phase="micro", replica=0, pair_id="tiny-s11", split="train",
                   case=dict(map_id="tiny"), job_id=f"tiny-{i}")
        ctx = dict(case_id="tiny-s11", task_id="tiny", solver_seed=11, proposal=plan["template"]["proposal"])
        row = episode_loop(job, q, env, state, ctx, folder / str(i), bundle if arm in {"trained_actor", "untrained_exploration"} else None)
        result_read(folder / str(i), p)
        rows.append(row)
    require(len({r["initial_fingerprint"] for r in rows}) == 1, "micro initial mismatch")
    require(rows[3]["final_fingerprint"] == rows[4]["final_fingerprint"], "same-policy repeat mismatch")
    def decisions(i):
        return [{k: e[k] for k in ("action", "selected_id", "probabilities", "uniform")} for e in trace_read(folder / str(i))]
    require(decisions(2) == decisions(3) == decisions(4), "zero residual/control/repeat differ")
    require(rows[5]["status"] == "censored" and rows[5]["decisions"] == 0, "safety stop not censored")
    result = dict(binding=plan["binding"], native_sha256=plan["config"]["native_sha256"],
                  episodes=6, same_initial=True, zero_residual_control_exact=True, repeat_exact=True,
                  safety_stop_censored=True, no_ttf=True,
                  files={str(i): sha256_file(folder / str(i) / "result.json") for i in range(6)})
    once(out / "micro.json", sealed(result))
    return result


def report():
    plan, out = verify()
    phase = "evaluation"
    audit_receipt = check_seal(read_json(out / "evaluation.audit.json"))
    require(audit_receipt["binding"] == plan["binding"], "evaluation audit")
    jobs = jobs_for(plan, phase, 2)
    audit_hashes = {r["job_id"]: r["result_sha256"] for r in audit_receipt["results"]}
    rows = []
    for j in jobs:
        folder = folder_for(out, phase, j)
        row = result_read(folder, plan)
        require(sha256_file(folder / "result.json") == audit_hashes[j["job_id"]], "evaluation changed after audit")
        rows.append(row)
    arms = plan["proposal"]["evaluation_arms"]
    tables = {a: {(r["pair_id"], r["replica"]): r for r in rows if r["arm"] == a} for a in arms}
    require(all(len(t) == 16 for t in tables.values()), "evaluation coverage")
    summary = {a: dict(episodes=len(t), completed=sum(r["success"] for r in t.values()),
                       censored=sum(r["status"] == "censored" for r in t.values())) for a, t in tables.items()}
    contrasts = {}
    for base in arms[:-1]:
        a, b = tables["trained_actor"], tables[base]
        require(set(a) == set(b), "unpaired evaluation")
        require(all(a[k]["initial_fingerprint"] == b[k]["initial_fingerprint"] for k in a), "paired initial mismatch")
        common = [k for k in a if a[k]["success"] and b[k]["success"]]
        ratios = {field: (sum(a[k][field] for k in common) / sum(b[k][field] for k in common)
                          if common and sum(b[k][field] for k in common) else None)
                  for field in ("generated", "decisions", "soc", "makespan")}
        contrasts[base] = dict(net_completed=sum(a[k]["success"] - b[k]["success"] for k in a),
                               gains=sum(a[k]["success"] and not b[k]["success"] for k in a),
                               losses=sum(b[k]["success"] and not a[k]["success"] for k in a),
                               common_success=len(common), common_success_ratios=ratios)
    signal = all(contrasts[b]["net_completed"] > 0 and
                 contrasts[b]["common_success_ratios"]["generated"] is not None and
                 contrasts[b]["common_success_ratios"]["generated"] <= 1.1
                 for b in ("dual16_sa", "untrained_exploration"))
    by_map = {m: {a: sum(r["success"] for r in rows if r["map_id"] == m and r["arm"] == a)
                  for a in arms} for m in sorted({r["map_id"] for r in rows})}
    trace_keys = {(j["pair_id"], j["replica"], j["arm"]): folder_for(out, phase, j) for j in jobs}
    identical = 0
    for pair, rep in tables["trained_actor"]:
        decisions = []
        for arm in ("trained_actor", "untrained_exploration"):
            decisions.append([(e["before"], e["selected_id"]) for e in trace_read(trace_keys[pair, rep, arm])])
        identical += decisions[0] == decisions[1]
    result = dict(binding=plan["binding"], summary=summary, contrasts=contrasts, by_map=by_map,
                  identical_trained_exploration_trajectories=identical, episodes=rows, no_ttf=True,
                  automatic_promotion=False, decision="development_signal_requires_independent_confirmation" if signal else "no_development_advantage",
                  only_two_viewed_holdout_maps=True)
    once(out / "report.json", sealed(result))
    return {k: result[k] for k in ("summary", "contrasts", "decision", "no_ttf")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "verify", "qualify", "initialize", "parity", "micro", "collect", "audit", "update", "report", "stop"))
    parser.add_argument("--round", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--evaluation", action="store_true")
    parser.add_argument("--torch", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    phase = "evaluation" if args.evaluation else f"train-{args.round}"
    if args.phase == "stop":
        cfg = read_json(ROOT / CONFIG)
        write_json(ROOT / cfg["output"] / "STOP_AFTER_BATCH", dict(requested=True))
        result = dict(stop_after_current_batch=True)
    elif args.phase == "prepare": result = prepare()
    elif args.phase == "verify": result = dict(binding=verify()[0]["binding"], verified=True)
    elif args.phase == "qualify": result = collect("qualify", resume=args.resume)
    elif args.phase == "collect": result = collect(phase, args.round, args.resume)
    elif args.phase == "audit": result = audit(phase, args.round)
    else:
        from experiments.repair_collection import _CollectionRunLock
        plan, out = verify()
        with _CollectionRunLock(out, plan["binding"], "onpolicy-" + args.phase):
            if args.phase == "initialize": result = initialize()
            elif args.phase == "parity": result = parity(args.round, args.torch)
            elif args.phase == "micro": result = micro()
            elif args.phase == "report": result = report()
            else: result = update(args.round)
    print(json.dumps(result, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
