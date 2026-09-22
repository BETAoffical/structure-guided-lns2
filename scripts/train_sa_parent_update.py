"""One registered terminal-credit update from the frozen actor-1 Train batch."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts import collect_sa_second_batch as collection
from scripts import run_sa_crossfit_update as cache_tools
from scripts import run_sa_onpolicy as run
from scripts import recover_sa_onpolicy as recovery
from experiments import sa_uncapped_training_contract as credit
from experiments.sa_onpolicy_actor import NumpyActor, torch_actor, torch_distribution
from experiments.sa_parent_update import padded_with_prior, log_distribution, guarded_parent_update

CONFIG = "configs/sa_parent_update.json"
REGISTRATION = "training_registration.json"
CODE = (CONFIG, "scripts/train_sa_parent_update.py", "experiments/sa_parent_update.py",
        "tests/evaluation/test_sa_parent_update.py", "docs/SA_PARENT_UPDATE_PROTOCOL_ZH.md",
        "experiments/sa_crossfit_update.py", "experiments/sa_onpolicy_actor.py",
        "scripts/run_sa_crossfit_update.py", "experiments/sa_uncapped_training_contract.py")


def fixed_scope(cfg):
    expected = dict(training_arm="uncapped_condition", output_arm="second_condition", parent_iteration=1,
        output_iteration=2, expected_episodes=96, workers=20, max_decisions=None, node_budget=25000000,
        maximum_updates=1, solver_calls=0, formal_ttf=False, heldout_evaluation=False, automatic_promotion=False)
    run.require(all(cfg[k] == v for k, v in expected.items()), "fixed Train continuation scope")
    run.require(cfg["update"] == run.read_json(ROOT/collection.source_training.CONFIG)["update"], "fixed update limits")


def evidence():
    reg, source, src = collection.verify()
    cfg = run.read_json(ROOT/CONFIG)
    fixed_scope(cfg)
    run.require(src == ROOT/cfg["source"] and run.sha256_file(src/"report.json") == cfg["source_report_sha256"], "source report identity")
    report = run.check_seal(run.read_json(src/"report.json"))
    audit = run.check_seal(run.read_json(src/"audit.json"))
    complete = run.check_seal(run.read_json(src/"collection.complete.json"))
    jobs = collection.ready_jobs(reg, source, src)
    run.require(report["binding"] == audit["binding"] == complete["binding"] == reg["binding"] and
        report["audit_sha256"] == run.sha256_file(src/"audit.json") and report["no_update"] and
        report["training_arm"] == cfg["training_arm"] and report["control_excluded_from_gradient"], "source analysis/audit")
    hashes = {r["job_id"]: r["result_sha256"] for r in audit["results"] if r["status"] == "ok"}
    run.require(len(audit["results"]) == complete["jobs"] == 192 and
        set(hashes) == set(complete["files"]) == {j["job_id"] for j in jobs}, "full source audit")
    actor = run.read_json(ROOT/reg["config"]["models"][cfg["training_arm"]]["path"])
    policy = run.validate_bundle(actor)
    run.require(actor["iteration"] == 1 and np.any(actor["w2"]), "frozen nonzero actor-1")
    entries = []
    for j in jobs:
        if j["comparison_arm"] != cfg["training_arm"]:
            continue
        folder = collection.compare.prior.folder_for(reg["config"], j)
        row = collection.compare.read_result(reg, source, j)
        digest = run.sha256_file(folder/"result.json")
        run.require(hashes[j["job_id"]] == complete["files"][j["job_id"]] == digest and
            row["split"] == "train" and row["policy_sha256"] == policy and row["status"] == "ok", "audited current-policy Train only")
        credit.terminal_return(row, max_decisions=None, node_budget=cfg["node_budget"])
        entries.append(dict(job_id=j["job_id"], folder=folder.relative_to(ROOT).as_posix(), result_sha256=digest, episode=row))
    run.require(len(entries) == cfg["expected_episodes"] and report["training_information"]["one_update_eligible"], "complete batch with terminal credit")
    return cfg, reg, source, report, actor, sorted(entries, key=lambda e: e["job_id"])


def prepare():
    cfg, source_reg, source, report, actor, entries = evidence()
    out = ROOT/cfg["output"]
    run.require(not out.exists(), "existing output; no implicit retraining")
    inputs = {n: run.sha256_file(ROOT/n) for n in CODE}
    src = ROOT/cfg["source"]
    for n in ("registration.json", "qualification.json", "collection.complete.json", "audit.json", "report.json"):
        inputs[(src/n).relative_to(ROOT).as_posix()] = run.sha256_file(src/n)
    for e in entries:
        for n in ("result.json", *e["episode"]["files"]):
            path = ROOT/e["folder"]/n
            inputs[path.relative_to(ROOT).as_posix()] = run.sha256_file(path)
    model_spec = source_reg["config"]["models"][cfg["training_arm"]]
    inputs[model_spec["path"]] = model_spec["file_sha256"]
    body = dict(schema=cfg["schema"], config=cfg, inputs=inputs, entries=entries,
        source_binding=source_reg["binding"], scientific_binding=source["binding"], parent_policy=run.validate_bundle(actor),
        credits=report["training_information"]["coefficients"],
        source_commit=run.subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        no_solver=True, no_heldout=True, no_ttf=True)
    body["binding"] = run.json_fingerprint(body)
    with recovery.strict_lock(out, body["binding"], "parent-update-prepare"):
        run.publish_actor(out, source, actor, dict(copied_not_trained=True, registration_binding=body["binding"]))
        run.once(out/REGISTRATION, run.sealed(body))
        run.write_json(out/"run_status.json", dict(status="registered", binding=body["binding"]))
    return dict(registered=True, binding=body["binding"], episodes=96, updates_allowed=1, no_solver=True)


def verify():
    cfg, source_reg, source, report, actor, entries = evidence()
    out = ROOT/cfg["output"]
    reg = run.check_seal(run.read_json(out/REGISTRATION))
    run.require(reg["binding"] == run.json_fingerprint({k: v for k, v in reg.items() if k not in ("binding", "integrity")}), "registration hash")
    run.require(reg["config"] == cfg and reg["entries"] == entries and reg["source_binding"] == source_reg["binding"] and
        reg["scientific_binding"] == source["binding"] and reg["parent_policy"] == run.validate_bundle(actor) and
        reg["credits"] == report["training_information"]["coefficients"], "registration/source identity")
    for name, digest in reg["inputs"].items():
        run.require(run.sha256_file(run.contained_file(ROOT, name, field="parent update input")) == digest, "changed input: " + name)
    run.require(run.actor_load(out, source, 1) == actor, "parent copy changed")
    return reg, source, out, actor


def validate_credit(reg, rows):
    groups = {e["episode"]["pair_id"]: e["episode"]["map_id"] for e in reg["entries"]}
    weights = credit.gradient_coefficients(rows, policy_sha256=reg["parent_policy"], expected_groups=groups,
        replicas=4, max_decisions=None, node_budget=25000000)
    run.require(weights == reg["credits"], "recomputed complete terminal credit changed")
    return {r["episode_id"]: r for r in weights}


def extract():
    reg, source, out, actor = verify()
    with recovery.strict_lock(out, reg["binding"], "parent-update-extract"):
        run.require(not (out/"cache").exists() and not (out/"cache.json").exists(), "partial extraction needs inspection")
        run.write_json(out/"run_status.json", dict(status="extracting", binding=reg["binding"]))
        jobs = [dict(e, plan=source, actor=actor, output=reg["config"]["output"]) for e in reg["entries"]]
        rows = []
        with ProcessPoolExecutor(max_workers=reg["config"]["workers"]) as pool:
            for row in pool.map(cache_tools.extract_worker, jobs):
                rows.append(row)
                if len(rows) % 12 == 0:
                    print(f"Extracted {len(rows)}/96", flush=True)
        weights = validate_credit(reg, [dict(r["episode"], steps=r["steps"]) for r in rows])
        for row in rows:
            row.pop("steps")
            row["credit"] = weights[row["episode"]["episode_id"]]
        run.once(out/"cache.json", run.sealed(dict(binding=reg["binding"], rows=rows)))
        run.write_json(out/"run_status.json", dict(status="extracted", binding=reg["binding"]))
    return dict(extracted=len(rows), control_episodes=0, no_solver=True)


def cached(reg, out):
    rows = cache_tools.cached(reg, out)
    entries = {e["job_id"]: e for e in reg["entries"]}
    weights = {w["episode_id"]: w for w in reg["credits"]}
    run.require(len(rows) == len({r["episode"]["job_id"] for r, _ in rows}) == len(entries) == 96, "unique full cache")
    for r, data in rows:
        e = entries[r["episode"]["job_id"]]
        run.require(r["episode"] == e["episode"] and r["source_sha256"] == e["result_sha256"] and
            r["credit"] == weights[r["episode"]["episode_id"]], "cache source/credit changed")
        run.require(len(data["lengths"]) == r["episode"]["decisions"], "complete cached trajectory")
    return rows


def numerical_check():
    import torch
    torch.set_num_threads(1)
    reg, _, out, parent = verify()
    run.require(torch.__version__ == parent["training_library"], "fixed Torch")
    rows = cached(reg, out)
    maps = sorted({r["episode"]["map_id"] for r, _ in rows})
    picked = []
    for m in maps:
        candidates = [(r, d) for r, d in rows if r["episode"]["map_id"] == m]
        picked.extend((candidates[0], candidates[-1]))
    chunks, lengths, probabilities, anchors, selected, fixtures = [], [], [], [], [], []
    for r, data in picked:
        offsets = np.r_[0, np.cumsum(data["lengths"])]
        for i in sorted({0, len(data["lengths"])-1} | ({256} if len(data["lengths"]) > 256 else set())):
            a, b = offsets[i:i+2]
            x = data["x"][a:b]
            chunks.append(x)
            lengths.append(b-a)
            probabilities.extend(data["probabilities"][a:b])
            anchors.append(int(data["anchors"][i]))
            selected.append(int(data["selected"][i]))
            fixtures.append(dict(job_id=r["episode"]["job_id"], decision=i, candidate_ids=data["ids"][a:b].tolist(),
                anchor_id=str(data["ids"][a+data["anchors"][i]]),
                features=[dict(zip(parent["feature_names"], v.tolist())) for v in x]))
    data = dict(x=np.concatenate(chunks), lengths=np.asarray(lengths), probabilities=np.asarray(probabilities),
                anchors=np.asarray(anchors))
    padded, prior = padded_with_prior(data, parent)
    model = torch_actor(parent)
    weights = torch.linspace(.2, .9, len(fixtures), dtype=torch.float64)
    indices = torch.as_tensor(selected, dtype=torch.long)
    logp = log_distribution(model, parent, padded, prior)
    probability_error = float(np.max(np.abs(logp.exp().detach().numpy()-padded[1])))
    run.require(probability_error <= 1e-12, "nonzero packed behavior mismatch")
    packed_loss = -(weights * logp[torch.arange(len(fixtures)), indices]).sum()
    g = torch.autograd.grad(packed_loss, tuple(model.parameters()))
    def scalar_loss():
        return sum(-weights[i] * torch_distribution(model, parent, f["candidate_ids"], f["anchor_id"], f["features"])
            .log_prob(indices[i]) for i, f in enumerate(fixtures))
    h = torch.autograd.grad(scalar_loss(), tuple(model.parameters()))
    gradient_error = max(float((a-b).abs().max()) for a, b in zip(g, h))
    run.require(gradient_error <= 1e-10, "packed/single-state gradient mismatch")
    differences = []
    for parameter, analytic in zip(model.parameters(), g):
        index = int(analytic.abs().argmax())
        flat = parameter.view(-1)
        original = float(flat[index].detach())
        try:
            with torch.no_grad():
                flat[index] = original + 1e-6
                plus = float(scalar_loss())
                flat[index] = original - 1e-6
                minus = float(scalar_loss())
        finally:
            with torch.no_grad():
                flat[index] = original
        actual, target = (plus-minus)/2e-6, float(analytic.flatten()[index])
        run.require(abs(actual-target) <= 1e-7 * max(1., abs(target)), "finite difference mismatch")
        differences.append(dict(index=index, analytic=target, finite_difference=actual, absolute_error=abs(actual-target)))
    result = dict(binding=reg["binding"], cache_sha256=run.sha256_file(out/"cache.json"), parent_policy=reg["parent_policy"],
        fixtures=len(fixtures), probability_max_error=probability_error, gradient_max_error=gradient_error,
        finite_differences=differences, no_update=True)
    with recovery.strict_lock(out, reg["binding"], "parent-update-numerical-check"):
        run.once(out/"numerical_check.json", run.sealed(result))
    return result


def train():
    import torch
    torch.set_num_threads(1)
    reg, source, out, parent = verify()
    run.require(torch.__version__ == parent["training_library"], "fixed Torch")
    proof = run.check_seal(run.read_json(out/"numerical_check.json"))
    run.require(proof["binding"] == reg["binding"] and proof["parent_policy"] == reg["parent_policy"] and
        proof["cache_sha256"] == run.sha256_file(out/"cache.json"), "numerical check identity")
    with recovery.strict_lock(out, reg["binding"], "parent-update-train"):
        run.require(not (out/"update.json").exists() and not run.actor_file(out, 2).exists(), "one fixed update only")
        run.write_json(out/"run_status.json", dict(status="training", binding=reg["binding"]))
        model = torch_actor(parent)
        gradient = [torch.zeros_like(p) for p in model.parameters()]
        packs, maximum_error = [], 0.
        for i, (r, data) in enumerate(cached(reg, out)):
            padded, prior = padded_with_prior(data, parent)
            logp = log_distribution(model, parent, padded, prior)
            error = float(np.max(np.abs(logp.exp().detach().numpy()-padded[1])))
            maximum_error = max(maximum_error, error)
            run.require(error <= 1e-12, "full parent behavior replay")
            selected = logp[torch.arange(len(data["lengths"])), torch.as_tensor(data["selected"], dtype=torch.long)]
            raw = torch.autograd.grad(-r["credit"]["coefficient"] * selected.sum(), tuple(model.parameters()))
            for total, value in zip(gradient, raw):
                total += value
            packs.append(dict(padded=padded, prior=prior, weight=r["credit"]["episode_weight"],
                              draws=data["draws"], lengths=data["lengths"], selected=data["selected"]))
            if (i+1) % 12 == 0:
                print(f"Gradient {i+1}/96", flush=True)
        magnitude = sum(float((g*g).sum()) for g in gradient)**.5
        bundle, diagnostic = (None, dict(decision="zero_gradient_no_update", gradient_norm=0.)) if magnitude == 0 else guarded_parent_update(
            parent, gradient, packs, reg["config"]["update"], reg["binding"], reg["config"]["output_arm"])
        result = dict(binding=reg["binding"], parent_policy=reg["parent_policy"], updated=bundle is not None,
            diagnostic=diagnostic, replay_probability_max_error=maximum_error,
            cache_sha256=run.sha256_file(out/"cache.json"), numerical_check_sha256=run.sha256_file(out/"numerical_check.json"),
            decision="fixed_parent_update_not_evaluated" if bundle is not None else diagnostic["decision"],
            no_solver=True, no_heldout=True, no_ttf=True, automatic_promotion=False)
        if bundle is not None:
            run.require(bundle["iteration"] == 2 and bundle["parent_policy"] == reg["parent_policy"], "export lineage")
            run.publish_actor(out, source, bundle, result)
            result.update(model_sha256=run.sha256_file(run.actor_file(out, 2)), policy_sha256=run.validate_bundle(bundle))
        run.once(out/"update.json", run.sealed(result))
        run.write_json(out/"run_status.json", dict(status="update_recorded_not_evaluated", updated=result["updated"], binding=reg["binding"]))
    return result


def parity(torch_side=False):
    reg, source, out, parent = verify()
    update = run.check_seal(run.read_json(out/"update.json"))
    run.require(update["binding"] == reg["binding"] and update["updated"], "no updated model")
    bundle = run.actor_load(out, source, 2)
    run.require(bundle["parent_policy"] == reg["parent_policy"] and run.validate_bundle(bundle) == update["policy_sha256"] and
        run.sha256_file(run.actor_file(out, 2)) == update["model_sha256"], "child model identity")
    actor = NumpyActor(bundle)
    with recovery.strict_lock(out, reg["binding"], "parent-update-parity"):
        fixtures = []
        if torch_side:
            import torch
            torch.set_num_threads(1)
            run.require(torch.__version__ == parent["training_library"], "fixed Torch")
            model = torch_actor(bundle)
            for entry in reg["entries"]:
                for event in run.trace_read(ROOT/entry["folder"]):
                    if event["decision"] not in {0, 32, 128, 255, 256, entry["episode"]["decisions"]-1}:
                        continue
                    ids = event["candidate_ids"]
                    values = torch_distribution(model, bundle, ids, event["anchor_id"], event["features"]).probs.detach().numpy().tolist()
                    fixtures.append(dict(job_id=entry["job_id"], decision=event["decision"], candidate_ids=ids,
                        anchor_id=event["anchor_id"], features=event["features"], probabilities=dict(zip(ids, values))))
        else:
            run.native_runtime(source)
            ref = run.check_seal(run.read_json(out/"torch-parity.json"))
            run.require(ref["binding"] == reg["binding"] and ref["policy_sha256"] == actor.sha and
                ref["update_sha256"] == run.sha256_file(out/"update.json"), "parity identity")
            fixtures = ref["fixtures"]
        error = 0.
        for f in fixtures:
            actual = actor.probabilities(f["candidate_ids"], f["anchor_id"], f["features"])
            error = max(error, max(abs(actual[k]-f["probabilities"][k]) for k in actual))
            run.require(error <= 1e-12, "portable probability mismatch")
            for i in range(16):
                draw = run.random.Random(i).random()
                run.require(run.select_with_draw(actual, draw) == run.select_with_draw(f["probabilities"], draw), "portable choice mismatch")
        run.require(fixtures and {f["job_id"] for f in fixtures} == {e["job_id"] for e in reg["entries"]}, "full episode parity coverage")
        result = dict(binding=reg["binding"], policy_sha256=actor.sha, update_sha256=run.sha256_file(out/"update.json"),
            fixture_count=len(fixtures), decisions_after_256=sum(f["decision"] >= 256 for f in fixtures),
            max_error=error, choices_checked=16*len(fixtures), no_solver=True, no_closed_loop=True)
        run.once(out/("torch-parity.json" if torch_side else "parity.json"), run.sealed(dict(result, fixtures=fixtures) if torch_side else result))
        if not torch_side:
            run.write_json(out/"run_status.json", dict(status="verified_not_evaluated", binding=reg["binding"],
                model_sha256=update["model_sha256"], no_active_workers=True))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "verify", "extract", "check", "train", "parity"))
    parser.add_argument("--torch", action="store_true")
    args = parser.parse_args()
    try:
        if args.phase == "verify":
            result = dict(verified=True, binding=verify()[0]["binding"])
        elif args.phase == "check":
            result = numerical_check()
        elif args.phase == "parity":
            result = parity(args.torch)
        else:
            result = globals()[args.phase]()
    except BaseException as error:
        out = ROOT/run.read_json(ROOT/CONFIG)["output"]
        if (out/REGISTRATION).exists():
            run.write_json(out/"run_status.json", dict(status="needs_inspection", error=repr(error), no_retry=True))
        raise
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
