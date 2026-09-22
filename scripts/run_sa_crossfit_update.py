"""Frozen Train-only critic ablation and bounded actor-0 fork, without solver calls."""
import argparse
from collections import defaultdict
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
from experiments.sa_paired_completion import require
from experiments.sa_onpolicy_actor import NumpyActor, vectorize, torch_actor, select_with_draw
from experiments.sa_crossfit_update import (state_columns, state_vector, fold_masks, fit_value, predict_value,
    padded_episode, tensor_distribution, guarded_update)
from scripts import recover_sa_onpolicy as recovery
from scripts import run_sa_onpolicy as run
from scripts.audit_sa_onpolicy_gradient import PINS

CONFIG = "configs/sa_onpolicy_crossfit.json"
CODE = (CONFIG, "scripts/run_sa_crossfit_update.py", "experiments/sa_crossfit_update.py",
        "tests/evaluation/test_sa_crossfit_update.py", "docs/SA_ONPOLICY_CROSSFIT_PROTOCOL_ZH.md")


def prepare():
    reg, plan, source = recovery.verify()
    config = run.read_json(ROOT/CONFIG)
    require(config["solver_calls"] == 0 and config["maximum_updates_per_arm"] == 1 and config["fork_from"] == "actor-0", "scope")
    output = ROOT/config["output"]
    require(not output.exists(), "existing prototype output")
    for name, digest in PINS.items():
        require(run.sha256_file(source/name) == digest, "source evidence changed")
    batch = recovery.audited_batch(reg, plan, source)
    inputs = {name: run.sha256_file(ROOT/name) for name in CODE}
    for name in (*PINS, "registration.json", "batch.audit.json", "torch-parity-1.json"):
        inputs[(source/name).relative_to(ROOT).as_posix()] = run.sha256_file(source/name)
    entries = [dict(folder=folder.relative_to(ROOT).as_posix(), result_sha256=run.sha256_file(folder/"result.json"),
                    job_id=row["job_id"]) for _, folder, row in batch]
    body = dict(schema=config["schema"], config=config, source=source.relative_to(ROOT).as_posix(),
                source_binding=reg["binding"], inputs=inputs, entries=entries,
                code_commit=run.subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    body["binding"] = run.json_fingerprint(body)
    with recovery.strict_lock(output, body["binding"], "crossfit-prepare"):
        run.once(output/"registration.json", run.sealed(body))
    return dict(registered=True, episodes=len(entries), no_solver=True)


def verify():
    source_reg, plan, source = recovery.verify()
    cfg = run.read_json(ROOT/CONFIG)
    output = ROOT/cfg["output"]
    reg = run.check_seal(run.read_json(output/"registration.json"))
    require(reg["binding"] == run.json_fingerprint({k:v for k,v in reg.items() if k not in ("binding", "integrity")}), "registration")
    require(reg["config"] == cfg and reg["source_binding"] == source_reg["binding"], "configuration")
    for name, digest in reg["inputs"].items():
        require(run.sha256_file(ROOT/name) == digest, "changed registered input: "+name)
    return reg, plan, source, output


def extract_worker(job):
    folder = ROOT/job["folder"]
    row = run.result_read(folder, job["plan"])
    require(run.sha256_file(folder/"result.json") == job["result_sha256"], "episode changed")
    require(row["status"] == "ok" and row["split"] == "train", "uncensored Train only")
    actor = NumpyActor(job["actor"])
    require(row["policy_sha256"] == actor.sha, "behavior policy")
    columns = state_columns(actor.bundle["feature_names"])
    xs, states, lengths, selected, draws, probabilities, ids, anchors, steps = [], [], [], [], [], [], [], [], []
    for e in run.trace_read(folder):
        candidates = e["candidate_ids"]
        require(candidates == sorted(set(candidates)), "candidate order")
        x = vectorize(e["features"], actor.bundle["feature_names"])
        expected = actor.probabilities(candidates, e["anchor_id"], e["features"])
        require(max(abs(expected[k] - e["probabilities"][k]) for k in candidates) < 1e-12, "behavior replay")
        xs.append(x)
        states.append(state_vector(x, columns))
        lengths.append(len(candidates))
        selected.append(candidates.index(e["selected_id"]))
        anchors.append(candidates.index(e["anchor_id"]))
        draws.append(e["selection_draw"])
        probabilities.extend(expected[k] for k in candidates)
        ids.extend(candidates)
        steps.append({k:e[k] for k in ("decision", "policy_sha256", "probabilities", "selected_id", "behavior_log_probability")})
    require(len(steps) == row["decisions"] and len(steps) > 0, "complete nonempty trajectory")
    path = ROOT/job["output"]/"cache"/(row["job_id"]+".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), "existing extraction")
    temporary = path.with_suffix(".tmp")
    with temporary.open("xb") as f:
        np.savez_compressed(f, x=np.concatenate(xs), state=np.asarray(states), lengths=lengths, selected=selected,
            anchors=anchors, draws=draws, probabilities=probabilities, ids=np.asarray(ids))
    os.replace(temporary, path)
    return dict(episode=row, steps=steps, cache=path.relative_to(ROOT).as_posix(), sha256=run.sha256_file(path),
                source_sha256=job["result_sha256"])


def extract():
    reg, plan, source, out = verify()
    with recovery.strict_lock(out, reg["binding"], "crossfit-extract"):
        require(not (out/"cache").exists(), "no silent partial extraction retry")
        actor = run.actor_load(source, plan, 0)
        jobs = [dict(e, plan=plan, actor=actor, output=reg["config"]["output"]) for e in reg["entries"]]
        rows = []
        with ProcessPoolExecutor(max_workers=reg["config"]["workers"]) as pool:
            for row in pool.map(extract_worker, jobs):
                rows.append(row)
                if len(rows) % 12 == 0: print(f"Extracted {len(rows)}/96", flush=True)
        coefficients = run.gradient_coefficients([dict(r["episode"], steps=r["steps"]) for r in rows],
            policy_sha256=run.validate_bundle(actor), expected_groups={r["episode"]["pair_id"]:r["episode"]["map_id"] for r in rows},
            replicas=4, max_decisions=256, node_budget=25000000)
        require(len(rows) == 96, "batch count")
        weights = {r["episode_id"]:r for r in coefficients}
        for row in rows:
            row.pop("steps")
            row["credit"] = weights[row["episode"]["episode_id"]]
        run.once(out/"cache.json", run.sealed(dict(binding=reg["binding"], rows=rows)))
    return dict(extracted=96, no_solver=True)


def cached(reg, out):
    manifest = run.check_seal(run.read_json(out/"cache.json"))
    require(manifest["binding"] == reg["binding"] and len(manifest["rows"]) == 96, "cache coverage")
    result = []
    for row in manifest["rows"]:
        require(run.sha256_file(ROOT/row["cache"]) == row["sha256"], "cache changed")
        with np.load(ROOT/row["cache"], allow_pickle=False) as data:
            result.append((row, dict(data)))
    return result


def baseline():
    import sklearn
    reg, plan, source, out = verify()
    require(sklearn.__version__ == "1.5.0", "fixed installed sklearn")
    with recovery.strict_lock(out, reg["binding"], "crossfit-baseline"):
        require(not (out/"baseline.json").exists(), "existing baseline")
        rows = cached(reg, out)
        metadata = [r["episode"] for r, _ in rows]
        predictions, folds = {}, []
        for held in reg["config"]["replica_folds"]:
            train, test = fold_masks(metadata, held)
            train_rows = [r for r, use in zip(rows, train) if use]
            x = np.concatenate([d["state"] for _,d in train_rows])
            y = np.concatenate([np.full(len(d["state"]), float(r["episode"]["success"])) for r,d in train_rows])
            w = np.concatenate([np.full(len(d["state"]), r["credit"]["episode_weight"]/len(d["state"])) for r,d in train_rows])
            model = fit_value(x, y, w, reg["config"]["critic"]["alpha"])
            for (r,d), use in zip(rows, test):
                if use: predictions[r["episode"]["episode_id"]] = predict_value(model, d["state"]).tolist()
            folds.append(dict(held_replica=held, model=model,
                training_episodes=[m["episode_id"] for m,use in zip(metadata, train) if use],
                predicted_episodes=[m["episode_id"] for m,use in zip(metadata, test) if use]))
        brier = {"condition":0., "state":0.}
        for r,d in rows:
            y = float(r["episode"]["success"])
            weight = r["credit"]["episode_weight"]
            brier["condition"] += weight*(y-r["credit"]["baseline"])**2
            brier["state"] += weight*float(np.mean((y-np.asarray(predictions[r["episode"]["episode_id"]]))**2))
        names = run.actor_load(source, plan, 0)["feature_names"]
        result = run.sealed(dict(binding=reg["binding"], cache_sha256=run.sha256_file(out/"cache.json"),
            feature_names=[names[i] for i in state_columns(names)],
            sklearn=sklearn.__version__, predictions=predictions, folds=folds, brier=brier, no_map_generalization_claim=True))
        run.once(out/"baseline.json", result)
    return dict(folds=4, predicted_episodes=len(predictions), brier=brier)


def train():
    import torch
    torch.set_num_threads(1)
    reg, plan, source, out = verify()
    with recovery.strict_lock(out, reg["binding"], "crossfit-update"):
        require(not (out/"models").exists() and not (out/"update.json").exists(), "one update per fork")
        critic = run.check_seal(run.read_json(out/"baseline.json"))
        require(critic["binding"] == reg["binding"] and critic["cache_sha256"] == run.sha256_file(out/"cache.json"), "baseline identity")
        actor = run.actor_load(source, plan, 0)
        require(not np.any(actor["w2"]) and not np.any(actor["b2"]), "actor-0 only")
        require(torch.__version__ == actor["training_library"], "frozen Torch environment")
        model = torch_actor(actor)
        arms = reg["config"]["arms"]
        gradients = {a:[torch.zeros_like(p) for p in model.parameters()] for a in arms}
        packs, episode_gradients = [], []
        for row, data in cached(reg, out):
            e = row["episode"]
            padded = padded_episode(data, actor)
            logp = tensor_distribution(model, actor, padded)
            require(np.max(np.abs(logp.exp().detach().numpy()-padded[1])) < 1e-12, "packed behavior mismatch")
            selected_logp = logp[torch.arange(len(data["lengths"])), torch.as_tensor(data["selected"], dtype=torch.long)]
            per_episode = {}
            for arm in arms:
                base = row["credit"]["baseline"] if arm == "bounded_condition" else np.asarray(critic["predictions"][e["episode_id"]])
                advantage = torch.as_tensor(float(e["success"])-base, dtype=torch.float64)
                loss = -(advantage*selected_logp).sum()
                raw = torch.autograd.grad(loss, tuple(model.parameters()), retain_graph=arm != arms[-1])
                for total, value in zip(gradients[arm], raw): total += row["credit"]["episode_weight"]*value
                per_episode[arm] = torch.cat([g.flatten() for g in raw]).numpy().tolist()
            episode_gradients.append(dict(pair_id=e["pair_id"], replica=e["replica"], gradients=per_episode))
            packs.append(dict(padded=padded, weight=row["credit"]["episode_weight"], draws=data["draws"],
                              lengths=data["lengths"], selected=data["selected"]))
        original = run.check_seal(run.read_json(source/"update-0.json"))
        condition_norm = float(torch.sqrt(sum((g*g).sum() for g in gradients["bounded_condition"])))
        require(abs(condition_norm-original["gradient_norm_before_clip"]) < 1e-10, "original gradient reproduction")
        results = {}
        for arm in arms:
            bundle, diagnostics = guarded_update(actor, gradients[arm], packs, reg["config"]["update"], reg["binding"], arm)
            if bundle is None:
                run.once(out/(arm+"-no-step.json"), run.sealed(dict(binding=reg["binding"], **diagnostics)))
                raise ValueError("no bounded step; no automatic resource or parameter changes")
            run.once(out/"models"/(arm+".json"), bundle)
            groups = defaultdict(list)
            for e in episode_gradients: groups[e["pair_id"]].append(e["gradients"][arm])
            dispersion = float(np.mean([np.var(v, axis=0, ddof=1).sum() for v in groups.values()]))
            results[arm] = dict(**diagnostics, gradient_dispersion_proxy=dispersion,
                file_sha256=run.sha256_file(out/"models"/(arm+".json")), policy_sha256=run.validate_bundle(bundle))
        update = dict(binding=reg["binding"], baseline_sha256=run.sha256_file(out/"baseline.json"), arms=results,
                      brier=critic["brier"], original_actor1_unchanged=True, no_closed_loop=True)
        run.once(out/"update.json", run.sealed(update))
    return update


def parity(torch_side=False):
    reg, plan, source, out = verify()
    receipt = run.check_seal(run.read_json(out/"update.json"))
    require(receipt["binding"] == reg["binding"] and receipt["baseline_sha256"] == run.sha256_file(out/"baseline.json"), "update receipt")
    fixtures = run.check_seal(run.read_json(source/"torch-parity-1.json"))["rows"]
    with recovery.strict_lock(out, reg["binding"], "crossfit-parity"):
        outputs, max_error = {}, 0.
        reference = None if torch_side else run.check_seal(run.read_json(out/"torch-parity.json"))
        if reference is not None: require(reference["binding"] == reg["binding"], "parity receipt")
        for arm in reg["config"]["arms"]:
            path = out/"models"/(arm+".json")
            require(run.sha256_file(path) == receipt["arms"][arm]["file_sha256"], "model changed")
            bundle = run.read_json(path)
            actor, values = NumpyActor(bundle), []
            if torch_side:
                from experiments.sa_onpolicy_actor import torch_distribution
                model = torch_actor(bundle)
            for i, f in enumerate(fixtures):
                actual = actor.probabilities(f["candidate_ids"], f["anchor_id"], f["features"])
                if torch_side:
                    target = dict(zip(f["candidate_ids"], torch_distribution(model, bundle, f["candidate_ids"], f["anchor_id"], f["features"]).probs.detach().numpy().tolist()))
                else: target = reference["probabilities"][arm][i]
                max_error = max(max_error, max(abs(actual[k]-target[k]) for k in actual))
                require(max_error <= 1e-12, "portable probability mismatch")
                for seed in range(16):
                    draw = run.random.Random(seed).random()
                    require(select_with_draw(actual, draw) == select_with_draw(target, draw), "portable action mismatch")
                values.append(target)
            outputs[arm] = values
        result = dict(binding=reg["binding"], update_sha256=run.sha256_file(out/"update.json"),
                      fixtures_per_arm=len(fixtures), max_error=max_error, no_solver=True)
        if torch_side:
            run.once(out/"torch-parity.json", run.sealed(dict(result, probabilities=outputs)))
        else:
            run.native_runtime(plan)
            require(reference["update_sha256"] == result["update_sha256"], "stale Torch parity")
            run.once(out/"parity.json", run.sealed(result))
            run.write_json(out/"run_status.json", dict(status="prototype_verified_not_evaluated", no_active_workers=True))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("prepare", "extract", "baseline", "train", "parity"))
    p.add_argument("--torch", action="store_true")
    args = p.parse_args()
    try:
        result = parity(args.torch) if args.phase == "parity" else globals()[args.phase]()
    except Exception as error:
        output = ROOT/run.read_json(ROOT/CONFIG)["output"]
        if (output/"registration.json").exists():
            run.write_json(output/"run_status.json", dict(status="error", error=repr(error), no_retry=True))
        raise
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
