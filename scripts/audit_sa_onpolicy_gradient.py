"""Read-only first-update score decomposition; no solver, optimizer or model export."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
import math
import os
from pathlib import Path
import sys

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts import recover_sa_onpolicy as recovery
from scripts import run_sa_onpolicy as run
from experiments.sa_onpolicy_actor import NumpyActor, vectorize
from experiments.sa_paired_completion import require

OUTPUT = "build/sa-onpolicy-gradient-audit-v1b"
PINS = {
    "report.json": "cdd4ad9cc812871dae2738fab10611d146c8f31710166db66e21428f8f926c80",
    "update-0.json": "a2813cddabf87b6ad427e847bc23d6101225d6b2af9c74f8cf54b26a31af3485",
    "models/actor-0.json": "7413bea77ca58e4604bce1fa0d6cfe923bb8668885ff9fc700db7804e2e240b2",
    "models/actor-1.json": "133e10805ac0eb87cb548c4eed55c525d8f2e9cd63fa5a89f78a261a24981b7e",
}
CODE = ("scripts/audit_sa_onpolicy_gradient.py", "tests/evaluation/test_sa_onpolicy_gradient.py",
        "docs/SA_ONPOLICY_GRADIENT_PROTOCOL_ZH.md")
STAGES = ("0-31", "32-127", "128-255")


def norm(value):
    return float(np.linalg.norm(value))


def cosine(a, b):
    denominator = norm(a) * norm(b)
    return None if denominator <= 1e-20 else float(np.clip(np.dot(a, b) / denominator, -1, 1))


def zero_output_score(hidden, probabilities, selected):
    """Exact score of w2 ONLY at w2=b2=0: 2*(h_selected-E[h]).

    This diagnostic identity is checked against torch autograd and the saved SGD
    step. It is not an alternative differentiation or training implementation.
    """
    hidden, p = np.asarray(hidden, dtype=float), np.asarray(probabilities, dtype=float)
    require(hidden.ndim == 2 and p.shape == (len(hidden),), "score shape")
    require(np.isfinite(hidden).all() and np.isfinite(p).all(), "nonfinite score")
    require((p > 0).all() and abs(float(p.sum()) - 1) <= 1e-12, "score probabilities")
    require(0 <= selected < len(hidden), "score selection")
    return 2 * (hidden[selected] - p @ hidden)


def require_zero_output(actor):
    require(not np.any(actor.w2) and not np.any(actor.b2), "identity only valid at zero output")


def summarize_rows(rows, replicas=(0, 1, 2, 3)):
    """Recompute independent same-condition baselines for each fixed subset."""
    require(len(replicas) >= 2 and len(replicas) == len(set(replicas)), "replica subset")
    groups = defaultdict(list)
    for row in rows:
        require(row["split"] == "train", "heldout input prohibited")
        if row["replica"] in replicas:
            groups[row["pair_id"]].append(row)
    require(groups, "empty groups")
    maps = defaultdict(list)
    for pair, values in groups.items():
        require(sorted(r["replica"] for r in values) == sorted(replicas), "incomplete group")
        require(len({r["map_id"] for r in values}) == 1, "mixed group maps")
        maps[values[0]["map_id"]].append(pair)
    vectors, weights, info = {}, {}, {}
    for pair, values in sorted(groups.items()):
        values = sorted(values, key=lambda r: r["replica"])
        m = values[0]["map_id"]
        weight = 1 / (len(maps) * len(maps[m]) * len(replicas))
        total = math.fsum(r["return_value"] for r in values)
        vector = np.zeros(32)
        for r in values:
            baseline = (total - r["return_value"]) / (len(replicas) - 1)
            coefficient = weight * (r["return_value"] - baseline)
            weights[r["episode_id"]] = coefficient
            vector += coefficient * np.asarray(r["score"])
        vectors[pair] = vector
        info[pair] = dict(map_id=m, successes=total, replicas=len(replicas), gradient_norm=norm(vector))
    return sum(vectors.values(), np.zeros(32)), vectors, weights, info


def directional_summary(rows):
    full, vectors, weights, groups = summarize_rows(rows)
    active = [pair for pair in vectors if norm(vectors[pair]) > 1e-15]
    total_norms = math.fsum(norm(vectors[pair]) for pair in active)
    omitted = [dict(pair_id=pair, held_out_gradient_norm=norm(vectors[pair]),
                    remaining_norm=norm(full - vectors[pair]),
                    held_out_vs_remaining_cosine=cosine(vectors[pair], full - vectors[pair])) for pair in active]
    halves = []
    for left, right in (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))):
        a, _, _, _ = summarize_rows(rows, left)
        b, _, _, _ = summarize_rows(rows, right)
        halves.append(dict(left=list(left), right=list(right), left_norm=norm(a), right_norm=norm(b), cosine=cosine(a, b)))
    return dict(full_gradient=full.tolist(), norm=norm(full), groups=groups,
                active_conditions=len(active), condition_coherence_ratio=norm(full)/total_norms if total_norms else None,
                leave_condition_out=omitted, split_replicas=halves,
                nonzero_credit_episodes=sum(v != 0 for v in weights.values())), weights


def episode_summary(job):
    folder = ROOT / job["folder"]
    row = run.result_read(folder, job["plan"])
    require(run.sha256_file(folder / "result.json") == job["result_sha256"], "source changed")
    require(row["split"] == "train" and row["status"] == "ok", "not an uncensored Train episode")
    actor = NumpyActor(job["actor"])
    require_zero_output(actor)
    require(row["policy_sha256"] == actor.sha, "policy mismatch")
    stages = {name: dict(count=0, score=np.zeros(32), saturation=0., score_norm=0.,
                         fisher_trace=0., hidden_spread=0.) for name in STAGES}
    total, steps, fixtures = np.zeros(32), [], []
    for e in run.trace_read(folder):
        ids = e["candidate_ids"]
        require(ids == sorted(set(ids)), "noncanonical pool")
        expected = actor.probabilities(ids, e["anchor_id"], e["features"])
        require(max(abs(expected[k] - e["probabilities"][k]) for k in ids) <= 1e-12, "behavior mismatch")
        x = vectorize(e["features"], actor.bundle["feature_names"])
        h = np.tanh(((x - actor.mean) / actor.scale) @ actor.w1.T + actor.b1)
        p = np.asarray([expected[k] for k in ids])
        score = zero_output_score(h, p, ids.index(e["selected_id"]))
        total += score
        stage = STAGES[0] if e["decision"] < 32 else STAGES[1] if e["decision"] < 128 else STAGES[2]
        s = stages[stage]
        s["count"] += 1
        s["score"] += score
        s["saturation"] += float(np.mean(np.abs(h) >= .99))
        s["score_norm"] += norm(score)
        s["hidden_spread"] += float(np.max(np.ptp(h, axis=0)))
        s["fisher_trace"] += float(4 * (p[:, None] * (h - p @ h)**2).sum())
        steps.append({k: e[k] for k in ("decision", "policy_sha256", "probabilities", "selected_id", "behavior_log_probability")})
        if not any(f["stage"] == stage for f in fixtures):
            fixtures.append(dict(stage=stage, ids=ids, anchor=e["anchor_id"], features=e["features"],
                                 selected_id=e["selected_id"], score=score.tolist()))
    require(len(steps) == row["decisions"], "incomplete trace")
    for s in stages.values():
        s["score"] = s["score"].tolist()
    return dict(episode=row, steps=steps, score=total.tolist(), stages=stages, fixtures=fixtures,
                result_sha256=job["result_sha256"])


def autograd_checks(actor, summaries):
    import torch
    from experiments.sa_onpolicy_actor import torch_actor, torch_distribution
    torch.set_num_threads(1)
    model = torch_actor(actor)
    errors, other, count = 0., 0., 0
    for summary in summaries:
        for f in summary["fixtures"]:
            distribution = torch_distribution(model, actor, f["ids"], f["anchor"], f["features"])
            gradient = torch.autograd.grad(distribution.log_prob(torch.tensor(f["ids"].index(f["selected_id"]))), tuple(model.parameters()))
            errors = max(errors, float(np.max(np.abs(gradient[2].detach().numpy()[0] - f["score"]))))
            other = max(other, *(float(t.abs().max()) for i, t in enumerate(gradient) if i != 2))
            count += 1
    require(errors < 1e-12 and other < 1e-12, "analytic diagnostic vs autograd mismatch")
    return dict(fixtures=count, score_max_error=errors, other_parameter_gradient_max=other)


def prepare(workers):
    reg, p, source = recovery.verify()
    require(1 <= workers <= 20, "workers must be 1..20")
    out = ROOT / OUTPUT
    require(not out.exists(), "audit output already exists")
    for name, digest in PINS.items():
        require(run.sha256_file(source / name) == digest, "first-update evidence changed: " + name)
    batch = recovery.audited_batch(reg, p, source)
    require(len(batch) == 96 and all(row["split"] == "train" for _, _, row in batch), "Train-only batch")
    inputs = {n: run.sha256_file(ROOT / n) for n in CODE}
    for name in (*PINS, "batch.audit.json", "registration.json"):
        inputs[(source / name).relative_to(ROOT).as_posix()] = run.sha256_file(source / name)
    jobs = [dict(folder=folder.relative_to(ROOT).as_posix(), result_sha256=run.sha256_file(folder / "result.json"))
            for _, folder, _ in batch]
    body = dict(schema="lns2.sa.onpolicy_gradient_audit.v1", inputs=inputs, jobs=jobs, workers=workers,
                source=source.relative_to(ROOT).as_posix(), source_binding=reg["binding"],
                source_commit=run.subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                no_solver=True, no_parameter_updates=True, no_heldout=True)
    body["binding"] = run.json_fingerprint(body)
    with recovery.strict_lock(out, body["binding"], "gradient-prepare"):
        run.once(out / "registration.json", run.sealed(body))
    return dict(registered=True, jobs=len(jobs), workers=workers)


def analyze():
    reg, p, source = recovery.verify()
    out = ROOT / OUTPUT
    registration = run.check_seal(run.read_json(out / "registration.json"))
    require(registration["binding"] == run.json_fingerprint({k: v for k, v in registration.items() if k not in ("binding", "integrity")}), "registration identity")
    require(registration["source_binding"] == reg["binding"], "source registration")
    for name, digest in registration["inputs"].items():
        require(run.sha256_file(ROOT / name) == digest, "changed registered input: " + name)
    actor = run.actor_load(source, p, 0)
    newer = run.actor_load(source, p, 1)
    require_zero_output(NumpyActor(actor))
    with recovery.strict_lock(out, registration["binding"], "gradient-analysis"):
        require(not (out / "report.json").exists(), "existing result")
        run.write_json(out / "run_status.json", dict(status="running", jobs=96, workers=registration["workers"]))
        jobs = [dict(j, plan=p, actor=actor) for j in registration["jobs"]]
        summaries = []
        with ProcessPoolExecutor(max_workers=registration["workers"]) as pool:
            for summary in pool.map(episode_summary, jobs, chunksize=1):
                summaries.append(summary)
                if len(summaries) % 12 == 0:
                    print(f"Audited {len(summaries)}/96 trajectories", flush=True)
        rows = [dict(s["episode"], steps=s["steps"], score=s["score"],
                     return_value=float(s["episode"]["success"])) for s in summaries]
        coefficients = run.gradient_coefficients(rows, policy_sha256=run.validate_bundle(actor),
            expected_groups={e["job"]["pair_id"]: e["job"]["case"]["map_id"] for e in reg["entries"].values()},
            replicas=4, max_decisions=256, node_budget=25000000)
        direction, weights = directional_summary(rows)
        require(all(abs(weights[r["episode_id"]] - r["coefficient"]) < 1e-15 for r in coefficients), "weight reconstruction")
        gradient = np.asarray(direction["full_gradient"])
        update = run.check_seal(run.read_json(source / "update-0.json"))
        require(abs(norm(gradient) - update["gradient_norm_before_clip"]) < 1e-10, "gradient norm reconstruction")
        expected = .001 * gradient * min(1., 1. / (norm(gradient) + 1e-6))
        movement = (np.asarray(newer["w2"]) - actor["w2"])[0]
        error = float(np.max(np.abs(movement - expected)))
        require(error < 1e-12, "saved SGD step reconstruction")
        require(actor["w1"] == newer["w1"] and actor["b1"] == newer["b1"] and
                actor["mean"] == newer["mean"] and actor["scale"] == newer["scale"], "unexpected representation update")
        checks = autograd_checks(actor, summaries)
        stages = {}
        for name in STAGES:
            count = sum(s["stages"][name]["count"] for s in summaries)
            v = sum((weights[s["episode"]["episode_id"]] * np.asarray(s["stages"][name]["score"]) for s in summaries), np.zeros(32))
            stages[name] = dict(decisions=count, gradient_norm=norm(v), gradient_cosine_to_full=cosine(v, gradient),
                **{"mean_"+k: math.fsum(s["stages"][name][k] for s in summaries)/count if count else None
                   for k in ("saturation", "score_norm", "fisher_trace", "hidden_spread")})
        # Relative to actor-0, every logit lies in [-B,B]. TV <= tanh(B/2).
        bound = 2 * math.tanh(float(np.abs(newer["w2"]).sum() + np.abs(newer["b2"]).sum()))
        original_report = run.check_seal(run.read_json(source / "report.json"))
        report = dict(schema=registration["schema"], binding=registration["binding"], gradient=direction,
            stages=stages, autograd=checks, saved_update_max_error=error,
            universal_probability_tv_upper_bound=math.tanh(bound/2),
            old_state_probability_tv_mean=original_report["mean_total_variation"],
            sampled_action_changes=original_report["same_draw_changed_actions"],
            decisions=original_report["decisions_replayed"],
            nonzero_episode_ids=[k for k, v in weights.items() if v != 0],
            decision="posthoc_gradient_diagnostic_not_performance_evidence",
            no_solver=True, no_new_training=True, no_heldout=True, no_learning_rate_search=True)
        run.once(out / "episodes.json", run.sealed(dict(binding=registration["binding"],
            rows=[dict(episode_id=r["episode_id"], pair_id=r["pair_id"], map_id=r["map_id"], replica=r["replica"],
                       return_value=r["return_value"], score=r["score"], coefficient=weights[r["episode_id"]]) for r in rows])))
        run.once(out / "report.json", run.sealed(report))
        for name, digest in registration["inputs"].items():
            require(run.sha256_file(ROOT / name) == digest, "source changed during audit")
        run.write_json(out / "run_status.json", dict(status="complete", no_workers=True, no_new_training=True))
    return {k: report[k] for k in ("decision", "saved_update_max_error", "autograd", "universal_probability_tv_upper_bound", "stages", "gradient")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "analyze"))
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()
    try:
        value = prepare(args.workers) if args.phase == "prepare" else analyze()
    except Exception as error:
        if args.phase == "analyze" and (ROOT / OUTPUT / "registration.json").exists():
            run.write_json(ROOT / OUTPUT / "run_status.json", dict(status="error", error=repr(error), no_automatic_retry=True))
        raise
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
