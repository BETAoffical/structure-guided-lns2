"""Read-only diagnosis of a stopped batch; never supplies training admission."""
from collections import Counter, defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_sa_onpolicy as run
import numpy as np
from experiments.sa_onpolicy_actor import NumpyActor, vectorize


def hidden_metrics(features, actor):
    x = vectorize(features, actor.bundle["feature_names"])
    z = (x - actor.mean) / actor.scale
    h = np.tanh(z @ actor.w1.T + actor.b1)
    spread = float(np.max(np.ptp(h, axis=0)))
    return dict(saturated_fraction=float(np.mean(np.abs(h) >= .99)),
                normalized_over_10_fraction=float(np.mean(np.abs(z) > 10)),
                max_candidate_hidden_spread=spread,
                nearly_action_invariant=spread <= 1e-6)


def diagnose():
    p, out = run.verify()
    actor = NumpyActor(run.actor_load(out, p, 0))
    groups, stages, rows = defaultdict(list), defaultdict(list), []
    jobs = run.jobs_for(p, "train-0", 0)
    for j in jobs:
        folder = run.folder_for(out, "train-0", j)
        if not (folder / "result.json").exists():
            continue
        row = run.result_read(folder, p)
        initial_nodes = run.read_json(folder / "initial.json")["low_level"]["generated"]
        current_nodes = initial_nodes
        pp_seconds, native_seconds = 0., 0.
        last = None
        for e in run.trace_read(folder):
            stage = "0-31" if e["decision"] < 32 else "32-127" if e["decision"] < 128 else "128-255"
            stages[stage].append(hidden_metrics(e["features"], actor))
            m = e["metrics"]
            pp_seconds += m["pp_replan_seconds"]
            native_seconds += m["native_step_seconds"]
            last = dict(decision=e["decision"], before_generated=current_nodes-initial_nodes,
                        requested_pp_seconds=m["requested_pp_time_limit_seconds"],
                        actual_pp_seconds=m["pp_replan_seconds"], failure_reason=m["pp_failure_reason"],
                        acceptance_evaluated=m["acceptance_evaluated"], rolled_back=m["pp_rolled_back"],
                        attempted_agents=m["pp_attempted_agent_count"], neighborhood_size=len(m["neighborhood"]))
            current_nodes = e["delta"]["top_set"].get("low_level", {"generated": current_nodes})["generated"]
        item = {k: row[k] for k in ("job_id", "pair_id", "map_id", "replica", "status", "stop", "success", "decisions", "generated", "final_conflicts", "seconds_diagnostic")}
        item.update(pp_seconds=pp_seconds, native_seconds=native_seconds, last_step=last,
                    result_sha256=run.sha256_file(folder/"result.json"))
        rows.append(item)
        groups[row["pair_id"]].append(item)
    group_rows = []
    for key, items in sorted(groups.items()):
        eligible = len(items) == p["proposal"]["train_replicas_per_condition"] and all(x["status"] == "ok" for x in items)
        group_rows.append(dict(pair_id=key, collected=len(items), uncensored_complete=eligible,
                               completed=sum(x["success"] for x in items),
                               contrasting_returns=eligible and len({x["success"] for x in items}) > 1))
    stage_rows = {}
    for key, values in stages.items():
        stage_rows[key] = dict(decisions=len(values),
            mean_saturated_fraction=float(np.mean([v["saturated_fraction"] for v in values])),
            mean_normalized_over_10_fraction=float(np.mean([v["normalized_over_10_fraction"] for v in values])),
            nearly_action_invariant_count=sum(v["nearly_action_invariant"] for v in values),
            median_max_candidate_hidden_spread=float(np.median([v["max_candidate_hidden_spread"] for v in values])))
    result = dict(schema="lns2.sa.onpolicy_batch_diagnostic.v1", binding=p["binding"],
        script_sha256=run.sha256_file(Path(__file__)), policy_sha256=actor.sha,
        expected=len(jobs), collected=len(rows), unstarted=len(jobs)-len(rows),
        stop_counts=dict(Counter(r["stop"] for r in rows)), groups=group_rows, stages=stage_rows,
        episodes=rows, no_update=True, no_ttf=True, posthoc_training_inputs_only=True,
        decision="incomplete_censored_batch_not_a_model_performance_result")
    run.once(out/"train-0.diagnostic.json", run.sealed(result))
    return {k: result[k] for k in ("expected", "collected", "unstarted", "stop_counts", "stages", "decision")}


if __name__ == "__main__":
    import json
    print(json.dumps(diagnose(), indent=2))
