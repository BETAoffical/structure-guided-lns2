"""Bounded acceptance diagnostic, not a production policy or CPLNS reproduction."""

import math
import random

from experiments.state_analysis import summarize_initial_state_complexity

ARMS = ("standard", "complete_greedy", "annealed")


def temperature(iteration):
    if type(iteration) is not int or iteration < 0:
        raise ValueError("invalid temperature iteration")
    return 1000.0 * 0.99 ** iteration


def probability(delta, temp):
    if not math.isfinite(temp) or temp < 0:
        raise ValueError("invalid temperature")
    return 1.0 if delta <= 0 else (math.exp(-delta / temp) if temp else 0.0)


def acceptance_draw(seed):
    return random.Random(seed).random()


def validate_transition(before, after, metrics, selected, arm, temp, draw):
    if arm not in ARMS or not 0 <= draw < 1:
        raise ValueError("invalid diagnostic request")
    if not metrics["action_valid"] or not metrics["step_applied"] or sorted(selected) != sorted(metrics["neighborhood"]):
        raise ValueError("modified or invalid action")
    old = {a["id"]: a["path"] for a in before["agents"]}
    new = {a["id"]: a["path"] for a in after["agents"]}
    if old.keys() != new.keys() or any(old[k] != new[k] for k in old if k not in selected):
        raise ValueError("external paths changed")
    if metrics["pp_rolled_back"] and (old != new or before["sum_of_costs"] != after["sum_of_costs"]):
        raise ValueError("rollback mismatch")
    if before["num_of_colliding_pairs"] != metrics["conflicts_before"] or after["num_of_colliding_pairs"] != metrics["conflicts_after"]:
        raise ValueError("transition conflict count mismatch")
    delta = metrics["pp_attempt_conflict_pair_count"] - metrics["pp_old_conflict_pair_count"]
    if arm == "standard":
        if after["num_of_colliding_pairs"] > before["num_of_colliding_pairs"]:
            raise ValueError("standard conflicts increased")
    else:
        if metrics["experimental_acceptance"] != arm or metrics["acceptance_temperature"] != temp or metrics["acceptance_uniform"] != draw:
            raise ValueError("acceptance request mismatch")
        if metrics["acceptance_evaluated"]:
            if metrics["pp_inserted_agent_count"] != len(selected):
                raise ValueError("accepted incomplete planning")
            expected = probability(delta, temp if arm == "annealed" else 0)
            if not math.isclose(expected, metrics["acceptance_probability"], abs_tol=1e-14, rel_tol=1e-12):
                raise ValueError("probability mismatch")
            if metrics["replan_success"] != (delta <= 0 or draw < expected):
                raise ValueError("acceptance decision mismatch")
        elif metrics["replan_success"]:
            raise ValueError("success without acceptance evaluation")
    if metrics["replan_success"] and after["num_of_colliding_pairs"] - before["num_of_colliding_pairs"] != delta:
        raise ValueError("accepted global delta mismatch")
    # Reconstruct vertex/edge/terminal-wait conflicts from actual paths.
    summary = summarize_initial_state_complexity(after)
    if summary["total_path_cost"] != after["sum_of_costs"]:
        raise ValueError("path cost mismatch")


def summarize(rows):
    tables = {arm: {} for arm in ARMS}
    for row in rows:
        key = (row["case_id"], row["trial"])
        if row["arm"] not in tables or key in tables[row["arm"]]:
            raise ValueError("duplicate or unknown arm")
        tables[row["arm"]][key] = row
    keys = set(tables["standard"])
    if not keys or any(set(t) != keys for t in tables.values()):
        raise ValueError("incomplete paired results")
    result = {}
    for arm, table in tables.items():
        result[arm] = {"episodes": len(keys), "feasible": sum(r["feasible"] for r in table.values()),
                       "censored": sum(r["censored"] for r in table.values()),
                       "generated": sum(r["generated"] for r in table.values()),
                       "final_conflicts": sum(r["conflicts"][-1] for r in table.values()),
                       "auc_sum": sum(r["auc"] for r in table.values()),
                       "accepted_increases": sum(r["accepted_increases"] for r in table.values())}
    comparisons = {}
    for base in ("standard", "complete_greedy"):
        gains = [k for k in sorted(keys) if tables["annealed"][k]["feasible"] and not tables[base][k]["feasible"]]
        losses = [k for k in sorted(keys) if tables[base][k]["feasible"] and not tables["annealed"][k]["feasible"]]
        comparisons[base] = {"gains": gains, "losses": losses,
                             "generated_ratio": result["annealed"]["generated"] / max(1, result[base]["generated"])}
    passed = (not any(r["censored"] for r in rows) and all(
        len(c["gains"]) >= 2 and len({k[0] for k in c["gains"]}) >= 2 and not c["losses"] and
        c["generated_ratio"] <= 1.25 for c in comparisons.values()))
    return {"arms": result, "annealed_vs": comparisons,
            "decision": "development_signal_only" if passed else (
                "inconclusive_resource" if any(r["censored"] for r in rows) else "no_go_this_pilot"),
            "no_ttf_or_promotion": True}
