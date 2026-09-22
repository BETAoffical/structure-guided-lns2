"""Versioned episodic credit with no decision limit; historical credit is frozen."""
from collections import Counter, defaultdict
import math

from experiments._common import strict_int
from experiments.sa_onpolicy_contract import _number, _sha
from experiments.sa_paired_completion import require


def terminal_return(row, *, max_decisions, node_budget):
    require(max_decisions is None, "training must not impose a decision cap")
    strict_int(node_budget, field="node_budget", minimum=1)
    strict_int(row["decisions"], field="decisions")
    nodes = strict_int(row["generated"], field="generated")
    conflicts = strict_int(row["final_conflicts"], field="final_conflicts")
    require(type(row["success"]) is bool, "success must be boolean")
    stop, status = row["stop"], row["status"]
    if stop in {"wall_safety", "incomplete_pp", "external_timeout", "user_stop"}:
        require(status == "censored", "interruption must remain censored")
        return None
    require(status == "ok", "unexplained execution error")
    require(row["success"] == (stop == "feasible") == (conflicts == 0), "terminal state mismatch")
    if stop == "feasible":
        return 1.0
    require(stop == "node_budget" and nodes >= node_budget, "unknown or premature terminal reason")
    return 0.0


def gradient_coefficients(episodes, *, policy_sha256, expected_groups, replicas,
                          max_decisions, node_budget):
    """Map-equal episode weights with independent leave-one-replica-out baselines.

    Each coefficient multiplies SUM(log pi(a_t | s_t)) for one complete episode.
    This validates declared metadata, not native paths or recomputed probabilities.
    """
    require(_sha(policy_sha256), "policy SHA required")
    strict_int(replicas, field="replicas", minimum=2)
    require(expected_groups and all(isinstance(k, str) and k and isinstance(v, str) and v
                                   for k, v in expected_groups.items()), "expected groups required")
    require(len(episodes) == len(expected_groups) * replicas, "incomplete batch")
    groups = defaultdict(list)
    seen = set()
    for row in episodes:
        require(isinstance(row["episode_id"], str) and row["episode_id"] not in seen, "duplicate episode")
        seen.add(row["episode_id"])
        require(row["split"] == "train", "evaluation data cannot update policy")
        require(row["policy_sha256"] == policy_sha256, "mixed or stale policy batch")
        pair = row["pair_id"]
        require(pair in expected_groups and row["map_id"] == expected_groups[pair], "group/map mismatch")
        require(_sha(row["initial_fingerprint"]) and _sha(row["rng_stream_id"]), "initial/RNG identity required")
        strict_int(row["replica"], field="replica")
        require(row["replica"] < replicas, "replica out of range")
        value = terminal_return(row, max_decisions=max_decisions, node_budget=node_budget)
        require(value is not None, "censored batch: do not drop or impute episodes")
        require(len(row["steps"]) == row["decisions"], "missing trajectory steps")
        for decision, step in enumerate(row["steps"]):
            strict_int(step["decision"], field="decision")
            require(step["decision"] == decision, "trajectory order")
            require(step["policy_sha256"] == policy_sha256, "policy changed inside episode")
            probabilities = step["probabilities"]
            require(isinstance(probabilities, dict) and probabilities, "missing action probabilities")
            require(all(isinstance(c, str) and c and _number(p) and 0 < p <= 1
                        for c, p in probabilities.items()), "invalid action support")
            require(math.isclose(math.fsum(probabilities.values()), 1, rel_tol=0, abs_tol=1e-12), "probability sum")
            require(step["selected_id"] in probabilities, "action outside pool")
            logp = step["behavior_log_probability"]
            require(_number(logp) and math.isclose(logp, math.log(probabilities[step["selected_id"]]),
                                                  rel_tol=0, abs_tol=1e-12), "behavior log probability")
        groups[pair].append((row, value))
    require(set(groups) == set(expected_groups), "missing task group")
    maps = Counter(expected_groups.values())
    result = []
    for pair in sorted(groups):
        rows = sorted(groups[pair], key=lambda item: item[0]["replica"])
        require([r["replica"] for r, _ in rows] == list(range(replicas)), "duplicate or missing replica")
        require(len({r["initial_fingerprint"] for r, _ in rows}) == 1, "replicas changed initial state")
        require(len({r["rng_stream_id"] for r, _ in rows}) == replicas, "replicas reused random stream")
        total = math.fsum(v for _, v in rows)
        for row, value in rows:
            baseline = (total - value) / (replicas - 1)
            weight = 1 / (len(maps) * maps[row["map_id"]] * replicas)
            result.append(dict(episode_id=row["episode_id"], return_value=value,
                               baseline=baseline, episode_weight=weight,
                               coefficient=weight * (value - baseline)))
    return result


def signal_change(previous, current, old_credit, new_credit):
    """A changed coefficient or new nonzero-credit suffix can change the gradient."""
    old = {r["episode_id"]:r for r in previous}
    new = {r["episode_id"]:r for r in current}
    a = {r["episode_id"]:r for r in old_credit}
    b = {r["episode_id"]:r for r in new_credit}
    require(set(old) == set(new) == set(a) == set(b), "signal comparison coverage")
    returns = [k for k in old if old[k]["success"] != new[k]["success"]]
    coefficients = [k for k in old if a[k]["coefficient"] != b[k]["coefficient"]]
    suffixes = {k:new[k]["decisions"]-old[k]["decisions"] for k in old
                if new[k]["decisions"] > old[k]["decisions"] and b[k]["coefficient"] != 0}
    require(all(new[k]["decisions"] >= old[k]["decisions"] for k in old), "short recovered episode")
    possible = bool(coefficients or suffixes)
    return dict(changed_returns=returns, changed_coefficients=coefficients, credited_suffixes=suffixes,
                possible_gradient_change=possible,
                decision="one_fixed_update_eligible" if possible else "no_new_gradient_information_skip_retraining")
