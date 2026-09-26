"""Versioned episodic credit with no decision limit; historical credit is frozen."""

from experiments._common import strict_int
from experiments.sa_onpolicy_contract import gradient_coefficients as shared_gradient_coefficients
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
    """Reuse the credit formula with the uncapped terminal validator."""
    return shared_gradient_coefficients(
        episodes, policy_sha256=policy_sha256, expected_groups=expected_groups,
        replicas=replicas, max_decisions=max_decisions, node_budget=node_budget,
        terminal_validator=terminal_return)


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
