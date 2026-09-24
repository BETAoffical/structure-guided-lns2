"""Isolated raw-logit policy contract; no optimizer, rollout or default integration."""
from copy import deepcopy
import math
import numpy as np

from experiments._common import json_fingerprint
from experiments.sa_crossfit_update import probability_stats, within_budget
from experiments.sa_onpolicy_actor import NumpyActor, select_with_draw, validate_bundle as validate_base, vectorize
from experiments.sa_paired_completion import require

SCHEMA = "lns2.sa.raw_residual_actor.v1"
LIMITS = dict(mean_kl=.002, mean_trajectory_kl=.1, max_state_kl=.02)


def validate_bundle(bundle):
    require(bundle["schema"] == SCHEMA, "raw actor schema")
    base = bundle["base"]
    require(validate_base(base) == bundle["base_policy_sha256"], "frozen base changed")
    require(len(base["feature_names"]) == 129, "registered 129-feature schema")
    require(bundle["normalization"] == "frozen_base" and bundle["output"] == "raw_logits", "raw policy contract")
    require(type(bundle["iteration"]) is int and bundle["iteration"] >= 0, "raw iteration")
    parent = bundle["parent_policy"]
    require(parent is None if bundle["iteration"] == 0 else
            isinstance(parent, str) and len(parent) == 64 and all(c in "0123456789abcdef" for c in parent), "raw lineage")
    for name, shape in {"w1":(32,129), "b1":(32,), "w2":(1,32), "b2":(1,)}.items():
        value = np.asarray(bundle["correction"][name], dtype=np.float64)
        require(value.shape == shape and np.isfinite(value).all(), "finite raw layer: " + name)
    return json_fingerprint(bundle)


def initial_bundle(base, binding):
    """No new random initialization: copy the base hidden layer, zero the raw head."""
    validate_base(base)
    result = dict(schema=SCHEMA, binding=binding, iteration=0, parent_policy=None,
                  base=deepcopy(base), base_policy_sha256=validate_base(base),
                  normalization="frozen_base", output="raw_logits",
                  correction=dict(w1=deepcopy(base["w1"]), b1=deepcopy(base["b1"]),
                                  w2=[[0.] * 32], b2=[0.]))
    validate_bundle(result)
    return result


def canonical_inputs(base, ids, anchor, features):
    require(ids and all(isinstance(c, str) and c for c in ids) and len(ids) == len(set(ids))
            and anchor in ids and len(ids) == len(features), "candidate identity/alignment")
    order = sorted(range(len(ids)), key=ids.__getitem__)
    ids, features = [ids[i] for i in order], [features[i] for i in order]
    x = vectorize(features, base["feature_names"])
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        z = (x - np.asarray(base["mean"])) / np.asarray(base["scale"])
    require(np.isfinite(z).all(), "nonfinite normalized features")
    return ids, features, z


def corrected_probabilities(parent, corrections):
    ids = sorted(parent)
    require(ids and set(corrections) == set(ids), "raw correction coverage")
    require(all(math.isfinite(parent[c]) and parent[c] > 0 and math.isfinite(corrections[c]) for c in ids)
            and math.isclose(math.fsum(parent.values()), 1., rel_tol=0, abs_tol=1e-12), "finite positive base distribution")
    # Preserve bitwise parent probabilities for a zero/common offset, including all CDF boundaries.
    if all(corrections[c] == corrections[ids[0]] for c in ids):
        return {c:parent[c] for c in ids}
    logits = {c:math.log(parent[c]) + corrections[c] for c in ids}
    require(all(math.isfinite(v) for v in logits.values()), "nonfinite corrected logits")
    peak = max(logits.values())
    mass = {c:math.exp(logits[c] - peak) for c in ids}
    total = math.fsum(mass.values())
    result = {c:mass[c] / total for c in ids}
    require(all(v > 0 and math.isfinite(v) for v in result.values()), "raw probability underflow; do not clip or fall back")
    return result


class RawResidualActor:
    def __init__(self, bundle):
        self.sha = validate_bundle(bundle)
        self.bundle = deepcopy(bundle)
        self.base = NumpyActor(self.bundle["base"])
        for key, value in self.bundle["correction"].items():
            setattr(self, key, np.asarray(value, dtype=np.float64))

    def probabilities(self, ids, anchor, features):
        ids, features, z = canonical_inputs(self.bundle["base"], ids, anchor, features)
        parent = self.base.probabilities(ids, anchor, features)
        with np.errstate(over="ignore", invalid="ignore"):
            hidden = np.tanh(z @ self.w1.T + self.b1)
            raw = (hidden @ self.w2.T + self.b2)[:, 0]
        return corrected_probabilities(parent, dict(zip(ids, raw.tolist())))


def torch_correction(bundle):
    import torch
    validate_bundle(bundle)
    with torch.random.fork_rng(devices=[]):
        model = torch.nn.Sequential(torch.nn.Linear(129, 32, dtype=torch.float64),
                                    torch.nn.Tanh(), torch.nn.Linear(32, 1, dtype=torch.float64))
    with torch.no_grad():
        for layer, prefix in ((model[0], "1"), (model[2], "2")):
            layer.weight.copy_(torch.tensor(bundle["correction"]["w" + prefix], dtype=torch.float64))
            layer.bias.copy_(torch.tensor(bundle["correction"]["b" + prefix], dtype=torch.float64))
    return model


def torch_log_distribution(model, bundle, ids, anchor, features):
    import torch
    ids, features, z = canonical_inputs(bundle["base"], ids, anchor, features)
    parent = NumpyActor(bundle["base"]).probabilities(ids, anchor, features)
    raw = model(torch.tensor(z, dtype=torch.float64)).squeeze(-1)
    require(raw.shape == (len(ids),) and bool(torch.isfinite(raw).all()), "raw tensor shape/values")
    log_parent = torch.tensor([math.log(parent[c]) for c in ids], dtype=torch.float64)
    logs = torch.log_softmax(log_parent + raw, dim=0)
    require(bool(torch.isfinite(logs).all()) and bool((logs.exp() > 0).all()), "tensor probability underflow")
    # No zero-head shortcut here: gradients must still flow at the exact-parent initialization.
    return logs


def export_correction(model, previous, binding):
    result = deepcopy(previous)
    result.update(iteration=previous["iteration"] + 1, parent_policy=validate_bundle(previous), update_binding=binding)
    for layer, prefix in ((model[0], "1"), (model[2], "2")):
        result["correction"]["w" + prefix] = layer.weight.detach().cpu().numpy().tolist()
        result["correction"]["b" + prefix] = layer.bias.detach().cpu().numpy().tolist()
    validate_bundle(result)
    return result


def evaluate_step(previous, proposed, episodes, limits=LIMITS):
    """Measure an update against its immediate behavior policy, not forever against the base.

    This returns a diagnostic. It cannot authorize training or establish safety on unseen states.
    Each episode supplies its weight, input states, behavior probabilities and sampling draws.
    """
    old_sha, _ = validate_bundle(previous), validate_bundle(proposed)
    require(proposed["base"] == previous["base"] and proposed["base_policy_sha256"] == previous["base_policy_sha256"], "base/normalization changed")
    require(proposed["parent_policy"] == old_sha and proposed["iteration"] == previous["iteration"] + 1, "immediate parent required")
    require(limits == LIMITS and episodes, "fixed step limits/coverage")
    old, new = RawResidualActor(previous), RawResidualActor(proposed)
    packs, probabilities = [], []
    seen = set()
    weights = [e["weight"] for e in episodes]
    require(all(math.isfinite(w) and w > 0 for w in weights) and
            math.isclose(math.fsum(weights), 1., abs_tol=1e-12, rel_tol=0), "episode weights")
    for episode in episodes:
        require(episode["episode_id"] not in seen and episode["split"] == "train" and
                episode["policy_sha256"] == old_sha, "unique Train behavior episodes")
        seen.add(episode["episode_id"])
        states = episode["states"]
        if not states:
            require(episode.get("initially_feasible") is True, "unexplained empty episode")
            continue
        lengths = np.asarray([len(s["ids"]) for s in states], dtype=int)
        width = int(lengths.max())
        a, b = np.zeros((len(states), width)), np.zeros((len(states), width))
        mask, selected, draws = np.zeros_like(a, dtype=bool), [], []
        for i, state in enumerate(states):
            ids = sorted(state["ids"])
            p = old.probabilities(state["ids"], state["anchor"], state["features"])
            require(set(state["probabilities"]) == set(ids) and
                    max(abs(p[c] - state["probabilities"][c]) for c in ids) <= 1e-12, "stale behavior probabilities")
            require(select_with_draw(p, state["draw"]) == state["selected"], "behavior draw mismatch")
            q = new.probabilities(state["ids"], state["anchor"], state["features"])
            a[i, :len(ids)], b[i, :len(ids)] = [p[c] for c in ids], [q[c] for c in ids]
            mask[i, :len(ids)] = True
            selected.append(ids.index(state["selected"]))
            draws.append(state["draw"])
        packs.append(dict(padded=(None, a, mask), weight=episode["weight"], lengths=lengths,
                          selected=np.asarray(selected), draws=np.asarray(draws)))
        probabilities.append(b)
    active_weight = math.fsum(p["weight"] for p in packs)
    if packs:
        for pack in packs:
            pack["weight"] /= active_weight
        stats = probability_stats(packs, probabilities)
        for key in ("mean_kl", "mean_trajectory_kl", "mean_tv"):
            stats[key] *= active_weight
    else:
        stats = dict(mean_kl=0., mean_trajectory_kl=0., max_state_kl=0., mean_tv=0., changed_actions=0, decisions=0)
    return dict(within_step_budget=bool(packs) and within_budget(stats, limits), stats=stats,
                zero_action_episodes=len(episodes)-len(packs),
                reference_policy_sha256=old_sha, no_performance_claim=True)
