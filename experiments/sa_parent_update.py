"""Fixed-prior episodic update for a nonzero residual parent policy."""
import math

import numpy as np

from experiments.sa_crossfit_update import padded_episode, probability_stats, within_budget
from experiments.sa_onpolicy_actor import export_actor, torch_actor, validate_bundle
from experiments.sa_paired_completion import require


def padded_with_prior(data, parent):
    validate_bundle(parent)
    require(np.asarray(data["lengths"]).dtype.kind in "iu", "integer candidate counts")
    padded = padded_episode(data, parent)
    _, behavior, valid = padded
    lengths = np.asarray(data["lengths"])
    anchors = np.asarray(data["anchors"])
    require(anchors.shape == lengths.shape and anchors.dtype.kind in "iu" and
            np.all((anchors >= 0) & (anchors < lengths)), "anchor indices")
    prior = np.where(valid, .1 / lengths[:, None], 0.)
    prior[np.arange(len(lengths)), anchors] += .9
    require(prior.shape == behavior.shape, "prior/behavior alignment")
    return padded, prior


def log_distribution(model, parent, padded, prior):
    import torch
    x, _, valid = padded
    counts = valid.sum(axis=1)
    require(prior.shape == valid.shape and (counts > 0).all() and np.isfinite(prior).all(), "prior shape")
    expected = np.where(valid, .1 / counts[:, None], 0.)
    expected[np.arange(len(counts)), np.argmax(prior, axis=1)] += .9
    require(np.allclose(prior, expected, atol=1e-15, rtol=0), "use fixed prior, not behavior probability")
    x = torch.as_tensor(x, dtype=torch.float64)
    x = (x - torch.tensor(parent["mean"], dtype=torch.float64)) / torch.tensor(parent["scale"], dtype=torch.float64)
    logits = torch.as_tensor(prior, dtype=torch.float64).clamp_min(1e-300).log()
    logits = logits + 2 * torch.tanh(model(x).squeeze(-1))
    return torch.log_softmax(logits.masked_fill(~torch.as_tensor(valid), -torch.inf), dim=1)


def guarded_parent_update(parent, gradient, packs, config, binding, arm):
    import torch
    parent_sha = validate_bundle(parent)
    require(parent["iteration"] >= 1 and (np.any(parent["w2"]) or np.any(parent["b2"])), "nonzero parent required")
    parameters = tuple(torch_actor(parent).parameters())
    require(len(gradient) == len(parameters) and all(g.shape == p.shape and torch.isfinite(g).all()
            for g, p in zip(gradient, parameters)), "finite aligned gradient")
    magnitude = math.sqrt(sum(float((g * g).sum()) for g in gradient))
    require(magnitude > 0 and math.isfinite(magnitude), "no finite update direction")
    history = []
    for attempt in range(config["backtracks"]):
        distance = config["initial_l2"] / 2**attempt
        model = torch_actor(parent)
        optimizer = torch.optim.SGD(model.parameters(), lr=distance / magnitude)
        for parameter, grad in zip(model.parameters(), gradient):
            parameter.grad = grad.clone()
        optimizer.step()
        with torch.no_grad():
            probabilities = [log_distribution(model, parent, p["padded"], p["prior"]).exp().numpy() for p in packs]
        stats = probability_stats(packs, probabilities)
        history.append(dict(parameter_l2=distance, **stats))
        require(validate_bundle(parent) == parent_sha, "parent mutated")
        if within_budget(stats, config):
            bundle = export_actor(model, parent, binding)
            bundle["prototype_arm"] = arm
            return bundle, dict(gradient_norm=magnitude, chosen=history[-1], backtracking=history,
                                one_gradient=True, no_closed_loop_evidence=True)
    return None, dict(gradient_norm=magnitude, backtracking=history, decision="no_step_within_budget")
