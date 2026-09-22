"""Action-independent cross-fit baseline and a single KL-guarded gradient step."""
import math
import numpy as np

from experiments.sa_paired_completion import require
from experiments.sa_onpolicy_actor import export_actor, torch_actor


def state_columns(names):
    columns = [i for i, name in enumerate(names) if name.startswith(("state.", "sa.", "budget."))]
    require(columns, "empty state schema")
    return columns


def state_vector(x, columns):
    values = np.asarray(x, dtype=float)[:, columns]
    require(len(values) and np.isfinite(values).all(), "invalid state features")
    require(np.array_equal(values, np.broadcast_to(values[0], values.shape)), "candidate-dependent baseline feature")
    return values[0]


def fold_masks(metadata, held):
    require(held in range(4) and metadata, "cross-fit fold")
    require(all(r["split"] == "train" and r["replica"] in range(4) for r in metadata), "Train replicas only")
    train = np.asarray([r["replica"] != held for r in metadata])
    require(train.any() and (~train).any(), "empty fold")
    require(not ({r["episode_id"] for r, yes in zip(metadata, train) if yes} &
                 {r["episode_id"] for r, yes in zip(metadata, train) if not yes}), "episode leakage")
    return train, ~train


def fit_value(x, y, weights, alpha):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    require(alpha == 1. and np.isfinite(x).all() and np.isfinite(y).all(), "fixed finite critic")
    weights = np.asarray(weights, dtype=float)
    require((weights > 0).all(), "critic weights")
    weights /= weights.sum()
    scaler = StandardScaler().fit(x, sample_weight=weights)
    model = Ridge(alpha=alpha, solver="svd").fit(scaler.transform(x), y, sample_weight=weights)
    bundle = dict(mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(),
                  coefficient=model.coef_.tolist(), intercept=float(model.intercept_), alpha=alpha)
    np.testing.assert_allclose(predict_value(bundle, x), np.clip(model.predict(scaler.transform(x)), 0, 1), atol=1e-12, rtol=0)
    return bundle


def predict_value(bundle, x):
    return np.clip(((np.asarray(x) - bundle["mean"]) / bundle["scale"]) @ np.asarray(bundle["coefficient"]) + bundle["intercept"], 0, 1)


def padded_episode(data, actor):
    lengths = np.asarray(data["lengths"], dtype=int)
    require(len(lengths) > 0 and (lengths > 0).all() and int(lengths.sum()) == len(data["x"]), "ragged episode")
    n, width, dimension = len(lengths), int(lengths.max()), len(actor["feature_names"])
    x, old, valid = np.zeros((n, width, dimension)), np.zeros((n, width)), np.zeros((n, width), dtype=bool)
    offset = 0
    for i, count in enumerate(lengths):
        x[i, :count] = data["x"][offset:offset+count]
        old[i, :count] = data["probabilities"][offset:offset+count]
        valid[i, :count] = True
        offset += count
    require(np.allclose(old.sum(axis=1), 1, atol=1e-12, rtol=0) and (old[valid] > 0).all(), "behavior distribution")
    require(np.isfinite(x).all() and dimension == data["x"].shape[1], "actor schema")
    return x, old, valid


def tensor_distribution(model, bundle, padded):
    import torch
    x, old, valid = padded
    x = torch.as_tensor(x, dtype=torch.float64)
    x = (x - torch.tensor(bundle["mean"], dtype=torch.float64)) / torch.tensor(bundle["scale"], dtype=torch.float64)
    mask = torch.as_tensor(valid)
    q = torch.as_tensor(old, dtype=torch.float64)
    # The two prototype arms both start at actor-0, whose residual is exactly zero.
    logits = q.clamp_min(1e-300).log() + 2 * torch.tanh(model(x).squeeze(-1))
    return torch.log_softmax(logits.masked_fill(~mask, -torch.inf), dim=1)


def probability_stats(packs, new_probabilities):
    require(len(packs) == len(new_probabilities) and len(packs) > 0, "probability coverage")
    require(all(p["weight"] > 0 for p in packs) and
            math.isclose(sum(p["weight"] for p in packs), 1., abs_tol=1e-12, rel_tol=0), "map-equal probability weights")
    mean_kl, trajectory_kl, max_kl, tv, changed, count = 0., 0., 0., 0., 0, 0
    for pack, prob in zip(packs, new_probabilities):
        _, old, valid = pack["padded"]
        require(prob.shape == old.shape and (prob[valid] > 0).all() and np.isfinite(prob).all(), "new distribution")
        require(np.allclose(prob.sum(axis=1), 1, atol=1e-12, rtol=0), "new probability mass")
        term = np.zeros_like(old)
        term[valid] = old[valid] * np.log(old[valid] / prob[valid])
        kl = np.maximum(term.sum(axis=1), 0)
        weight = pack["weight"]
        mean_kl += weight * float(kl.mean())
        trajectory_kl += weight * float(kl.sum())
        max_kl = max(max_kl, float(kl.max()))
        tv += weight * float(.5 * np.abs(prob-old).sum(axis=1).mean())
        chosen = (np.cumsum(prob, axis=1) <= pack["draws"][:, None]).sum(axis=1)
        chosen = np.minimum(chosen, pack["lengths"] - 1)
        changed += int(np.sum(chosen != pack["selected"]))
        count += len(prob)
    return dict(mean_kl=mean_kl, mean_trajectory_kl=trajectory_kl, max_state_kl=max_kl,
                mean_tv=tv, changed_actions=changed, decisions=count)


def within_budget(stats, config):
    return all(stats[key] <= config[key] for key in ("mean_kl", "mean_trajectory_kl", "max_state_kl"))


def guarded_update(parent, gradient, packs, config, binding, arm):
    import torch
    require(not np.any(parent["w2"]) and not np.any(parent["b2"]), "fork must start at actor-0")
    magnitude = math.sqrt(sum(float((g*g).sum()) for g in gradient))
    require(math.isfinite(magnitude) and magnitude > 0, "no finite update direction")
    history = []
    for attempt in range(config["backtracks"]):
        distance = config["initial_l2"] / 2**attempt
        model = torch_actor(parent)
        optimizer = torch.optim.SGD(model.parameters(), lr=distance/magnitude)
        for parameter, grad in zip(model.parameters(), gradient):
            parameter.grad = grad.clone()
        optimizer.step()
        with torch.no_grad():
            probabilities = [tensor_distribution(model, parent, p["padded"]).exp().numpy() for p in packs]
        stats = probability_stats(packs, probabilities)
        history.append(dict(parameter_l2=distance, **stats))
        if within_budget(stats, config):
            bundle = export_actor(model, parent, binding)
            bundle["prototype_arm"] = arm
            return bundle, dict(gradient_norm=magnitude, chosen=history[-1], backtracking=history,
                                one_gradient=True, no_closed_loop_evidence=True)
    return None, dict(gradient_norm=magnitude, backtracking=history, decision="no_step_within_budget")
