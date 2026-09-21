"""Small episodic residual actor, with numpy inference and lazy torch training."""

import math
import numpy as np

from experiments._common import json_fingerprint
from experiments.sa_onpolicy_contract import anchor_distribution
from experiments.sa_paired_completion import require

SCHEMA = "lns2.sa.onpolicy_actor.v1"


def validate_bundle(bundle):
    require(bundle["schema"] == SCHEMA, "actor schema")
    names = bundle["feature_names"]
    require(names == sorted(set(names)) and names, "actor feature schema")
    d = len(names)
    for key, shape in {"mean": (d,), "scale": (d,), "w1": (32, d), "b1": (32,),
                       "w2": (1, 32), "b2": (1,)}.items():
        array = np.asarray(bundle[key], dtype=np.float64)
        require(array.shape == shape and np.isfinite(array).all(), "invalid actor array: " + key)
    require(np.all(np.asarray(bundle["scale"]) > 0), "normalization scale")
    require(bundle["epsilon"] == .1 and bundle["residual_bound"] == 2., "actor probability contract")
    return json_fingerprint(bundle)


def vectorize(features, names):
    require(features and all(set(f) == set(names) for f in features), "feature leakage/schema mismatch")
    values = np.asarray([[f[n] for n in names] for f in features], dtype=np.float64)
    require(np.isfinite(values).all(), "nonfinite actor features")
    return values


def initial_bundle(features, seed, binding):
    import torch
    names = sorted(features[0])
    values = vectorize(features, names)
    require(len(names) == 129, "production actor needs 129 registered inputs")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        first = torch.nn.Linear(len(names), 32, dtype=torch.float64)
    scale = values.std(axis=0)
    scale[scale < 1e-12] = 1.
    bundle = dict(schema=SCHEMA, binding=binding, iteration=0, parent_policy=None, seed=seed,
                  feature_names=names, mean=values.mean(axis=0).tolist(), scale=scale.tolist(),
                  input_min=values.min(axis=0).tolist(), input_max=values.max(axis=0).tolist(),
                  w1=first.weight.detach().numpy().tolist(), b1=first.bias.detach().numpy().tolist(),
                  w2=[[0.] * 32], b2=[0.], epsilon=.1, residual_bound=2.,
                  training_library=torch.__version__, normalization="train_initial_only")
    validate_bundle(bundle)
    return bundle


class NumpyActor:
    def __init__(self, bundle):
        self.sha = validate_bundle(bundle)
        self.bundle = bundle
        for key in ("mean", "scale", "w1", "b1", "w2", "b2"):
            setattr(self, key, np.asarray(bundle[key], dtype=np.float64))

    def probabilities(self, ids, anchor, features):
        x = vectorize(features, self.bundle["feature_names"])
        require(len(ids) == len(x), "candidate/vector alignment")
        hidden = np.tanh(((x - self.mean) / self.scale) @ self.w1.T + self.b1)
        residuals = 2. * np.tanh(hidden @ self.w2.T + self.b2)
        return anchor_distribution(ids, anchor, dict(zip(ids, residuals[:, 0].tolist())), .1)

    def out_of_range_fraction(self, features):
        if "input_min" not in self.bundle:
            return None
        values = vectorize(features, self.bundle["feature_names"])
        return float(np.mean((values < np.asarray(self.bundle["input_min"])) |
                             (values > np.asarray(self.bundle["input_max"]))))


def select_with_draw(probabilities, draw):
    require(math.isfinite(draw) and 0 <= draw < 1, "selection draw")
    require(probabilities and all(math.isfinite(v) and v > 0 for v in probabilities.values()), "probability support")
    require(math.isclose(math.fsum(probabilities.values()), 1., abs_tol=1e-12, rel_tol=0), "probability sum")
    total = 0.
    for cid in sorted(probabilities):
        total += probabilities[cid]
        if draw < total:
            return cid
    return sorted(probabilities)[-1]


def torch_actor(bundle):
    import torch
    validate_bundle(bundle)
    # Keep the autograd implementation in the installed ML library.
    model = torch.nn.Sequential(torch.nn.Linear(len(bundle["feature_names"]), 32, dtype=torch.float64),
                                torch.nn.Tanh(), torch.nn.Linear(32, 1, dtype=torch.float64))
    with torch.no_grad():
        for layer, prefix in ((model[0], "1"), (model[2], "2")):
            layer.weight.copy_(torch.tensor(bundle["w" + prefix], dtype=torch.float64))
            layer.bias.copy_(torch.tensor(bundle["b" + prefix], dtype=torch.float64))
    return model


def torch_distribution(model, bundle, ids, anchor, features):
    import torch
    require(ids == sorted(set(ids)) and anchor in ids, "canonical actor IDs")
    x = torch.tensor(vectorize(features, bundle["feature_names"]), dtype=torch.float64)
    require(len(x) == len(ids), "candidate/vector alignment")
    x = (x - torch.tensor(bundle["mean"], dtype=torch.float64)) / torch.tensor(bundle["scale"], dtype=torch.float64)
    residuals = 2. * torch.tanh(model(x).squeeze(-1))
    q = torch.tensor([.1 / len(ids) + (.9 if cid == anchor else 0.) for cid in ids], dtype=torch.float64)
    return torch.distributions.Categorical(logits=q.log() + residuals)


def export_actor(model, parent, binding):
    result = dict(parent, iteration=parent["iteration"] + 1, parent_policy=json_fingerprint(parent),
                  update_binding=binding)
    for layer, prefix in ((model[0], "1"), (model[2], "2")):
        result["w" + prefix] = layer.weight.detach().numpy().tolist()
        result["b" + prefix] = layer.bias.detach().numpy().tolist()
    validate_bundle(result)
    return result


def update_once(bundle, weighted_episodes, trace_reader, binding):
    """One gradient step, no epoch reuse, with behavior probability replay."""
    import torch
    torch.set_num_threads(1)
    model = torch_actor(bundle)
    optimizer = torch.optim.SGD(model.parameters(), lr=.001)
    optimizer.zero_grad()
    actor = NumpyActor(bundle)
    count, max_error, total_loss = 0, 0., 0.
    for row in weighted_episodes:
        coefficient = row["coefficient"]
        for event in trace_reader(row["episode_id"]):
            ids = event["candidate_ids"]
            require(event["policy_sha256"] == actor.sha, "stale training trajectory")
            distribution = torch_distribution(model, bundle, ids, event["anchor_id"], event["features"])
            expected = actor.probabilities(ids, event["anchor_id"], event["features"])
            actual = distribution.probs.detach().numpy()
            error = max(abs(actual[i] - event["probabilities"][cid]) for i, cid in enumerate(ids))
            require(error <= 1e-12 and max(abs(expected[c] - event["probabilities"][c]) for c in ids) <= 1e-12,
                    "behavior probabilities changed")
            selected = torch.tensor(ids.index(event["selected_id"]))
            loss = -coefficient * distribution.log_prob(selected)
            loss.backward()
            max_error = max(max_error, error)
            total_loss += float(loss.detach())
            count += 1
    require(count > 0, "no decisions to train")
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    require(float(norm) > 0, "zero gradient: no update")
    optimizer.step()
    output = export_actor(model, bundle, binding)
    movement = math.sqrt(sum(float(((np.asarray(output[k]) - np.asarray(bundle[k])) ** 2).sum())
                             for k in ("w1", "b1", "w2", "b2")))
    return output, dict(decisions=count, gradient_norm_before_clip=float(norm),
                        parameter_l2_change=movement, replay_probability_max_error=max_error,
                        loss_diagnostic=total_loss, updates=1, no_performance_claim=True)
