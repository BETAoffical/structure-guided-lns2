"""One complete-terminal raw-logit update; no dense or counterfactual labels."""
import math
import numpy as np

from experiments import sa_raw_residual_actor as raw
from experiments.sa_crossfit_update import probability_stats, within_budget
from experiments.sa_onpolicy_actor import select_with_draw
from experiments.sa_paired_completion import require

UPDATE = dict(initial_l2=.25, backtracks=12, **raw.LIMITS)


def pack_episode(bundle, events):
    """Canonicalize and replay behavior before constructing differentiable arrays."""
    actor = raw.RawResidualActor(bundle)
    events = list(events)
    if not events:
        return None
    n, width = len(events), max(len(e['candidate_ids']) for e in events)
    x = np.zeros((n, width, 129))
    base, behavior = np.zeros((n, width)), np.zeros((n, width))
    valid = np.zeros((n, width), dtype=bool)
    selected, draws, lengths = [], [], []
    for i, e in enumerate(events):
        require(e['decision'] == i and e['policy_sha256'] == actor.sha, "behavior sequence/identity")
        ids, features, z = raw.canonical_inputs(bundle['base'], e['candidate_ids'], e['anchor_id'], e['features'])
        p = actor.probabilities(ids, e['anchor_id'], features)
        require(set(e['probabilities']) == set(ids) and
                max(abs(p[c]-e['probabilities'][c]) for c in ids) <= 1e-12, "stale behavior")
        require(select_with_draw(p, e['selection_draw']) == e['selected_id'], "behavior draw")
        require(math.isclose(e['behavior_log_probability'], math.log(p[e['selected_id']]),
                             rel_tol=0, abs_tol=1e-12), "behavior likelihood")
        prior = actor.base.probabilities(ids, e['anchor_id'], features)
        count = len(ids)
        x[i, :count], valid[i, :count] = z, True
        base[i, :count] = [prior[c] for c in ids]
        behavior[i, :count] = [p[c] for c in ids]
        selected.append(ids.index(e['selected_id']))
        lengths.append(count)
        draws.append(e['selection_draw'])
    return dict(padded=(x, behavior, valid), base=base, selected=np.asarray(selected),
                lengths=np.asarray(lengths), draws=np.asarray(draws))


def log_distribution(model, pack):
    import torch
    x, _, valid = pack['padded']
    base = pack['base']
    require(base.shape == valid.shape and np.isfinite(base).all() and (base[valid] > 0).all(), "base distribution")
    require(np.allclose(base.sum(axis=1), 1., atol=1e-12, rtol=0), "base mass")
    # Padding has neutral log mass then is masked; never clip real probabilities.
    logs = np.zeros_like(base)
    logs[valid] = np.log(base[valid])
    logits = torch.as_tensor(logs, dtype=torch.float64) + model(torch.as_tensor(x, dtype=torch.float64)).squeeze(-1)
    mask = torch.as_tensor(valid)
    require(bool(torch.isfinite(logits[mask]).all()), "nonfinite logits")
    result = torch.log_softmax(logits.masked_fill(~mask, -torch.inf), dim=1)
    require(bool(torch.isfinite(result[mask]).all()) and bool((result[mask].exp() > 0).all()), "probability underflow")
    return result


def step_statistics(packs, probabilities):
    active_weight = math.fsum(p['weight'] for p in packs)
    require(0 < active_weight <= 1. + 1e-12, "active episode mass")
    normalized = [dict(p, weight=p['weight']/active_weight) for p in packs]
    stats = probability_stats(normalized, probabilities)
    for key in ('mean_kl', 'mean_trajectory_kl', 'mean_tv'):
        stats[key] *= active_weight
    return stats


def terminal_gradient(bundle, packs):
    import torch
    model = raw.torch_correction(bundle)
    parameters = tuple(model.parameters())
    gradient = [torch.zeros_like(p) for p in parameters]
    for pack in packs:
        logs = log_distribution(model, pack)
        require(np.max(np.abs(logs.detach().exp().numpy()-pack['padded'][1])) <= 1e-12, "tensor behavior replay")
        if pack['coefficient'] == 0:
            continue
        chosen = logs[torch.arange(len(pack['selected'])), torch.as_tensor(pack['selected'], dtype=torch.long)]
        loss = -pack['coefficient'] * chosen.sum()
        for total, grad in zip(gradient, torch.autograd.grad(loss, parameters)):
            total.add_(grad)
    require(all(bool(torch.isfinite(g).all()) for g in gradient), "nonfinite terminal gradient")
    return gradient


def guarded_update(bundle, packs, binding):
    import torch
    before = raw.validate_bundle(bundle)
    gradient = terminal_gradient(bundle, packs)
    magnitude = math.sqrt(sum(float((g*g).sum()) for g in gradient))
    if not packs or magnitude == 0:
        return None, dict(decision='zero_terminal_gradient_no_update', gradient_norm=magnitude)
    attempts = []
    for attempt in range(UPDATE['backtracks']):
        distance = UPDATE['initial_l2'] / 2**attempt
        model = raw.torch_correction(bundle)
        with torch.no_grad():
            for parameter, g in zip(model.parameters(), gradient):
                parameter.add_(g, alpha=-distance/magnitude)
            probabilities = [log_distribution(model, p).exp().numpy() for p in packs]
        stats = step_statistics(packs, probabilities)
        attempts.append(dict(parameter_l2=distance, **stats))
        require(raw.validate_bundle(bundle) == before, "parent mutated")
        if within_budget(stats, raw.LIMITS):
            return raw.export_correction(model, bundle, binding), dict(
                decision='one_terminal_update_not_evaluated', gradient_norm=magnitude,
                attempts=attempts, chosen=attempts[-1], reference_policy_sha256=before)
    return None, dict(decision='no_step_within_budget', gradient_norm=magnitude, attempts=attempts)
