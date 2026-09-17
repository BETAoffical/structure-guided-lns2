"""Bounded one-intervention ablation, not a deployable gating policy."""
from experiments.sa_paired_completion import require


def select_once(anchor, prediction, already_used):
    selected = anchor if already_used else prediction
    used_now = selected != anchor
    return selected, already_used or used_now, used_now


def first_disagreement(events):
    for i, event in enumerate(events):
        require(event["decision"] == i, "non-contiguous source decisions")
        if event["ranking"]["selected"] != event["anchor_id"]:
            return i
    return None


def horizon_outcome(result, decision, horizon=32):
    """Never impute a censored or budget-short trajectory as an H32 failure."""
    if decision is None:
        return None
    counts = result["conflicts"]
    require(0 <= decision < len(counts) - 1, "intervention outside trace")
    target = decision + horizon
    if target < len(counts):
        return dict(known=True, success=counts[target] == 0, final_conflicts=counts[target])
    if result["success"]:
        return dict(known=True, success=True, final_conflicts=0)
    return dict(known=False, success=None, final_conflicts=None)


def completion_contrast(rows, baseline, bootstrap, seed):
    import numpy as np
    maps = sorted({r["map_id"] for r in rows})
    censored = any(r["unknown"][a] for r in rows for a in ("single", baseline))
    values = []
    bounds = []
    for map_id in maps:
        group = [r for r in rows if r["map_id"] == map_id]
        values.append(sum(r["success"]["single"] - r["success"][baseline] for r in group) / len(group))
        bounds.append([
            sum(r["success"]["single"] - min(1, r["success"][baseline] + r["unknown"][baseline]) for r in group) / len(group),
            sum(min(1, r["success"]["single"] + r["unknown"]["single"]) - r["success"][baseline] for r in group) / len(group),
        ])
    v = np.asarray(values)
    draws = np.random.default_rng(seed).integers(0, len(maps), (bootstrap, len(maps)))
    return dict(observed_delta=float(v.mean()),
                ci95=None if censored else np.quantile(v[draws].mean(axis=1), [.025, .975]).tolist(),
                sample_difference_bounds=np.asarray(bounds).mean(axis=0).tolist(),
                censored=censored, map_wins=int(sum(v > 0)), map_losses=int(sum(v < 0)),
                map_ties=int(sum(v == 0)))


def interpretation(contrasts):
    if any(c["censored"] for c in contrasts.values()):
        return "incomplete_evidence_resource_censoring"
    continuous = contrasts["paired"]
    if continuous["observed_delta"] > 0:
        return ("limited_evidence_repeated_takeover_hurts" if continuous["ci95"][0] > 0
                else "possible_takeover_effect_uncertain")
    return "single_intervention_does_not_resolve_failure"
