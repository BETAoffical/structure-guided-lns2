"""Input-only sampling for a bounded, developmental completion-label pilot."""
from experiments._common import json_fingerprint
from experiments.sa_paired_completion import require


def choose_candidates(pool, anchor_id, state_id, seed, limit=4):
    ids = [c["candidate_id"] for c in pool]
    memberships = [tuple(sorted(c["agents"])) for c in pool]
    require(len(ids) == len(set(ids)) and len(memberships) == len(set(memberships)), "duplicate candidates")
    require(anchor_id in ids and len(ids) >= 2 and limit >= 2, "candidate coverage")
    require(all(m and len(m) == len(set(m)) for m in memberships), "invalid members")
    ordered = sorted(pool, key=lambda c: (json_fingerprint([seed, state_id, c["candidate_id"]]), c["candidate_id"]))
    chosen = [anchor_id]
    # Only the required old-policy control uses the old ranking. Challengers
    # depend on membership size and a registered hash, never on model scores.
    for bucket in (lambda n: n <= 4, lambda n: 4 < n <= 8, lambda n: n > 8):
        options = [c["candidate_id"] for c in ordered if bucket(len(c["agents"])) and c["candidate_id"] not in chosen]
        if options and len(chosen) < limit:
            chosen.append(options[0])
    for c in ordered:
        if c["candidate_id"] not in chosen and len(chosen) < limit:
            chosen.append(c["candidate_id"])
    return chosen


def choose_roots(available, maps, excluded_episodes, seed):
    require(len({r["id"] for r in available}) == len(available), "duplicate source occurrence")
    selected, coverage = [], []
    for map_id in sorted(maps):
        pool = [r for r in available if r["map_id"] == map_id and r["item"]["job_id"] not in excluded_episodes]
        selected_map = []
        used = set()
        # Select the rarer continuing stratum before reserving an early root.
        for phase in ("continuing", "early"):
            choices = [r for r in pool if r["phase"] == phase and r["item"]["job_id"] not in used]
            if choices:
                pick = min(choices, key=lambda r: (json_fingerprint([seed, phase, r["id"]]), r["id"]))
                selected_map.append(pick)
                used.add(pick["item"]["job_id"])
        selected.extend(selected_map)
        coverage.append(dict(map_id=map_id, early=sum(r["phase"] == "early" for r in pool),
                             continuing=sum(r["phase"] == "continuing" for r in pool), selected=len(selected_map)))
    require(len({r["item"]["job_id"] for r in selected}) == len(selected), "reused episode")
    return sorted(selected, key=lambda r: (r["map_id"], r["phase"], r["id"])), coverage


def work_budget(roots, cfg):
    branches = sum(len(r["selected"]) for r in roots) * cfg["trials"]
    prefix = sum(r["decision"] for r in roots)
    return dict(states=len(roots), maps=len({r["map_id"] for r in roots}),
                branch_jobs=branches, original_controls=len(roots), resets=len(roots),
                prefix_repairs=prefix, continuation_repairs_upper=branches*cfg["horizon"],
                all_repairs_upper=prefix+len(roots)+branches*cfg["horizon"],
                branch_wall_safety_sum_seconds=branches*cfg["branch_seconds"],
                workers=cfg["workers"], timing_claim=False,
                note="resource ceilings, not runtime forecasts; no reliable CPU or disk lower bound")
