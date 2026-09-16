"""Read-only completion-label diagnostics; never used for live decisions."""
from itertools import combinations
from statistics import mean, variance

from experiments._common import json_fingerprint
from experiments.repair_collection import state_fingerprint
from experiments.sa_paired_completion import require


def signatures(state):
    # Audit-only views. The full replay/proposal identity is never replaced.
    return dict(full=state_fingerprint(state),
                counter_free=state_fingerprint(dict(state, low_level={})),
                physical=state_fingerprint(dict(state, low_level={}, iteration=0)),
                conflicts=state["num_of_colliding_pairs"])


def pool_signature(event):
    return dict(full=json_fingerprint(event["pool"]),
                members=sorted(tuple(sorted(c["agents"])) for c in event["pool"]),
                selected=tuple(sorted(event["action"]["agents"])),
                proposal_seeds=sorted({v for c in event["pool"] for v in c.get("proposal_seeds", [])}))


def split_diagnostics(values, anchor):
    ids = sorted(values)
    require(len(ids) >= 2 and anchor in ids, "candidate coverage")
    require(all(len(values[c]) == 8 and all(type(v) is int and v in (0,1) for v in values[c]) for c in ids),
            "eight complete binary trials required")
    rates = {c:sum(values[c])/8 for c in ids}
    partitions = []
    for half in combinations(range(8),4):
        if 0 not in half:
            continue
        other = tuple(i for i in range(8) if i not in half)
        halves = [{c:sum(values[c][i] for i in part)/4 for c in ids} for part in (half,other)]
        best = [{c for c in ids if h[c] == max(h.values())} for h in halves]
        chosen = [min(h, key=lambda c:(-h[c], c != anchor, c)) for h in halves]
        strict, agreed = 0,0
        for a,b in combinations(ids,2):
            d = [h[a]-h[b] for h in halves]
            strict += all(x != 0 for x in d)
            agreed += d[0]*d[1] > 0
        partitions.append(dict(half=list(half), jaccard=len(best[0]&best[1])/len(best[0]|best[1]),
            cross_half_rate=(halves[1][chosen[0]]+halves[0][chosen[1]])/2,
            strict_pairs=strict, concordant_pairs=agreed))
    pair_variances = []
    for a,b in combinations(ids,2):
        delta = [x-y for x,y in zip(values[a],values[b])]
        independent = variance(values[a])+variance(values[b])
        paired = variance(delta)
        pair_variances.append(dict(left=a,right=b,paired_variance=paired,independent_variance=independent,
                                   covariance=(independent-paired)/2))
    require(len(partitions) == 35, "partition enumeration incomplete")
    return dict(rates=rates, frozen_rate=rates[anchor], empirical_oracle=max(rates.values()),
                informative=len(set(rates.values())) > 1, partitions=partitions,
                mean_jaccard=mean(p["jaccard"] for p in partitions),
                cross_half_rate=mean(p["cross_half_rate"] for p in partitions),
                variance_pairs=pair_variances)


def compare_branches(left,right):
    require(left["trial"] == right["trial"], "unpaired branch trial")
    a,b = left["states"],right["states"]
    require(a and b, "missing observed states")
    same = a[0]["counter_free"] == b[0]["counter_free"]
    exact = a[0]["full"] == b[0]["full"]
    counter_only = same and not exact
    next_pair = left["next_pool"] is not None and right["next_pool"] is not None
    if exact and next_pair:
        require(left["next_pool"] == right["next_pool"], "same full state has different next proposal/selection")
    comparable = counter_only and next_pair
    future = [(i,x,y) for i,(x,y) in enumerate(zip(a,b)) if i>0 and x["conflicts"]>0 and y["conflicts"]>0]
    mergers = [i for i,x,y in future if x["counter_free"] == y["counter_free"]]
    return dict(trial=left["trial"],left=left["candidate_id"],right=right["candidate_id"],
        same_first_paths=same, same_first_full_state=exact, first_counters_only=counter_only,
        counters_only_with_continuation=comparable,
        next_members_differ=comparable and left["next_pool"]["members"] != right["next_pool"]["members"],
        next_seed_sets_differ=comparable and left["next_pool"]["proposal_seeds"] != right["next_pool"]["proposal_seeds"],
        next_selected_members_differ=comparable and left["next_pool"]["selected"] != right["next_pool"]["selected"],
        different_completion=left["completed"] != right["completed"],
        later_nonterminal_merge_after_first_difference=not same and bool(mergers),
        later_nonterminal_full_merge=not exact and any(x["full"]==y["full"] for _,x,y in future),
        physical_redivergence_after_merge=bool(mergers) and any(
            x["counter_free"]!=y["counter_free"] for i,x,y in future if i>mergers[0]))


def aggregate(roots):
    require(roots and len({r["id"] for r in roots}) == len(roots), "root coverage")
    informative = [r for r in roots if r["split"]["informative"]]
    pairs = [p for r in roots for p in r["pairs"]]
    branches = [b for r in roots for b in r["branches"]]
    flags = ("same_first_paths","same_first_full_state","first_counters_only", "counters_only_with_continuation",
             "next_members_differ","next_seed_sets_differ","next_selected_members_differ", "different_completion",
             "later_nonterminal_merge_after_first_difference","later_nonterminal_full_merge","physical_redivergence_after_merge")
    same_counter_pairs = [p for p in pairs if p["counters_only_with_continuation"]]
    by_split = [dict(half=roots[0]["split"]["partitions"][i]["half"],
                    cross_half_rate=mean(r["split"]["partitions"][i]["cross_half_rate"] for r in roots),
                    jaccard=mean(r["split"]["partitions"][i]["jaccard"] for r in roots),
                    informative_jaccard=mean(r["split"]["partitions"][i]["jaccard"] for r in informative) if informative else None)
                for i in range(35)]
    return dict(roots=len(roots),maps=len({r["map_id"] for r in roots}),branches=len(branches),paired_comparisons=len(pairs),
        counts={k:sum(p[k] for p in pairs) for k in flags},
        counters_only_later_completion_disagreement=sum(p["different_completion"] for p in same_counter_pairs),
        unchanged_first_paths=sum(b["unchanged_first_paths"] for b in branches),
        accepted_unchanged_first_paths=sum(b["unchanged_first_paths"] and b["first_accepted"] for b in branches),
        branches_with_nonterminal_physical_recurrence=sum(b["physical_recurrence"] for b in branches),
        informative_roots=len(informative),partitions=by_split,
        frozen_rate=mean(r["split"]["frozen_rate"] for r in roots),
        cross_half_rate=mean(r["cross_half_rate"] for r in by_split),
        informative_jaccard=mean(r["mean_jaccard"] for r in (x["split"] for x in informative)) if informative else None,
        strict_pair_partition_comparisons=sum(p["strict_pairs"] for r in roots for p in r["split"]["partitions"]),
        concordant_pair_partition_comparisons=sum(p["concordant_pairs"] for r in roots for p in r["split"]["partitions"]),
        mean_paired_variance=mean(mean(v["paired_variance"] for v in r["split"]["variance_pairs"]) for r in roots),
        mean_independent_variance=mean(mean(v["independent_variance"] for v in r["split"]["variance_pairs"]) for r in roots))
