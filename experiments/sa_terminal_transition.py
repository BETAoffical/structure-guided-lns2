"""Outcome-blind OD-preserving Train subsets, not solver improvements or new maps."""
from collections import Counter, defaultdict

from experiments._common import json_fingerprint
from experiments.sa_paired_completion import require

PER_AGENT = ("flow_assignments", "required_bottlenecks", "required_intersections",
             "required_intersection_component_ids", "actual_shortest_distances")


def retained_ids(task, seed):
    n = len(task["starts"])
    require(n >= 10 and len(task["goals"]) == n, "complete source endpoints")
    meta = task["metadata"]
    require(all(len(meta[k]) == n for k in PER_AGENT), "source per-agent metadata")
    require(not meta["swapped_agent_pairs"], "swap pairs require a separate subset contract")
    groups = defaultdict(list)
    for i,(flow,bottleneck) in enumerate(zip(meta["flow_assignments"],meta["required_bottlenecks"])):
        require(isinstance(flow,str) and flow, "source OD label")
        groups[(flow,bottleneck is not None)].append(i)
    count = (9*n+5)//10
    quotas = {g:9*len(ids)//10 for g,ids in groups.items()}
    order = sorted(groups, key=lambda g:(-(9*len(groups[g])%10),json_fingerprint([seed,task["task_id"],list(g)])))
    for g in order[:count-sum(quotas.values())]:
        quotas[g] += 1
    selected = []
    strata = []
    for g in sorted(groups):
        ids = sorted(groups[g],key=lambda i:(json_fingerprint([seed,task["task_id"],i,
            task["starts"][i],task["goals"][i]]),i))
        selected.extend(ids[:quotas[g]])
        strata.append(dict(flow=g[0],bottleneck_constrained=g[1],source=len(ids),retained=quotas[g]))
    selected.sort()
    require(len(selected) == len(set(selected)) == count and count < n, "exact retained count")
    return selected,strata


def subset_document(task, ids, strata, *, seed, free_cells, source_sha256):
    require(ids == sorted(set(ids)) and all(type(i) is int and 0 <= i < len(task["starts"]) for i in ids),
            "ordered parent ID mapping")
    require((ids,strata) == retained_ids(task,seed), "subset must use registered deterministic allocation")
    meta = task["metadata"]
    rows = {k:[meta[k][i] for i in ids] for k in PER_AGENT}
    count = len(ids)
    metadata = dict(schema_version=2,task_semantics_version=2,derivation="stratified_retain90_v1",role="train_only",
        source_task_id=task["task_id"],source_task_sha256=source_sha256,parent_agent_ids=ids,subset_seed=seed,
        strata=strata,agent_count=count,agent_density_free_cells=count/free_cells,
        source_agent_count=len(task["starts"]),flow_type=meta["flow_type"],scenario_type=meta["scenario_type"],
        realized_flow_counts=dict(Counter(rows["flow_assignments"])),
        required_bottleneck_crossing_ratio=sum(v is not None for v in rows["required_bottlenecks"])/count,
        mean_shortest_distance=sum(rows["actual_shortest_distances"])/count,
        minimum_actual_shortest_distance=min(rows["actual_shortest_distances"]),
        maximum_actual_shortest_distance=max(rows["actual_shortest_distances"]),**rows)
    return dict(schema_version=2,task_id=task["task_id"]+"__retain90_v1",map_id=task["map_id"],seed=seed,
        starts=[task["starts"][i] for i in ids],goals=[task["goals"][i] for i in ids],metadata=metadata)


def signal_decision(information, hard_success):
    mixed, maps = len(information["mixed_conditions"]),len(information["effective_maps"])
    if mixed >= 4 and maps >= 3:
        decision = "additional_terminal_signal_available_separate_training_preregistration_required"
    elif mixed:
        decision = "localized_terminal_signal_preserve_no_automatic_expansion"
    else:
        decision = "no_within_condition_terminal_signal_stop_this_transition"
    return dict(decision=decision,mixed_conditions=mixed,effective_maps=maps,
                hard_task_subset_has_success=hard_success > 0,original_hard_task_improved=False,
                no_training_this_stage=True,not_algorithm_improvement=True)
