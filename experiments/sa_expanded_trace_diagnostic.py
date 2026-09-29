"""Retrospective trace descriptors, not counterfactual labels or solver calls."""
from collections import Counter
import math

from experiments._common import json_fingerprint
from experiments.closed_loop_trace_storage import apply_state_delta
from experiments.state_analysis import reconstruct_conflicts


def require(condition, message):
    if not condition:
        raise ValueError(message)


class PathIdentity:
    """Exclude counters/history from recurrence; cache unchanged agent paths."""
    def __init__(self):
        self.paths = {}
        self.hashes = {}

    def update(self, state):
        paths = {a['id']: a['path'] for a in state['agents']}
        require(len(paths) == len(state['agents']), 'duplicate agent ID')
        if self.paths:
            require(paths.keys() == self.paths.keys(), 'agent IDs changed')
        for aid, path in paths.items():
            require(bool(path), 'empty path')
            if self.paths.get(aid) != path:
                self.hashes[aid] = json_fingerprint(path)
        self.paths = paths
        return json_fingerprint(sorted(self.hashes.items()))


def longest_run(values):
    longest = current = 0
    previous = object()
    for value in values:
        current = current + 1 if value == previous else 1
        longest = max(longest, current)
        previous = value
    return longest


def window_summary(rows):
    n = len(rows)
    counts = Counter(x['members'] for x in rows)
    attempts = [x for x in rows if x['attempt_delta'] is not None and x['attempt_delta'] > 0]
    return dict(decisions=n, unique_neighborhoods=len(counts),
        top_neighborhood_fraction=max(counts.values())/n if n else None,
        longest_same_neighborhood=longest_run(x['members'] for x in rows),
        longest_same_conflict_edges=longest_run(x['edges_before'] for x in rows),
        unique_path_states=len({x['path_before'] for x in rows}),
        unchanged_paths=sum(x['path_before']==x['path_after'] for x in rows),
        accepted=sum(x['accepted'] for x in rows),
        path_changed_without_conflict_count_change=sum(x['path_before']!=x['path_after'] and x['before']==x['after'] for x in rows),
        conflict_decreased=sum(x['after']<x['before'] for x in rows),
        conflict_increased=sum(x['after']>x['before'] for x in rows),
        worsening_attempts=len(attempts), worsening_accepted=sum(x['accepted'] for x in attempts),
        failure_reasons=dict(Counter(x['reason'] for x in rows)),
        temperatures=[rows[0]['temperature'],rows[-1]['temperature']] if rows else [],
        selected_anchor=sum(x['anchor_selected'] is True for x in rows),
        mean_anchor_probability=sum(x['anchor_probability'] for x in rows if x['anchor_probability'] is not None)/n
            if n and all(x['anchor_probability'] is not None for x in rows) else None,
        mean_max_probability=sum(x['max_probability'] for x in rows)/n
            if n and all(x['max_probability'] is not None for x in rows) else None,
        mean_out_of_range_fraction=sum(x['out_of_range'] for x in rows)/n
            if n and all(x['out_of_range'] is not None for x in rows) else None,
        pp_seconds=sum(x['pp_seconds'] for x in rows),
        generated=sum(x['generated'] for x in rows),
        sizes=dict(Counter(str(len(x['members'])) for x in rows)),
        full_conflict_endpoint_coverage=sum(x['covers_all'] for x in rows))


def final_events(state):
    events = reconstruct_conflicts(state['agents'])
    pairs = {tuple(sorted((e.left,e.right))) for e in events}
    require(pairs == {tuple(sorted(e)) for e in state['conflict_edges']}, 'final conflict reconstruction')
    require(len(pairs) == state['num_of_colliding_pairs'], 'final pair count')
    paths = {a['id']:a['path'] for a in state['agents']}
    holds = {tuple(sorted((e.left,e.right))) for e in events
             if any(e.time > len(paths[a])-1 for a in (e.left,e.right))}
    return dict(events=len(events), pairs=len(pairs), goal_hold_pairs=sorted(holds),
        vertex_events=sum(e.kind == 'vertex' for e in events),
        swap_events=sum(e.kind != 'vertex' for e in events),
        first_time=min((e.time for e in events), default=None),
        last_time=max((e.time for e in events), default=None))


def analyze(initial, final, events, result):
    state, identity = initial, PathIdentity()
    path_hash = identity.update(state)
    initial_hash = path_hash
    rows, signatures = [], []
    seen = {path_hash}
    revisits = 0
    best = state['num_of_colliding_pairs']
    best_at = result['reset_seconds']
    hits = {str(t):(best_at if best <= t else None) for t in (20,10,3,1,0)}
    for event in events:
        m = event['metrics']
        require(event['decision'] == len(rows), 'noncontiguous decision')
        require(m['action_valid'] and m['step_applied'], 'invalid action')
        after = apply_state_delta(state, event['delta'])
        require(m['conflicts_before'] == state['num_of_colliding_pairs'] and
                m['conflicts_after'] == after['num_of_colliding_pairs'], 'transition count')
        elapsed = event['elapsed_seconds']
        require(math.isfinite(elapsed) and elapsed >= (rows[-1]['elapsed'] if rows else result['reset_seconds']), 'elapsed order')
        members = tuple(sorted(m['neighborhood']))
        require(len(members) == len(set(members)) and set(members) <= identity.paths.keys(), 'members')
        requested = event['action'].get('agents')
        require(requested is None or tuple(sorted(requested)) == members, 'changed explicit action')
        old_paths = identity.paths
        new_hash = identity.update(after)
        require(all(old_paths[a] == identity.paths[a] for a in old_paths.keys()-set(members)), 'outsider path changed')
        edges = tuple(sorted(tuple(sorted(e)) for e in state['conflict_edges']))
        require(len(edges) == len(set(edges)) == m['conflicts_before'], 'edge count')
        active = {a for e in edges for a in e}
        probabilities = event.get('probabilities')
        anchor = event.get('anchor_id')
        if probabilities is not None:
            pool = {p['candidate_id']:p for p in event['pool']}
            require(len(pool)==len(event['pool']) and set(pool)==set(probabilities), 'pool/probability mismatch')
            require(all(math.isfinite(p) and p>=0 for p in probabilities.values()) and
                    math.isclose(sum(probabilities.values()),1.,abs_tol=1e-12), 'invalid probability')
            require(tuple(sorted(pool[event['selected_id']]['agents']))==members, 'pool choice')
        attempt = (m['pp_attempt_conflict_pair_count']-m['pp_old_conflict_pair_count']) if m.get('acceptance_evaluated') else None
        row = dict(decision=event['decision'], elapsed=elapsed, before=m['conflicts_before'], after=m['conflicts_after'],
            path_before=path_hash, path_after=new_hash, edges_before=edges, members=members,
            accepted=bool(m['replan_success']), reason=m['pp_failure_reason'], attempt_delta=attempt,
            temperature=event['temperature'], pp_seconds=m['native_replan_seconds'],
            generated=after['low_level']['generated']-state['low_level']['generated'], covers_all=active <= set(members),
            anchor_selected=event['selected_id']==anchor if probabilities is not None else None,
            anchor_probability=probabilities.get(anchor) if probabilities is not None else None,
            max_probability=max(probabilities.values()) if probabilities is not None else None,
            out_of_range=event.get('out_of_range_fraction'))
        if probabilities is not None:
            signatures.append(dict(decision=event['decision'], path_before=path_hash, path_after=new_hash,
                members=list(members), pp_seed=event['action'].get('random_seed'), draw=event.get('selection_draw'),
                pool=json_fingerprint(sorted((k,sorted(v['agents'])) for k,v in pool.items()))))
        if new_hash != path_hash and new_hash in seen:
            revisits += 1
        seen.add(new_hash)
        if row['after'] < best:
            best, best_at = row['after'], elapsed
        for threshold in hits:
            if hits[threshold] is None and row['after'] <= int(threshold):
                hits[threshold] = elapsed
        rows.append(row)
        state, path_hash = after, new_hash
    require(len(rows)==result['decisions'], 'decision count')
    require(state['num_of_colliding_pairs']==result['final_conflicts'], 'final count')
    require(PathIdentity().update(final)==path_hash, 'final paths')
    require(state['conflict_edges']==final['conflict_edges'], 'final edges')
    require(math.isclose(sum(x['pp_seconds'] for x in rows),result['native_pp_seconds'],rel_tol=1e-10,abs_tol=1e-8), 'PP sum')
    require(sum(x['generated'] for x in rows)==result['generated'], 'generated sum')
    late = [x for x in rows if x['elapsed'] >= result['search_end_seconds']-30]
    progress = {}
    for t in (30,60,90,120):
        reached = [x for x in rows if x['elapsed'] <= t]
        progress[str(t)] = dict(last_returned_conflicts=reached[-1]['after'] if reached else initial['num_of_colliding_pairs'],
            decisions=len(reached), feasible_by_t=bool(result['success_within_budget'] and result['ttf_seconds']<=t),
            last_returned_seconds=reached[-1]['elapsed'] if reached else result['reset_seconds'])
    return dict(job_id=result['job_id'], pair_id=result['pair_id'], map_id=result['map_id'], arm=result['comparison_arm'],
        success=result['success_within_budget'], initial_conflicts=result['initial_conflicts'], final_conflicts=result['final_conflicts'],
        initial_path_identity=initial_hash, best_conflicts=best, best_at_seconds=best_at,
        seconds_after_last_new_best=result['search_end_seconds']-best_at, first_at_most_conflicts=hits,
        changed_path_revisits=revisits, unique_path_states=len(seen), progress=progress,
        all=window_summary(rows), last_30s=window_summary(late), last_32_decisions=window_summary(rows[-32:]),
        final_events=final_events(final), decision_signatures=signatures)


def first_divergence(parent, completion):
    require(parent['pair_id']==completion['pair_id'] and parent['initial_path_identity']==completion['initial_path_identity'], 'unpaired actors')
    left, right = parent['decision_signatures'], completion['decision_signatures']
    for a,b in zip(left,right):
        require(a['decision']==b['decision'], 'unaligned decisions')
        if a['path_before'] != b['path_before']:
            return dict(kind='different_path_state', decision=a['decision'])
        if a['members'] != b['members']:
            return dict(kind='selection', decision=a['decision'], same_pool=a['pool']==b['pool'],
                same_draw=a['draw']==b['draw'], same_pp_seed=a['pp_seed']==b['pp_seed'],
                parent_members=a['members'], completion_members=b['members'])
        if a['path_after'] != b['path_after']:
            return dict(kind='same_choice_different_repair', decision=a['decision'], same_pp_seed=a['pp_seed']==b['pp_seed'])
    return dict(kind='identical_until_stop', matched_decisions=min(len(left),len(right)),
                parent_decisions=len(left), completion_decisions=len(right))
