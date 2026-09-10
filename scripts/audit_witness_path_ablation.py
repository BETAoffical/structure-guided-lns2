"""Finite old/new path recombination, not PP or online policy evaluation."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import gzip
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from scripts.audit_preentry_trace import native_path, read, sha
from experiments.closed_loop_trace_storage import apply_state_delta
from experiments.repair_collection import state_fingerprint
from experiments.state_analysis import reconstruct_conflicts

OUT = ROOT / 'build/initlns-witness-path-ablation-v1'


def pair_conflicts(left, right):
    if not left or not right:
        raise ValueError('empty path')
    for t in range(max(len(left), len(right))):
        a = left[min(t, len(left)-1)]; b = right[min(t, len(right)-1)]
        if a == b:
            return True
        if t and a == right[min(t-1, len(right)-1)] and b == left[min(t-1, len(left)-1)]:
            return True
    return False


def full_edges(paths):
    return {(e.left, e.right) for e in reconstruct_conflicts(
        [dict(id=i, path=p) for i,p in sorted(paths.items())])}


def compile_choices(old, new):
    if old.keys() != new.keys():
        raise ValueError('agent identity changed')
    variables = sorted(i for i in old if old[i] != new[i])
    if len(variables) > 16:
        raise ValueError('finite diagnostic limited to 16 changed paths')
    indices = {i:n for n,i in enumerate(variables)}
    before = full_edges(old); after = full_edges(new)
    fixed = {p for p in before if not set(p) & set(variables)}
    assert fixed == {p for p in after if not set(p) & set(variables)}
    table = []
    # Only changed/changed pairs need mixed-path collision checks. Boundary
    # pairs inherit their old/new truth from the two full reconstructions.
    for a,b in itertools.combinations(sorted(old), 2):
        if a not in indices and b not in indices:
            continue
        edge = (a,b)
        if a in indices and b in indices:
            truth = [edge in before, pair_conflicts(new[a],old[b]),
                     pair_conflicts(old[a],new[b]), edge in after]
            i,j = indices[a],indices[b]
        else:
            v = a if a in indices else b
            truth = [edge in before, edge in after, False, False]
            i,j = indices[v],-1
        if any(truth):
            table.append((edge,i,j,truth))
    return variables, fixed, table, before, after


def selected_edges(mask, fixed, table):
    return fixed | {edge for edge,i,j,truth in table
                    if truth[((mask>>i)&1) + (2*((mask>>j)&1) if j>=0 else 0)]}


def enumerate_choices(variables, fixed, table, target):
    minimum = None; best = []; exact_min = None; exact = []; count = 0
    for mask in range(1<<len(variables)):
        edges = selected_edges(mask,fixed,table)
        key = (len(edges), mask.bit_count())
        if minimum is None or key < minimum:
            minimum = key; best = [mask]
        elif key == minimum:
            best.append(mask)
        if edges == target:
            count += 1
            if exact_min is None or mask.bit_count() < exact_min:
                exact_min = mask.bit_count(); exact = [mask]
            elif mask.bit_count() == exact_min:
                exact.append(mask)
    def unpack(m):
        return [a for i,a in enumerate(variables) if m>>i&1]
    return dict(combinations=1<<len(variables), minimum_conflicts=minimum[0],
        minimum_conflict_minimum_changed_paths=minimum[1],
        best_count=len(best), best_examples=[unpack(m) for m in best[:4]],
        exact_post_edges_count=count, exact_post_edges_minimum_changed_paths=exact_min,
        exact_minimal_count=len(exact), exact_minimal_examples=[unpack(m) for m in exact[:4]]), best[:4]+exact[:4]


def load_states(detail):
    path = ROOT/detail['source_trace']
    assert sha(path) == detail['source_sha256']
    fingerprint = None
    with gzip.open(native_path(path),'rt',encoding='utf-8') as stream:
        for line in stream:
            event = json.loads(line)
            if event['event'] == 'initial':
                episode = path.parents[2]
                with gzip.open(native_path(episode/event['state_blob']),'rt',encoding='utf-8') as blob:
                    state = json.load(blob)
                fingerprint = state_fingerprint(state)
                assert fingerprint == event['state_fingerprint']
                continue
            if event['event'] != 'transition':
                continue
            assert event['before_fingerprint'] == fingerprint
            before = state; state = apply_state_delta(before,event['state_delta'])
            fingerprint = event['after_fingerprint']
            if event['decision_index']+1 == detail['exit_transition']['step']:
                assert state_fingerprint(before) == event['before_fingerprint']
                assert state_fingerprint(state) == event['after_fingerprint']
                return before,state,event
    raise ValueError('missing witness transition')


def audit(detail):
    before,after,event = load_states(detail)
    old = {a['id']:a['path'] for a in before['agents']}
    new = {a['id']:a['path'] for a in after['agents']}
    meta = {a['id']:a for a in before['agents']}
    for identity in old:
        for path in (old[identity],new[identity]):
            assert path[0] == meta[identity]['start'] and path[-1] == meta[identity]['goal']
            assert all(0<=p<len(before['obstacles']) and not before['obstacles'][p] for p in path)
            cols = before['cols']
            assert all(abs(a//cols-b//cols)+abs(a%cols-b%cols)<=1 for a,b in zip(path,path[1:]))
    variables,fixed,table,initial,target = compile_choices(old,new)
    assert initial == {tuple(p) for p in before['conflict_edges']}
    assert target == {tuple(p) for p in after['conflict_edges']}
    selected = sorted(event['metrics']['neighborhood'])
    assert set(variables) <= set(selected)
    full = (1<<len(variables))-1
    assert selected_edges(0,fixed,table) == initial
    assert selected_edges(full,fixed,table) == target
    single = []
    checked = {0,full}
    for i,agent in enumerate(variables):
        mask = full ^ (1<<i); checked.add(mask)
        paths = dict(new); paths[agent] = old[agent]
        collisions = reconstruct_conflicts([dict(id=k,path=v) for k,v in paths.items()])
        observed = {(e.left,e.right) for e in collisions}
        assert observed == selected_edges(mask,fixed,table)
        single.append(dict(agent=agent, initially_conflicting=meta[agent]['conflict_degree']>0,
            remaining_conflicts=len(observed), new_edges=sorted(observed-target),
            removed_edges=sorted(target-observed), preserves_exact_post_edges=observed==target,
            first_new_events=[dict(time=e.time,kind=e.kind,pair=[e.left,e.right],cells=list(e.cells))
                              for e in collisions if (e.left,e.right) in observed-target][:8]))
    summary,examples = enumerate_choices(variables,fixed,table,target)
    for mask in set(examples)-checked:
        paths = {a:new[a] if a in variables and mask>>variables.index(a)&1 else old[a] for a in old}
        assert full_edges(paths) == selected_edges(mask,fixed,table)
    result = dict(case_id=detail['case_id'], source_trace=detail['source_trace'],
        source_trace_sha256=detail['source_sha256'], step=event['decision_index']+1,
        before_fingerprint=event['before_fingerprint'], after_fingerprint=event['after_fingerprint'],
        episode_success=detail['success'], conflicts_before=len(initial), conflicts_after=len(target),
        selected_agents=selected, changed_agents=variables,
        selected_unchanged_agents=sorted(set(selected)-set(variables)),
        before_zero_degree_selected=[a for a in selected if not meta[a]['conflict_degree']],
        single_undo=single, finite_combinations=summary,
        new_repairs=0, interpretation='Fixed old/new path witness compatibility only; not agent necessity under PP or an online candidate label.')
    (OUT/(detail['case_id']+'.json')).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result


if __name__ == '__main__':
    evidence = read(ROOT/'artifacts/initlns-preentry-trace-audit-v1/evidence.json')
    source = ROOT/evidence['report_file']
    assert sha(source) == evidence['report_sha256']
    cases = [c for c in read(source)['cases'] if c['role']!='fast_control' and c['exit_transition']]
    assert len(cases)==3
    OUT.mkdir(exist_ok=True)
    if (OUT/'report.json').exists():
        raise RuntimeError('Diagnostic report already exists; do not overwrite')
    with ProcessPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(audit,cases))
    report = dict(schema='lns2.witness_path_ablation.v1', cases=results, errors=0,
        script_sha256=sha(Path(__file__)), source_report_sha256=sha(source),
        solver_runs=0, native_validation=False, timing_experiment=False)
    temp = OUT/'report.tmp'
    temp.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    temp.replace(OUT/'report.json')
    for result in results:
        print(json.dumps(dict(case_id=result['case_id'],finite_combinations=result['finite_combinations'])),flush=True)
