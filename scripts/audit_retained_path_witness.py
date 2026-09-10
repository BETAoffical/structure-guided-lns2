"""Check stored path witnesses and binary old/reference path splices; no search."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import multiprocessing as mp
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.dont_write_bytecode = True

from experiments._common import read_json, write_json
from experiments.local_path_compatibility import digest, run_lock, sha256_file
from experiments.local_path_search import validate_state, validate_witness, witness_paths
from experiments.native_path_compatibility import check_seal, modules, paths_of, seal
from experiments.repair_collection import state_fingerprint
from experiments.state_analysis import reconstruct_conflicts
from lns2_selector.evaluation.path_quality_execution import read_artifact
from scripts.diagnose_single_release_opportunity import load as load_prior

CONFIG = ROOT/'configs/retained_path_witness_v1.json'
REG = ROOT/'artifacts/initlns-retained-path-witness-v1/registration.json'
TIMED = ROOT/'build/path-quality-pressure-deadline-recovery-v1'
LOCAL = ROOT/'build/initlns-local-path-compatibility-v1-final'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def edges(agents):
    return {(e.left, e.right) for e in reconstruct_conflicts(agents)}


def task_signature(state):
    return dict(rows=state['rows'], cols=state['cols'], obstacles=state['obstacles'],
                agents=sorted((a['id'], a['start'], a['goal']) for a in state['agents']))


def dependency_row(old, reference, aid):
    agents = [dict(a, path=reference[aid]) if a['id']==aid else a for a in old['agents']]
    return sorted({right if left==aid else left for left, right in edges(agents) if aid in (left, right)})


def closure(graph, roots):
    require(set(roots) <= set(graph), 'unknown closure root')
    selected = set(roots); pending = list(roots)
    while pending:
        node = pending.pop()
        require(set(graph[node]) <= set(graph), 'unknown dependency endpoint')
        for other in graph[node]:
            if other not in selected:
                selected.add(other); pending.append(other)
    return sorted(selected)


def check_splice(old, replacement, original_neighborhood):
    require(set(replacement) <= {a['id'] for a in old['agents']}, 'unknown replacement agent')
    # This validator uses the complete original grid and permanent goal occupancy.
    check = validate_witness(old, {str(i): p for i, p in replacement.items()})
    changed = sorted(a['id'] for a in old['agents'] if a['id'] in replacement and a['path']!=replacement[a['id']])
    final = [dict(a, path=replacement.get(a['id'], a['path'])) for a in old['agents']]
    after = edges(final); before = edges(old['agents'])
    require(after <= before, 'new conflict in splice')
    return dict(changed_agents=changed, external_changed_agents=sorted(set(changed)-set(original_neighborhood)),
                conflicts_before=len(before), conflicts_after=len(after), removed_edges=sorted(before-after),
                global_feasible=not after, sum_of_costs=check['sum_of_costs'],
                makespan=max(len(a['path'])-1 for a in final), paths_sha256=digest(paths_of(dict(agents=final))))


def prepare():
    cfg = read_json(CONFIG); prior, prior_out = load_prior()
    schedule = [json.loads(s) for s in (TIMED/'execution_schedule.jsonl').read_text().splitlines() if s.strip()]
    files = dict(prior['files']); pairs = []
    for c in prior['cases']:
        match = [s for s in schedule if s['task_id']==c['task_id'] and s['solver_seed']==c['solver_seed']
                 and s['controller']==cfg['reference_controller'] and s['protocol']==cfg['reference_protocol']
                 and s['budget_seconds']==cfg['reference_budget_seconds']]
        require(len(match)==1, 'unique preselected reference required')
        ref = TIMED/'episodes'/match[0]['job_id']
        original = TIMED/'episodes'/c['source_job_id']
        condition = next(j for j in prior['conditions'] if j['case']==c)
        pairs.append(dict(case=c, reference_job=match[0], reference_dir=ref.relative_to(ROOT).as_posix(),
                          original_dir=original.relative_to(ROOT).as_posix(), original_order=condition['order'],
                          expected_native_sha256=prior['native']['native_sha256']))
        for directory, names in ((ref, ('binding.json', 'initial.json', cfg['path_capture'], 'result.json')),
                                 (original, ('binding.json', 'initial.json'))):
            for name in names:
                path = directory/name
                require(path.exists(), 'fixed reference unavailable; do not choose another: '+str(path))
                files[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    # Include all local pair results for these cases, including unknown and negative rows.
    local = read_json(LOCAL/'manifest.json'); verification = read_json(LOCAL/'verification.json')
    require(sha256_file(LOCAL/'manifest.json')=='42a4a3d69c920fd93b9236261a64ab2904516bca3f534d7df576df98c6abe0e2', 'local manifest SHA')
    require(sha256_file(LOCAL/'verification.json')=='fbcc5584c9a485fe2e5a46b4e1a6c21befaf3d19fe1558e60c2130c1cbe8795e', 'local verification SHA')
    case_ids = {p['case']['case_id'] for p in pairs}; local_jobs = []
    for j in local['jobs']:
        if j['case_id'] not in case_ids:
            continue
        file = 'results/'+j['job_id']+'.json'
        require(sha256_file(LOCAL/file)==verification['result_files'][file], 'local result SHA')
        local_case = next(c for c in local['cases'] if c['case_id']==j['case_id'])
        local_jobs.append(dict(job=j, file=(LOCAL/file).relative_to(ROOT).as_posix(), state_file=local_case['state_file']))
        files[(LOCAL/file).relative_to(ROOT).as_posix()] = verification['result_files'][file]
        files[local_case['state_file']] = local['files'][local_case['state_file']]
    paths = [CONFIG, Path(__file__), ROOT/'tests/test_retained_path_witness.py',
             ROOT/'docs/RETAINED_PATH_WITNESS_PROTOCOL_ZH.md', TIMED/'execution_schedule.jsonl',
             LOCAL/'manifest.json', LOCAL/'verification.json', prior_out/'plan.json', prior_out/'report.json',
             ROOT/'artifacts/initlns-single-release-opportunity-v1/evidence.json',
             ROOT/'lns2_selector/evaluation/path_quality_execution.py']
    for path in paths:
        files[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    plan = seal(dict(schema=cfg['schema'], config=cfg, pairs=pairs, local_jobs=local_jobs, files=files, native=prior['native']))
    out = ROOT/cfg['output']; out.mkdir(parents=True, exist_ok=True)
    require(not (out/'plan.json').exists(), 'existing frozen output')
    write_json(out/'plan.json', plan)
    write_json(REG, dict(plan_file=cfg['output']+'/plan.json', plan_sha256=sha256_file(out/'plan.json'), schema=cfg['schema']))
    return dict(cases=len(pairs), reference_files=len(pairs), local_jobs=len(local_jobs), workers=cfg['workers'],
                solver_calls=0, plan_sha256=sha256_file(out/'plan.json'))


def load():
    reg = read_json(REG); path = ROOT/reg['plan_file']
    require(sha256_file(path)==reg['plan_sha256'], 'registered plan SHA')
    plan = read_json(path); check_seal(plan)
    for p, h in plan['files'].items():
        require(sha256_file(ROOT/p)==h, 'frozen input changed: '+p)
    return plan, path.parent


def read_pair(pair):
    case = pair['case']; old = read_json(ROOT/case['state_file'])
    validate_state(old)
    require(state_fingerprint(old)==case['full_state_fingerprint'], 'anchor state fingerprint')
    refdir = ROOT/pair['reference_dir']; original = ROOT/pair['original_dir']
    binding = read_json(refdir/'binding.json'); oldbinding = read_json(original/'binding.json')
    require(binding['item']==pair['reference_job'], 'reference task binding')
    for key in ('task_id', 'solver_seed', 'map_id'):
        require(binding['item'][key]==oldbinding['item'][key]==case[key], 'paired identity: '+key)
    require(binding['native_sha256']==oldbinding['native_sha256']==pair['expected_native_sha256'], 'paired native identity')
    initial = read_artifact(refdir/'initial.json', binding['binding'])
    source_initial = read_artifact(original/'initial.json', oldbinding['binding'])
    for capture in (initial, source_initial):
        require(state_fingerprint(capture['observation'])==capture['state_fingerprint'], 'initial payload fingerprint')
        require(task_signature(capture['observation'])==task_signature(old), 'task grid/agent mismatch')
    require(initial['state_fingerprint']==source_initial['state_fingerprint'], 'paired initial states differ')
    final = read_artifact(refdir/'first_feasible.json', binding['binding'])
    reference = final['observation']; validate_state(reference)
    require(state_fingerprint(reference)==final['state_fingerprint'], 'reference path fingerprint')
    require(task_signature(reference)==task_signature(old), 'reference belongs to another task')
    require(reference['num_of_colliding_pairs']==0 and reference['feasible'], 'reference not globally feasible')
    return old, {a['id']: a['path'] for a in reference['agents']}


_CONTEXT = None


def init_context(pairs):
    global _CONTEXT
    _CONTEXT = {p['case']['case_id']: read_pair(p) for p in pairs}


def row_job(job):
    cid, aid = job; old, reference = _CONTEXT[cid]
    return cid, aid, dependency_row(old, reference, aid)


def root_job(job):
    cid, root, selected, original_order = job; old, reference = _CONTEXT[cid]
    return dict(case_id=cid, root=root, adopted_reference_ids=selected,
                **check_splice(old, {i:reference[i] for i in selected}, original_order))


def audit():
    plan, out = load()
    with run_lock(out):
        states = {p['case']['case_id']: read_pair(p) for p in plan['pairs']}
        graphs = defaultdict(dict); rows = []
        jobs = [(cid, a['id']) for cid, (s, _) in states.items() for a in s['agents']]
        write_json(out/'run_status.json', dict(status='running', phase='dependency_rows', total=len(jobs), completed=0))
        try:
            with mp.get_context('spawn').Pool(plan['config']['workers'], initializer=init_context,
                                              initargs=(plan['pairs'],)) as pool:
                for index, (cid, aid, neighbors) in enumerate(pool.imap(row_job, jobs, chunksize=2), 1):
                    graphs[cid][aid] = neighbors
                    if index % 200 == 0:
                        progress = dict(status='running', phase='dependency_rows', total=len(jobs), completed=index)
                        write_json(out/'run_status.json', progress); print(json.dumps(progress), flush=True)
                root_jobs = []
                for p in plan['pairs']:
                    cid = p['case']['case_id']; old, _ = states[cid]
                    roots = sorted({i for pair in edges(old['agents']) for i in pair})
                    root_jobs.extend((cid, root, closure(graphs[cid], [root]), p['original_order']) for root in roots)
                rows = list(pool.imap(root_job, root_jobs, chunksize=1))
            historical = []
            for entry in plan['local_jobs']:
                r = read_json(ROOT/entry['file']); caseid = entry['job']['case_id']
                old = states[caseid][0]; local = read_json(ROOT/entry['state_file'])
                require(state_fingerprint(local)==state_fingerprint(old), 'historical witness from different state')
                require(r['job']==entry['job'], 'historical result identity')
                original_order = next(p['original_order'] for p in plan['pairs'] if p['case']['case_id']==caseid)
                checks = []
                for w in witness_paths(r):
                    checks.append(check_splice(old, {int(i):path for i,path in w.items()}, original_order))
                historical.append(dict(file=entry['file'], job=entry['job'], status=r['status'], witnesses=checks))
            # Only import paths for validation. No reset(seed), PP, or candidate generation calls.
            native, _ = modules(plan['native']); summaries = []; native_checks = []
            for p in plan['pairs']:
                case = p['case']; cid = case['case_id']; old, reference = states[cid]
                options = [r for r in rows if r['case_id']==cid]
                best = min(options, key=lambda r:(len(r['changed_agents']), len(r['external_changed_agents']), r['root']))
                selected = best['adopted_reference_ids']
                replacement = {i:reference[i] for i in selected}
                paths = [replacement.get(a['id'], a['path']) for a in old['agents']]
                env = native.LNS2RepairEnv(str(ROOT/case['map_file']), str(ROOT/case['scenario_file']), len(old['agents']), time_limit=60)
                observed = env.reset_paths(paths, seed=case['solver_seed'])
                require(paths_of(observed)==paths and observed['num_of_colliding_pairs']==best['conflicts_after'], 'native path validation')
                native_checks.append(dict(case_id=cid, root=best['root'], paths_sha256=digest(paths), passed=True))
                summary = dict(case_id=cid, map_id=case['map_id'], conflicts_before=old['num_of_colliding_pairs'],
                               conflict_endpoint_roots=len(options), smallest_reference_splice=best,
                               fixed_reference_only=True,
                               small=len(best['changed_agents']) <= plan['config']['maximum_small_changed_agents'])
                summaries.append(summary)
                write_json(out/'witnesses'/(cid+'.json'), seal(dict(plan=plan['content_sha256'], case_id=cid,
                    original_state=case['state_file'], reference_source=p['reference_dir']+'/first_feasible.json',
                    adopted_reference_ids=selected, paths=paths, check=best)))
            result = seal(dict(schema=plan['schema'], plan=plan['content_sha256'], dependency_rows=len(jobs),
                roots=len(rows), solver_search_calls=0, native_validation_only=len(native_checks), cases=summaries,
                root_results=rows, historical=historical, native_checks=native_checks,
                decision='existence_diagnostic_only_no_controller_promotion'))
            write_json(out/'dependencies.json', seal(dict(plan=plan['content_sha256'], graphs=graphs)))
            write_json(out/'report.json', result)
            write_json(out/'run_status.json', dict(status='complete', dependency_rows=len(jobs), roots=len(rows), error=None))
            return dict(cases=summaries, historical_feasible_witnesses=sum(len(r['witnesses']) for r in historical),
                        decision=result['decision'], report_sha256=sha256_file(out/'report.json'))
        except BaseException as exc:
            write_json(out/'run_status.json', dict(status='error', error=repr(exc))); raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'audit'))
    args = parser.parse_args()
    print(json.dumps(prepare() if args.phase=='prepare' else audit(), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
