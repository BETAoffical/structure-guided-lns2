"""Frozen, bounded queued-node observation; no repair intervention."""
import importlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments._common import read_json, write_json
from experiments.local_path_compatibility import sha256_file
from experiments.native_path_compatibility import paths_of, seal
from experiments.full_neighborhood_recovery import SearchBudget, run_sequence, check_paths
from experiments.search_frontier_observation import FrontierProbe
from experiments.search_occupancy_observation import rank_contacts
from scripts import diagnose_search_occupancy as runner
from scripts.diagnose_reservation_mediation import require

MODULE = 'build/linux/search-frontier-observer-v1/lns2_search_frontier_native.cpython-310-x86_64-linux-gnu.so'
OUT = ROOT / 'build/initlns-search-frontier-observer-v1'


def configure():
    runner.OUT = OUT
    runner.REG = ROOT / 'artifacts/initlns-search-frontier-observer-v1/registration.json'
    runner.__file__ = __file__


def prepare():
    from scripts.diagnose_search_occupancy_snapshot import configure as previous
    previous()
    plan = runner.load()
    configure()
    plan['schema'] = 'lns2.search_frontier.v1'
    plan['module_file'] = MODULE
    plan['event_cap'] = 32768
    plan['primary_profile'] = 'queued_envelope'
    plan['gates'] = dict(witness_top4=6, maximum_mean_owner_fraction=0.25)
    plan['parent_plan_sha256'] = sha256_file(ROOT / 'build/initlns-search-occupancy-observer-v2/plan.json')
    for p in (ROOT / 'src/search_frontier').glob('*'):
        if p.is_file(): plan['files'][p.relative_to(ROOT).as_posix()] = sha256_file(p)
    for name in (MODULE, 'build/linux/search-frontier-observer-v1/observed_sipp.cpp',
                 'build/linux/search-frontier-observer-v1/CMakeCache.txt',
                 'experiments/search_frontier_observation.py', 'scripts/diagnose_search_frontier.py',
                 'tests/test_search_frontier.py', 'docs/SEARCH_FRONTIER_OBSERVER_PROTOCOL_ZH.md'):
        plan['files'][name] = sha256_file(ROOT / name)
    OUT.mkdir(exist_ok=True)
    require(not (OUT / 'plan.json').exists(), 'plan exists')
    write_json(OUT / 'plan.json', plan)
    print(json.dumps(dict(plan_sha256=sha256_file(OUT / 'plan.json'), module_sha256=plan['files'][MODULE], jobs=len(plan['jobs']))))


def worker(phase, job_id):
    plan = runner.load()
    job = next(j for j in plan['jobs'] if j['id'] == job_id and j['phase'] == phase)
    expected = read_json(ROOT / job['expected_file'])
    expected = expected['data']['sequences']['compressed11'] if phase == 'witness' else expected['base']
    state = read_json(ROOT / job['state_file'])
    original = {a['id']: a['path'] for a in state['agents']}
    sys.path.insert(0, str((ROOT / MODULE).parent))
    module = importlib.import_module('lns2_search_frontier_native')
    require(Path(module.__file__).resolve() == (ROOT / MODULE).resolve(), 'wrong module')
    require(module.observer_schema == 'lns2.queued_positive_transitions.v1', 'wrong schema')
    runs = {}
    external = set(original) - set(job['order'])
    for mode in ('off', 'on'):
        native = module.NativePathProbe(str(ROOT / job['map_file']), str(ROOT / job['scenario_file']), paths_of(state))
        probe = FrontierProbe(module, native, original, external, mode == 'on', plan['event_cap'])
        result = run_sequence(probe, state, job['order'], job['seed'], SearchBudget(plan['search_seconds'], plan['call_seconds']))
        require(runner.signature(result) == runner.signature(expected), 'frozen/frontier signature mismatch: ' + mode)
        check_paths(state, result['paths'], job['order'])
        runs[mode] = result
    ranked = {name: rank_contacts(values) for name, values in probe.summaries.items()}
    ranks = {}
    for name, rows in ranked.items():
        ids = [r['agent'] for r in rows]
        ranks[name] = ids.index(plan['witness_agent'])+1 if plan['witness_agent'] in ids and phase == 'witness' else None
    raw = OUT / phase / (job_id + '.json.gz')
    runner.write_gzip(raw, dict(job=job, on=runs['on'], off=runs['off'], captures=probe.captures, summaries=probe.summaries))
    main = plan['primary_profile']
    summary = dict(schema='lns2.search_frontier_result.v1', job=job, status='ok', parity_passed=True,
                   plan_sha256=sha256_file(OUT / 'plan.json'), raw_file=raw.relative_to(ROOT).as_posix(), raw_sha256=sha256_file(raw),
                   truncated_queries=sum(c['capture']['truncated'] for c in probe.captures),
                   unmapped_flags=sum(len(s['unmatched']) for s in probe.summaries['queued_all']),
                   ranking=ranked[main], witness_rank=ranks[main], profile_rankings=ranked, profile_witness_ranks=ranks,
                   external_count=len(external), searches=len(probe.captures),
                   records=sum(len(c['capture']['events']) for c in probe.captures),
                   offered=sum(c['capture']['offered'] for c in probe.captures),
                   virtual_goal_visits=sum(c['capture']['virtual_goals'] for c in probe.captures))
    write_json(OUT / phase / (job_id + '.json'), seal(summary))


def analyze(phase):
    plan = runner.load()
    rows = [runner.read_result(j) for j in plan['jobs'] if j['phase'] == phase]
    integrity = all(r['parity_passed'] and not r['truncated_queries'] and not r['unmapped_flags'] for r in rows)
    compact = statistics.mean(len(r['ranking']) / r['external_count'] for r in rows)
    top4 = sum(r['witness_rank'] is not None and r['witness_rank'] <= 4 for r in rows)
    passed = integrity and compact <= plan['gates']['maximum_mean_owner_fraction']
    if phase == 'witness': passed = passed and top4 >= plan['gates']['witness_top4']
    report = dict(schema='lns2.search_frontier_report.v1', phase=phase, jobs=len(rows), integrity_passed=integrity,
                  mean_owner_fraction=compact, witness_top4=top4 if phase == 'witness' else None,
                  next_phase_allowed=bool(passed and phase == 'witness'),
                  decision='observation_only' if passed else 'no_go_frontier_observation',
                  truncated_searches=sum(r['truncated_queries'] for r in rows),
                  unmapped_flags=sum(r['unmapped_flags'] for r in rows),
                  searches=sum(r['searches'] for r in rows), records=sum(r['records'] for r in rows),
                  profiles={name: dict(owner_counts=[len(r['profile_rankings'][name]) for r in rows],
                      witness_ranks=[r['profile_witness_ranks'][name] for r in rows] if phase == 'witness' else None)
                      for name in ('queued_all', 'queued_envelope', 'popped_envelope')},
                  result_sha256={r['job']['id']: sha256_file(OUT / phase / (r['job']['id'] + '.json')) for r in rows})
    write_json(OUT / (phase + '-report.json'), report)
    print(json.dumps(report))


def main():
    configure()
    if len(sys.argv) > 1 and sys.argv[1] == 'prepare': prepare()
    elif len(sys.argv) > 1 and sys.argv[1] == '_worker': worker(sys.argv[2], sys.argv[3])
    elif len(sys.argv) > 1 and sys.argv[1] == 'analyze': analyze(sys.argv[2])
    else: runner.main()


if __name__ == '__main__': main()
