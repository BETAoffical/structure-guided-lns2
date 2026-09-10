"""Read-only independent reconstruction and failure analysis of frontier records."""
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments._common import read_json, write_json
from experiments.local_path_compatibility import sha256_file
from experiments.native_path_compatibility import check_seal
from scripts import diagnose_search_frontier as collector
from scripts import diagnose_search_occupancy as runner
from scripts.audit_search_occupancy import reconstruct_contacts, independent_ranking
from scripts.diagnose_reservation_mediation import require


def convert_and_filter(rows, goal_conflicts, goal_cost, name):
    events = []
    for row in rows:
        require(len(row) == 12 and row[0] in (1, 2) and not row[11], 'invalid event')
        kind, src, previous_time, dst, tick, g, h, c, previous_c, v, _, _ = row
        require(tick > previous_time and c > previous_c, 'invalid transition')
        edge = c - previous_c - v
        require(v in (0, 1) and edge in (0, 1), 'invalid collision increment')
        if name == 'popped_envelope' and kind != 2:
            continue
        if name != 'popped_envelope' and kind != 1:
            continue
        if name != 'queued_all' and (c > goal_conflicts + 1 or g + h > goal_cost):
            continue
        events.append([0, src, dst, tick, tick+1, tick, tick+1, tick+1, v, edge])
    return events


def contact_set(summary):
    return {(int(owner), *contact) for owner, values in summary['contacts'].items() for contact in values}


def verify_job(job):
    collector.configure()
    result = runner.read_result(job)
    with gzip.open(ROOT / result['raw_file'], 'rt') as stream:
        raw = json.load(stream)
    state = read_json(ROOT / job['state_file'])
    original = {a['id']: a['path'] for a in state['agents']}
    fixed = {i: p for i, p in original.items() if i not in job['order']}
    external = set(fixed)
    require(raw['job'] == job, 'raw job mismatch')
    source = read_json(ROOT / job['expected_file'])
    first_divergent = source['data']['explanation']['first_divergent_agent'] if job['phase'] == 'witness' else None
    expected = source['data']['sequences']['compressed11'] if job['phase'] == 'witness' else source['base']
    require(runner.signature(expected) == runner.signature(raw['on']) == runner.signature(raw['off']), 'frozen parity')
    prior_dir = ROOT / 'build/initlns-search-occupancy-observer-v2'
    prior_result = read_json(prior_dir / job['phase'] / (job['id'] + '.json'))
    check_seal(prior_result)
    prior_report = read_json(prior_dir / (job['phase'] + '-report.json'))
    require(sha256_file(prior_dir / job['phase'] / (job['id'] + '.json')) == prior_report['result_sha256'][job['id']], 'prior report binding')
    require(sha256_file(ROOT / prior_result['raw_file']) == prior_result['raw_sha256'], 'prior raw hash')
    with gzip.open(ROOT / prior_result['raw_file'], 'rt') as stream:
        prior = json.load(stream)
    require(runner.signature(prior['on']) == runner.signature(raw['on']), 'prior path signature')
    profiles = ('queued_all', 'queued_envelope', 'popped_envelope')
    require(len(raw['captures']) == len(raw['on']['records']) == len(prior['summaries']), 'capture count')
    summaries = {name: [] for name in profiles}
    searches = []
    target = read_json(collector.OUT / 'plan.json')['witness_agent']
    for index, (capture, record) in enumerate(zip(raw['captures'], raw['on']['records'])):
        aid = record['agent']; c = capture['capture']
        require(aid == capture['agent'] == job['order'][index] and capture['fixed'] == list(fixed), 'PP prefix mismatch')
        require(c['popped'] == record['search']['expanded'], 'expanded counter')
        require(c['goal_cost'] == record['search']['cost'] and c['goal_conflicts'] == record['search']['low_level_collisions'], 'goal values')
        require(not c['truncated'] and len(c['events']) == c['offered'], 'truncated evidence')
        values = {}
        for name in profiles:
            events = convert_and_filter(c['events'], c['goal_conflicts'], c['goal_cost'], name)
            summary = reconstruct_contacts(aid, fixed, record['search']['path'], events, external)
            require(summary == raw['summaries'][name][index] and not summary['unmatched'], 'contact reconstruction')
            summaries[name].append(summary)
            values[name] = dict(owners=len(summary['contacts']), target_contacts=len(summary['contacts'].get(str(target), [])))
        require(contact_set(summaries['queued_all'][-1]) <= contact_set(prior['summaries'][index]), 'queued contact absent from offered intervals')
        require(contact_set(summaries['popped_envelope'][-1]) <= contact_set(summaries['queued_envelope'][-1]), 'popped contact not enqueued')
        searches.append(dict(agent=aid, goal_cost=c['goal_cost'], goal_conflicts=c['goal_conflicts'],
                             enqueued=c['enqueued'], popped=c['popped'], virtual_goal_visits=c['virtual_goals'], profiles=values))
        fixed[aid] = record['search']['path']
    rankings = {name: independent_ranking(values) for name, values in summaries.items()}
    require(rankings == result['profile_rankings'], 'profile rankings')
    for name, ranking in rankings.items():
        ids = [r['agent'] for r in ranking]
        rank = ids.index(target) + 1 if target in ids else None
        require(result['profile_witness_ranks'][name] == rank, 'witness rank')
    profile_details = {}
    for name, ranking in rankings.items():
        target_row = next((r for r in ranking if r['agent'] == target), None)
        max_single = max(s['profiles'][name]['owners'] for s in searches)
        profile_details[name] = dict(union_owners=len(ranking), maximum_single_search_owners=max_single,
                                    target=target_row, leader=ranking[0] if ranking else None)
    return dict(id=job['id'], seed=job['seed'], searches=searches, profiles=profile_details,
                historical_first_divergent_agent=first_divergent,
                historical_first_divergent_profiles=next(s['profiles'] for s in searches if s['agent'] == first_divergent),
                raw_sha256=result['raw_sha256'], result_sha256=sha256_file(collector.OUT / 'witness' / (job['id'] + '.json')))


def main():
    collector.configure()
    plan = runner.load()
    report = read_json(collector.OUT / 'witness-report.json')
    jobs = [j for j in plan['jobs'] if j['phase'] == 'witness']
    for j in jobs:
        require(sha256_file(collector.OUT / 'witness' / (j['id'] + '.json')) == report['result_sha256'][j['id']], 'report binding')
    with ProcessPoolExecutor(max_workers=20) as pool:
        rows = list(pool.map(verify_job, jobs))
    summaries = {}
    for name in ('queued_all', 'queued_envelope', 'popped_envelope'):
        summaries[name] = dict(mean_union_owners=statistics.mean(r['profiles'][name]['union_owners'] for r in rows),
                              mean_maximum_single_owners=statistics.mean(r['profiles'][name]['maximum_single_search_owners'] for r in rows),
                              target_planner_counts=[r['profiles'][name]['target']['planner_count'] if r['profiles'][name]['target'] else 0 for r in rows],
                              historical_first_divergent_target_contact_counts=[r['historical_first_divergent_profiles'][name]['target_contacts'] for r in rows],
                              leader_planner_counts=[r['profiles'][name]['leader']['planner_count'] if r['profiles'][name]['leader'] else 0 for r in rows])
    output = dict(schema='lns2.search_frontier_independent_audit.v1', verified_jobs=len(rows),
                  implementation_sha256=sha256_file(Path(__file__)), solver_runs=0,
                  report_sha256=sha256_file(collector.OUT / 'witness-report.json'),
                  raw_owner_and_prefix_errors=0, nested_contact_sets_verified=True,
                  summaries=summaries, rows=rows,
                  interpretation='Frequency and competitive-envelope readiness failed; no intervention or runtime claim.')
    write_json(collector.OUT / 'independent-audit.json', output)
    print(json.dumps({k: v for k, v in output.items() if k != 'rows'}))


if __name__ == '__main__': main()
