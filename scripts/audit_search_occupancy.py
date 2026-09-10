"""Independently audit saved observer evidence; never run a solver."""
from collections import defaultdict
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
from scripts.diagnose_search_occupancy import OUT, load, read_result, signature
from scripts.diagnose_reservation_mediation import require
from scripts.audit_reservation_outcomes import canonical_role


def reconstruct_contacts(planner, fixed, returned, events, external):
    # Construct only requested occupancy keys, then fill them from fixed paths.
    vertices = {(e[2], e[5]): set() for e in events if e[8]}
    swaps = {(e[2], e[1], e[5]): set() for e in events if e[9]}
    terminal_ticks = defaultdict(set)
    for cell, tick in vertices:
        terminal_ticks[cell].add(tick)
    for aid, path in fixed.items():
        require(bool(path), 'empty fixed path')
        for tick, cell in enumerate(path):
            if (cell, tick) in vertices:
                vertices[cell, tick].add(aid)
            if tick and path[tick - 1] != cell:
                key = (path[tick - 1], cell, tick)
                if key in swaps:
                    swaps[key].add(aid)
        for tick in terminal_ticks[path[-1]]:
            if tick >= len(path):
                vertices[path[-1], tick].add(aid)
    contacts = defaultdict(set)
    unmatched = []
    flagged = 0
    for event in events:
        require(len(event) == 10 and event[5] >= 0, 'invalid event')
        kind, src, dst, _, _, tick, _, _, vertex, edge = event
        require(kind in (0, 1, 2) and vertex in (0, 1) and edge in (0, 1), 'invalid flags')
        for label, enabled, owners in (
            ('vertex', vertex, vertices.get((dst, tick), set())),
            ('edge', edge, swaps.get((dst, src, tick), set())),
        ):
            if not enabled:
                continue
            flagged += 1
            if not owners:
                unmatched.append(dict(event=event, kind=label))
            on_path = returned[min(tick, len(returned) - 1)] == dst
            if label == 'edge':
                on_path = on_path and tick > 0 and returned[min(tick - 1, len(returned) - 1)] == src
            if not on_path:
                for owner in owners.intersection(external):
                    contacts[owner].add((planner, label, src, dst, tick))
    return dict(flagged=flagged, unmatched=unmatched,
                contacts={str(k): [list(v) for v in sorted(values)] for k, values in contacts.items()})


def independent_ranking(summaries):
    union = defaultdict(set)
    for summary in summaries:
        for aid, values in summary['contacts'].items():
            union[int(aid)].update(tuple(v) for v in values)
    counts = [(aid, len({v[0] for v in values}), len(values)) for aid, values in union.items()]
    counts.sort(key=lambda v: (-v[1], -v[2], v[0]))
    return [dict(agent=a, planner_count=p, contact_count=c) for a, p, c in counts]


def audit_job(job):
    result = read_result(job)
    with gzip.open(ROOT / result['raw_file'], 'rt') as stream:
        raw = json.load(stream)
    require(raw['job'] == job, 'raw job changed')
    predecessor = ROOT / 'build/initlns-search-occupancy-observer-v1'
    prior_report = read_json(predecessor / (job['phase'] + '-report.json'))
    prior_path = predecessor / job['phase'] / (job['id'] + '.json')
    require(sha256_file(prior_path) == prior_report['result_sha256'][job['id']], 'predecessor binding')
    prior_result = read_json(prior_path)
    check_seal(prior_result)
    require(sha256_file(ROOT / prior_result['raw_file']) == prior_result['raw_sha256'], 'predecessor raw SHA')
    with gzip.open(ROOT / prior_result['raw_file'], 'rt') as stream:
        prior = json.load(stream)
    require(prior['job'] == job and prior['summaries'] == raw['summaries'], 'snapshot changed contacts')
    require(len(prior['captures']) == len(raw['captures']), 'snapshot changed call count')
    for previous, current in zip(prior['captures'], raw['captures']):
        require(previous['capture'] == current['capture'] and previous['agent'] == current['agent'],
                'snapshot changed native observations')
    for mode in ('on', 'off'):
        require(signature(prior[mode]) == signature(raw[mode]), 'snapshot changed scientific result')
    state = read_json(ROOT / job['state_file'])
    original = {a['id']: a['path'] for a in state['agents']}
    expected_source = read_json(ROOT / job['expected_file'])
    expected = (expected_source['data']['sequences']['compressed11']
                if job['phase'] == 'witness' else expected_source['base'])
    require(signature(raw['off']) == signature(raw['on']) == signature(expected), 'raw parity')
    order = job['order']
    fixed = {aid: p for aid, p in original.items() if aid not in order}
    external = set(fixed)
    records = raw['on']['records']
    require(len(raw['captures']) == len(records) == len(raw['summaries']), 'search count')
    summaries = []
    clipped = []
    within_arrival = set()
    late_records = 0
    terminal_contacts = 0
    contact_total = 0
    for index, (record, capture, saved) in enumerate(zip(records, raw['captures'], raw['summaries'])):
        aid = record['agent']
        require(aid == order[index] == capture['agent'], 'planner order')
        require(capture['fixed'] == list(fixed), 'fixed PP prefix changed')
        require(record['search']['status'] == 'path', 'non-path search')
        data = capture['capture']
        require(data['offered'] >= len(data['events']) and
                bool(data['truncated']) == (data['offered'] > len(data['events'])), 'invalid capture counters')
        if data['truncated']:
            clipped.append(dict(agent=aid, offered=data['offered'], retained=len(data['events'])))
        summary = reconstruct_contacts(aid, fixed, record['search']['path'], data['events'], external)
        require(summary == saved, 'owner/contact reconstruction differs')
        arrival = len(record['search']['path']) - 1
        late_records += sum(event[5] > arrival for event in data['events'])
        for owner, contacts in summary['contacts'].items():
            if any(contact[-1] <= arrival for contact in contacts):
                within_arrival.add(int(owner))
            terminal_contacts += sum(contact[-1] >= len(fixed[int(owner)]) - 1 for contact in contacts)
            contact_total += len(contacts)
        summaries.append(summary)
        fixed[aid] = record['search']['path']
    ranking = independent_ranking(summaries)
    require(ranking == result['ranking'], 'ranking differs')
    require(sum(s['flagged'] for s in summaries) == result['flagged'], 'flag count')
    require(sum(len(s['unmatched']) for s in summaries) == result['unmapped_flags'], 'unmapped count')
    require(sum(len(c['capture']['events']) for c in raw['captures']) == result['records'], 'record count')
    incident = set()
    for r in records:
        for event in r['incident_events']:
            incident.update({event['left'], event['right']} & external)
    observed = {r['agent'] for r in ranking}
    return dict(id=job['id'], phase=job['phase'], case_id=job['case_id'], map_id=job['map_id'],
                role=canonical_role(job['role']), external_agents=len(external), observed_owners=len(observed),
                owner_fraction=len(observed) / len(external) if external else 0.,
                returned_external_owners=len(incident), observed_not_in_returned=len(observed - incident),
                truncated_searches=clipped, complete_capture=not clipped,
                predecessor_observation_equal=True,
                fixed_snapshots_corrected=sum(a['fixed'] != b['fixed'] for a, b in zip(prior['captures'], raw['captures'])),
                exploratory=dict(late_records=late_records,
                                 late_record_fraction=late_records / result['records'] if result['records'] else 0.,
                                 within_returned_arrival_owners=len(within_arrival),
                                 within_returned_arrival_fraction=len(within_arrival) / len(external) if external else 0.,
                                 terminal_contacts=terminal_contacts, external_contacts=contact_total,
                                 terminal_contact_fraction=terminal_contacts / contact_total if contact_total else 0.),
                records=result['records'], queries=result['queries'], searches=result['searches'],
                witness_rank=result['witness_rank'], raw_sha256=result['raw_sha256'],
                result_sha256=sha256_file(OUT / job['phase'] / (job['id'] + '.json')))


def distribution(values):
    return dict(minimum=min(values), median=statistics.median(values),
                mean=statistics.mean(values), maximum=max(values))


def summarize(rows):
    return dict(jobs=len(rows), states=len({r['case_id'] for r in rows}),
                maps=len({r['map_id'] for r in rows}),
                owner_fraction=distribution([r['owner_fraction'] for r in rows]),
                observed_owners=distribution([r['observed_owners'] for r in rows]),
                returned_external_owners=distribution([r['returned_external_owners'] for r in rows]),
                observed_not_in_returned=distribution([r['observed_not_in_returned'] for r in rows]),
                exploratory_late_fraction=distribution([r['exploratory']['late_record_fraction'] for r in rows]),
                exploratory_within_arrival_fraction=distribution([r['exploratory']['within_returned_arrival_fraction'] for r in rows]),
                exploratory_terminal_contact_fraction=distribution([r['exploratory']['terminal_contact_fraction'] for r in rows]),
                total_searches=sum(r['searches'] for r in rows),
                total_records=sum(r['records'] for r in rows))


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', choices=('v1', 'v2'), default='v2')
    args = parser.parse_args()
    if args.version == 'v2':
        from scripts.diagnose_search_occupancy_snapshot import configure
        configure()
        global OUT
        OUT = ROOT / 'build/initlns-search-occupancy-observer-v2'
    plan = load()
    for phase in ('witness', 'development'):
        report = read_json(OUT / (phase + '-report.json'))
        for job in plan['jobs']:
            if job['phase'] == phase:
                require(sha256_file(OUT / phase / (job['id'] + '.json')) == report['result_sha256'][job['id']],
                        'phase report binding changed')
    with ProcessPoolExecutor(max_workers=20) as pool:
        rows = list(pool.map(audit_job, plan['jobs']))
    development = [r for r in rows if r['phase'] == 'development']
    report = dict(schema='lns2.search_occupancy_independent_audit.v1',
                  implementation_sha256=sha256_file(Path(__file__)),
                  plan_sha256=sha256_file(OUT / 'plan.json'), verified_jobs=len(rows),
                  raw_reconstruction_errors=0, solver_runs=0,
                  complete_capture_jobs=sum(r['complete_capture'] for r in rows),
                  truncated_searches=sum(len(r['truncated_searches']) for r in rows),
                  completeness_gate_passed=all(r['complete_capture'] for r in rows),
                  predecessor_observation_equal_jobs=sum(r['predecessor_observation_equal'] for r in rows),
                  fixed_snapshots_corrected=sum(r['fixed_snapshots_corrected'] for r in rows),
                  partial_capture_note='Owner counts are lower bounds where capture is truncated; never impute missing events.',
                  phases={p: summarize([r for r in rows if r['phase'] == p]) for p in ('witness', 'development')},
                  roles={role: summarize([r for r in development if r['role'] == role])
                         for role in sorted({r['role'] for r in development})},
                  maps={mid: summarize([r for r in development if r['map_id'] == mid])
                        for mid in sorted({r['map_id'] for r in development})},
                  complete_development=summarize([r for r in development if r['complete_capture']])
                      if any(r['complete_capture'] for r in development) else None,
                  witness_top4=sum(r['witness_rank'] is not None and r['witness_rank'] <= 4
                                   for r in rows if r['phase'] == 'witness'),
                  witness_top8=sum(r['witness_rank'] is not None and r['witness_rank'] <= 8
                                   for r in rows if r['phase'] == 'witness'),
                  interpretation='Observability only; no causal selectivity or recovery benefit established.',
                  rows=rows)
    write_json(OUT / 'independent-audit.json', report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('rows', 'maps')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
