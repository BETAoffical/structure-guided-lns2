"""Recheck saved native diagnostics and expose applicability to historical neighborhoods."""
from pathlib import Path
import argparse
from collections import defaultdict
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.native_path_compatibility import check_result, check_seal, load, result_file, seal
from experiments.local_path_compatibility import ROOT, contained, digest, sha256_file
from experiments.local_path_search import validate_witness
from experiments._common import read_json, write_json


def verify(output):
    m = load(output)
    report = read_json(output/'report.json')
    check_seal(report)
    if not report['complete'] or not report['parity_pass'] or report['decision'] != 'hard_pair_mechanism_only':
        raise ValueError('completed, error-free parity and mechanism results required')
    cases = {c['case_id']: c for c in m['cases']}
    states = {key: read_json(contained(c['state_file'])) for key, c in cases.items()}
    neighborhoods = {key: set(read_json(contained(c['historical_transition_file']))['action']['agents'])
                     for key, c in cases.items()}
    groups = defaultdict(list)
    recovered, witnesses, parity_count = [], 0, 0
    for job in m['jobs']:
        file = result_file(output, job)
        result = read_json(file)
        check_result(m, job, result)
        if report['result_files'][file.relative_to(ROOT).as_posix()] != sha256_file(file):
            raise ValueError('report/result SHA mismatch')
        if job['kind'] == 'parity':
            if result['status'] != 'pass' or not all(result['checks'].values()):
                raise ValueError('parity failure')
            parity_count += 1
            continue
        if result['status'] not in ('feasible', 'not_found', 'unknown'):
            raise ValueError('invalid hard-pair result')
        groups[(job['case_id'], tuple(job['order']))].append(result)
        if result['status'] == 'feasible':
            if validate_witness(states[job['case_id']], result['paths']) != result['witness']:
                raise ValueError('witness summary mismatch')
            if not result.get('native_witness_verified') or result['first_cost_delta'] != 0:
                raise ValueError('missing native check or increased first-path cost')
            witnesses += 1
            if len(result['attempts']) > 1:
                original = {a['id']: a['path'] for a in states[job['case_id']]['agents']}
                changed = {int(a) for a, path in result['paths'].items() if path != original[int(a)]}
                recovered.append(dict(case_id=job['case_id'], order=job['order'], method=job['method'],
                    outside_historical_neighborhood=sorted(set(job['order'])-neighborhoods[job['case_id']]),
                    changed_outside_historical_neighborhood=sorted(changed-neighborhoods[job['case_id']]),
                    globally_feasible=result['witness']['globally_feasible']))
    for key, rows in groups.items():
        if {r['job']['method'] for r in rows} != {'ordinary', 'random_paths', 'directed_paths'} or len(rows) != 3:
            raise ValueError('missing paired method')
        signatures = []
        for row in rows:
            base = row['attempts'][0]['result']
            signatures.append(digest({k: None if v is None else
                {field: v[field] for field in ('status', 'path', 'cost', 'expanded', 'generated', 'low_level_collisions')}
                for k, v in base.items()}))
        if len(set(signatures)) != 1:
            raise ValueError('ordinary baseline mismatch across methods: ' + str(key))
    value = seal(dict(schema='lns2.native_path_compatibility_verification.v1',
        fingerprint=m['content_sha256'], passed=True, parity_count=parity_count,
        baseline_triplets=len(groups), verified_witness_count=witnesses, recovered=recovered,
        report_sha256=sha256_file(output/'report.json'), verifier_sha256=sha256_file(Path(__file__)),
        direct_recoveries_within_historical_neighborhood=sum(
            r['method']=='directed_paths' and not r['outside_historical_neighborhood'] for r in recovered)))
    write_json(output/'verification.json', value)
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='build/initlns-native-path-compatibility-v1')
    args = parser.parse_args()
    print(verify(contained(args.output)))
