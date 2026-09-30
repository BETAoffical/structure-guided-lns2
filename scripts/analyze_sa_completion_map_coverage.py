"""Read-only posthoc descriptors of the completed node-budget comparison."""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import argparse
import csv
import io
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_sa_completion_map_coverage as cli
from experiments._common import atomic_write_text
from experiments.closed_loop_trace_storage import apply_state_delta
from experiments.sa_expanded_trace_diagnostic import (
    PathIdentity, final_events, first_divergence, window_summary,
)


def trace_worker(job):
    row = cli.old.previous.read_result(job)
    folder = cli.old.previous.folder(job)
    state = cli.run.read_json(folder / 'initial.json')
    final = cli.run.read_json(folder / 'final.json')
    identity = PathIdentity()
    path_hash = identity.update(state)
    first_hash = path_hash
    seen = {path_hash}
    records, signatures = [], []
    revisits = 0
    best = state['num_of_colliding_pairs']
    for e in cli.run.trace_read(folder):
        m = e['metrics']
        cli.require(e['decision'] == len(records), 'trace order')
        after = apply_state_delta(state, e['delta'])
        cli.require(m['conflicts_before'] == state['num_of_colliding_pairs'] and
                    m['conflicts_after'] == after['num_of_colliding_pairs'], 'conflict counts')
        members = tuple(sorted(m['neighborhood']))
        old_paths = identity.paths
        new_hash = identity.update(after)
        cli.require(all(old_paths[a] == identity.paths[a] for a in old_paths.keys() - set(members)),
                    'outsider path changed')
        if new_hash != path_hash and new_hash in seen:
            revisits += 1
        seen.add(new_hash)
        edges = tuple(sorted(tuple(sorted(x)) for x in state['conflict_edges']))
        active = {a for edge in edges for a in edge}
        probs = e['probabilities']
        anchor = e['anchor_id']
        records.append(dict(
            members=members, edges_before=edges, path_before=path_hash, path_after=new_hash,
            before=m['conflicts_before'], after=m['conflicts_after'],
            accepted=bool(m['replan_success']), reason=m['pp_failure_reason'],
            attempt_delta=(m['pp_attempt_conflict_pair_count'] - m['pp_old_conflict_pair_count'])
                if m['acceptance_evaluated'] else None,
            temperature=e['temperature'], pp_seconds=m['native_replan_seconds'],
            generated=after['low_level']['generated'] - state['low_level']['generated'],
            covers_all=active <= set(members), anchor_selected=e['selected_id'] == anchor,
            anchor_probability=probs[anchor], max_probability=max(probs.values()),
            out_of_range=e.get('out_of_range_fraction'),
        ))
        signatures.append(dict(
            decision=e['decision'], path_before=path_hash, path_after=new_hash,
            members=list(members), pp_seed=e['action'].get('random_seed'),
            draw=e['selection_draw'],
            pool=cli.run.json_fingerprint(sorted((c['candidate_id'], sorted(c['agents'])) for c in e['pool'])),
        ))
        best = min(best, m['conflicts_after'])
        state, path_hash = after, new_hash
    cli.require(len(records) == row['decisions'] and
                PathIdentity().update(final) == path_hash and
                state['num_of_colliding_pairs'] == row['final_conflicts'], 'final reconstruction')
    return dict(job_id=job['job_id'], pair_id=job['pair_id'], replica=job['replica'],
        map_id=row['map_id'], arm=job['comparison_arm'], success=row['success'],
        final_conflicts=row['final_conflicts'], best_conflicts=best,
        changed_path_revisits=revisits, initial_path_identity=first_hash,
        all=window_summary(records), last_32_decisions=window_summary(records[-32:]),
        final_events=final_events(final), decision_signatures=signatures)


def verify_existing(reg, out):
    body = cli.run.check_seal(cli.run.read_json(out / 'diagnostic.json'))
    cli.require(body['binding'] == reg['binding'] and body['source_report_sha256'] ==
                cli.run.sha256_file(out / 'report.json'), 'analysis identity')
    cli.require(body['csv_sha256'] == cli.run.sha256_file(out / 'comparison.csv') and
                body['diagnostic_source_sha256'] == cli.run.sha256_file(Path(__file__)),
                'changed CSV or analysis source')
    cli.require(len(list(cli.audited_rows(reg, out, 'comparison'))) == 144, 'analysis coverage')
    return dict(verified=True, paired_positions=body['paired_positions'],
                reviewed_episodes=body['reviewed_episodes'], no_solver_calls=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--verify-existing', action='store_true')
    args = parser.parse_args()
    reg, out = cli.verify()
    if args.verify_existing or (args.resume and (out / 'diagnostic.json').exists()):
        print(json.dumps(verify_existing(reg, out)))
        return
    cli.require(not (out / 'diagnostic.json').exists(), 'existing analysis; verify or resume')
    report = cli.run.check_seal(cli.run.read_json(out / 'report.json'))
    cli.require(report['binding'] == reg['binding'] and report['updated'], 'complete updated experiment')
    pairs = list(cli.audited_rows(reg, out, 'comparison'))
    train = list(cli.audited_rows(reg, out, 'train'))
    credit = cli.run.check_seal(cli.run.read_json(out / 'credit.json'))
    weights = {r['episode_id']: r for r in credit['coefficients']}
    coverage = defaultdict(lambda: dict(episodes=0, successes=0, credited_episodes=0, decisions=0))
    for job, row in train:
        key = job['case']['task_variant']
        coverage[key]['episodes'] += 1
        coverage[key]['successes'] += row['success']
        coverage[key]['decisions'] += row['decisions']
        coverage[key]['credited_episodes'] += weights[row['episode_id']]['coefficient'] != 0

    qual = {r['pair_id']: r for r in cli.run.check_seal(cli.run.read_json(out / 'qualification.json'))['rows']}
    fields = ['model', 'map_id', 'task_variant', 'solver_seed', 'replica', 'initial_conflicts',
              'success', 'stop', 'final_conflicts', 'generated_nodes', 'repair_decisions',
              'soc_steps', 'makespan_steps', 'wait_steps', 'result_sha256']
    csv_rows = []
    groups = defaultdict(dict)
    for job, row in pairs:
        groups[(row['pair_id'], row['replica'])][job['comparison_arm']] = (job, row)
        csv_rows.append(dict(model={'parent': 'A', 'reference': 'A2', 'completion': 'A-wide'}[job['comparison_arm']],
            map_id=row['map_id'], task_variant=job['case']['task_variant'], solver_seed=job['solver_seed'],
            replica=row['replica'], initial_conflicts=qual[row['pair_id']]['initial_conflicts'],
            success=row['success'], stop=row['stop'], final_conflicts=row['final_conflicts'],
            generated_nodes=row['generated'], repair_decisions=row['decisions'],
            soc_steps=row['soc'] if row['success'] else '',
            makespan_steps=row['makespan'] if row['success'] else '',
            wait_steps=row['wait_steps'] if row['success'] else '',
            result_sha256=cli.run.sha256_file(cli.old.previous.folder(job) / 'result.json')))
    csv_path = out / 'comparison.csv'
    text = io.StringIO(newline='')
    writer = csv.DictWriter(text, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows(csv_rows)
    if csv_path.exists():
        cli.require(args.resume and csv_path.read_bytes().decode('utf-8') == text.getvalue(),
                    'existing CSV requires matching explicit resume')
    else:
        atomic_write_text(csv_path, text.getvalue())

    selected = [g for _, g in sorted(groups.items()) if any(not r['success'] for _, r in g.values())]
    jobs = [j for group in selected for j, _ in group.values()]
    with ProcessPoolExecutor(max_workers=20) as pool:
        records = list(pool.map(trace_worker, jobs))
    traces = defaultdict(dict)
    for r in records:
        traces[(r['pair_id'], r['replica'])][r['arm']] = r
    differences = [dict(pair_id=key[0], replica=key[1],
        A_vs_Awide=first_divergence(group['parent'], group['completion']),
        A2_vs_Awide=first_divergence(group['reference'], group['completion']))
        for key, group in sorted(traces.items())]
    parent = reg['parent']
    wide = cli.run.read_json(out / 'models/completion.json')
    reference = reg['reference']
    def values(model):
        return [x for key in ('w1', 'b1', 'w2', 'b2') for x in cli.old.flat(model['correction'][key])]
    a, b, c = (values(m) for m in (parent, wide, reference))
    dw, dr = [y-x for x,y in zip(a,b)], [y-x for x,y in zip(a,c)]
    nw, nr = (math.sqrt(math.fsum(x*x for x in d)) for d in (dw, dr))
    cosine = math.fsum(x*y for x,y in zip(dw,dr)) / (nw*nr) if nw*nr else None
    body = dict(binding=reg['binding'], source_report_sha256=cli.run.sha256_file(out / 'report.json'),
        retrospective=True, no_solver_calls=True, no_training=True, no_ttf=True,
        diagnostic_source_sha256=cli.run.sha256_file(Path(__file__)),
        not_a_causal_intervention=True, not_all_episode_frequency_estimate=True,
        train_by_density=dict(coverage), paired_positions=len(groups), reviewed_pairs=len(selected),
        reviewed_episodes=len(records), csv_sha256=cli.run.sha256_file(csv_path),
        parameter_l2_Awide_from_A=nw, parameter_l2_A2_from_A=nr,
        update_direction_cosine=cosine, first_divergences=differences,
        records=[{k:v for k,v in r.items() if k != 'decision_signatures'} for r in records])
    cli.run.once(out / 'diagnostic.json', cli.run.sealed(body))
    print(json.dumps({k:v for k,v in body.items() if k not in ('records', 'first_divergences')}))


if __name__ == '__main__':
    main()
