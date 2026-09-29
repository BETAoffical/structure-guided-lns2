"""Read only completed four-arm traces. No native import, training or solver use."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments._common import read_json, sha256_file, write_json, json_fingerprint
from experiments.sa_expanded_trace_diagnostic import analyze, first_divergence, require

SOURCE = ROOT/'build/sa-expanded-four-arm-ttf-v1'
OUTPUT = ROOT/'build/sa-expanded-tail-diagnostic-v1'
REPORT_SHA = '4130102b16d29460cd5de76e2373f1140161af8dd9b90539a6aa3f3dbde0b86a'
CODE = ('scripts/audit_sa_expanded_traces.py','experiments/sa_expanded_trace_diagnostic.py',
        'tests/evaluation/test_sa_expanded_trace_diagnostic.py','docs/SA_EXPANDED_TAIL_PROTOCOL_ZH.md')
HISTORY = ('docs/SA_TAIL_MEMBERSHIP_AUDIT_ZH.md','docs/SA_WORK_CLOCK_RESULT_ZH.md',
           'docs/SA_BUDGET_COOLING_RESULT_ZH.md','docs/SA_BOUNDED_REHEAT_RESULTS_ZH.md',
           'docs/SA_TERMINAL_EFFICIENCY_RESULT_ZH.md','docs/SA_COMPLETION_INDEPENDENT_TTF_RESULT_ZH.md')


def check_source_row(row):
    folder = SOURCE/'episodes'/row['job_id']
    require(read_json(folder/'result.json')==row, 'source result changed')
    for name, expected in row['files'].items():
        require(sha256_file(folder/name)==expected, 'changed trace/state '+name)
    proof=read_json(SOURCE/'audit'/(row['job_id']+'.json'))
    require(proof['result_sha256']==sha256_file(folder/'result.json'),'stale original audit')
    return folder


def worker(item):
    row, binding = item
    folder=check_source_row(row)
    initial, final = read_json(folder/'initial.json'),read_json(folder/'final.json')
    with gzip.open(folder/'trace.jsonl.gz','rt',encoding='utf-8') as stream:
        record = analyze(initial,final,(json.loads(line) for line in stream),row)
    record['binding'] = binding
    record['integrity'] = json_fingerprint(record)
    write_json(OUTPUT/'episodes'/(row['job_id']+'.json'),record)
    return record


def summarize(records):
    groups = defaultdict(dict)
    for row in records:
        groups[row['pair_id']][row['arm']]=row
    require(len(groups)==72 and all(set(g)=={'official','official_sa','parent','completion'} for g in groups.values()), 'missing pairing')
    pairs = []
    for key,g in sorted(groups.items()):
        require(len({r['initial_path_identity'] for r in g.values()})==1,'unpaired initial paths')
        pairs.append(dict(pair_id=key, success={a:r['success'] for a,r in g.items()},
            divergence=first_divergence(g['parent'],g['completion'])))
    summaries = {}
    for arm in ('official','official_sa','parent','completion'):
        for success in (False,True):
            rs = [r for r in records if r['arm']==arm and r['success']==success]
            summaries[f'{arm}/'+('success' if success else 'failure')] = dict(episodes=len(rs),
                decisions=sum(r['all']['decisions'] for r in rs),
                late_decisions=sum(r['last_30s']['decisions'] for r in rs),
                late_worsening_attempts=sum(r['last_30s']['worsening_attempts'] for r in rs),
                late_worsening_accepted=sum(r['last_30s']['worsening_accepted'] for r in rs),
                changed_path_revisits=sum(r['changed_path_revisits'] for r in rs),
                unchanged_paths=sum(r['all']['unchanged_paths'] for r in rs),
                final_goal_hold_pairs=sum(len(r['final_events']['goal_hold_pairs']) for r in rs),
                final_pairs=sum(r['final_events']['pairs'] for r in rs))
    return dict(schema='lns2.sa_expanded_tail_audit.v1', episodes=len(records), pairs=72,
        summary=summaries, actor_pairs=pairs,
        divergences=dict(Counter(p['divergence']['kind'] for p in pairs)),
        failures=[{k:v for k,v in r.items() if k!='decision_signatures'} for r in records if not r['success']],
        observational=True, no_solver_calls=True, no_training=True, no_controller_change=True,
        retrospective_windows_not_predictive_gates=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers',type=int,default=20)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    require(1<=args.workers<=20,'workers outside 1..20')
    require(sha256_file(SOURCE/'report.json')==REPORT_SHA,'changed official source report')
    source=read_json(SOURCE/'report.json')
    require(read_json(SOURCE/'run_status.json')['status']=='complete','unfinished source')
    identity={n:sha256_file(ROOT/n) for n in CODE+HISTORY}
    identity['source_report']=REPORT_SHA
    binding=json_fingerprint(identity)
    if OUTPUT.exists():
        require(args.resume and read_json(OUTPUT/'registration.json')['binding']==binding,'existing output or changed implementation')
    else:
        require(not args.resume,'missing resume output')
        OUTPUT.mkdir(parents=True)
        write_json(OUTPUT/'registration.json',dict(binding=binding,inputs=identity,observational=True))
    lock=OUTPUT/'analysis.lock'
    with lock.open('x',encoding='ascii') as f:
        f.write(binding)
    try:
        rows,pending=[],[]
        for row in source['episodes']:
            saved=OUTPUT/'episodes'/(row['job_id']+'.json')
            if saved.exists():
                check_source_row(row)
                r=read_json(saved)
                require(r['binding']==binding,'changed receipt')
                require(r['integrity']==json_fingerprint({k:v for k,v in r.items() if k!='integrity'}),'corrupt receipt')
                rows.append(r)
            else:
                pending.append((row,binding))
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(worker,j) for j in pending]
            for future in as_completed(futures):
                rows.append(future.result())
                write_json(OUTPUT/'progress.json',dict(completed=len(rows),total=288,status='running',binding=binding))
                if len(rows)%24==0:
                    print(f'Analyzed {len(rows)}/288',flush=True)
        report=summarize(sorted(rows,key=lambda x:x['job_id']))
        report['binding']=binding
        report['source_report_sha256']=REPORT_SHA
        report['receipts']={r['job_id']:sha256_file(OUTPUT/'episodes'/(r['job_id']+'.json')) for r in rows}
        write_json(OUTPUT/'report.json',report)
        write_json(OUTPUT/'progress.json',dict(completed=288,total=288,status='complete',binding=binding))
        print(json.dumps(dict(episodes=288,source_unchanged=sha256_file(SOURCE/'report.json')==REPORT_SHA,
            divergences=report['divergences'],report_sha256=sha256_file(OUTPUT/'report.json'))),flush=True)
    except BaseException as exc:
        write_json(OUTPUT/'error.json',dict(kind=type(exc).__name__,message=str(exc),binding=binding))
        write_json(OUTPUT/'progress.json',dict(completed=len(rows),total=288,status='failed',binding=binding))
        raise
    finally:
        lock.unlink()


if __name__=='__main__':
    main()
