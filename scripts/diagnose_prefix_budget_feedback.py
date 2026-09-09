"""Run an isolated same-budget prefix feedback mechanism comparison."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.local_path_compatibility import ROOT, contained
from experiments.prefix_feedback_collection import CONFIG, analyze, collect, load, prepare, worker


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('prepare','dry-run','collect','analyze','_job'))
    parser.add_argument('--config',default=CONFIG)
    parser.add_argument('--output',default='build/initlns-prefix-budget-feedback-v1')
    parser.add_argument('--workers',type=int)
    parser.add_argument('--max-jobs',type=int)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--job-id')
    args=parser.parse_args()
    p=Path(args.output)
    output=p.resolve() if p.is_absolute() else contained(args.output)
    if output==ROOT/'build' or not output.is_relative_to(ROOT/'build'):
        raise ValueError('output must be inside build')
    if args.mode=='prepare':
        result=prepare(contained(args.config))
    elif args.mode=='dry-run':
        m=load(output)
        result=dict(jobs=len(m['jobs']),states=len(m['cases']),workers=m['config']['workers'],
                    max_search_seconds_per_job=m['config']['job_seconds'],timing_allowed=False)
    elif args.mode=='collect':
        collect(output,args.workers,args.resume,args.max_jobs)
        return
    elif args.mode=='_job':
        worker(output,args.job_id)
        return
    else:
        result=analyze(output)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
