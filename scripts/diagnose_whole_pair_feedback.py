"""Prepare, run and verify an isolated whole-pair development mechanism test."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.local_path_compatibility import contained
from experiments.whole_pair_collection import prepare, load, worker, collect, analyze


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare','dry-run','collect','verify','analyze','_job'))
    parser.add_argument('--config',default='configs/whole_pair_feedback_v1.json')
    parser.add_argument('--output',default='build/initlns-whole-pair-feedback-v1')
    parser.add_argument('--workers',type=int,default=20)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--max-jobs',type=int)
    parser.add_argument('--partial',action='store_true')
    parser.add_argument('--job-id')
    args=parser.parse_args()
    output=contained(args.output)
    if args.phase=='prepare': value=prepare(contained(args.config))
    elif args.phase=='dry-run':
        m=load(output)
        value=dict(jobs=len(m['jobs']),reused_controls=len(m['controls']),workers=args.workers,
                   per_job_seconds=m['config']['job_seconds'],per_process_fuse=m['config']['process_fuse_seconds'],
                   timing_allowed=False,solver_calls=0)
    elif args.phase=='_job': value=worker(output,args.job_id)
    elif args.phase=='collect': value=collect(output,args.workers,args.resume,args.max_jobs)
    else: value=analyze(output,args.partial)
    print(json.dumps(value,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
