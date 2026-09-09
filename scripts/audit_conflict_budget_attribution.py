"""Audit saved cumulative conflict budgets; never invokes a MAPF solver."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.conflict_budget_attribution import OUTPUT, run
from experiments.local_path_compatibility import contained


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default=OUTPUT)
    parser.add_argument('--workers',type=int,default=20)
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    report = run(contained(args.output),args.workers,args.resume)
    print(json.dumps({k:v for k,v in report.items() if k!='condition_hashes'},indent=2))


if __name__=='__main__':
    main()
