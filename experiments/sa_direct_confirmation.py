"""Five frozen methods and a pure schedule; execution requires separate authority."""
from collections import Counter,defaultdict
from copy import deepcopy

from scripts import run_sa_onpolicy as run
from experiments import sa_direct_selector as direct
from experiments.sa_shared_features import SharedFeaturePolicy

ARMS=('direct','a','official','official_sa','dual16_sa')
LABELS={'direct':'Frozen direct selector, round-2 + SA',
        'a':'Frozen completion-only A + SA',
        'official':'MAPF-LNS2 Official Adaptive, standard PP',
        'official_sa':'MAPF-LNS2 Official Adaptive + SA',
        'dual16_sa':'Frozen Dual16 + SA'}


def order(index):
    # An odd-size Williams design: five cyclic rows and their reversals.
    base=(0,1,4,2,3)
    shift=index%5
    row=tuple(ARMS[(x+shift)%5] for x in base)
    return row if index%10<5 else row[::-1]


def make_schedule(cases,cfg,models,plan,binding):
    run.require(tuple(cfg['comparison_arms'])==ARMS,'fixed five-method comparison')
    jobs=[]
    for case in sorted(cases,key=lambda c:c['task_id']):
        for seed in cfg['solver_seeds']:
            pair=f"{case['task_id']}-s{seed}"
            for arm in order(len(jobs)//5):
                model=models.get(arm)
                job=dict(case=deepcopy(case),solver_seed=seed,pair_id=pair,replica=0,
                    comparison_arm=arm,arm='trained_actor' if model else arm,
                    model=deepcopy(model),iteration=2 if arm=='a' else 0,
                    phase=cfg['phase'],output=cfg['output'],plan=deepcopy(plan),
                    timing_binding=binding,budget_seconds=cfg['budget_seconds'],
                    timing_mode=cfg['timing_mode'],max_decisions=None,execution_node_budget=None,
                    split='independent_confirmation',expected_initial=None,
                    execution_authorized=False,requires_initial_qualification=True,
                    schedule_index=len(jobs),within_pair_position=len(jobs)%5,
                    job_id=run.json_fingerprint(['direct-five-arm-v1',pair,arm])[:24])
                jobs.append(job)
    check_schedule(jobs,cfg)
    return jobs


def check_schedule(jobs,cfg):
    expected=cfg['maps']*len(cfg['densities'])*len(cfg['solver_seeds'])
    run.require(len(jobs)==expected*5 and len({j['job_id'] for j in jobs})==len(jobs),'job coverage')
    groups=defaultdict(list)
    for j in jobs:
        run.require(j['execution_authorized'] is False and j['expected_initial'] is None,
                    'preparation must not authorize execution')
        run.require(j['budget_seconds']==120. and j['max_decisions'] is None and
                    j['execution_node_budget'] is None,'hidden limit or changed clock')
        groups[j['pair_id']].append(j)
    run.require(len(groups)==expected,'condition coverage')
    for group in groups.values():
        run.require(len(group)==5 and {j['comparison_arm'] for j in group}==set(ARMS),'missing method')
        for field in ('case','solver_seed','replica','phase','plan','timing_binding'):
            run.require(all(j[field]==group[0][field] for j in group),'unpaired '+field)
    positions={a:Counter(j['within_pair_position'] for j in jobs if j['comparison_arm']==a) for a in ARMS}
    run.require(all(max(c.values())-min(c.values())<=1 and len(c)==5 for c in positions.values()),
                'unbalanced method positions')
    return dict(conditions=len(groups),episodes=len(jobs),positions={a:dict(c) for a,c in positions.items()})


def timed_worker(job):
    # Preparation cannot reach the native runtime, even if this function is called by mistake.
    run.require(job.get('execution_authorized') is True,'timing not authorized; preparation only')
    run.require(isinstance(job.get('expected_initial'),str) and len(job['expected_initial'])==64,
                'new conditions must pass initial qualification first')
    arm=job['comparison_arm']
    run.require(arm in ARMS,'unknown confirmation method')
    if arm=='direct':return direct._bind(direct.lean.lean_worker,Policy=direct.DirectPolicy)(job)
    if arm=='a':return direct._bind(direct.lean.lean_worker,Policy=SharedFeaturePolicy)(job)
    return direct.lean.WORKERS[arm](job)
