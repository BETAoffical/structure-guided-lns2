"""Frozen-A prior intervention; retain score inputs and both learned residuals."""
from collections import Counter
import math
import random
import numpy as np

from experiments import sa_lean_timing as lean
from experiments import sa_raw_timed_runtime as reference
from experiments import sa_raw_timing_metrics as metrics
from experiments import sa_terminal_efficiency_timing as comparison
from experiments.sa_completion_acceptance import MODEL_SHA
from experiments.sa_raw_selection_fast import _bind
from experiments.sa_shared_features import SharedFeaturePolicy
from scripts import run_sa_onpolicy as run

ARMS = ('original', 'uniform_prior', 'random_prior')
LABELS = dict(original='Frozen A: original Dual prior',
              uniform_prior='Frozen A: uniform prior, Dual scores retained',
              random_prior='Frozen A: random anchor prior, Dual scores retained')
TIMING_MODE = lean.TIMING_MODE
PURPOSE = 'prior-anchor-v1'


def identity(job):
    run.require(job['comparison_arm'] in ARMS and job['prior_mode'] == job['comparison_arm'], 'prior identity')
    run.require(job['arm'] == 'trained_actor' and job['model']['sha256'] == MODEL_SHA, 'frozen A identity')
    run.require(job.get('max_decisions') is None and job.get('execution_node_budget') is None, 'unexpected work cap')


def prior(ids, anchor):
    run.require(ids and ids == sorted(set(ids)) and (anchor is None or anchor in ids), 'prior candidates')
    return {c: 1. / len(ids) if anchor is None else .1 / len(ids) + (.9 if c == anchor else 0.) for c in ids}


def replace_prior(probabilities, anchor, replacement):
    """p_A/q_old*q_new preserves both frozen neural residuals algebraically."""
    ids = sorted(probabilities)
    old, new = prior(ids, anchor), prior(ids, replacement)
    run.require(all(math.isfinite(p) and p > 0 for p in probabilities.values()) and
                math.isclose(math.fsum(probabilities.values()), 1., abs_tol=1e-12), 'invalid probabilities')
    if replacement == anchor:
        return dict(probabilities)
    logits = {c: math.log(probabilities[c]) - math.log(old[c]) + math.log(new[c]) for c in ids}
    peak = max(logits.values())
    mass = {c: math.exp(logits[c] - peak) for c in ids}
    total = math.fsum(mass.values())
    result = {c: mass[c] / total for c in ids}
    run.require(all(math.isfinite(p) and p > 0 for p in result.values()), 'prior underflow')
    return result


def random_anchor(ids, draw):
    run.require(ids == sorted(set(ids)) and ids and math.isfinite(draw) and 0 <= draw < 1, 'prior draw')
    return ids[min(int(draw * len(ids)), len(ids)-1)]


def direct_probabilities(actor, ids, features, replacement):
    """Independent audit formula: new prior + bounded base residual + raw residual."""
    base = actor.base
    x = np.asarray([[f[n] for n in actor.bundle['base']['feature_names']] for f in features], dtype=np.float64)
    z = (x-base.mean)/base.scale
    inner = (2.*np.tanh(np.tanh(z @ base.w1.T+base.b1) @ base.w2.T+base.b2))[:, 0]
    outer = (np.tanh(z @ actor.w1.T+actor.b1) @ actor.w2.T+actor.b2)[:, 0]
    q = prior(ids, replacement)
    logits = np.asarray([math.log(q[c]) for c in ids])+inner+outer
    mass = np.exp(logits-np.max(logits))
    values = mass/np.sum(mass)
    return dict(zip(ids, values.tolist()))


class PriorActor:
    def __init__(self, actor, job):
        self.actor, self.job = actor, job
        self.sha, self.bundle = actor.sha, actor.bundle
        run.require('source.score' in self.bundle['base']['feature_names'], 'missing retained Dual score')
        self.decision, self.last = 0, None

    def out_of_range_fraction(self, features):
        return self.actor.out_of_range_fraction(features)

    def probabilities(self, ids, anchor, features):
        run.require(ids == sorted(set(ids)), 'canonical candidate ordering required')
        j, d = self.job, self.decision
        draw = run.stream_draw(j['plan'], j['phase'], j['pair_id'], j['replica'], d, PURPOSE)
        sampled = random_anchor(ids, draw)
        mode = j['prior_mode']
        run.require(mode in ARMS, 'unknown prior mode')
        replacement = anchor if mode == 'original' else (None if mode == 'uniform_prior' else sampled)
        original = self.actor.probabilities(ids, anchor, features)
        result = replace_prior(original, anchor, replacement)
        self.last = dict(mode=mode, decision=d, true_anchor_id=anchor, prior_anchor_id=replacement,
                         random_anchor_id=sampled, prior_draw=draw, true_anchor_probability=result[anchor],
                         original_true_anchor_probability=original[anchor])
        self.decision += 1
        return result


class PriorPolicy(SharedFeaturePolicy):
    def __init__(self, job, q, ctx):
        identity(job)
        super().__init__(job, q, ctx)
        self.actor = PriorActor(self.actor, job)

    def choose(self, env, state, d):
        run.require(self.actor.decision == d, 'prior clock drift')
        event = super().choose(env, state, d)
        event['prior_intervention'] = dict(self.actor.last)
        return event


class AuditPolicy(reference.Policy):
    def __init__(self, job, q, ctx):
        identity(job)
        super().__init__(job, q, ctx)
        self.actor = PriorActor(self.actor, job)

    def observe(self, before, event, after):
        run.require(event['prior_intervention'] == self.actor.last, 'intervention metadata drift')
        alternate = direct_probabilities(self.actor.actor, event['candidate_ids'], event['features'],
                                         self.actor.last['prior_anchor_id'])
        run.require(max(abs(alternate[c]-event['probabilities'][c]) for c in alternate) < 1e-12,
                    'independent prior algebra mismatch')
        run.require(event['behavior_log_probability'] == math.log(event['probabilities'][event['selected_id']]),
                    'behavior log probability mismatch')
        super().observe(before, event, after)


_timed = _bind(lean.lean_worker, Policy=PriorPolicy)
_audit = _bind(reference.audit_worker, Policy=AuditPolicy)


def timed_worker(job):
    identity(job)
    return _timed(job)


def preflight_worker(job):
    from scripts.train_sa_history_selector import die_with_parent
    die_with_parent(job['parent_pid'])
    identity(job)
    folder = run.ROOT/job['source_folder']
    for name, digest in job['source_files'].items():
        run.require(run.sha256_file(folder/name) == digest, 'source artifact changed')
    q, env, ctx = reference.prepare_environment(job)
    before = q._plain(env.reset(seed=job['solver_seed']))
    fp = q.state_fingerprint(before)
    run.require(before['initial_solution_complete'] and fp == job['expected_initial'], 'preflight initial')
    policy = PriorPolicy(job, q, ctx)
    policy.start(before)
    reader = run.trace_read(folder)
    original = next(reader, None)
    reader.close()
    if before['feasible']:
        run.require(original is None, 'unexpected source repairs')
        return dict(status='ok', job_id=job['job_id'], native_repairs=0)
    run.require(original is not None, 'missing historical first action')
    event = policy.choose(env, before, 0)
    run.require(q.state_fingerprint(q._plain(env.get_state())) == fp, 'proposal mutated state')
    for key in ('pool','proposal_order','anchor_id','selection_draw','features','candidate_ids'):
        run.require(event[key] == original[key], 'non-prior input changed: '+key)
    saved_p = policy.actor.actor.probabilities(event['candidate_ids'], event['anchor_id'], event['features'])
    run.require(saved_p == original['probabilities'], 'unmodified A does not reproduce source')
    if job['prior_mode'] == 'original':
        for key in ('action','selected_id','probabilities','behavior_log_probability'):
            run.require(event[key] == original[key], 'original A changed: '+key)
    direct = direct_probabilities(policy.actor.actor, event['candidate_ids'], event['features'],
                                  event['prior_intervention']['prior_anchor_id'])
    error = max(abs(direct[c]-event['probabilities'][c]) for c in direct)
    run.require(error < 1e-12, 'prior intervention algebra')
    after, m, temp, uniform = reference.transition(job, q, env, before, event, 0, 30.)
    run.require(m['pp_failure_reason'] != 'time_limit', 'preflight PP safety interruption')
    q.validate_transition(before, after, m, event['action']['agents'], 'annealed', temp, uniform)
    run.require(temp == original['temperature'] and uniform == original['uniform'] and
                event['action']['random_seed'] == original['action']['random_seed'], 'repair/SA RNG changed')
    if event['selected_id'] == original['selected_id']:
        saved = q.apply_state_delta(before, original['delta'])
        run.require(q.state_fingerprint(after) == q.state_fingerprint(saved), 'same action different path')
        run.require(after['low_level'] == saved['low_level'], 'same action different search work')
    return dict(status='ok', job_id=job['job_id'], native_repairs=1, initial_fingerprint=fp,
                input_and_score_unchanged=True, independent_algebra_error=error,
                prior_intervention=event['prior_intervention'], selected_id=event['selected_id'])


def audit_worker(job):
    identity(job)
    folder = run.ROOT/job['output']/'episodes'/job['job_id']
    row = run.check_seal(run.read_json(folder/'result.json'))
    run.require(row['timing_mode'] == TIMING_MODE and row['trace_seconds'] == 0 and
                row['post_delivery_trace_seconds'] >= 0, 'audit charged to TTF')
    proof = _audit(job)
    count, hits, probabilities, same_random = 0, 0, 0., 0
    for event in run.trace_read(folder):
        count += 1
        hits += event['selected_id'] == event['anchor_id']
        probabilities += event['probabilities'][event['anchor_id']]
        same_random += event['prior_intervention']['random_anchor_id'] == event['anchor_id']
    return proof | dict(prior_mode=job['prior_mode'], true_anchor_hits=hits,
                        true_anchor_probability_sum=probabilities, random_matches_true=same_random,
                        source_score_retained=True, independent_prior_algebra_verified=True)


_validate = _bind(metrics._validate, ARMS=ARMS)


def summarize(rows, bootstrap=5000, seed=2026100401):
    rows = list(rows)
    run.require(bootstrap > 0 and all(r.get('timing_mode') == TIMING_MODE for r in rows), 'summary scope')
    run.require(len({r['policy_sha256'] for r in rows}) == 1, 'mixed weights')
    groups = _validate(rows)
    maps = sorted({g[ARMS[0]]['map_id'] for g in groups})
    rng = random.Random(seed)
    draws = [Counter(rng.choices(maps, k=len(maps))) for _ in range(bootstrap)]

    def summary(gs, ds):
        contrasts = {}
        for target, baseline in (('original','uniform_prior'), ('original','random_prior'),
                                 ('uniform_prior','random_prior')):
            value = comparison.contrast([dict(parent=g[baseline], completion=g[target]) for g in gs], ds)
            value.update(target=target, baseline=baseline,
                         statistics_keys=dict(parent=baseline, completion=target))
            contrasts[target+'_vs_'+baseline] = value
        return dict(pairs=len(gs), arms={a:metrics._arm([g[a] for g in gs]) for a in ARMS}, contrasts=contrasts)

    return summary(groups, draws) | dict(
        by_map={m:summary([g for g in groups if g[ARMS[0]]['map_id'] == m], []) for m in maps},
        bootstrap=bootstrap, bootstrap_unit='map', runtime_labels=LABELS, timing_mode=TIMING_MODE,
        frozen_model=True, source_score_retained=True, prior_only=True, no_training=True,
        automatic_promotion=False, failure_time_imputation=False, reused_maps_development_only=True,
        scientific_audit_outside_ttf=True, independent_network_training_not_tested=True,
        dual_scoring_cost_still_included=True)
