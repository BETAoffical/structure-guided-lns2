"""Independent terminal-score confirmation with baselines fitted on a prior batch only."""
import math

import numpy as np

from experiments import sa_terminal_control_variate as prior
from experiments.sa_paired_completion import require


def freeze_baselines(rows):
    y, score = prior.validate_rows(rows)
    require(len(rows) == 16, 'sixteen prior trajectories required')
    energy = np.diag(score @ score.T)
    return dict(pair_id=rows[0]['pair_id'], initial_fingerprint=rows[0]['initial_fingerprint'],
        policy_sha256=rows[0]['policy_sha256'], parameter_count=score.shape[1],
        fit_episodes=16, fit_successes=int(y.sum()),
        fit_episode_ids=sorted(r['episode_id'] for r in rows),
        fit_rng_streams=sorted(r['rng_stream_id'] for r in rows),
        baselines={method: prior.estimate_baseline(y, energy, method)
                   for method in ('mean', 'score_squared')})


def validate_confirmation(rows, frozen, known_streams):
    y, score = prior.validate_rows(rows)
    require(len(rows) == 16 and sorted(r['replica'] for r in rows) == list(range(16)),
            'complete sixteen-replica confirmation required')
    require(frozen['fit_episodes'] == 16 and score.shape[1] == frozen['parameter_count'], 'fit/score dimensions')
    for key in ('pair_id', 'initial_fingerprint', 'policy_sha256'):
        require(rows[0][key] == frozen[key], 'frozen identity changed: ' + key)
    require(not set(frozen['fit_episode_ids']) & {r['episode_id'] for r in rows}, 'evaluation overlaps fit')
    require(not set(known_streams) & {r['rng_stream_id'] for r in rows}, 'reused historical stream')
    require(not set(frozen['fit_rng_streams']) & {r['rng_stream_id'] for r in rows}, 'reused fit stream')
    require(set(frozen['baselines']) == {'mean', 'score_squared'} and
            all(math.isfinite(b) and 0 <= b <= 1 for b in frozen['baselines'].values()), 'invalid frozen baseline')
    return y, score


def condition_report(rows, frozen, known_streams):
    y, score = validate_confirmation(rows, frozen, known_streams)
    energy = np.diag(score @ score.T)
    models = {}
    gradients = {}
    for method, b in frozen['baselines'].items():
        advantage = y - b
        moment = advantage**2 * energy
        gradient = -advantage @ score / len(y)
        second = float(moment.mean())
        models[method] = dict(baseline=b, second_moment=second,
            within_replica_dispersion=max(second - float(gradient @ gradient), 0.),
            mean_gradient_norm=float(np.linalg.norm(gradient)),
            largest_episode_moment_share=float(moment.max()/moment.sum()) if moment.sum() > 0 else None,
            second_moment_by_replica=moment.tolist(), fits_evaluation_data=False)
        gradients[method] = gradient
    denominator = float(np.linalg.norm(gradients['mean']) * np.linalg.norm(gradients['score_squared']))
    cosine = float(gradients['mean'] @ gradients['score_squared']/denominator) if denominator > 0 else None
    return dict(episodes=16, successes=int(y.sum()), failures=int(16-y.sum()),
        mixed=bool(0 < y.sum() < 16), fit_successes=frozen['fit_successes'],
        methods=models, relative_change=prior.relative_change(
            models['score_squared']['second_moment'], models['mean']['second_moment']),
        gradient_cosine=cosine, no_refit=True, not_true_gradient_variance=True,
        stop_counts={stop: sum(r['stop'] == stop for r in rows) for stop in sorted({r['stop'] for r in rows})},
        decision_quantiles=np.quantile([r['decisions'] for r in rows], [0., .5, 1.]).tolist(),
        generated_mean=float(np.mean([r['generated'] for r in rows])))


def condition_safe(condition, maximum_increase):
    a = condition['methods']['mean']['second_moment']
    b = condition['methods']['score_squared']['second_moment']
    # A zero reference cannot justify a positive weighted moment. Neither case is discarded.
    return b == 0 if a == 0 else b/a - 1 <= maximum_increase


def summarize(conditions, minimum_reduction=.1, maximum_condition_increase=.1, bootstrap=2000, seed=2026100102):
    require(len(conditions) == 4 and 0 < minimum_reduction < 1 and maximum_condition_increase >= 0
            and bootstrap > 0, 'fixed confirmation scope')
    values = list(conditions.values())
    mean = float(np.mean([c['methods']['mean']['second_moment'] for c in values]))
    weighted = float(np.mean([c['methods']['score_squared']['second_moment'] for c in values]))
    change = prior.relative_change(weighted, mean)
    guards = {pair: condition_safe(c, maximum_condition_increase) for pair, c in conditions.items()}
    passed = change is not None and change <= -minimum_reduction and all(guards.values())
    rng = np.random.default_rng(seed)
    samples = {m: np.zeros(bootstrap) for m in ('mean', 'score_squared')}
    for c in values:
        ix = rng.integers(0, 16, size=(bootstrap, 16))
        for method in samples:
            moment = np.asarray(c['methods'][method]['second_moment_by_replica'])
            samples[method] += moment[ix].mean(axis=1)/4
    valid = samples['mean'] > 0
    ratios = samples['score_squared'][valid]/samples['mean'][valid] - 1
    return dict(conditions=conditions, episodes=64, successes=sum(c['successes'] for c in values),
        failures=sum(c['failures'] for c in values), errors=0, censored=0,
        condition_equal_second_moment=dict(mean=mean, score_squared=weighted, relative_change=change),
        condition_safety=guards, confirmation_passed=bool(passed),
        thresholds=dict(minimum_reduction=minimum_reduction, maximum_condition_increase=maximum_condition_increase),
        conditional_replica_bootstrap=dict(replicates=bootstrap, seed=seed,
            relative_change_ci95=np.quantile(ratios, [.025, .975]).tolist() if len(ratios) else None,
            undefined_replicates=int((~valid).sum()), not_map_population_inference=True, not_a_gate=True),
        decision='fixed_baseline_signal_repeated_needs_separate_update_gate' if passed else
                 'fixed_baseline_signal_not_repeated_keep_A',
        no_training=True, no_ttf=True, no_promotion=True, no_refit=True,
        not_solver_performance=True, not_independent_map_generalization=True)


def compare_derived(actual, expected):
    """SHA binds bytes; tiny platform BLAS differences are allowed only for derived float checks."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), 'derived keys')
        for key, value in expected.items():
            compare_derived(actual[key], value)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), 'derived list')
        for a, b in zip(actual, expected):
            compare_derived(a, b)
    elif isinstance(expected, float):
        require(isinstance(actual, (int, float)) and not isinstance(actual, bool) and
                math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12), 'derived float')
    else:
        require(type(actual) is type(expected) and actual == expected, 'derived value')
