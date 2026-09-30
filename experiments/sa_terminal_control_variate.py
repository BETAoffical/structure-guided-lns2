"""Whole-terminal-score control variates; read-only, never an optimizer."""
from itertools import combinations
import math

import numpy as np

from experiments.sa_completion_replicability import conditional_loss_gradient
from experiments.sa_paired_completion import require


def validate_rows(rows):
    require(len(rows) in (4, 16), 'fixed original/fresh replica count')
    for key in ('episode_id', 'rng_stream_id', 'replica'):
        require(len({r[key] for r in rows}) == len(rows), 'duplicate ' + key)
    for key in ('pair_id', 'initial_fingerprint', 'policy_sha256'):
        require(len({r[key] for r in rows}) == 1, 'changed condition ' + key)
    require(all(r['split'] == 'train' and r['status'] == 'ok' for r in rows),
            'heldout or unknown is not a terminal label')
    y = np.asarray([r['success'] for r in rows], dtype=float)
    score = np.asarray([r['score'] for r in rows], dtype=float)
    require(np.isin(y, [0., 1.]).all() and score.ndim == 2 and len(score) == len(y)
            and score.shape[1] > 0 and np.isfinite(score).all(), 'invalid terminal score')
    return y, score


def balanced_partitions(count):
    require(count >= 4 and count % 2 == 0, 'even replica count >=4 required')
    # Fix row zero in the first half so complementary partitions are not duplicated.
    masks = np.zeros((math.comb(count - 1, count // 2 - 1), count), dtype=bool)
    for i, others in enumerate(combinations(range(1, count), count // 2 - 1)):
        masks[i, (0,) + others] = True
    return masks


def estimate_baseline(returns, energy, method):
    y, e = np.asarray(returns, dtype=float), np.asarray(energy, dtype=float)
    require(y.ndim == e.ndim == 1 and len(y) == len(e) > 0 and np.isin(y, [0., 1.]).all()
            and np.isfinite(e).all() and (e >= 0).all(), 'invalid baseline fit')
    require(method in ('mean', 'score_squared'), 'unknown baseline')
    if method == 'mean' or e.sum() == 0:
        return float(y.mean())
    return float(y @ e / e.sum())


def crossfit_baselines(returns, energy, masks, method):
    y, e = np.asarray(returns, dtype=float), np.asarray(energy, dtype=float)
    masks = np.asarray(masks, dtype=bool)
    require(method in ('mean', 'score_squared'), 'unknown baseline')
    require(y.ndim == e.ndim == 1 and len(y) == len(e) and masks.ndim == 2
            and masks.shape[1] == len(y) and np.isfinite(e).all() and (e >= 0).all()
            and np.isin(y, [0., 1.]).all(), 'invalid crossfit inputs')
    require(((masks.sum(axis=1) == len(y) // 2) & (len(y) % 2 == 0)).all(), 'unequal halves')
    left, right = masks.astype(float), (~masks).astype(float)
    half = len(y) // 2
    mean_left, mean_right = left @ y / half, right @ y / half
    if method == 'mean':
        fit_left, fit_right = mean_left, mean_right
        zero_energy_fits = 0
    else:
        energy_left, energy_right = left @ e, right @ e
        fit_left = np.divide(left @ (y * e), energy_left, out=mean_left.copy(), where=energy_left > 0)
        fit_right = np.divide(right @ (y * e), energy_right, out=mean_right.copy(), where=energy_right > 0)
        zero_energy_fits = int((energy_left == 0).sum() + (energy_right == 0).sum())
    # Only opposite-half terminal outcomes and scores may determine an evaluated row's baseline.
    baseline = np.where(masks, fit_right[:, None], fit_left[:, None])
    return baseline, zero_energy_fits


def relative_change(new, reference):
    return float(new / reference - 1) if reference > 0 else None


def coefficient_metrics(advantages, gram, reference_coefficients):
    """Use the replica Gram matrix, not a large split-by-parameter tensor."""
    a = np.asarray(advantages, dtype=float)
    if a.ndim == 1:
        a = a[None, :]
    n = a.shape[1]
    coefficients = -a / n
    norm2 = np.maximum(np.einsum('bi,ij,bj->b', coefficients, gram, coefficients), 0.)
    second = (a * a) @ np.diag(gram) / n
    dispersion = np.maximum(second - norm2, 0.)
    ref_norm2 = max(float(reference_coefficients @ gram @ reference_coefficients), 0.)
    denominator = np.sqrt(norm2 * ref_norm2)
    usable = denominator > 0
    direction = coefficients[usable] @ gram @ reference_coefficients / denominator[usable]
    mean = coefficients.mean(axis=0)
    mean_norm2 = max(float(mean @ gram @ mean), 0.)
    quantiles = lambda x: np.quantile(x, [0., .1, .5, .9, 1.]).tolist()
    return dict(second_moment_mean=float(second.mean()), second_moment_quantiles=quantiles(second),
        within_replica_dispersion_mean=float(dispersion.mean()),
        gradient_norm_quantiles=quantiles(np.sqrt(norm2)),
        direction_to_loro_quantiles=quantiles(direction) if len(direction) else None,
        uninformative_directions=int((~usable).sum()),
        partition_gradient_dispersion=max(float(norm2.mean() - mean_norm2), 0.),
        mean_partition_gradient_norm=math.sqrt(mean_norm2),
        not_independent_samples=True, not_true_gradient_variance=True)


def condition_report(rows):
    y, score = validate_rows(rows)
    n = len(y)
    gram = score @ score.T
    energy = np.diag(gram)
    masks = balanced_partitions(n)
    loro = y - (y.sum() - y) / (n - 1)
    reference = -loro / n
    full_gradient, _ = conditional_loss_gradient(y, score)
    require(np.allclose(reference @ score, full_gradient, rtol=1e-11, atol=1e-12), 'LORO mismatch')
    results = {}
    for method in ('mean', 'score_squared'):
        baseline, fallback = crossfit_baselines(y, energy, masks, method)
        if method == 'mean':
            require(np.allclose(baseline.mean(axis=0), (y.sum() - y)/(n-1),
                                rtol=1e-12, atol=1e-12), 'partition average must reproduce LORO')
        results[method] = coefficient_metrics(y[None, :] - baseline, gram, reference)
        results[method].update(baseline_quantiles=np.quantile(baseline, [0., .1, .5, .9, 1.]).tolist(),
                               zero_energy_fits=fallback)
    b = estimate_baseline(y, energy, 'score_squared')
    oracle = coefficient_metrics(y - b, gram, reference)
    oracle.update(baseline=b, uses_evaluation_outcomes=True, not_deployable=True,
                  role='in_sample_best_single_constant_not_a_bound_for_other_baseline_classes')
    ratio = relative_change(results['score_squared']['second_moment_mean'],
                            results['mean']['second_moment_mean'])
    return dict(episodes=n, successes=int(y.sum()), failures=int(n-y.sum()),
        mixed=bool(0 < y.sum() < n), partitions=len(masks),
        parameter_count=score.shape[1], score_energy_quantiles=np.quantile(energy, [0., .1, .5, .9, 1.]).tolist(),
        loro=coefficient_metrics(loro, gram, reference), crossfit=results, oracle=oracle,
        weighted_vs_mean_second_moment_change=ratio,
        weighted_vs_mean_within_dispersion_change=relative_change(
            results['score_squared']['within_replica_dispersion_mean'],
            results['mean']['within_replica_dispersion_mean']),
        terminal_target_unchanged=True, no_length_weighting=True)


def summarize(conditions, minimum_reduction, maximum_condition_increase):
    require(len(conditions) == 4 and 0 < minimum_reduction < 1 and maximum_condition_increase >= 0,
            'fixed scope and descriptive thresholds')
    fresh = [c['fresh'] for c in conditions.values()]
    mixed = [c for c in fresh if c['mixed']]
    mean = float(np.mean([c['crossfit']['mean']['second_moment_mean'] for c in fresh]))
    weighted = float(np.mean([c['crossfit']['score_squared']['second_moment_mean'] for c in fresh]))
    change = relative_change(weighted, mean)
    signal = bool(mixed and change is not None and change <= -minimum_reduction and
        all(c['weighted_vs_mean_second_moment_change'] is not None and
            c['weighted_vs_mean_second_moment_change'] <= maximum_condition_increase for c in mixed))
    return dict(conditions=conditions, fresh_episodes=sum(c['episodes'] for c in fresh),
        fresh_successes=sum(c['successes'] for c in fresh), mixed_conditions=len(mixed),
        fresh_condition_equal_second_moment=dict(mean=mean, score_squared=weighted, relative_change=change),
        descriptive_signal=signal, thresholds=dict(minimum_reduction=minimum_reduction,
                                                  maximum_condition_increase=maximum_condition_increase),
        decision='descriptive_signal_needs_fresh_confirmation' if signal else 'no_crossfit_control_variate_gain_keep_A',
        no_training=True, no_ttf=True, no_promotion=True,
        split_enumeration_is_not_new_trajectory_data=True)
