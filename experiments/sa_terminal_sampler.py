"""Explicit, non-trainable sampling distribution for a Train-only diagnostic."""
from contextlib import contextmanager
import math
import threading

from experiments.sa_onpolicy_actor import NumpyActor, validate_bundle
from experiments._common import json_fingerprint
from experiments.sa_paired_completion import require

SCHEMA = "lns2.sa.terminal_sampler.v1"


def make_sampler(parent):
    return dict(schema=SCHEMA, binding=parent["binding"], iteration=parent["iteration"],
                parent_policy=validate_bundle(parent), base_actor=parent,
                uniform_mix=.5, trainable=False)


def validate_sampler(bundle):
    require(bundle["schema"] == SCHEMA and bundle["uniform_mix"] == .5 and
            bundle["trainable"] is False, "fixed sampling contract")
    parent = bundle["base_actor"]
    require(validate_bundle(parent) == bundle["parent_policy"] and
            parent["binding"] == bundle["binding"] and parent["iteration"] == bundle["iteration"],
            "sampling parent identity")
    return json_fingerprint(bundle)


def mixture(probabilities, weight=.5):
    require(weight == .5 and probabilities and
            all(math.isfinite(p) and p > 0 for p in probabilities.values()) and
            math.isclose(math.fsum(probabilities.values()), 1., abs_tol=1e-12, rel_tol=0),
            "sampling probability support")
    return {cid: (1.-weight)*p + weight/len(probabilities) for cid, p in probabilities.items()}


class TerminalSampler:
    def __init__(self, bundle):
        self.sha = validate_sampler(bundle)
        self.actor = NumpyActor(bundle["base_actor"])

    def probabilities(self, ids, anchor, features):
        return mixture(self.actor.probabilities(ids, anchor, features))

    def out_of_range_fraction(self, features):
        return self.actor.out_of_range_fraction(features)


@contextmanager
def sampler_runtime():
    # The historical runtime is SHA-pinned. Scope its two policy hooks to one
    # isolated worker, restoring them even on failure; never patch the solver.
    from scripts import run_sa_onpolicy as run
    require(threading.current_thread() is threading.main_thread(), "sampler requires process isolation")
    require(run.NumpyActor is NumpyActor and run.validate_bundle is validate_bundle, "nested sampler adapter")
    run.NumpyActor, run.validate_bundle = TerminalSampler, validate_sampler
    try:
        yield
    finally:
        run.NumpyActor, run.validate_bundle = NumpyActor, validate_bundle


def coverage(parent, sampled):
    """Count new terminal information without rewarding damage to all-success groups."""
    require(parent.keys() == sampled.keys(), "condition coverage mismatch")
    require(all(type(v) is int and 0 <= v <= 4 for counts in (parent, sampled) for v in counts.values()),
            "four complete replicas required")
    counts = dict(parent_mixed=sum(0 < v < 4 for v in parent.values()),
                  sampled_mixed=sum(0 < v < 4 for v in sampled.values()),
                  parent_success=sum(parent.values()), sampled_success=sum(sampled.values()))
    counts.update(
        recovered_all_failure=[g for g in sorted(parent) if parent[g] == 0 and sampled[g] > 0],
        new_mixed_from_all_failure=[g for g in sorted(parent) if parent[g] == 0 and 0 < sampled[g] < 4],
        damaged_all_success=[g for g in sorted(parent) if parent[g] == 4 and sampled[g] < 4])
    recovered = bool(counts["recovered_all_failure"])
    if recovered and counts["sampled_success"] >= counts["parent_success"]:
        decision = "terminal_coverage_signal_needs_independent_sampling_confirmation"
    elif recovered:
        decision = "coverage_efficiency_tradeoff_no_training_authorization"
    else:
        decision = "no_new_hard_terminal_support_stop_this_sampler"
    return dict(**counts, decision=decision, no_training=True, automatic_promotion=False)
