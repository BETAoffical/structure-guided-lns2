"""Process-local opt-in adapter; the historical runtime and native stay frozen."""
from contextlib import contextmanager
from threading import Lock, current_thread, main_thread

from experiments import sa_raw_residual_actor as raw
from experiments.sa_paired_completion import require

_scope_lock = Lock()


class RuntimeActor(raw.RawResidualActor):
    def out_of_range_fraction(self, features):
        return self.base.out_of_range_fraction(features)


@contextmanager
def actor_scope(bundle, output, plan):
    """Bind an explicitly verified raw bundle only inside one isolated worker."""
    from scripts import run_sa_onpolicy as run
    sha = raw.validate_bundle(bundle)
    require(current_thread() is main_thread(), "raw adapter requires process isolation")
    require(_scope_lock.acquire(blocking=False), "nested/concurrent raw adapter")
    saved = run.NumpyActor, run.validate_bundle, run.actor_load

    def load(folder, requested_plan, iteration):
        require(folder == output and requested_plan == plan and iteration == bundle['iteration'],
                "raw loader identity mismatch")
        require(raw.validate_bundle(bundle) == sha, "raw bundle mutated")
        return bundle

    try:
        run.NumpyActor, run.validate_bundle, run.actor_load = RuntimeActor, raw.validate_bundle, load
        yield
    finally:
        run.NumpyActor, run.validate_bundle, run.actor_load = saved
        _scope_lock.release()
