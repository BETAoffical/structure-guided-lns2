"""Process-local cooling adapter for a diagnostic, never a default controller."""
from contextlib import contextmanager
import math
import threading

from experiments.nonmonotonic_repair import temperature as decision_temperature

CLOCKS = ("decision", "nodes")
NODE_BUDGET = 25000000
END_TEMPERATURE = -1.0 / math.log(.05)


def temperature(clock, decision, used, budget=NODE_BUDGET):
    if clock not in CLOCKS or type(decision) is not int or decision < 0 or used < 0 or budget <= 0:
        raise ValueError("invalid clock input")
    if clock == "decision":
        return decision_temperature(decision)
    fraction = min(1., used / budget)
    return 1000.0 * math.exp(math.log(END_TEMPERATURE / 1000.) * fraction)


class WorkClock:
    def __init__(self, clock, initial_nodes):
        self.clock, self.initial_nodes = clock, initial_nodes
        self.used, self.decision = 0, 0

    def __call__(self, decision):
        if decision != self.decision:
            raise ValueError("clock/decision discontinuity")
        return temperature(self.clock, decision, self.used)

    def observe(self, after):
        used = after["low_level"]["generated"] - self.initial_nodes
        if used < self.used:
            raise ValueError("nonmonotone search work")
        self.used = used
        self.decision += 1


class ClockedEnvironment:
    """Forward native calls unchanged; observe work only after PP returns."""
    def __init__(self, env, clock, plain):
        self.env, self.clock, self.plain = env, clock, plain

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step_experimental_pp(self, *args, **kwargs):
        result = self.env.step_experimental_pp(*args, **kwargs)
        self.clock.observe(self.plain(result)["observation"])
        return result


@contextmanager
def clock_scope(q, callback):
    # Selection, feature extraction and acceptance must use the same clock.
    # Every experiment job has its own process; no module or file is changed on disk.
    if threading.current_thread() is not threading.main_thread():
        raise ValueError("clock adapter requires isolated process main thread")
    original = q.temperature
    q.temperature = callback
    try:
        yield
    finally:
        q.temperature = original
