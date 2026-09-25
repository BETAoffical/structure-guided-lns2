"""Opt-in raw dynamic-profile fast path; historical runtime stays immutable."""
import math
from types import FunctionType, SimpleNamespace

from experiments.sa_history_selector import edges, members
from experiments import sa_raw_timed_runtime as reference
from scripts import run_sa_onpolicy as run
from scripts.run_sa_paired_closed_loop import profile_features


def dynamic_features_for(state, candidates, engine, history, temperature, fingerprint):
    # The frozen dynamic profile discards every history.* value. Keep its guards
    # and phase inputs without constructing per-candidate outside-path hashes.
    known = {a['id'] for a in state['agents']}
    run.require(edges(state) == set(history.ages), 'history/state discontinuity')
    for candidate in candidates:
        run.require(set(members(candidate)) <= known, 'unknown candidate agent')
    phase = {'sa.log_temperature': math.log1p(temperature),
             'sa.log_decision': math.log1p(history.decision)}
    engine.prepare(state)
    rows, _ = engine.realized_rows(candidates, state_hash=fingerprint)
    return [profile_features(dict(base=row['features']['realized_dynamic'] |
            {'source.score': candidate['score']} | phase), 'dynamic')
            for row, candidate in zip(rows, candidates, strict=True)]


def _bind(function, **replacements):
    """Reuse frozen code with private bindings, never patch process globals."""
    bound = FunctionType(function.__code__, function.__globals__ | replacements,
                         function.__name__, function.__defaults__, function.__closure__)
    bound.__kwdefaults__ = function.__kwdefaults__
    return bound


_selection = _bind(run.selection, features_for=dynamic_features_for)
_private_run = SimpleNamespace(**(vars(run) | {'selection': _selection}))


class FastPolicy(reference.Policy):
    def __init__(self, job, q, ctx):
        super().__init__(job, q, ctx)
        if self.actor:
            names = self.actor.bundle['base']['feature_names']
            run.require(len(names) == 129 and not any(n.startswith('history.') for n in names)
                        and {'sa.log_decision', 'sa.log_temperature'} <= set(names),
                        'fast path requires frozen 129-column dynamic profile')

    choose = _bind(reference.Policy.choose, run=_private_run)

