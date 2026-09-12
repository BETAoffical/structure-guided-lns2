"""Isolated SA engineering variant; no global patches or default replacement."""

from types import FunctionType

from experiments.online_feature_engine import OnlineFeatureEngine, TopologyAnalysisCache
from experiments.repair_collection import state_fingerprint
from lns2_selector.runtime.online_selection import generate_online_candidates, score_online_candidates
from lns2_selector.runtime.structshell_dual16 import (
    generate_structshell_dual16_runtime_candidates, structshell_dual16_augmentation,
)
from scripts.run_sa_wall_clock import TimedPool


class SingleFullCheckPool(TimedPool):
    """Keep the final snapshot guard and the generator's native revision guard."""

    def select(self, env, state, decision):
        before = state_fingerprint(state)
        if self.engine is None:
            self.engine = OnlineFeatureEngine(state, backend="native", dense_output=True,
                required_features={"realized_dynamic": set(self.model.base_feature_names)})
        if self.topology is None:
            self.topology = TopologyAnalysisCache(state, static_grid=self.engine.static_grid, backend="native")
        else:
            self.topology.prepare(state)
        candidates, _ = generate_online_candidates(env, state, task_id=self.case["task_id"],
            solver_seed=self.case["solver_seed"], decision_index=decision,
            proposal_config=self.case["proposal"], state_hash=before,
            verify_full_state=False, proposal_backend="optimized")
        pool = list(generate_structshell_dual16_runtime_candidates(state, self.topology.analysis,
            v2_candidates=candidates, config=structshell_dual16_augmentation()).candidates)
        self.engine.prepare(state, prepared_native_analysis=self.topology.last_native_prepared)
        features, _ = self.engine.realized_rows(pool, state_hash=before)
        index, scores, _ = score_online_candidates(features, self.model)
        if state_fingerprint(env.get_state()) != before:
            raise ValueError("proposal mutated state")
        return int(index), [{**c, "score": float(s)} for c, s in zip(pool, scores)]


def isolated_worker(frozen_worker, selector, capture):
    # Reuse the exact frozen loop bytecode and timing boundaries, with private
    # dependency bindings. No shared module/global monkeypatch is installed.
    bindings = dict(frozen_worker.__globals__, TimedPool=selector, validate_result=capture)
    result = FunctionType(frozen_worker.__code__, bindings, frozen_worker.__name__,
                          frozen_worker.__defaults__, frozen_worker.__closure__)
    result.__kwdefaults__ = frozen_worker.__kwdefaults__
    return result
