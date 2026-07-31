from __future__ import annotations

import unittest
from types import SimpleNamespace

from lns2_selector.controllers import CONTROLLER_IDS
from lns2_selector.controllers.official import OfficialAdaptiveSelector
from lns2_selector.controllers.v2 import PairwiseV2Selector
from lns2_selector.controllers.v3_s3 import V3S3Selector
from lns2_selector.runtime.contracts import SelectionRequest, Selector


class DirectModel:
    def score_candidates(self, rows):
        return [float(row["score"]) for row in rows]


class DirectV3State:
    def __init__(self, index):
        self.index = index

    def select(self, candidates, rows, **context):
        del candidates, rows, context
        selection_kind = (
            "v3_sequence" if self.index is not None else "v3_stalled_no_candidate"
        )
        return self.index, {"selection_kind": selection_kind}


def request(scores=(1.0, 2.0)) -> SelectionRequest:
    candidates = tuple(
        {"candidate_id": f"candidate-{index}", "agents": [index]}
        for index in range(len(scores))
    )
    rows = tuple(
        {"candidate_key": f"candidate-{index}", "score": score}
        for index, score in enumerate(scores)
    )
    return SelectionRequest(
        candidates=candidates,
        candidate_rows=rows,
        before_fingerprint="state",
        agent_count=10,
    )


class SelectorContractTests(unittest.TestCase):
    def test_active_controller_ids_are_canonical(self) -> None:
        self.assertEqual(
            CONTROLLER_IDS,
            ("official_adaptive", "v2-full", "mixed-full-v2", "v3-s3"),
        )

    def test_request_rejects_mismatched_candidate_rows(self) -> None:
        with self.assertRaisesRegex(ValueError, "differ in length"):
            SelectionRequest(
                candidates=({},),
                candidate_rows=(),
                before_fingerprint="state",
            )

    def test_official_selector_routes_to_native_policy(self) -> None:
        selector = OfficialAdaptiveSelector()
        self.assertIsInstance(selector, Selector)
        decision = selector.select(request())
        self.assertTrue(decision.uses_native_adaptive)
        self.assertEqual(decision.fallback_reason, "native_policy")

    def test_v2_contract_selects_best_candidate(self) -> None:
        for controller_id in ("v2-full", "mixed-full-v2"):
            with self.subTest(controller_id=controller_id):
                bundle = SimpleNamespace(
                    main_models={"realized_dynamic": DirectModel()}
                )
                decision = PairwiseV2Selector(controller_id, bundle).select(
                    request()
                )
                self.assertEqual(decision.controller_id, controller_id)
                self.assertEqual(decision.candidate_index, 1)
                self.assertEqual(
                    decision.candidate["candidate_id"], "candidate-1"
                )
                self.assertIsNone(decision.fallback_reason)

    def test_v2_contract_falls_back_only_when_pool_is_empty(self) -> None:
        for controller_id in ("v2-full", "mixed-full-v2"):
            with self.subTest(controller_id=controller_id):
                bundle = SimpleNamespace(
                    main_models={"realized_dynamic": DirectModel()}
                )
                empty = SelectionRequest(
                    candidates=(), candidate_rows=(), before_fingerprint="state"
                )
                decision = PairwiseV2Selector(controller_id, bundle).select(empty)
                self.assertTrue(decision.uses_native_adaptive)
                self.assertEqual(decision.fallback_reason, "no_candidates")

    def test_v3_s3_contract_selects_the_state_decision(self) -> None:
        selector = V3S3Selector.__new__(V3S3Selector)
        selector.state = DirectV3State(1)
        self.assertIsInstance(selector, Selector)

        decision = selector.select(request())

        self.assertEqual(decision.controller_id, "v3-s3")
        self.assertEqual(decision.candidate_index, 1)
        self.assertEqual(decision.candidate["candidate_id"], "candidate-1")
        self.assertIsNone(decision.fallback_reason)

    def test_v3_s3_contract_reports_no_eligible_sequence(self) -> None:
        selector = V3S3Selector.__new__(V3S3Selector)
        selector.state = DirectV3State(None)

        decision = selector.select(request())

        self.assertIsNone(decision.candidate_index)
        self.assertIsNone(decision.candidate)
        self.assertEqual(decision.fallback_reason, "v3_stalled_no_candidate")
