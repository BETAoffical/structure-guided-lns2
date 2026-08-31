from __future__ import annotations

import unittest
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

from lns2_selector.controllers import (
    CONTROLLER_IDS,
    load_selector,
)
from lns2_selector.controllers.official import OfficialAdaptiveSelector
from lns2_selector.controllers.v2 import PairwiseV2Selector
from lns2_selector.controllers.v3_s3 import V3S3Selector
from lns2_selector.cli import main as selector_cli
from lns2_selector.runtime.contracts import (
    SelectionObservation,
    SelectionRequest,
    Selector,
    StatefulSelector,
)


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


class ObservableV3State:
    pending_candidate_id = "candidate-1"
    pending_agents = (1,)
    pending_before_fingerprint = "before"

    def observe(self, **values):
        self.values = values
        return True


def request(
    scores=(1.0, 2.0), *, profile: str = "realized_dynamic"
) -> SelectionRequest:
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
        profile=profile,
    )


class SelectorContractTests(unittest.TestCase):
    def test_active_controller_ids_are_canonical(self) -> None:
        self.assertEqual(
            CONTROLLER_IDS,
            (
                "official_adaptive",
                "v2-full",
                "mixed-full-v2",
                "v3-s3",
            ),
        )
    def test_historical_controller_aliases_are_not_executable(self) -> None:
        for alias in (
            "Adaptive",
            "proposal_dynamic",
            "realized_dynamic",
            "v2-stall-safe",
            "v3-full",
            "v3-h3",
        ):
            with self.subTest(alias=alias), self.assertRaisesRegex(
                ValueError, "unsupported controller"
            ):
                load_selector(alias)

    def test_rejected_diagnostic_controller_ids_are_not_executable(self) -> None:
        for controller_id in (
            "stride-control-v1",
            "stride-quality-v1",
            "stride-augcontrol-v1",
            "stride-guardrank-v1",
            "stride-maprank-v1",
        ):
            with self.subTest(controller_id=controller_id), self.assertRaisesRegex(
                ValueError, "unsupported controller"
            ):
                load_selector(controller_id)

    def test_pairwise_loader_rejects_a_noncanonical_bundle_identity(self) -> None:
        loaded = SimpleNamespace(
            manifest={"controller_id": "mixed-full-v2"},
            main_models={"realized_dynamic": DirectModel()},
        )
        with patch(
            "lns2_selector.controllers.load_controller_bundle",
            return_value=loaded,
        ), self.assertRaisesRegex(ValueError, "matching controller bundle"):
            load_selector("v2-full", "bundle")

    def test_cli_lists_only_canonical_controller_ids(self) -> None:
        output = StringIO()
        with redirect_stdout(output):
            status = selector_cli(["list-controllers"])
        self.assertEqual(status, 0)
        self.assertEqual(tuple(output.getvalue().splitlines()), CONTROLLER_IDS)

    def test_request_rejects_mismatched_candidate_rows(self) -> None:
        with self.assertRaisesRegex(ValueError, "differ in length"):
            SelectionRequest(
                candidates=({},),
                candidate_rows=(),
                before_fingerprint="state",
            )

    def test_request_rejects_swapped_candidate_feature_rows(self) -> None:
        with self.assertRaisesRegex(ValueError, "identities differ at index 0"):
            SelectionRequest(
                candidates=(
                    {"candidate_id": "candidate-a"},
                    {"candidate_id": "candidate-b"},
                ),
                candidate_rows=(
                    {"candidate_key": "candidate-b"},
                    {"candidate_key": "candidate-a"},
                ),
                before_fingerprint="state",
            )

    def test_restricted_request_requires_matching_generation_context(self) -> None:
        with self.assertRaisesRegex(ValueError, "generation template"):
            SelectionRequest(
                candidates=(),
                candidate_rows=(),
                before_fingerprint="state",
                candidate_pool_mode="restricted",
            )
        with self.assertRaisesRegex(ValueError, "differs"):
            SelectionRequest(
                candidates=(),
                candidate_rows=(),
                before_fingerprint="state",
                candidate_pool_mode="full",
                generation_context={"mode": "restricted"},
            )

    def test_request_rejects_a_missing_identity_on_either_side(self) -> None:
        cases = (
            (({"agents": [0]},), ({"candidate_id": "candidate-a"},), "candidate"),
            (({"candidate_id": "candidate-a"},), ({"score": 1.0},), "feature row"),
        )
        for candidates, rows, kind in cases:
            with self.subTest(kind=kind), self.assertRaisesRegex(
                ValueError, rf"{kind} identity is missing at index 0"
            ):
                SelectionRequest(
                    candidates=candidates,
                    candidate_rows=rows,
                    before_fingerprint="state",
                )

    def test_request_rejects_conflicting_identity_aliases(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "feature row identity fields differ at index 0"
        ):
            SelectionRequest(
                candidates=({"candidate_id": "candidate-a"},),
                candidate_rows=(
                    {
                        "candidate_id": "candidate-a",
                        "candidate_key": "candidate-b",
                    },
                ),
                before_fingerprint="state",
            )

    def test_request_and_observation_reject_nonstring_identities(self) -> None:
        for identity in (None, 7):
            with self.subTest(identity=identity), self.assertRaisesRegex(
                ValueError, "identity is missing"
            ):
                SelectionRequest(
                    candidates=({"candidate_id": identity},),
                    candidate_rows=({"candidate_id": identity},),
                    before_fingerprint="state",
                )
        with self.assertRaisesRegex(ValueError, "before_fingerprint"):
            SelectionRequest(
                candidates=(),
                candidate_rows=(),
                before_fingerprint=None,  # type: ignore[arg-type]
            )
        for field in ("candidate_id", "before_fingerprint", "after_fingerprint"):
            values = {
                "candidate_id": "candidate-1",
                "actual_agents": (1,),
                "before_fingerprint": "before",
                "after_fingerprint": "after",
                "replan_success": True,
                "conflicts_before": 2,
                "conflicts_after": 1,
                "feasible": False,
                "terminal": False,
                "total_seconds": 0.5,
            }
            values[field] = None
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, field
            ):
                SelectionObservation(**values)  # type: ignore[arg-type]

    def test_official_selector_routes_to_native_policy(self) -> None:
        selector = OfficialAdaptiveSelector()
        self.assertIsInstance(selector, Selector)
        decision = selector.select(request())
        self.assertTrue(decision.uses_native_adaptive)
        self.assertEqual(decision.fallback_reason, "native_policy")

    def test_v2_contract_selects_best_candidate(self) -> None:
        for controller_id in (
            "v2-full",
            "mixed-full-v2",
        ):
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

    def test_v2_contract_rejects_an_empty_pool(self) -> None:
        for controller_id in (
            "v2-full",
            "mixed-full-v2",
        ):
            with self.subTest(controller_id=controller_id):
                bundle = SimpleNamespace(
                    main_models={"realized_dynamic": DirectModel()}
                )
                empty = SelectionRequest(
                    candidates=(), candidate_rows=(), before_fingerprint="state"
                )
                with self.assertRaisesRegex(ValueError, "empty candidate pool"):
                    PairwiseV2Selector(controller_id, bundle).select(empty)

    def test_v2_contract_selects_the_requested_profile(self) -> None:
        selector = PairwiseV2Selector(
            "v2-full",
            {
                "proposal_dynamic": DirectModel(),
                "realized_dynamic": DirectModel(),
            },
        )
        decision = selector.select(request(profile="proposal_dynamic"))
        self.assertEqual(decision.candidate_index, 1)
        self.assertEqual(decision.diagnostics["profile"], "proposal_dynamic")

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
        self.assertFalse(decision.uses_native_adaptive)

    def test_v3_s3_observation_is_bound_to_selected_identity(self) -> None:
        selector = V3S3Selector.__new__(V3S3Selector)
        selector.state = ObservableV3State()
        self.assertIsInstance(selector, StatefulSelector)
        observation = SelectionObservation(
            candidate_id="candidate-1",
            actual_agents=(1,),
            before_fingerprint="before",
            after_fingerprint="after",
            replan_success=True,
            conflicts_before=2,
            conflicts_after=1,
            feasible=False,
            terminal=False,
            total_seconds=0.5,
        )

        diagnostic = selector.observe(observation)

        self.assertEqual(diagnostic["repair_outcome"], "conflict_reduced")
        self.assertTrue(diagnostic["continuation_expected"])
        self.assertEqual(selector.state.values["candidate_id"], "candidate-1")
        self.assertEqual(selector.state.values["actual_agents"], (1,))
        self.assertFalse(selector.state.values["terminal"])

    def test_v3_s3_observation_rejects_candidate_agent_and_state_mismatch(self) -> None:
        selector = V3S3Selector.__new__(V3S3Selector)
        selector.state = ObservableV3State()
        base = {
            "candidate_id": "candidate-1",
            "actual_agents": (1,),
            "before_fingerprint": "before",
            "after_fingerprint": "after",
            "replan_success": True,
            "conflicts_before": 2,
            "conflicts_after": 1,
            "feasible": False,
            "terminal": False,
            "total_seconds": 0.5,
        }
        for replacement, message in (
            ({"candidate_id": "other"}, "candidate ID"),
            ({"actual_agents": (2,)}, "agents"),
            ({"before_fingerprint": "other"}, "fingerprint"),
        ):
            with self.subTest(replacement=replacement), self.assertRaisesRegex(
                ValueError, message
            ):
                selector.observe(SelectionObservation(**{**base, **replacement}))
