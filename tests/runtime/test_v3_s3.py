from __future__ import annotations

import collections
import math
import unittest

from experiments.v3_s3 import (
    S3_ACTION_TEMPLATES,
    S3_TEMPORAL_FEATURE_NAMES,
    S3_WALL_TIME_FEATURE_NAMES,
    V3S3ControllerState,
    balanced_sequence_templates,
    candidate_template_indices,
    rank_s3_sequences,
    registered_runtime_sequences,
    s3_temporal_context,
    sequence_feature_row,
)


def _candidate(template, *, candidate_id="candidate"):
    return {
        "candidate_id": candidate_id,
        "candidate_key": candidate_id,
        "agents": [1, 2, 3, 4],
        "actual_size": template.requested_size,
        "selection_families": [template.family_key],
        "selection_rank_by_family": {
            template.family_key: template.representative
        },
    }


def _row():
    return {
        "candidate_id": "candidate",
        "candidate_key": "candidate",
        "features": {"realized_dynamic": {"state.agent_count": 100.0}},
    }


class _Bundle:
    feature_names = ("state.agent_count",)
    thresholds = {
        "minimum_template_valid_probability": 0.0,
        "maximum_no_progress_probability": 1.0,
        "maximum_sequence_no_progress_probability": 1.0,
    }
    prediction_intervals = {
        "conflict_reduction": 0.1,
        "total_seconds": 0.1,
    }
    continuation_calibration = {
        "schema": "lns2.v3_s3_continuation.v2",
        "coverage": 0.90,
        "minimum_cell_observations": 1,
        "cells": {},
        "fallback": {
            f"step{step}": {
                "observation_count": 1,
                "no_progress_threshold": 0.5,
                "no_progress_accuracy": 1.0,
                "reduction_relative_error": 0.1,
            }
            for step in range(1, 4)
        },
    }

    def __init__(self):
        self.calls = 0

    def predict(self, rows):
        self.calls += 1
        count = len(rows)
        result = {}
        for step in range(1, 4):
            result[f"step{step}_conflict_reduction"] = [1.0] * count
            result[f"step{step}_log_total_seconds"] = [math.log1p(1.0)] * count
            result[f"step{step}_total_seconds"] = [1.0] * count
            result[f"step{step}_no_progress_probability"] = [0.0] * count
            result[f"step{step}_template_valid_probability"] = [1.0] * count
        result["sequence_net_conflict_reduction"] = [3.0] * count
        result["sequence_log_total_seconds"] = [1.0] * count
        result["sequence_total_seconds"] = [3.0] * count
        result["sequence_no_progress_probability"] = [0.0] * count
        return result


class S3TemplateTests(unittest.TestCase):
    def test_current_temporal_schema_excludes_measured_wall_time(self) -> None:
        self.assertTrue(
            set(S3_WALL_TIME_FEATURE_NAMES).isdisjoint(S3_TEMPORAL_FEATURE_NAMES)
        )
        context = s3_temporal_context(
            [
                {
                    "conflict_reduction": 1.0,
                    "repair_seconds": 999.0,
                    "state_changed": True,
                    "no_progress": False,
                    "neighborhood_size": 8,
                }
            ],
            100,
        )
        self.assertTrue(set(S3_WALL_TIME_FEATURE_NAMES).isdisjoint(context))

    def test_registered_schedule_covers_every_first_action_twice(self) -> None:
        sequences = balanced_sequence_templates("state")
        self.assertEqual(len(sequences), 36)
        counts = collections.Counter(sequence[0].key for sequence in sequences)
        self.assertEqual(set(counts), {template.key for template in S3_ACTION_TEMPLATES})
        self.assertEqual(set(counts.values()), {2})

    def test_runtime_schedule_filters_unavailable_first_templates(self) -> None:
        available = tuple(template.key for template in S3_ACTION_TEMPLATES[:3])
        sequences = registered_runtime_sequences("state", available)
        self.assertEqual(len(sequences), 6)
        self.assertEqual(
            {sequence[0].key for sequence in sequences},
            set(available),
        )

    def test_candidate_template_uses_recorded_representative_rank(self) -> None:
        template = S3_ACTION_TEMPLATES[1]
        self.assertEqual(
            candidate_template_indices([_candidate(template)]),
            {template.key: 0},
        )

    def test_sequence_features_ignore_v2_metadata(self) -> None:
        row = _row()
        row["main_score"] = 999.0
        row["base_selected"] = True
        features = sequence_feature_row(
            row,
            {},
            tuple(S3_ACTION_TEMPLATES[:3]),
            agent_count=100,
            feature_names=("state.agent_count", "sequence.step1.requested_size"),
        )
        self.assertEqual(features["feature_values"][0], 100.0)
        self.assertNotIn("main_score", features["feature_names"])

    def test_sequence_features_reject_missing_required_base_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing required base features"):
            sequence_feature_row(
                _row(),
                {},
                tuple(S3_ACTION_TEMPLATES[:3]),
                agent_count=100,
                feature_names=(
                    "state.agent_count",
                    "proposal.actual_size",
                ),
            )


class S3SelectionTests(unittest.TestCase):
    def test_selection_never_uses_neighborhood_size_as_agent_count(self) -> None:
        state = V3S3ControllerState(_Bundle())
        template = S3_ACTION_TEMPLATES[0]
        row = _row()
        row["features"]["realized_dynamic"].clear()
        with self.assertRaisesRegex(ValueError, "explicit, consistent agent_count"):
            state.select(
                [_candidate(template)],
                [row],
                temporal_context={},
                before_fingerprint="before",
            )

    def test_explicit_agent_count_must_match_candidate_features(self) -> None:
        state = V3S3ControllerState(_Bundle())
        template = S3_ACTION_TEMPLATES[0]
        with self.assertRaisesRegex(ValueError, "explicit agent_count disagrees"):
            state.select(
                [_candidate(template)],
                [_row()],
                temporal_context={},
                before_fingerprint="before",
                agent_count=600,
            )

    def test_sequence_rank_uses_direct_sequence_utility(self) -> None:
        sequences = [
            tuple(S3_ACTION_TEMPLATES[:3]),
            tuple(S3_ACTION_TEMPLATES[3:6]),
        ]
        predictions = {}
        for step in range(1, 4):
            predictions[f"step{step}_conflict_reduction"] = [2.0, 3.0]
            predictions[f"step{step}_total_seconds"] = [1.0, 3.0]
            predictions[f"step{step}_no_progress_probability"] = [0.0, 0.0]
            predictions[f"step{step}_template_valid_probability"] = [1.0, 1.0]
        predictions["sequence_net_conflict_reduction"] = [6.0, 9.0]
        predictions["sequence_total_seconds"] = [3.0, 9.0]
        predictions["sequence_no_progress_probability"] = [0.0, 0.0]
        order = rank_s3_sequences(
            sequences,
            predictions,
            {
                "minimum_template_valid_probability": 0.5,
                "maximum_no_progress_probability": 0.5,
                "maximum_sequence_no_progress_probability": 0.5,
            },
        )
        self.assertEqual(order[0], 0)

    def test_risk_relaxation_is_explicit(self) -> None:
        sequences = [tuple(S3_ACTION_TEMPLATES[:3])]
        predictions = {
            "sequence_net_conflict_reduction": [3.0],
            "sequence_total_seconds": [1.0],
            "sequence_no_progress_probability": [0.9],
        }
        for step in range(1, 4):
            predictions[f"step{step}_conflict_reduction"] = [1.0]
            predictions[f"step{step}_total_seconds"] = [1.0]
            predictions[f"step{step}_no_progress_probability"] = [0.9]
            predictions[f"step{step}_template_valid_probability"] = [0.1]
        thresholds = {
            "minimum_template_valid_probability": 0.5,
            "maximum_no_progress_probability": 0.5,
            "maximum_sequence_no_progress_probability": 0.5,
        }
        self.assertEqual(rank_s3_sequences(sequences, predictions, thresholds), [])
        self.assertEqual(
            rank_s3_sequences(
                sequences,
                predictions,
                thresholds,
                allow_risk_relaxation=True,
            ),
            [0],
        )

    def test_expected_first_step_executes_continuation_without_new_prediction(self) -> None:
        bundle = _Bundle()
        state = V3S3ControllerState(bundle)
        first_template = S3_ACTION_TEMPLATES[0]
        selected, diagnostic = state.select(
            [_candidate(first_template)],
            [_row()],
            temporal_context={},
            before_fingerprint="before",
        )
        self.assertEqual(selected, 0)
        self.assertEqual(diagnostic["selection_kind"], "new-plan")
        self.assertEqual(bundle.calls, 1)
        expected = state.observe(
            before_fingerprint="before",
            after_fingerprint="after",
            repair_outcome="conflict_reduced",
            conflict_reduction=1.0,
            # Measured wall time is deliberately irrelevant to a v2
            # continuation decision.
            total_seconds=1_000_000.0,
            feasible=False,
        )
        self.assertTrue(expected)
        continuation = state.continuation_template
        self.assertIsNotNone(continuation)
        selected, diagnostic = state.select(
            [_candidate(continuation, candidate_id="continuation")],
            [_row()],
            temporal_context={},
            before_fingerprint="after",
        )
        self.assertEqual(selected, 0)
        self.assertEqual(diagnostic["selection_kind"], "direct-continuation")
        self.assertFalse(diagnostic["full_pool_scored"])
        self.assertEqual(bundle.calls, 1)

    def test_no_candidate_stalls_without_external_fallback(self) -> None:
        state = V3S3ControllerState(_Bundle())
        selected, diagnostic = state.select(
            [], [], temporal_context={}, before_fingerprint="state"
        )
        self.assertIsNone(selected)
        self.assertEqual(diagnostic["selection_kind"], "v3_stalled_no_candidate")
        self.assertEqual(diagnostic["v2_call_count"], 0)
        self.assertEqual(diagnostic["adaptive_call_count"], 0)

    def test_cache_hits_are_counted_by_the_shared_runtime_interface(self) -> None:
        state = V3S3ControllerState(_Bundle())
        state.note_cache_hit()
        self.assertEqual(state.summary()["cache_hit_count"], 1)

    def test_unexpected_result_discards_remaining_plan_and_replans(self) -> None:
        bundle = _Bundle()
        state = V3S3ControllerState(bundle)
        template = S3_ACTION_TEMPLATES[0]
        state.select(
            [_candidate(template)],
            [_row()],
            temporal_context={},
            before_fingerprint="before",
        )
        expected = state.observe(
            before_fingerprint="before",
            after_fingerprint="after",
            repair_outcome="conflict_reduced",
            conflict_reduction=9.0,
            total_seconds=1.0,
            feasible=False,
        )
        self.assertFalse(expected)
        self.assertIsNone(state.continuation_template)
        state.select(
            [_candidate(template, candidate_id="new-state")],
            [_row()],
            temporal_context={},
            before_fingerprint="after",
        )
        self.assertEqual(bundle.calls, 2)

    def test_no_progress_blacklists_agents_without_adaptive_fallback(self) -> None:
        state = V3S3ControllerState(_Bundle())
        template = S3_ACTION_TEMPLATES[0]
        state.select(
            [_candidate(template)],
            [_row()],
            temporal_context={},
            before_fingerprint="same",
        )
        state.observe(
            before_fingerprint="same",
            after_fingerprint="same",
            repair_outcome="accepted_noop",
            conflict_reduction=0.0,
            total_seconds=1.0,
            feasible=False,
        )
        selected, diagnostic = state.select(
            [_candidate(template)],
            [_row()],
            temporal_context={},
            before_fingerprint="same",
        )
        self.assertIsNone(selected)
        self.assertEqual(diagnostic["selection_kind"], "v3_stalled_no_candidate")
        self.assertEqual(state.summary()["adaptive_call_count"], 0)


if __name__ == "__main__":
    unittest.main()
