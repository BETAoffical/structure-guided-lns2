from __future__ import annotations

import dataclasses
import unittest

from lns2_selector.evaluation.hierarchical_admission_h1 import (
    ALL_DISTINCT,
    ALL_SHARED,
    COMPONENT_ARM,
    FORBIDDEN_H1_FIELDS,
    HOTSPOT_ARM,
    STRUCTURAL_CONSENSUS,
    V2_ARM,
    V2_COMPONENT_ALIAS,
    V2_HOTSPOT_ALIAS,
    CandidateArm,
    H1Trial,
    build_hierarchical_h1_labels,
    canonicalize_hierarchy,
    h1_trial_from_mapping,
)


def _arm(candidate_id: str, agents: list[int]) -> CandidateArm:
    return CandidateArm(candidate_id=candidate_id, agents=tuple(agents))


def _plan(
    v2: list[int], component: list[int], hotspot: list[int]
):
    return canonicalize_hierarchy(
        frozen_v2=_arm("frozen-v2-winner", v2),
        component=_arm("component-16", component),
        hotspot=_arm("hotspot-16", hotspot),
    )


def _trials(
    reduction: float | list[float],
    *,
    rolled_back: set[int] | None = None,
) -> list[H1Trial]:
    reductions = (
        [float(reduction)] * 16
        if isinstance(reduction, (int, float))
        else list(map(float, reduction))
    )
    if len(reductions) != 16:
        raise AssertionError("test fixture requires 16 reductions")
    rolled_back = rolled_back or set()
    rows = []
    for trial_index, value in enumerate(reductions):
        before = 100
        after = before - int(round(value * before))
        rows.append(
            H1Trial(
                trial_index=trial_index,
                pp_seed=10_000 + trial_index,
                before_conflicts=before,
                after_conflicts=after,
                strict_decrease=after < before,
                rolled_back=trial_index in rolled_back,
                normalized_conflict_reduction=(before - after) / before,
            )
        )
    return rows


def _product(plan, scores: dict[str, float | list[float]]):
    return {
        execution_key: _trials(scores[execution_key])
        for execution_key in plan.label_execution_keys
    }


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(map(str, value)) | set().union(
            *(_all_keys(nested) for nested in value.values()), set()
        )
    if isinstance(value, list):
        return set().union(*(_all_keys(nested) for nested in value), set())
    return set()


class HierarchicalAdmissionH1Tests(unittest.TestCase):
    def test_five_exact_set_partitions_are_explicit(self) -> None:
        fixtures = (
            (_plan([1, 2], [2, 1], [1, 2]), ALL_SHARED, 1, 0),
            (_plan([1, 2], [2, 1], [3, 4]), V2_COMPONENT_ALIAS, 2, 2),
            (_plan([1, 2], [3, 4], [2, 1]), V2_HOTSPOT_ALIAS, 2, 2),
            (_plan([1, 2], [3, 4], [4, 3]), STRUCTURAL_CONSENSUS, 2, 2),
            (_plan([1, 2], [3, 4], [5, 6]), ALL_DISTINCT, 3, 3),
        )
        for plan, partition, execution_count, label_execution_count in fixtures:
            with self.subTest(partition=partition):
                self.assertEqual(plan.partition, partition)
                self.assertEqual(len(plan.executions), execution_count)
                self.assertEqual(
                    len(plan.label_execution_keys), label_execution_count
                )

    def test_exact_agent_aliases_are_sorted_and_execute_once(self) -> None:
        plan = _plan([3, 1, 2], [2, 3, 1], [4, 3, 2])
        self.assertEqual(plan.frozen_v2.agents, (1, 2, 3))
        self.assertEqual(plan.partition, V2_COMPONENT_ALIAS)
        self.assertEqual(
            plan.execution_key_for(V2_ARM),
            plan.execution_key_for(COMPONENT_ARM),
        )
        self.assertNotEqual(
            plan.execution_key_for(V2_ARM),
            plan.execution_key_for(HOTSPOT_ARM),
        )
        self.assertEqual(len(plan.label_execution_keys), 2)
        self.assertEqual(
            plan.executions[0].arms,
            (V2_ARM, COMPONENT_ARM),
        )

    def test_duplicate_agents_and_inconsistent_candidate_ids_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate agent"):
            _arm("bad", [1, 1, 2])
        aliases = canonicalize_hierarchy(
            frozen_v2=_arm("exact-set-id", [1, 2]),
            component=_arm("exact-set-id", [2, 1]),
            hotspot=_arm("hotspot", [5, 6]),
        )
        self.assertEqual(aliases.partition, V2_COMPONENT_ALIAS)
        self.assertEqual(
            aliases.frozen_v2.candidate_id,
            aliases.component.candidate_id,
        )
        self.assertEqual(
            aliases.arm_candidate_ids,
            (
                (V2_ARM, "exact-set-id"),
                (COMPONENT_ARM, "exact-set-id"),
                (HOTSPOT_ARM, "hotspot"),
            ),
        )
        with self.assertRaisesRegex(ValueError, "cannot name distinct agent sets"):
            canonicalize_hierarchy(
                frozen_v2=_arm("same", [1, 2]),
                component=_arm("same", [3, 4]),
                hotspot=_arm("hotspot", [5, 6]),
            )

    def test_all_shared_is_audit_only_with_zero_label_pairs(self) -> None:
        plan = _plan([1, 2], [2, 1], [1, 2])
        labels = build_hierarchical_h1_labels(plan, {})
        self.assertEqual(labels.stage1.label, "audit_only")
        self.assertEqual(labels.stage1.comparisons, ())
        self.assertFalse(labels.stage2.applicable)
        self.assertEqual(labels.stage2.label, "not_applicable")

    def test_equal_structural_sets_have_stage1_but_never_stage2(self) -> None:
        plan = _plan([1, 2], [3, 4], [4, 3])
        product = _product(
            plan,
            {
                plan.execution_key_for(V2_ARM): 0.10,
                plan.execution_key_for(COMPONENT_ARM): 0.20,
            },
        )
        labels = build_hierarchical_h1_labels(plan, product)
        self.assertEqual(labels.stage1.label, "admit_structural_consensus")
        self.assertEqual(len(labels.stage1.comparisons), 1)
        self.assertFalse(labels.stage2.applicable)
        self.assertIsNone(labels.stage2.comparison)

    def test_distinct_structural_sets_require_both_to_support_admission(self) -> None:
        plan = _plan([1, 2], [3, 4], [5, 6])
        keys = {
            arm: plan.execution_key_for(arm)
            for arm in (V2_ARM, COMPONENT_ARM, HOTSPOT_ARM)
        }
        both_support = _product(
            plan,
            {
                keys[V2_ARM]: 0.10,
                keys[COMPONENT_ARM]: 0.20,
                keys[HOTSPOT_ARM]: 0.25,
            },
        )
        labels = build_hierarchical_h1_labels(plan, both_support)
        self.assertEqual(labels.stage1.label, "admit_structural_consensus")
        self.assertTrue(labels.stage2.applicable)
        self.assertEqual(labels.stage2.label, "hotspot")

        disagreement = _product(
            plan,
            {
                keys[V2_ARM]: 0.10,
                keys[COMPONENT_ARM]: 0.20,
                keys[HOTSPOT_ARM]: 0.05,
            },
        )
        labels = build_hierarchical_h1_labels(plan, disagreement)
        self.assertEqual(labels.stage1.label, "abstain")
        self.assertEqual(labels.stage2.label, "component")

    def test_incomplete_structural_support_abstains_in_stage1(self) -> None:
        plan = _plan([1, 2], [3, 4], [5, 6])
        product = _product(
            plan,
            {
                plan.execution_key_for(V2_ARM): 0.10,
                plan.execution_key_for(COMPONENT_ARM): 0.20,
                plan.execution_key_for(HOTSPOT_ARM): 0.10,
            },
        )
        labels = build_hierarchical_h1_labels(plan, product)
        self.assertEqual(labels.stage1.label, "abstain")
        self.assertNotEqual(labels.stage1.label, "admit_structural_consensus")

    def test_v2_alias_votes_for_v2_and_cannot_admit_by_one_structural_win(self) -> None:
        plan = _plan([1, 2], [2, 1], [3, 4])
        product = _product(
            plan,
            {
                plan.execution_key_for(V2_ARM): 0.10,
                plan.execution_key_for(HOTSPOT_ARM): 0.20,
            },
        )
        labels = build_hierarchical_h1_labels(plan, product)
        self.assertEqual(len(product), 2)
        self.assertEqual(labels.frozen_v2_candidate_id, "frozen-v2-winner")
        self.assertEqual(labels.stage1.label, "abstain")
        self.assertEqual(
            labels.stage1.role_votes,
            ((COMPONENT_ARM, "v2_alias"), (HOTSPOT_ARM, "structural")),
        )
        self.assertEqual(labels.stage2.label, "hotspot")

        v2_consensus = _product(
            plan,
            {
                plan.execution_key_for(V2_ARM): 0.20,
                plan.execution_key_for(HOTSPOT_ARM): 0.10,
            },
        )
        labels = build_hierarchical_h1_labels(plan, v2_consensus)
        self.assertEqual(labels.stage1.label, "keep_v2")

        changed_structural_candidates = _plan([1, 2], [7, 8], [9, 10])
        self.assertEqual(
            changed_structural_candidates.frozen_v2.candidate_id,
            plan.frozen_v2.candidate_id,
        )
        self.assertEqual(changed_structural_candidates.frozen_v2.agents, (1, 2))

    def test_pairing_requires_same_seed_across_arms_and_distinct_seed_per_trial(self) -> None:
        plan = _plan([1, 2], [3, 4], [4, 3])
        v2_key = plan.execution_key_for(V2_ARM)
        structural_key = plan.execution_key_for(COMPONENT_ARM)
        product = {
            v2_key: _trials(0.10),
            structural_key: _trials(0.20),
        }
        mismatched = dict(product)
        changed = list(mismatched[structural_key])
        changed[0] = dataclasses.replace(changed[0], pp_seed=99_999)
        mismatched[structural_key] = changed
        with self.assertRaisesRegex(ValueError, "same H1 trial index"):
            build_hierarchical_h1_labels(plan, mismatched)

        duplicate_seed = dict(product)
        changed = list(duplicate_seed[structural_key])
        changed[1] = dataclasses.replace(changed[1], pp_seed=changed[0].pp_seed)
        duplicate_seed[structural_key] = changed
        with self.assertRaisesRegex(ValueError, "seeds must differ"):
            build_hierarchical_h1_labels(plan, duplicate_seed)

    def test_rollback_or_fixed_half_instability_blocks_admission(self) -> None:
        plan = _plan([1, 2], [3, 4], [4, 3])
        v2_key = plan.execution_key_for(V2_ARM)
        structural_key = plan.execution_key_for(COMPONENT_ARM)
        unstable = _trials(0.20)
        unstable[0] = H1Trial(
            trial_index=0,
            pp_seed=10_000,
            before_conflicts=100,
            after_conflicts=100,
            strict_decrease=False,
            rolled_back=True,
            normalized_conflict_reduction=0.0,
        )
        labels = build_hierarchical_h1_labels(
            plan,
            {v2_key: _trials(0.10), structural_key: unstable},
        )
        self.assertEqual(labels.stage1.label, "abstain")

        split_direction = [0.20] * 8 + [0.00] * 8
        labels = build_hierarchical_h1_labels(
            plan,
            {
                v2_key: _trials(0.10),
                structural_key: _trials(split_direction),
            },
        )
        self.assertEqual(labels.stage1.label, "abstain")

    def test_trial_contract_rejects_future_or_runtime_label_fields(self) -> None:
        row = {
            "trial_index": 0,
            "pp_seed": 10_000,
            "before_conflicts": 100,
            "after_conflicts": 90,
            "strict_decrease": True,
            "rolled_back": False,
            "normalized_conflict_reduction": 0.10,
            "time_to_feasible": 1.5,
        }
        with self.assertRaisesRegex(ValueError, "forbidden H1 fields"):
            h1_trial_from_mapping(row)

        row.pop("time_to_feasible")
        self.assertEqual(h1_trial_from_mapping(row).after_conflicts, 90)
        row["unregistered_metric"] = 1
        with self.assertRaisesRegex(ValueError, "H1 trial fields changed"):
            h1_trial_from_mapping(row)

    def test_label_output_contains_only_current_step_evidence(self) -> None:
        plan = _plan([1, 2], [3, 4], [5, 6])
        labels = build_hierarchical_h1_labels(
            plan,
            _product(
                plan,
                {
                    plan.execution_key_for(V2_ARM): 0.10,
                    plan.execution_key_for(COMPONENT_ARM): 0.20,
                    plan.execution_key_for(HOTSPOT_ARM): 0.25,
                },
            ),
        )
        output_keys = _all_keys(dataclasses.asdict(labels))
        self.assertFalse(output_keys & FORBIDDEN_H1_FIELDS)


if __name__ == "__main__":
    unittest.main()
