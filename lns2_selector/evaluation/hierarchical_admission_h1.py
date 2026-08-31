"""Pure current-step labels for the Dual16 hierarchical-admission pilot.

This module deliberately does not generate candidates, score V2 candidates, run
native repairs, or register a controller.  Its input is one already-frozen V2
winner plus the Component16 and Hotspot16 candidates from the same state.  It
canonicalizes their exact agent sets, specifies the minimum deduplicated repair
product, and builds two conservative H1 labels from paired current-step repair
outcomes.

The first label only decides whether *all independent* structural arms support
admission over the frozen V2 action.  The second label compares Component with
Hotspot, and exists only when their exact agent sets differ.  No time-to-feasible,
future-trajectory, or remaining-round signal is accepted by this data contract.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


V2_ARM = "v2"
COMPONENT_ARM = "component"
HOTSPOT_ARM = "hotspot"
ARM_ORDER = (V2_ARM, COMPONENT_ARM, HOTSPOT_ARM)

ALL_SHARED = "all_shared"
V2_COMPONENT_ALIAS = "v2_component_alias"
V2_HOTSPOT_ALIAS = "v2_hotspot_alias"
STRUCTURAL_CONSENSUS = "structural_consensus"
ALL_DISTINCT = "all_distinct"
PARTITIONS = (
    ALL_SHARED,
    V2_COMPONENT_ALIAS,
    V2_HOTSPOT_ALIAS,
    STRUCTURAL_CONSENSUS,
    ALL_DISTINCT,
)

H1_TRIAL_INDICES = tuple(range(16))
FIRST_FIXED_HALF = tuple(range(8))
SECOND_FIXED_HALF = tuple(range(8, 16))

FORBIDDEN_H1_FIELDS = frozenset(
    {
        "cost_to_go",
        "future_repair_rounds",
        "future_trajectory",
        "native_step_seconds",
        "pp_replan_seconds",
        "receding_q",
        "remaining_repair_rounds",
        "repair_runtime",
        "time_to_feasible",
        "ttf",
    }
)

_TRIAL_FIELDS = frozenset(
    {
        "trial_index",
        "pp_seed",
        "before_conflicts",
        "after_conflicts",
        "strict_decrease",
        "rolled_back",
        "normalized_conflict_reduction",
    }
)


def _require_plain_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _canonical_agents(agents: Sequence[int]) -> tuple[int, ...]:
    canonical = tuple(
        _require_plain_int(agent, name="candidate agent") for agent in agents
    )
    if not canonical:
        raise ValueError("candidate agent set must not be empty")
    if any(agent < 0 for agent in canonical):
        raise ValueError("candidate agent ids must be nonnegative")
    if len(set(canonical)) != len(canonical):
        raise ValueError("candidate contains duplicate agent ids")
    return tuple(sorted(canonical))


def _execution_key(agents: tuple[int, ...]) -> str:
    return "agents:" + ",".join(map(str, agents))


@dataclass(frozen=True)
class CandidateArm:
    """One named generator's candidate, before cross-arm deduplication."""

    candidate_id: str
    agents: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise ValueError("candidate_id must be a nonempty string")
        object.__setattr__(self, "agents", _canonical_agents(self.agents))


@dataclass(frozen=True)
class CandidateExecution:
    """One exact agent set that needs at most one paired repair product."""

    execution_key: str
    agents: tuple[int, ...]
    arms: tuple[str, ...]
    candidate_ids: tuple[str, ...]


@dataclass(frozen=True)
class CanonicalHierarchy:
    """Canonical V/C/H relation with the V2 winner already frozen."""

    frozen_v2: CandidateArm
    component: CandidateArm
    hotspot: CandidateArm
    partition: str
    executions: tuple[CandidateExecution, ...]
    arm_execution_keys: tuple[tuple[str, str], ...]

    def execution_key_for(self, arm: str) -> str:
        try:
            return dict(self.arm_execution_keys)[arm]
        except KeyError as exc:
            raise ValueError(f"unknown hierarchy arm: {arm}") from exc

    def candidate_id_for(self, arm: str) -> str:
        candidates = {
            V2_ARM: self.frozen_v2,
            COMPONENT_ARM: self.component,
            HOTSPOT_ARM: self.hotspot,
        }
        try:
            return candidates[arm].candidate_id
        except KeyError as exc:
            raise ValueError(f"unknown hierarchy arm: {arm}") from exc

    @property
    def arm_candidate_ids(self) -> tuple[tuple[str, str], ...]:
        return tuple((arm, self.candidate_id_for(arm)) for arm in ARM_ORDER)

    @property
    def stage2_applicable(self) -> bool:
        return self.component.agents != self.hotspot.agents

    @property
    def stage1_structural_execution_keys(self) -> tuple[str, ...]:
        """Independent structural sets that must all support Stage 1 admission."""

        v2_key = self.execution_key_for(V2_ARM)
        structural_keys: list[str] = []
        for arm in (COMPONENT_ARM, HOTSPOT_ARM):
            key = self.execution_key_for(arm)
            if key != v2_key and key not in structural_keys:
                structural_keys.append(key)
        return tuple(structural_keys)

    @property
    def label_execution_keys(self) -> tuple[str, ...]:
        """Exact deduplicated repair product needed to build all applicable labels."""

        structural = self.stage1_structural_execution_keys
        if not structural:
            return ()
        needed = {self.execution_key_for(V2_ARM), *structural}
        if self.stage2_applicable:
            needed.update(
                {
                    self.execution_key_for(COMPONENT_ARM),
                    self.execution_key_for(HOTSPOT_ARM),
                }
            )
        return tuple(
            execution.execution_key
            for execution in self.executions
            if execution.execution_key in needed
        )


def canonicalize_hierarchy(
    *,
    frozen_v2: CandidateArm,
    component: CandidateArm,
    hotspot: CandidateArm,
) -> CanonicalHierarchy:
    """Freeze V and canonicalize exact V/C/H agent sets without ranking them.

    A deterministic candidate id may be shared by exact-set aliases.  One id
    may never name two different agent sets.  Role-to-id provenance is retained
    even when aliases share one execution key.
    """

    candidates = {
        V2_ARM: CandidateArm(frozen_v2.candidate_id, frozen_v2.agents),
        COMPONENT_ARM: CandidateArm(component.candidate_id, component.agents),
        HOTSPOT_ARM: CandidateArm(hotspot.candidate_id, hotspot.agents),
    }
    agents_by_candidate_id: dict[str, tuple[int, ...]] = {}
    for arm in ARM_ORDER:
        candidate = candidates[arm]
        previous = agents_by_candidate_id.setdefault(
            candidate.candidate_id, candidate.agents
        )
        if previous != candidate.agents:
            raise ValueError("one candidate id cannot name distinct agent sets")

    v2_agents = candidates[V2_ARM].agents
    component_agents = candidates[COMPONENT_ARM].agents
    hotspot_agents = candidates[HOTSPOT_ARM].agents
    if v2_agents == component_agents == hotspot_agents:
        partition = ALL_SHARED
    elif v2_agents == component_agents:
        partition = V2_COMPONENT_ALIAS
    elif v2_agents == hotspot_agents:
        partition = V2_HOTSPOT_ALIAS
    elif component_agents == hotspot_agents:
        partition = STRUCTURAL_CONSENSUS
    else:
        partition = ALL_DISTINCT

    grouped: dict[tuple[int, ...], list[str]] = {}
    for arm in ARM_ORDER:
        grouped.setdefault(candidates[arm].agents, []).append(arm)
    executions = tuple(
        CandidateExecution(
            execution_key=_execution_key(agents),
            agents=agents,
            arms=tuple(arms),
            candidate_ids=tuple(candidates[arm].candidate_id for arm in arms),
        )
        for agents, arms in grouped.items()
    )
    arm_execution_keys = tuple(
        (arm, _execution_key(candidates[arm].agents)) for arm in ARM_ORDER
    )
    return CanonicalHierarchy(
        frozen_v2=candidates[V2_ARM],
        component=candidates[COMPONENT_ARM],
        hotspot=candidates[HOTSPOT_ARM],
        partition=partition,
        executions=executions,
        arm_execution_keys=arm_execution_keys,
    )


@dataclass(frozen=True)
class H1Trial:
    """Permitted current-step observations for one paired PP seed."""

    trial_index: int
    pp_seed: int
    before_conflicts: int
    after_conflicts: int
    strict_decrease: bool
    rolled_back: bool
    normalized_conflict_reduction: float

    def __post_init__(self) -> None:
        trial_index = _require_plain_int(self.trial_index, name="trial_index")
        _require_plain_int(self.pp_seed, name="pp_seed")
        before = _require_plain_int(self.before_conflicts, name="before_conflicts")
        after = _require_plain_int(self.after_conflicts, name="after_conflicts")
        if trial_index not in H1_TRIAL_INDICES:
            raise ValueError("trial_index is outside the preregistered H1 product")
        if before <= 0 or after < 0:
            raise ValueError("H1 conflict counts must have before > 0 and after >= 0")
        if type(self.strict_decrease) is not bool or type(self.rolled_back) is not bool:
            raise ValueError("strict_decrease and rolled_back must be booleans")
        if self.strict_decrease is not (after < before):
            raise ValueError("strict_decrease differs from current-step conflicts")
        if self.rolled_back and after != before:
            raise ValueError("a rolled-back H1 trial must preserve the conflict count")
        reduction = float(self.normalized_conflict_reduction)
        expected = (before - after) / max(1, before)
        if not math.isfinite(reduction) or abs(reduction - expected) > 1e-12:
            raise ValueError("normalized_conflict_reduction differs from conflict counts")
        object.__setattr__(self, "normalized_conflict_reduction", reduction)


def h1_trial_from_mapping(row: Mapping[str, Any]) -> H1Trial:
    """Parse a strict H1 row and reject runtime or future-label leakage."""

    keys = frozenset(map(str, row))
    forbidden = keys & FORBIDDEN_H1_FIELDS
    if forbidden:
        raise ValueError(f"forbidden H1 fields: {sorted(forbidden)}")
    if keys != _TRIAL_FIELDS:
        raise ValueError(
            "H1 trial fields changed: "
            f"expected {sorted(_TRIAL_FIELDS)}, got {sorted(keys)}"
        )
    return H1Trial(
        trial_index=row["trial_index"],
        pp_seed=row["pp_seed"],
        before_conflicts=row["before_conflicts"],
        after_conflicts=row["after_conflicts"],
        strict_decrease=row["strict_decrease"],
        rolled_back=row["rolled_back"],
        normalized_conflict_reduction=row["normalized_conflict_reduction"],
    )


@dataclass(frozen=True)
class StabilityPolicy:
    minimum_mean_advantage: float = 0.02
    minimum_paired_win_fraction: float = 0.75
    tie_epsilon: float = 1e-12

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.minimum_mean_advantage)
            or self.minimum_mean_advantage <= 0.0
        ):
            raise ValueError("minimum_mean_advantage must be finite and positive")
        if not 0.5 < self.minimum_paired_win_fraction <= 1.0:
            raise ValueError("minimum_paired_win_fraction must be in (0.5, 1]")
        if not math.isfinite(self.tie_epsilon) or self.tie_epsilon < 0.0:
            raise ValueError("tie_epsilon must be finite and nonnegative")


@dataclass(frozen=True)
class PairwiseH1Evidence:
    reference_execution_key: str
    challenger_execution_key: str
    trial_count: int
    mean_normalized_reduction_delta: float
    first_fixed_half_mean_delta: float
    second_fixed_half_mean_delta: float
    challenger_paired_win_fraction: float
    reference_paired_win_fraction: float
    challenger_strict_decrease_rate: float
    reference_strict_decrease_rate: float
    challenger_rollback_rate: float
    reference_rollback_rate: float
    verdict: str


@dataclass(frozen=True)
class Stage1H1Label:
    label: str
    structural_execution_keys: tuple[str, ...]
    comparisons: tuple[PairwiseH1Evidence, ...]
    role_votes: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class Stage2H1Label:
    applicable: bool
    label: str
    comparison: PairwiseH1Evidence | None


@dataclass(frozen=True)
class HierarchicalH1Labels:
    partition: str
    frozen_v2_candidate_id: str
    stage1: Stage1H1Label
    stage2: Stage2H1Label


def _coerce_trial(row: H1Trial | Mapping[str, Any]) -> H1Trial:
    if isinstance(row, H1Trial):
        return row
    if not isinstance(row, Mapping):
        raise ValueError("H1 trial rows must be H1Trial objects or mappings")
    return h1_trial_from_mapping(row)


def _validated_trial_product(
    plan: CanonicalHierarchy,
    trials_by_execution: Mapping[str, Sequence[H1Trial | Mapping[str, Any]]],
) -> dict[str, tuple[H1Trial, ...]]:
    required = set(plan.label_execution_keys)
    observed = set(map(str, trials_by_execution))
    if observed != required:
        raise ValueError(
            "deduplicated H1 execution product changed: "
            f"expected {sorted(required)}, got {sorted(observed)}"
        )
    if not required:
        return {}

    product: dict[str, tuple[H1Trial, ...]] = {}
    for execution_key in plan.label_execution_keys:
        rows = tuple(
            sorted(
                (_coerce_trial(row) for row in trials_by_execution[execution_key]),
                key=lambda row: row.trial_index,
            )
        )
        if tuple(row.trial_index for row in rows) != H1_TRIAL_INDICES:
            raise ValueError("each H1 execution requires exactly trial indices 0..15")
        seeds = tuple(row.pp_seed for row in rows)
        if len(set(seeds)) != len(seeds):
            raise ValueError("H1 PP seeds must differ across trial indices")
        product[execution_key] = rows

    anchor = product[plan.label_execution_keys[0]]
    before_conflicts = anchor[0].before_conflicts
    for trial_index in H1_TRIAL_INDICES:
        paired_seed = anchor[trial_index].pp_seed
        for rows in product.values():
            row = rows[trial_index]
            if row.pp_seed != paired_seed:
                raise ValueError("the same H1 trial index must use one paired PP seed")
            if row.before_conflicts != before_conflicts:
                raise ValueError("all H1 repairs must start from the same conflict count")
    return product


def _rate(values: Sequence[bool]) -> float:
    return statistics.fmean(float(value) for value in values)


def _stable_direction(
    *,
    deltas: Sequence[float],
    challenger: Sequence[H1Trial],
    reference: Sequence[H1Trial],
    policy: StabilityPolicy,
) -> bool:
    epsilon = policy.tie_epsilon
    return (
        statistics.fmean(deltas)
        >= policy.minimum_mean_advantage - epsilon
        and _rate([delta > epsilon for delta in deltas])
        >= policy.minimum_paired_win_fraction
        and statistics.fmean(deltas[:8]) > epsilon
        and statistics.fmean(deltas[8:]) > epsilon
        and _rate([row.rolled_back for row in challenger])
        <= _rate([row.rolled_back for row in reference]) + epsilon
        and _rate([not row.strict_decrease for row in challenger])
        <= _rate([not row.strict_decrease for row in reference]) + epsilon
    )


def _pairwise_evidence(
    *,
    reference_key: str,
    challenger_key: str,
    product: Mapping[str, tuple[H1Trial, ...]],
    policy: StabilityPolicy,
) -> PairwiseH1Evidence:
    reference = product[reference_key]
    challenger = product[challenger_key]
    deltas = tuple(
        challenger_row.normalized_conflict_reduction
        - reference_row.normalized_conflict_reduction
        for reference_row, challenger_row in zip(reference, challenger)
    )
    challenger_stable = _stable_direction(
        deltas=deltas,
        challenger=challenger,
        reference=reference,
        policy=policy,
    )
    reverse_deltas = tuple(-delta for delta in deltas)
    reference_stable = _stable_direction(
        deltas=reverse_deltas,
        challenger=reference,
        reference=challenger,
        policy=policy,
    )
    if challenger_stable and reference_stable:
        raise RuntimeError("opposite H1 directions cannot both be stable")
    if challenger_stable:
        verdict = "challenger"
    elif reference_stable:
        verdict = "reference"
    else:
        mean_delta = statistics.fmean(deltas)
        first_half = statistics.fmean(deltas[:8])
        second_half = statistics.fmean(deltas[8:])
        directional_conflict = first_half * second_half < -(policy.tie_epsilon**2)
        strong_unstable_signal = (
            abs(mean_delta) >= policy.minimum_mean_advantage - policy.tie_epsilon
            or _rate([delta > policy.tie_epsilon for delta in deltas])
            >= policy.minimum_paired_win_fraction
            or _rate([delta < -policy.tie_epsilon for delta in deltas])
            >= policy.minimum_paired_win_fraction
        )
        verdict = (
            "ambiguous"
            if directional_conflict or strong_unstable_signal
            else "abstain"
        )

    return PairwiseH1Evidence(
        reference_execution_key=reference_key,
        challenger_execution_key=challenger_key,
        trial_count=len(deltas),
        mean_normalized_reduction_delta=statistics.fmean(deltas),
        first_fixed_half_mean_delta=statistics.fmean(deltas[:8]),
        second_fixed_half_mean_delta=statistics.fmean(deltas[8:]),
        challenger_paired_win_fraction=_rate(
            [delta > policy.tie_epsilon for delta in deltas]
        ),
        reference_paired_win_fraction=_rate(
            [delta < -policy.tie_epsilon for delta in deltas]
        ),
        challenger_strict_decrease_rate=_rate(
            [row.strict_decrease for row in challenger]
        ),
        reference_strict_decrease_rate=_rate(
            [row.strict_decrease for row in reference]
        ),
        challenger_rollback_rate=_rate([row.rolled_back for row in challenger]),
        reference_rollback_rate=_rate([row.rolled_back for row in reference]),
        verdict=verdict,
    )


def build_hierarchical_h1_labels(
    plan: CanonicalHierarchy,
    trials_by_execution: Mapping[str, Sequence[H1Trial | Mapping[str, Any]]],
    *,
    policy: StabilityPolicy = StabilityPolicy(),
) -> HierarchicalH1Labels:
    """Build conservative Stage 1 and conditional Stage 2 current-step labels."""

    product = _validated_trial_product(plan, trials_by_execution)
    structural_keys = plan.stage1_structural_execution_keys
    if not structural_keys:
        return HierarchicalH1Labels(
            partition=plan.partition,
            frozen_v2_candidate_id=plan.frozen_v2.candidate_id,
            stage1=Stage1H1Label(
                label="audit_only",
                structural_execution_keys=(),
                comparisons=(),
                role_votes=(
                    (COMPONENT_ARM, "v2_alias"),
                    (HOTSPOT_ARM, "v2_alias"),
                ),
            ),
            stage2=Stage2H1Label(
                applicable=False,
                label="not_applicable",
                comparison=None,
            ),
        )

    v2_key = plan.execution_key_for(V2_ARM)
    stage1_comparisons = tuple(
        _pairwise_evidence(
            reference_key=v2_key,
            challenger_key=structural_key,
            product=product,
            policy=policy,
        )
        for structural_key in structural_keys
    )
    comparison_by_key = {
        comparison.challenger_execution_key: comparison
        for comparison in stage1_comparisons
    }
    role_votes: list[tuple[str, str]] = []
    for arm in (COMPONENT_ARM, HOTSPOT_ARM):
        execution_key = plan.execution_key_for(arm)
        if execution_key == v2_key:
            vote = "v2_alias"
        else:
            vote = {
                "challenger": "structural",
                "reference": "v2",
                "abstain": "abstain",
                "ambiguous": "abstain",
            }[comparison_by_key[execution_key].verdict]
        role_votes.append((arm, vote))
    votes = {vote for _arm, vote in role_votes}
    if votes == {"structural"}:
        stage1_label = "admit_structural_consensus"
    elif votes <= {"v2", "v2_alias"}:
        stage1_label = "keep_v2"
    else:
        stage1_label = "abstain"

    stage2_comparison: PairwiseH1Evidence | None = None
    if plan.stage2_applicable:
        stage2_comparison = _pairwise_evidence(
            reference_key=plan.execution_key_for(COMPONENT_ARM),
            challenger_key=plan.execution_key_for(HOTSPOT_ARM),
            product=product,
            policy=policy,
        )
        stage2_label = {
            "reference": "component",
            "challenger": "hotspot",
            "abstain": "abstain",
            "ambiguous": "ambiguous",
        }[stage2_comparison.verdict]
    else:
        stage2_label = "not_applicable"

    return HierarchicalH1Labels(
        partition=plan.partition,
        frozen_v2_candidate_id=plan.frozen_v2.candidate_id,
        stage1=Stage1H1Label(
            label=stage1_label,
            structural_execution_keys=structural_keys,
            comparisons=stage1_comparisons,
            role_votes=tuple(role_votes),
        ),
        stage2=Stage2H1Label(
            applicable=plan.stage2_applicable,
            label=stage2_label,
            comparison=stage2_comparison,
        ),
    )


__all__ = [
    "ALL_DISTINCT",
    "ALL_SHARED",
    "COMPONENT_ARM",
    "CandidateArm",
    "CandidateExecution",
    "CanonicalHierarchy",
    "FIRST_FIXED_HALF",
    "FORBIDDEN_H1_FIELDS",
    "H1Trial",
    "H1_TRIAL_INDICES",
    "HOTSPOT_ARM",
    "HierarchicalH1Labels",
    "PARTITIONS",
    "PairwiseH1Evidence",
    "SECOND_FIXED_HALF",
    "STRUCTURAL_CONSENSUS",
    "StabilityPolicy",
    "Stage1H1Label",
    "Stage2H1Label",
    "V2_ARM",
    "V2_COMPONENT_ALIAS",
    "V2_HOTSPOT_ALIAS",
    "build_hierarchical_h1_labels",
    "canonicalize_hierarchy",
    "h1_trial_from_mapping",
]
