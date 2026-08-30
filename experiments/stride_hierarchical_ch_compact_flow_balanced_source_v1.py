"""Balanced-coverage identity for the registered compact-flow source product.

The implementation remains in the original compact-flow orchestrator so the
task product, no-backfill behavior, and worker semantics cannot drift.  The
balanced config selects the independent output schemas and split-level OD
coverage gate profile.
"""

from experiments.stride_hierarchical_ch_compact_flow_source_v1 import (
    ALLOWED_SPLITS,
    BALANCED_COLLECTION_TRUST_SCHEMA as COLLECTION_TRUST_SCHEMA,
    BALANCED_CONFIG_SCHEMA as CONFIG_SCHEMA,
    BALANCED_EXPERIMENT_ID as EXPERIMENT_ID,
    BALANCED_GATE_PROFILE as QUALIFICATION_GATE_PROFILE,
    BALANCED_MATERIALIZATION_SCHEMA as MATERIALIZATION_SCHEMA,
    BALANCED_PLAN_SCHEMA as PLAN_SCHEMA,
    BALANCED_QUALIFICATION_AUDIT_SCHEMA as QUALIFICATION_AUDIT_SCHEMA,
    DEFAULT_WORKERS,
    EXPECTED_EPISODE_COUNT,
    EXPECTED_TASK_COUNT,
    MAXIMUM_WORKERS,
    TASK_MATERIALIZER_SCHEMA,
    collect_source_episodes,
    load_registered_source_context,
    materialize_source_dataset,
    plan_source_collection,
)


__all__ = [
    "ALLOWED_SPLITS",
    "COLLECTION_TRUST_SCHEMA",
    "CONFIG_SCHEMA",
    "DEFAULT_WORKERS",
    "EXPECTED_EPISODE_COUNT",
    "EXPECTED_TASK_COUNT",
    "EXPERIMENT_ID",
    "MATERIALIZATION_SCHEMA",
    "MAXIMUM_WORKERS",
    "PLAN_SCHEMA",
    "QUALIFICATION_AUDIT_SCHEMA",
    "QUALIFICATION_GATE_PROFILE",
    "TASK_MATERIALIZER_SCHEMA",
    "collect_source_episodes",
    "load_registered_source_context",
    "materialize_source_dataset",
    "plan_source_collection",
]
