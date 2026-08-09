# STRIDE ClosurePool Long-tail Forensics v1

## Question

This retrospective study asks which agent, conflict-graph, temporal-path, and
map-topology mechanisms distinguish harmful structural first divergences from
matched V2 actions. It is the first stage of the new `stride-closurepool-v1`
line. It does not change candidate generation, fit a model, run a solver, or
select a runtime threshold.

The motivating defect is narrower than “one-step ranking is inaccurate.” A
structural neighborhood may remove many current conflicts while repairing only
part of a bottleneck queue or conflict dependency component. Replanned agents
can then acquire paths that conflict with unchanged agents outside the
neighborhood, leaving a small persistent residual conflict.

## Frozen evidence

The analysis retains all 58 existing V2-versus-StructPool/SlotPool comparisons
over 29 keys. Comparisons without a first divergence remain in the coverage
denominator. Every divergent comparison is reconstructed from the saved
compact trace, including the exact initial state blob and all path/conflict
deltas.

Two additional comparisons reuse the single known Maze regression:
StructPool versus V2 and SlotPool versus V2. This case is explicitly marked
`known_regression_only`; it is not included in mechanism contrasts, feature
selection, threshold selection, or generalization claims.

Outcome groups are descriptive only. A challenger is called severe when it
requires at least ten additional repairs and at least twice the V2 repair
count. Any positive repair-iteration delta is adverse. These definitions must
not become training labels without a new registration and independent data.

## Diagnostics

At the first matched-state divergence, both actions use the same PP seed. For
the realized V2 and structural neighborhoods the analyzer records:

- internal, boundary, and external conflict-pair/event coverage;
- complete and partial conflict-component coverage;
- unselected conflicting agents that share articulation or degree-at-most-two
  path cells with selected agents;
- path low-degree, articulation, and wait ratios;
- retained, removed, and newly created conflict pairs after one repair;
- residual boundary conflicts and repeated-event concentration.

Across the remaining trajectory it records conflict-pair and agent persistence,
pair churn, new persistent pairs, the maximum no-progress streak, and the
maximum one-conflict plateau. Persistent-pair event locations are described
relative to articulation and low-degree cells.

## Integrity and decision boundary

All registered file and trace hashes, reconstructed state fingerprints,
conflict trajectories, realized neighborhoods, first-divergence states, and PP
seeds must match. The analyzer writes only diagnostic JSON/JSONL artifacts.

Completion does not authorize a ClosurePool rule. A human cross-case review
must first determine whether the same pre-action mechanism separates adverse
and non-adverse divergences across more than one map/task group. Only then may
a new document freeze an interpretable dependency-closure rule and test it on
data not used to identify the mechanism.
