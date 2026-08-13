# STRIDE ShellBudget Audit v1 result

## Decision

The preregistered ranker-free audit completed with every identity and integrity
check passing. Budgets 6, 8, and 12 all failed the frozen readiness gates.
Therefore no compressed budget is adopted: the candidate contract remains the
full exact-deduplicated equal `8/16/24/32` structural pool, with the complete V2
pool and V2 anchor preserved separately.

The frozen report is
`build/stride-shellbudget-audit-v1/shellbudget_audit_report.json`, with SHA-256
`dd90dff6c19ccb5cf2af21823cda025e07cf4fd3d622ee85200f44ed95a14f99`.

## Results

| Metric | Budget 6 | Budget 8 | Budget 12 | Gate |
|---|---:|---:|---:|---:|
| Non-tail global-best retention | 57.14% | 65.31% | 85.71% | at least 90% |
| Non-tail mean regret | 0.03070 | 0.02052 | 0.00690 | at most 0.02 |
| First-half best retention | 58.16% | 64.29% | 89.80% | at least 85% |
| Second-half best retention | 55.10% | 62.24% | 84.69% | at least 85% |
| Maze opportunity-state recall | 82.98% | 91.49% | 95.74% | at least 90% |
| Maze robust-action recall | 36.84% | 47.37% | 64.59% | at least 80% |
| Beneficial PreTail membership | 38.30% | 61.70% | 72.34% | at least 80% |

Budget 12 is close on ordinary one-step quality but still discards 74 of the
209 robust Maze actions and 13 of the 47 previously beneficial bounded
continuation actions. This is precisely why one-step regret cannot stand in for
self-loop avoidance. The smaller budgets lose both ordinary quality and
escape-capable coverage.

The size-balanced reducer itself behaved as registered. Where compression was
required, the maximum size-count spread was one. States whose entire pool was
already no larger than the nominal budget retained every candidate.

## Execution correction

The first report was rejected before acceptance because the executor continued
filling a budget after one size bucket was exhausted, despite the registered
size-spread gate. The accepted implementation stops before a further addition
would violate that gate and exempts states for which no compression occurs.
The correction is documented in
`STRIDE_SHELLBUDGET_AUDIT_V1_EXECUTION_AMENDMENT.md`. No threshold, ordering
criterion, state, candidate, or outcome changed.

## Interpretation and next step

The artificial per-family preferred sizes are now removed without asking a
model to predict one exact size. The four sizes are symmetric alternatives, and
the data show that their apparently redundant actions cannot yet be compressed
safely by outcome-blind structural diversity.

This result does not validate the current ranker, prove long-tail avoidance, or
authorize runtime/TTF work. The next stage must leave this full candidate pool
unchanged and address selection before the first harmful action. It must jointly
preserve immediate repair quality and estimate transition into a persistent
conflict cycle; a risk-only objective, post-failure fallback, or online PP probe
is not allowed. Before model training, the transition label must be shown stable
on frozen paired continuations and must retain decision occurrences rather than
deduplicating state fingerprints.
