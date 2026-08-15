# STRIDE HybridStructPool runtime optimization v3

## Correction from v2

The v2 engineering pilot proved that deleting structural candidates which had
never won was not selection-neutral under probabilistic Copeland aggregation.
The lean pool completed 152/152 episodes without execution errors, but success
fell from 73.68% to 63.16%, normalized fixed AUC rose from 0.16486 to 0.17694,
and restricted mean repair decisions rose from 29.09 to 33.31.  At every
observed first divergence the old winner was still present and the retained
CausalClosure set was identical; removing other structural opponents reversed
the pairwise aggregate score.

V3 therefore restores all six structural families at all four registered sizes.
It is a semantics-preserving engineering experiment, not a compressed Pool.

## Retained acceleration

- Causal occupancy and transition indices cover only the composed temporal
  window reachable from current conflict events while retaining terminal-path
  semantics.
- Localized conflict closures and temporal-neighbor evidence are reused across
  the five Causal family variants.
- V2 realized feature rows are reused; features are computed only for new
  structural and Causal challengers.
- Pure candidate-generation time is separated from challenger feature time so
  controller accounting cannot count the same feature construction twice.

The complete V2 pool, complete 24-cell structural grid, and complete retained
CausalClosure frontier remain present at every activated state.

## Gates

The old full-union hybrid block consumed 17.04469 mean seconds over the frozen
76 Hybrid episodes.  V3 must reduce this to at most 12.78352 seconds, preserve
the old full-union success, AUC, repair decisions and platform rate, and pass
the original paired V2 gates.  Every map's capped continuation wall must be at
most 1.10 times its paired V2 arm.

Passing authorizes the next semantics-preserving optimization: an incremental
Causal path/contact index updated only for agents changed by the preceding PP
action.  It does not authorize candidate compression, a new ranker, raw TTF, or
default-Pool promotion.
