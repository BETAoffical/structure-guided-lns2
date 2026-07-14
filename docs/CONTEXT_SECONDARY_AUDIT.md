# InitLNS Context Secondary Audit

## Purpose

The secondary audit was pre-registered after the first context audit failed. It
uses the existing 7,344 counterfactual candidates only. The original Train and
previously inspected Validation states are merged into a development set and
evaluated by map-grouped three-fold cross-validation; they are no longer eligible
to serve as a final confirmation set.

The audit changes the diagnostic protocol without changing the LNS2 solver or
counterfactual outcomes:

- neighborhood size is encoded categorically as `4`, `8`, or `16`;
- both pairwise dominance GBDT and direct Pareto-membership GBDT are reported;
- the primary Pareto definition excludes wall-clock runtime, while a sensitivity
  definition adds runtime back;
- static context is permuted as a complete task-level bundle 500 times while
  action, seed, and dynamic-state features remain fixed;
- confidence intervals resample maps rather than treating states from the same
  map as independent;
- an oracle coverage check rejects unsupported collapse to one neighborhood size.

The direct Pareto-membership model is the pre-registered primary learner. The
pairwise model is retained as a robustness result rather than selected after
observing the scores.

## Reproduce

```powershell
python scripts/run_context_confirmation.py `
  --collection build/repair-experience-calibration-v2 `
  --dataset build/repair-transfer-pilot-v2 `
  --output build/initlns-context-secondary-final `
  --mode development `
  --permutations 500
```

The expected non-zero exit status records a failed pre-registered gate. Detailed
results are written to `secondary_audit.json/.md`; generated artifacts remain
under the ignored `build/` directory.

## 2026-07-14 Result

| Learner | Features | Pareto top-1 | Mean AUC regret | Maximum size share |
| --- | --- | ---: | ---: | ---: |
| Pairwise | Action + seed | 12.0% | 1.1401 | 100% |
| Pairwise | Dynamic | 14.0% | 1.0346 | 100% |
| Pairwise | Full context | 17.5% | 1.0532 | 100% |
| Pareto membership | Action + seed | 12.0% | 1.2818 | 93% |
| Pareto membership | Dynamic | 13.5% | 1.2331 | 92% |
| Pareto membership | Full context | 10.0% | 1.2123 | 91% |

Three prerequisites passed. Action/seed labels beat the random-candidate baseline,
dynamic state reduced AUC regret relative to Action + seed, and oracle Pareto
families differed across layout, task, and agent-count groups. The latter means
that action outcomes are heterogeneous; it does not by itself show that the
current static features predict those differences after dynamic state is known.

The two decisive gates failed:

- Real-context top-1 gain ranked at only the `6.2` percentile of 500 task-context
  permutations; AUC-regret reduction ranked at the `59.2` percentile. Both were
  required to reach at least the `95` percentile.
- The primary full-context model selected one neighborhood size in `91%` of
  states, above the `90%` collapse limit, even though sizes 4, 8, and 16 appeared
  in a true primary Pareto set in 88.5%, 40.0%, and 38.0% of states respectively.

The runtime-sensitive analysis did not rescue the conclusion: both learners had
worse full-context AUC regret than their dynamic counterparts. The secondary
development gate is therefore **FAIL**.

## Decision

Per the pre-registered stopping rule, the independent 12/6-map confirmation set
and its maximum 23,328 counterfactual outcomes are not generated. Closed-loop
policy evaluation, expanded collection, and RL remain paused. The result supports
a narrower statement: the current hand-crafted static context does not provide
demonstrated incremental ranking value for this InitLNS action space and model
family, despite observable oracle heterogeneity.

The frozen confirmation protocol remains in
`configs/context_confirmation_dataset.json` and
`configs/context_confirmation_collection.json`. It can only be activated after a
new, independently justified representation hypothesis, not by retuning this
audit against the already inspected development outcomes.
