# STRIDE HybridStructPool budget v1 result

## Decision

None of the frozen challenger budgets 6, 8 or 12 passed the preregistered
membership gates.  The full HybridStructPool union therefore remains the
research candidate contract.  The reducer must not be retuned on these 78
states, and no compressed pool is eligible for runtime integration.

The audit made no solver or PP calls.  All 78 states, three maps and frozen
membership identities passed integrity checks.  The complete V2 pool remained
outside the challenger budget and was preserved exactly.

## Results

| Metric | Budget 6 | Budget 8 | Budget 12 | Gate |
| --- | ---: | ---: | ---: | ---: |
| Legacy opportunity-state recall | 72.34% | 82.98% | 85.11% | at least 90% |
| Legacy robust-action recall | 25.84% | 32.06% | 51.20% | at least 80% |
| Worst-map opportunity recall | 62.50% | 66.67% | 66.67% | at least 80% |
| Beneficial PreTail membership recall | 27.66% | 27.66% | 53.19% | at least 80% |
| CausalClosure-only best-action recall | 0/3 | 0/3 | 1/3 | 3/3 |
| V2 base recall | 100% | 100% | 100% | 100% |

Budget 12 was the strongest treatment but still failed every challenger
membership gate.  Its map-specific structural opportunity recall was 66.67%
on `maze-128-128-1` and 93.75% on each of the other two maps.  The failure is
therefore not explained by budget count alone: an outcome-blind semantic and
set-diversity rule still cannot know which member of each family/size shell or
compact causal group must survive.

## Interpretation

The complete candidate-space problem is now separated from runtime selection:

- the full union fixes the omission problem and preserves all registered old
  and new candidate membership;
- a small transparent reducer does not preserve that union's useful actions;
- this result does not justify training another one-step ranker, using outcome
  labels to retain the successful states, or reverting to fixed family sizes.

For the next post-failure repair study, the full union is frozen as the set of
available proposal mechanisms.  The study must not claim that all 44 actions
should be evaluated online.  Runtime compression and selection remain a later,
separate problem that requires end-to-end evidence.

## Artifacts

- `hybridstructpool_budget_report.json` SHA-256:
  `3a383a0df84c31228ec6b2dcfce94ec62b0dd87411c7d8d61a2257866f0ef9d0`;
- `budget_state_rows.jsonl` SHA-256:
  `21f9e3db6478d746f371e9f8a974543c491fcb88870e4c3e97b671be89a4776d`.

## Claim boundary

`selected_budget=null`, `full_union_required=true`, and
`runtime_replacement_ready=false`.  No selector, platform-prevention, runtime,
TTF or default-controller claim follows from this audit.
