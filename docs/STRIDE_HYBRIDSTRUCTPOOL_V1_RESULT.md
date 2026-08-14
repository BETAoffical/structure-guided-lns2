# STRIDE HybridStructPool v1 result

## Decision

The zero-solver candidate-contract audit passed every preregistered gate.  The
full exact-set union is now eligible to serve as the frozen proposal-space
contract for a separate budget audit.  It is not yet a runtime replacement for
StructPool or `v2-full` because no runtime budget reducer or selector has been
validated.

## Frozen union

- states: 78 across all three registered Maze maps;
- complete V2 candidates: 1,367;
- equal-four-size structural candidates: 1,135;
- compact CausalClosure v2 candidates: 930;
- exact-deduplicated union: 3,432 candidates, mean 44 per state;
- union candidate size: minimum 1, median 8, mean 11.5862, maximum 58.

There were no cross-source exact duplicates in this cohort.  This is a result
of the frozen candidate geometries, not a rule that assumes future cohorts will
also have zero overlap.  The runtime merge still performs exact-set
deduplication and retains all source provenance.

## Membership gates

All registered recalls were exactly 100%:

| Membership | Recall |
| --- | ---: |
| Complete V2 pool | 1.0000 |
| Equal-four-size structural pool | 1.0000 |
| 209 robust legacy structural actions | 1.0000 |
| Compact CausalClosure pool | 1.0000 |
| Best action in three CausalClosure-only opportunity states | 1.0000 |

All three maps contained candidates from all three sources.  State, candidate,
source and registered-input identity checks passed.  No future trajectory,
runtime, TTF or PP-order field entered the output, and no candidate outcome was
used to construct or prune the union.

## Interpretation

This resolves the candidate-space omission created by trying to replace
StructPool with CausalClosure alone.  The contract keeps the old topology
families without their hard-coded per-family preferred sizes, and adds the
small causal actions without discarding legacy structural opportunity.

It does not resolve the runtime problem.  Evaluating all 44 candidates per
state would add selection and feature overhead, while the previous learned
six-slot reducer did not improve raw TTF.  The next milestone is therefore a
separate outcome-blind budget audit for budgets 6, 8 and 12.  Its purpose is
membership preservation and diversity, not one-step winner prediction or a
platform-prevention claim.  If no budget preserves the registered membership,
the full union remains the research contract and must be evaluated lazily.

Only after the candidate contract and budget boundary are frozen should the
project resume the separate native-order post-failure repair study.

## Artifacts

- `hybridstructpool_audit_report.json` SHA-256:
  `478e5e10a0b2eae90db55ba5750afd3777ae1a947956d20261d702d7ed56e06c`;
- `hybridstructpool_states.jsonl` SHA-256:
  `a15ac7e1f5b73e13b03aa47f3673c753c0a68c1e50d3e67337f0ae71cbb39462`.

## Claim boundary

`candidate_contract_ready=true`, but `runtime_replacement_ready=false`.  No
model training, runtime integration, platform-prevention claim, TTF test or
default-controller replacement is authorized by this result.
