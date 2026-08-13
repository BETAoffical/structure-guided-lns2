# STRIDE ShellBudget Audit v1 preregistration

## Purpose

StructShell v1 rejected both the original fixed family sizes and a single
outcome-blind structural cutpoint. This audit asks whether the complete equal
`8/16/24/32` structural pool can be reduced to a budget of 6, 8, or 12 without
discarding the actions that make the pool useful.

It is a candidate-pool audit, not a ranker, online controller, PP probe, TTF
test, or proof that a self-loop is avoided.

## Frozen reducer

The primary reducer is `size_family_balanced_maximin_v1`. It exact-deduplicates
the existing four-size candidates and greedily applies this fixed order:

1. prefer the nominal size currently represented least often;
2. prefer the least represented structural family supported by the candidate;
3. cover new families and then new family-size slots;
4. maximize minimum agent-set Jaccard distance from selected candidates;
5. use the equal-weight mean of four structural-coverage ratios; and
6. break the final tie by candidate ID.

This treats all four sizes symmetrically and cannot collapse to only small or
only large neighborhoods. It does not read PP scores, no-progress outcomes,
bounded-continuation labels, ranker scores, runtime, or future trajectories.
The complete V2 pool and its anchor remain separate and unchanged.

## Frozen evidence and gates

The audit reuses the immutable 98-state non-tail grid, 78 Maze difficult
states, 209 robust structural actions, and 180 bounded-continuation
comparisons. SHA-256 identities and the stage backup are frozen in
`configs/stride_shellbudget_audit_v1_registration.json`.

For each budget, every gate must pass:

- non-tail best retention at least 90%, mean regret at most 0.02, maximum map
  mean regret at most 0.05, and both seed-half best retentions at least 85%;
- Maze opportunity-state recall at least 90%, robust-action recall at least
  80%, and every map's opportunity recall at least 80%;
- previously beneficial PreTail action membership recall at least 80%;
- selected size-count spread at most one; and
- all registered identity and integrity checks.

The smallest passing budget is retained. Comparator or isolated-map results
cannot override a primary failure. If none passes, the full
exact-deduplicated equal four-size pool is retained.

## Claim boundary

Current-step quality is only a floor, not the definition of loop avoidance.
PreTail outcomes measure whether the reducer discarded already observed
beneficial actions; they do not prove an online selector would choose those
actions. A pass authorizes only a separately preregistered, result-blind,
bounded forced-continuation confirmation. It does not authorize training,
runtime integration, TTF measurement, or a long-tail claim.
