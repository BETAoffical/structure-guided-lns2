# STRIDE HybridStructPool runtime v1

## Milestone boundary

This milestone lands the audited full HybridStructPool union in the closed-loop
runtime without adding a long-tail retry or rollback intervention.  It is an
experimental candidate-space implementation, not a promoted default Pool.

The runtime preserves the registered contract:

- the complete V2 candidate pool and its V2-selected anchor;
- the complete StructShell grid at 8, 16, 24 and 32 agents;
- up to 12 CausalClosure candidates derived from the V2 anchor;
- exact-set deduplication with V2 rows authoritative;
- no 6, 8 or 12 candidate reducer, because every registered reducer failed the
  earlier membership gates;
- a hard cap of 64 total candidates, above the audited maximum of 54, with a
  terminal error rather than silent truncation if the contract is exceeded.

The existing outcome-blind high-stress gate remains the activation boundary.
When it does not pass, the complete V2 pool is used unchanged.  Candidate
generation, V2-anchor scoring, feature construction and final V2 ranking time
are all included in controller time and the trace records candidate provenance.

## PP random-seed semantics

Two seed policies are now explicit:

- `state_derived`: the historical candidate/state-derived seed used by frozen
  paired audits;
- `episode_stream`: a pseudorandom stream initialized once from the registered
  task, solver seed and episode identity, then advanced for every explicit PP
  action.

`episode_stream` matches the important official LNS2 behavior: reproducible for
a fixed solver seed, but not reset to a candidate-specific value at every
decision.  It deliberately does not use operating-system entropy.  Formal
paired experiments may still provide `pp_random_seed` after neighborhood-order
generation so competing controllers receive the same low-level PP stream.

## What is not included

No failure-informed blocker rescue, repeated-seed retry, semantic compaction or
repair-order intervention is enabled in this milestone.  Those mechanisms are
the second, separately testable step and may only be composed after this Pool
passes its own closed-loop validation.

