# STRIDE-LoopGuard v1 trigger audit

## Outcome

The preregistered existing-trajectory audit completed all 132 TailSwitch
continuation contrasts: 17 adverse, 11 beneficial and 104 neutral. Input
identity, all 132 treatment manifests and all 132 trace SHA-256 values match
the completed sequence-forensics evidence.

No trigger variant passed. Repetition is a high-recall warning sign, but it is
not a sufficiently precise rule for forcing a V2 fallback.

## Frozen trigger results

Every trigger is evaluated before the proposed action is repaired. It may read
the current proposed neighborhood and prior decisions, including the forced
first action, but it may not read the current action outcome, future trajectory,
TTF or runtime. The primary window is the first 20 continuation decisions.

| Frozen variant | Adverse | Beneficial false triggers | Neutral false triggers | Passed |
| --- | ---: | ---: | ---: | --- |
| Exact same candidate for three structural actions | 17/17 (100%) | 8/11 (72.73%) | 20/104 (19.23%) | no |
| Agent Jaccard at least 0.95 for a three-action structural run | 17/17 (100%) | 8/11 (72.73%) | 20/104 (19.23%) | no |
| Agent Jaccard at least 0.90 and pre-action conflict Jaccard at least 0.80 | 17/17 (100%) | 8/11 (72.73%) | 20/104 (19.23%) | no |
| Agent Jaccard at least 0.90 for a three-action structural run | 17/17 (100%) | 8/11 (72.73%) | 20/104 (19.23%) | no |

All variants covered three maps and both challengers among adverse cases. They
failed both precision gates:

- registered maximum beneficial false-trigger rate: 20%; observed 72.73%;
- registered maximum neutral false-trigger rate: 10%; observed 19.23%.

The four variants produce identical classifications because every high-overlap
triplet in this cohort is already an exact repeat of the same candidate. Adding
the frozen conflict-persistence condition does not remove a single false
trigger.

## Trigger timing

The median first exact-triplet trigger occurs at continuation decision 6.0 for
adverse pairs, 8.5 for beneficial pairs and 5.5 for neutral pairs. Triggering
earlier therefore does not separate the harmful sequences: neutral sequences
actually trigger earlier on average than adverse sequences.

## What this means for neighborhood selection

The evidence still supports a selection-loop defect, but narrows its form:

1. a memoryless augmented selector can repeatedly choose the same structural
   neighborhood;
2. every observed adverse continuation enters such a loop early;
3. the same behavior is also part of eight genuinely beneficial continuations;
4. therefore repetition itself is not the defect--the defect is failure to
   distinguish productive repetition from non-productive repetition.

This explains why a simple tabu, repeat penalty or third-repeat fallback would
be unsafe. It would cover the known tails, but would also suppress most of the
known beneficial structural continuations.

## Decision and next safe diagnostic

`STRIDE-LoopGuard-v1` is not implemented as a runtime controller from these
features. No TTF experiment, training or default-controller change is allowed.

The next defensible diagnostic must inspect information already available
before the proposed repeated action and ask whether the preceding repeated
repairs were productive. Candidate inputs include cumulative prior conflict
progress, persistent conflict-core turnover and conflict migration during the
previous two or three repairs. Those historical outcomes are available at the
next decision, but their definitions and thresholds must be preregistered
before the current report is queried for another rule.

## Integrity and claim boundary

- report SHA-256:
  `78fd31a772888b41192f6d0c5560a05499a7b2b13313779c86c57269e4eadda5`
- complete state count: 66;
- complete continuation contrast count: 132;
- no solver or PP rerun;
- no result-based state exclusion;
- no causal, generalization or TTF-improvement claim.
