# STRIDE SafeSlot First-Divergence Diagnostic v1 Result

## Outcome

The integrity audit passed for all 29 paired keys and both challenger
controllers, giving 58 comparisons and zero errors. This is a causal
first-action diagnostic, not a TTF comparison: the exact V2 baseline was
collected separately, so its wall-clock fields are provenance only.

Both full StructPool and SlotPool first diverged from V2 on 19/29 keys. Every
observed first divergence happened at decision zero. Full StructPool produced
17 immediately better, one tied, and one worse structural action; SlotPool
produced 17 immediately better and two worse structural actions.

Terminal repair behavior did not always agree with the immediate conflict
change. Three structural actions reduced at least as many conflicts as V2 at
the first step but still required more terminal repair rounds:

| Controller | Maze task / solver seed | Family / size | Immediate advantage | Eight-step advantage | Repairs: V2 -> challenger |
|---|---:|---|---:|---:|---:|
| SlotPool | 233 / 1 | bottleneck / 32 | +58 | -5 | 34 -> 40 |
| SlotPool | 277 / 3 | bottleneck / 32 | +12 | -16 | 18 -> 46 |
| full StructPool | 233 / 2 | boundary / 24 | +9 | 0 | 18 -> 42 |

The first two hazards become visible within eight subsequent repairs. The
third ends the eight-step window at the same conflict count as V2, but makes
strict progress on fewer repairs and has a longer no-progress streak. All
three contradictions occur on Maze; this is a warning about residual-state
hardness, not evidence for fitting a Maze-specific rule.

## V2-anchor finding

One full-StructPool state changed the identity of the best base candidate after
structural candidates were added. Candidate rows themselves were unchanged.
The cause is the current V2 pairwise Borda ranker: every candidate's score
depends on which other candidates are present, so reranking a mixed pool is not
equivalent to selecting V2 on the original pool and then challenging it.

SlotPool's logged base-only anchor matched the exact current-runtime V2 action
on all audited states. SafeSlot must therefore compute V2 once on the original
pool and keep that action fixed; structural candidates may only be compared
against this anchor with a separate pointwise gate.

## Decision

- StructPool and SlotPool remain one unified design problem: generate the
  non-manual family-by-size search space, then decide whether one structural
  action can safely replace the frozen V2 anchor.
- Immediate conflict reduction alone is not a sufficient replacement label.
- Cost-to-Go, remaining repair rounds, TTF, and the eight-step future are not
  allowed as primary training targets.
- The primary label may use robust paired one-step PP outcomes and post-repair
  residual-structure teacher fields. Runtime model inputs must remain available
  before executing the action.
- Uncertain challenges abstain to V2. Rollback remains a separate runtime
  safety mechanism, not a training label.

The immutable report SHA-256 is
`24147e3eb539a032f1e6db4cf3d5df460395a982fe64e73b980b4ef3aa1781fd`.
