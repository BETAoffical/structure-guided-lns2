# STRIDE CausalClosurePool structural-coverage diagnostic result

## Integrity

The frozen retrospective join completed without an identity or coverage error:

- 78/78 state fingerprints;
- all 1,367 V2 base candidates;
- all 1,135 legacy StructPool candidates; and
- all 930 CausalClosure candidates.

The completed CausalClosure opportunity report retained its registered SHA-256
and passed its own integrity checks.  The structural-coverage report SHA-256 is
`7cb2c7ba53766ef58d6eb4ca672841302c4161051439361ca7279f885b74c875`.

## Opportunity loss

Legacy StructPool robustly beat the best V2 base action in 41/78 states.
CausalClosure did so in 10/78 states, but only seven states were shared.  The
result is therefore:

- old-opportunity recall: 7/41 (`17.07%`);
- lost legacy opportunities: 34 states;
- new CausalClosure-only opportunities: three states; and
- CausalClosure opportunity precision relative to legacy opportunity: 7/10
  (`70%`).

Recall was `15.38%` on `maze-128-128-1`, `6.25%` on
`maze-128-128-2`, and `33.33%` on `maze-32-32-4`.  The loss is therefore
strongest on the two 128x128 groups and is not an aggregate-only effect.

## Agent-set coverage

There were 209 legacy structural actions which robustly beat the best V2 action
for their state.  Current CausalClosure candidates covered them as follows:

- exact agent-set equality: 0/209;
- maximum Jaccard at least `0.8`: 4/209 (`1.91%`);
- maximum Jaccard at least `0.6`: 11/209 (`5.26%`); and
- mean maximum Jaccard: `0.2658`.

The largest missing family was topology boundary: 71 robust actions, zero with
Jaccard at least `0.8`.  Conflict component contributed 60 robust actions,
bottleneck crossing 48, and spatiotemporal hotspot 45; family membership is
multi-label.  No path-overlap action robustly beat the best V2 action in this
cohort.

The missed opportunity is not confined to one medium size.  The 209 robust
legacy actions split into 31 size-8, 66 size-16, 60 size-24 and 52 size-32
actions.  Among the 41 state-wise legacy winners, 29 used size 24 or 32.  This
confirms that CausalClosure's median size four over-corrected the earlier size
inflation, but it does not justify restoring a preferred fixed size.

## Decision

All three frozen preservation gates failed, while the integrity gate passed.
CausalClosure alone therefore does not preserve the already demonstrated
StructPool one-step opportunity.  The next stage must design an outcome-blind
hybrid topology-plus-causal pool while retaining the full V2 base pool.

The diagnostic may inform generator design, but successful legacy agent sets
must not be copied or selected directly.  A successor must use state-derived
topology cores without the old fixed 8/16/24/32 preference and must undergo a
new paired PP opportunity audit followed by independent result-blind
continuation.  No ranker, runtime, TTF or long-tail claim is authorized here.
