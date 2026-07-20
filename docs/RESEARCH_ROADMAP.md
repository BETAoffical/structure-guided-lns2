# InitLNS Research Status

## Frozen Claim

The active claim is limited to dynamic explicit-neighborhood control during
InitLNS. The frozen v1 controller was validated in closed loop on unseen maps from
the same synthetic layout families and multiple solver seeds.

The standard MovingAI OOD confirmation produced broad positive evidence but did
not pass the preregistered threshold: v1 succeeded on 131/144 episodes versus
Adaptive on 123/144, while conflict-AUC improvement was 4.105% rather than the
required 5%.

## Unsupported Claims

- Static map topology, static OD, and agent density did not show reliable
  incremental value over the dynamic state representation.
- Policy-visited aggregation, larger tabular GBDTs, neural graph models, graph
  summary features, Horizon-4 labels, and a repair-order selector did not pass
  their registered gates.
- No RL policy was trained or validated.

These negative results remain part of the 24-entry evidence manifest and the
Chinese final report. Historical source, tests, protocols, and detailed notes are
available at Git tag `pre-minimal-runtime-2026-07-20`.

## Current Runtime

The supported learned controllers are `v1-full`, its exact accelerated
implementation `v2-full`, and `v2-stall-safe`. Official Adaptive, Target,
Collision, and Random remain the solver baselines.

No new learning experiment should be added to this branch. A future study must be
separately preregistered and should begin from one of these questions:

1. Can proposal generation be made much cheaper without changing selected actions?
2. Can a structured spatiotemporal representation demonstrate value on an
   independent data split before closed-loop use?
3. Can a carefully bounded sequential policy improve over the frozen controller
   without rewriting the existing MovingAI result?

The current evidence chain is generated with:

```bash
python scripts/consolidate_research_results.py \
  --config configs/result_consolidation.json --verify-build
```
