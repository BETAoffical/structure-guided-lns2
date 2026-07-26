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

## Current Runtime and V3 Research Chain

`v2-full` remains the default learned controller. Official Adaptive, Target,
Collision, and Random remain solver baselines; V2 and Adaptive are also allowed
as offline data sources or external comparisons for V3 studies.

`v3-S3` is the latest runnable V3 controller. It selects directly from its S3
candidate/sequence model and makes no online V2 or Adaptive call, but it did not
pass its promotion gate. The later receding-Q work is a label and stability
research chain, not a controller. Legacy `v3-full`/`v3-h3` code is retained as
historical evidence and is not in the S3 or receding-Q execution path.

All new V3 artifacts use producer/native identities and upgraded schemas.
Completed artifacts are semantically revalidated before reuse; a completed but
invalid artifact is preserved and stops the run. Old artifacts remain readable
but cannot be resumed across a schema, source, native binary, package-version,
controller-input, or native-audit platform change. Start a new output directory
instead.

Any promotion study must be separately preregistered and should begin from one
of these questions:

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
