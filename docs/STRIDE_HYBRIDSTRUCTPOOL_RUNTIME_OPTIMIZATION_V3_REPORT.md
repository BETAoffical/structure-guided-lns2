# STRIDE HybridStructPool runtime optimization v3 report

## Outcome

The registered 19-key, 152-episode paired run completed with 76 episodes per
arm and zero execution errors or process timeouts.  The complete HybridStructPool
candidate contract was restored; no structural family/size cell was removed.

The engineering optimization was material:

- mean Hybrid generation plus challenger-feature time: 17.0447 s to 9.6766 s
  per episode (-43.2%);
- mean candidate-generation time: 17.4942 s to 9.6254 s (-45.0%);
- mean controller time: 19.4151 s to 11.7175 s (-39.6%).

Quality returned to the prior complete-pool regime: success was 0.7368,
platform entry was 0.5263, normalized fixed AUC was 0.16470, and restricted
mean repair decisions were 29.2885.  Against the paired V2 arm, Hybrid improved
success, AUC, repair decisions, overall capped wall, and platform entry.

## Gate decision

The aggregate engineering gate did not pass.  Restricted mean repair decisions
were 0.1936 above the older complete-pool run, and Maze-128-128-1 Hybrid capped
wall was 1.203 times its paired V2 arm, above the preregistered 1.10 per-map cap.
The older complete-pool run already had a 1.287 ratio on that map, so this is a
pool/runtime-promotion limitation rather than evidence that v3 candidate
generation changed membership.

All 76 old/new Hybrid first transitions matched on candidate identity, agent
set, PP seed, before fingerprint, and after fingerprint.  Later same-action,
same-seed native divergence also occurred in the unchanged V2 arm (12 pairs)
and is retained as wall-clock/native variability, not used to relax the v3
gate.  V3 therefore remains an engineering result and is not a runtime-pool
promotion.

## Next bounded optimization

Profiling assigned approximately 1.227 of 1.328 seconds on a frozen Maze128
decision to CausalClosure generation.  The v4 follow-up only materializes
terminal waits once per context and accumulates temporal-contact counters in
mutable integers before constructing immutable evidence.  A frozen-state A/B
check found exact `HybridStructPoolResult` equality and a 1.218x median speedup
for this hot path.  A fresh paired run is required before accepting an
end-to-end engineering speed claim.
