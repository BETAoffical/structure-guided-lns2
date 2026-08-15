# STRIDE HybridStructPool runtime optimization v4 protocol

## Purpose

Measure a semantics-preserving CausalClosure hot-loop optimization after v3
restored the complete HybridStructPool and reduced its runtime materially.
This is an engineering experiment, not candidate compression or pool promotion.

## Frozen identity

- 19 registered Maze task/solver keys;
- two arms (`v2_full`, `hybridstructpool_full`), trials 0-3;
- 152 complete episodes, 16 workers;
- complete V2 pool, all 24 structural family/size cells, and complete retained
  CausalClosure frontier;
- 64 decisions or 180 seconds per episode; process timeout 240 seconds;
- episode-stream PP seeds and native PP order, with no retry or rescue.

## Allowed implementation changes

Only two internal CausalClosure operations change:

1. paths are extended with terminal waits once when building the sparse causal
   context, replacing repeated bounded position lookups;
2. temporal-contact evidence is accumulated in primitive integer counters and
   converted to `CausalContactEvidence` once per neighbor.

Candidate identities, agent sets, evidence counts, Pareto inputs, and online
features must remain exact.  Frozen-state reference output and tests must pass.

## Gates

The complete paired integrity and Hybrid-vs-V2 quality gates remain active.
Relative to the registered v3 engineering baseline, mean controller,
candidate-generation, and Hybrid-total time must each fall by at least 10%.
To separate native wall-clock variability from semantic drift, the frozen
engineering tolerances are: no success-rate decrease, AUC increase at most
0.002, restricted repair decisions increase at most 0.5, and platform entry
increase at most 0.05.  The per-map Hybrid/V2 capped-wall ratio remains bounded
at 1.30, matching the known complete-pool regime rather than asserting runtime
promotion.

Regardless of outcome, this experiment does not authorize default-pool
replacement, raw-TTF claims, non-Maze generalization, training, or rescue
integration.
