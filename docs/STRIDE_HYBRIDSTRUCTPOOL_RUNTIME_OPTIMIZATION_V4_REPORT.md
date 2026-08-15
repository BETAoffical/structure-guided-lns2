# STRIDE HybridStructPool runtime optimization v4 report

The 152-episode paired run completed with zero execution errors and zero
process timeouts.  All quality, integrity, per-map, candidate-generation, and
Hybrid-total gates passed.  Relative to v3, mean Hybrid-total time fell from
9.6766 to 8.4888 seconds (-12.27%) and mean candidate-generation time fell from
9.6254 to 8.4264 seconds (-12.46%).

The aggregate gate remained false because mean controller time fell from
11.7175 to 10.6173 seconds (-9.39%), narrowly missing the frozen 10% threshold
of 10.5458 seconds by 0.0716 seconds per episode.  The threshold is not relaxed.
V4 is therefore retained as a successful CausalClosure hot-loop optimization
but not a complete engineering-gate pass or runtime-pool promotion.

Profiling the remaining Python work found the causal Pareto front ranking made
repeated nested-dictionary comparisons.  The v5 follow-up pre-extracts the
same four frozen minimization coordinates and performs direct comparisons.
On the same 345 raw causal candidates, old and new ranks were exact and the
isolated median fell from 0.04182 to 0.00685 seconds per decision (6.1x).
