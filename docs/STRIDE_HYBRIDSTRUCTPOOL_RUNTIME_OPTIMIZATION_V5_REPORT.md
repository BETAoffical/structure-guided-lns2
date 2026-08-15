# STRIDE HybridStructPool runtime optimization v5 report

## Registered result

The paired 19-key experiment completed all 152 episodes (76 per arm) with zero
execution errors and zero process timeouts.  Report SHA-256:
`1c31711820da7cc2bced78240f5ff62bc7288e651629699a83dde7bb63cdf7a0`.

All preregistered gates passed.  Relative to the registered v3 full-union
engineering baseline:

- mean controller time: 11.7175 to 9.9329 seconds (-15.23%);
- mean candidate-generation time: 9.6254 to 7.8024 seconds (-18.94%);
- mean Hybrid generation plus challenger-feature time: 9.6766 to 7.8889
  seconds (-18.47%).

Relative to the original complete-pool runtime evidence, the cumulative
Hybrid-total reduction is 53.72% (17.0447 to 7.8889 seconds per episode).

The complete candidate contract remained active.  Hybrid success was 0.7368,
platform entry was 0.5263, normalized fixed AUC was 0.16464, and restricted
mean repair decisions were 29.4688.  The paired V2 arm was respectively
0.5789, 0.7895, 0.20738, and 41.0625.  Overall capped continuation wall was
63.73 seconds for Hybrid and 84.86 seconds for V2.

## Boundary

This validates a semantics-preserving engineering acceleration of the complete
HybridStructPool research candidate space on the registered, outcome-enriched
Maze cohort.  It does not promote HybridStructPool as the default online pool,
does not establish raw TTF or result-blind/OOD benefit, and does not resolve the
known Maze-128-128-1 per-map overhead relative to V2 (86.74 versus 73.50 seconds
in this capped continuation run).

The next scientifically valid step is not further candidate deletion.  It is a
separately preregistered result-blind runtime/TTF confirmation or an online
selector that can preserve the complete-union Copeland context without paying
full exact scoring cost.
