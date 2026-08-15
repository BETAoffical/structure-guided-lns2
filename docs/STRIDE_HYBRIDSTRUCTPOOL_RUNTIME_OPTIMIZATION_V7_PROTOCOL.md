# STRIDE HybridStructPool runtime optimization v7 protocol

V7 preserves the complete HybridStructPool candidate universe, the frozen V2
pairwise ranker, all candidate rows, and all deterministic tie breaks. It does
not delete neighborhoods, change their sizes, retune the selector, or cache
anything across decisions.

Two exact CausalClosure hot-loop changes are evaluated:

1. Ordinary and bottleneck-only temporal-contact evidence is accumulated in a
   single reservation scan and then exposed as the same two evidence maps.
2. Private occupancy and transition indexes use collision-free integer keys
   derived from non-negative cell ids, avoiding millions of temporary tuple
   allocations and hashes. Invalid negative cell ids are rejected explicitly.

Before the paired run, the targeted runtime tests and a frozen 500-agent Room
state must produce the same serialized `HybridStructPoolResult` SHA-256 as V6.
An alternating three-run microbenchmark must show at least a 5% median
improvement on that heavy state.

The full 19-key, 152-episode paired engineering run uses 16 workers and repeats
the registered V6 cohort from the same decision-zero paths. Relative to the
frozen V6 report, mean controller, candidate-generation, and Hybrid-total time
must each improve by at least 5%. All success, AUC, repair-decision, platform,
integrity, and per-map gates remain active. Failure does not authorize candidate
filtering, threshold changes, raw-TTF claims, or runtime promotion.
