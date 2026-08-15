# STRIDE HybridStructPool raw-TTF Quick v1 protocol

The v5 engineering audit leaves no additional low-risk, semantics-preserving
micro-optimization with material expected benefit. Candidate generation remains
the dominant overhead; a cross-decision incremental causal index would be a new
engineering milestone requiring separate equivalence evidence.

This development Quick therefore compares the complete engineered
HybridStructPool against frozen V2 and official Adaptive LNS2. It uses eight
paired high-load keys: two Maze tasks and two Room tasks, each with solver seeds
1 and 2. Controller order rotates within keys, while timed episodes execute
strictly one at a time. Qualification may use 16 workers because it is outside
the TTF clock.

The primary clock is reset-inclusive run-to-completion raw wall TTF. Candidate
generation, feature construction, inference, native PP, and controller overhead
are all included. PP uses the ordinary episode seed stream and native ordering;
there is no deterministic candidate replay or rescue. A 900-second process fuse
prevents unbounded execution; a fuse event is a terminal failure, not a capped
TTF observation.

The Quick passes only if Hybrid preserves success against both baselines, lowers
mean raw TTF against both, wins at least half the pairs against both, does not
regress either map group by more than 10%, and does not increase repair rounds
relative to V2. Passing authorizes only a larger result-blind paired
confirmation. It does not promote HybridStructPool or establish a formal speed
claim.
