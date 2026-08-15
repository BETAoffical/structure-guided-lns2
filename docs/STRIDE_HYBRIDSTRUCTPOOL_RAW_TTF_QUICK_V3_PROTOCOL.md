# STRIDE HybridStructPool raw-TTF Quick v3 protocol

Quick v3 repeats the exact eight development keys and rotating, strictly serial
three-controller timing protocol from Quick v2. Official Adaptive LNS2, frozen
V2, and full HybridStructPool are all rerun. Timed episodes use one worker and
include reset, complete candidate generation, features, Copeland inference,
native PP, and all controller overhead.

The only challenger change is the V8 exact CausalClosure engineering
optimization that passed all 152 registered engineering episodes. Candidate
membership, neighborhood sizes, activation gate, features, V2 ranker, PP seed
policy, native order, and run-to-completion stopping rule remain unchanged.

All 24 episodes must complete without execution error or process timeout. The
same success, mean raw-TTF, paired-faster, per-group regression, and repair-round
gates remain frozen. Passing is development evidence only and authorizes at most
a fresh result-blind confirmation; failure forbids promotion or cohort tuning.
