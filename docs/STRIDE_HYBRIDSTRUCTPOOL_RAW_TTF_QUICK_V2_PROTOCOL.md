# STRIDE HybridStructPool raw-TTF Quick v2 protocol

Quick v2 repeats the exact eight paired development keys and the rotating,
strictly serial three-controller order from Quick v1.  Timed episodes use one
worker and include reset, candidate generation, features, Copeland inference,
native PP, and all controller overhead.  Official Adaptive LNS2 and frozen V2
are rerun rather than copied from Quick v1.

The only challenger implementation difference is the V6 conflict-audit index
and allocation-elision optimization that passed the 152-episode engineering
gate.  Candidate membership, features, model, PP seed policy, native PP order,
and stopping rule are unchanged.

All 24 episodes must complete without execution error or process timeout.  The
same success, mean raw-TTF, paired-faster, per-group regression, and repair-round
gates remain frozen.  Failure stops automatic expansion.  Passing remains
development evidence only and authorizes at most a separately preregistered,
fresh result-blind confirmation.
