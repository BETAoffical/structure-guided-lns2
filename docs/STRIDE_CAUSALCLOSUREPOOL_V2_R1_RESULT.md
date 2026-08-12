# STRIDE CausalClosurePool v2 compactness diagnostic result

The first localized generator materialized all 78 frozen states with no
identity or execution error.  Relative to RepairClosurePool v1, candidate mean
size fell from 51.57 to 10.89 and median size fell from 54 to 4.  No agent was
admitted only because of whole-path overlap.

The frozen compactness gate nevertheless failed: 466/3,180 core-family
closures (14.65%) exceeded 64 agents, above the 10% limit.  Of these, 274 were
the broad near-temporal family.  More importantly, 32 cores exceeded 64 even
for direct conflict closure because conflicts at unrelated times were joined
through the static conflict component.

The result is retained as a failed development diagnostic.  Its gates are not
relaxed and no PP, ranker, continuation or TTF experiment follows from it.  The
separately registered r2 audit restricts direct-conflict propagation to the
event-time neighborhood of conflicts incident to the core and records an
intrinsically oversized component core once instead of repeating it across all
five families.
