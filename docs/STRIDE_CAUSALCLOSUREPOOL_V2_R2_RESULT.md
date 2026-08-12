# STRIDE CausalClosurePool v2 r2 compactness diagnostic result

The event-local direct-conflict revision again materialized all 78 states with
no execution or identity error.  Median candidate size remained 4 and mean
size was 12.27 versus 51.57 in RepairClosurePool v1.  Whole-path-only admission
remained zero.

The oversized-family rate improved from 14.65% to 11.19%, but still exceeded
the frozen 10% gate.  257/348 remaining oversized closures came from the broad
near-temporal family.  The cause was aggregation across every event time
incident to a core: individually local edges still formed a global temporal
union before closure.

The r2 result remains failed.  No threshold was relaxed and no PP, ranking,
continuation or TTF run follows from it.  The separately registered r3 audit
creates one closure per conflict-event time slice so unrelated time windows
cannot connect solely through a shared core.
