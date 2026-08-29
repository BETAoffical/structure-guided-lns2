# Warehouse Disruption Recovery Load Extension V1

## Outcome

The extension generated six byte-distinct, history-disjoint station-centric maps
and 18 fixed tasks. All 36 authenticated disturbed-state checkpoints passed the
16 conflict-pair / 32 active-agent / 16 largest-component gate, covering all 18
map-by-load cells. Official LNS2 and frozen Dual16 then completed all 72
checkpoint-restore-inclusive timed episodes in a rotating strict-serial schedule.
All episodes succeeded and all identity, clock and integrity gates passed.

| Load band | Density | Pairs | Dual16 wins | Official mean / median TTF | Dual16 mean / median TTF | Mean TTF improvement | Success/hour improvement | Map-cluster 95% CI | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| medium_high | 0.100 | 12 | 10 (83.3%) | 0.428 / 0.384 s | 0.397 / 0.367 s | +7.3% | +7.5% | [-37.2%, +31.2%] | Fail |
| high | 0.125 | 12 | 8 (66.7%) | 0.564 / 0.557 s | 0.647 / 0.444 s | -14.9% | -17.6% | [-70.1%, +30.0%] | Fail |
| very_high | 0.150 | 12 | 9 (75.0%) | 1.213 / 0.909 s | 0.758 / 0.570 s | +37.5% | +56.4% | [+13.8%, +50.0%] | Pass |

The preregistered band gate required at least 70% paired wins, 15% mean capped-TTF
improvement, 20% success/hour improvement, no success-rate loss and no additional
timeout/censor. Only `very_high` passed. There, Dual16 reduced mean repair
iterations from 38.75 to 8.42 and mean native repair time from 0.885 s to 0.264 s.
At lower loads, the repair problem is too short for the structural controller's
fixed work to yield a large reliable end-to-end gain; the `high` mean was also
hurt by a Dual16 tail episode despite a faster median.

The first collection attempt stopped safely after 48 episodes when one task's
ordinary reset was already feasible. The runner was corrected to qualify each
timed lane against the authenticated disturbed checkpoint rather than that
unrelated ordinary reset. The 48 completed timed results were retained unchanged;
the anchor was rebound to all 36 checkpoint fingerprints, conflict summaries and
identity hashes, and the remaining suffix resumed. Final pairing and initial-state
identity checks passed for 72/72 episodes.

## Decision

Route only the fresh `very_high` station-centric warehouse band to Dual16 in this
extension. Do not expand the existing targeted expert to medium-high/high and do
not claim global/default replacement. Confirm the 0.15-density boundary with a
second fresh map-disjoint cohort before treating it as a stable deployment rule.

Evidence: `build/stride-warehouse-disruption-recovery-load-extension-v1/ttf/load_extension_ttf_report.json`.

