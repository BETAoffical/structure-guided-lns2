# STRIDE-LNS Stage 2 stability gate

Stage 2 produced valid four-seed labels, but the preregistered seed-stability
gate did not pass. Stage 3 expansion and model training therefore remain
blocked.

## Protocol

- Select 48 states (20 percent of the 240-state Pilot) without reading repair
  outcomes.
- Use 24 official-Adaptive and 24 `v2-full` source states, at most one state per
  episode.
- Cover all 14 maps that produced conflicts. The V2 half contains eight early,
  eight middle, and eight late states.
- Add PP trial indices 4 through 7 for every candidate. Trial indices 0 through
  3 retain the original seed namespace, so each candidate has two independent
  four-seed halves.
- Compare dominance direction only on pairs that are decisive in at least one
  half. Pairs undecided in both halves cannot inflate agreement.
- Compare ranked Top-3 sets using intersection divided by three. This is more
  permissive than Top-3 Jaccard and therefore does not exaggerate failure.
- Gates: pairwise consistency at least 70 percent and mean Top-3 overlap at
  least 80 percent.

## Result

- 48/48 extension states completed with zero collection errors.
- 845 candidates and 6,760 total outcomes across both halves.
- 6,256 decisive-pair union; 4,580 directions agreed.
- pairwise consistency: `0.7320971867` -- pass.
- mean Top-3 overlap: `0.7361111111` -- fail.
- Top-3 overlap distribution: 16 states at 1.0, 26 at 2/3, and 6 at 1/3.
- 29/48 states individually reached 0.70 pairwise consistency; 19 did not.
- mean per-state pairwise consistency was `0.7771` for official-Adaptive source
  states and `0.6776` for V2 source states.

Artifacts:

- stability selection SHA-256:
  `487bb4f14627155431f34a70b2dacaf2990a480fd04be3e4574fbb32fa47fcd5`.
- extra-trial SHA-256:
  `318caa489ca0734d7a8cc35d8eac4b8958bfeebd0429e2ee36a2a1ff03382de3`.
- deterministic report SHA-256:
  `19ad148895a17af849195c1b04c04d2418b671aacdf31c3a5cc0f248d684be27`.

## Decision

`stride-quality-v1` is not trained or promoted, and the 600-state Stage 3
collection is not started. The current four-seed robust label can identify many
pairwise preferences, but it does not define a sufficiently stable Top-3 set.

Changing the label, increasing training seeds from four to eight, or switching
from a unique ranker to a set-valued selector would change the registered
research question. Any such change requires a new label version and a fresh
held-out stability cohort; the current extra four seeds must not be reused both
to redesign and to claim confirmation.
