# STRIDE-LNS Stage 3 data expansion protocol

Stage 3 is enabled by the independent `stride-quality-v2` label-stability
confirmation. It expands label coverage without changing candidate generation,
features, PP, or the registered quality score.

## Target cohort

- 600 states total: 300 official-Adaptive source states and 300 frozen-V2
  source states;
- retain the result-blind Stage 2 cohort of 240 states and add 360 states from
  new source episodes;
- at least 24 conflict-producing maps in the labelled cohort;
- source decision indices remain restricted to 0 through 11;
- the 360-state extension uses at most one state per source episode;
- retain low/mid (100/200 agents) and high (400/600 agents) strata, source
  policy, decision stage, conflict band, map, and layout metadata.

The existing labelled cohort covers 14 conflict-producing maps. Ten new
generated maps are added so the labelled target can cover 24 active maps. The
development inventory therefore contains 25 maps in total, including the
historical `den520d` map that produced no conflicting Stage 2 state.

## New maps and tasks

The extension uses master seed `2026080203`, ten 72x88 maps, and eight tasks per
map. Layout counts are three regular beltway, three compartmentalized, and four
dead-end-aisle maps. The dimensions, seed, jitter, shelf ranges, gate count, and
dead-end geometry differ from Stage 2. Agent counts remain 100, 200, 400, and
600, with balanced, bottleneck-pressure, uniform, and swap-heavy task modes.

All 12 formal MovingAI/OOD map IDs remain excluded. The six MovingAI
development maps from Stage 2 remain available for training. Because `den520d`
produced no conflicting state, the 24-active-map target consists of 19
generated maps and five MovingAI maps; the inventory still contains all six
MovingAI development maps.

## Source and repair collection

- source solver seeds: 101, 202, and 303;
- source policies: official Adaptive and frozen `v2-full`;
- source episode budget: 60 seconds and at most 12 eligible decisions;
- every selected state must pass three independent replay/candidate preflights;
- candidate pool: the unchanged deduplicated full pool of up to 18 candidates;
- label trials: eight deterministic paired PP seeds per candidate;
- label: `lns2.stride.quality_label.v2`; runtime remains excluded.

The first 240 states are reused rather than recomputed. Ninety-six already have
indices 0--7. The other 144 receive only missing indices 4--7. New states first
receive indices 0--3 and then 4--7. Reuse is accepted only when state,
candidate, seed-cartesian coverage, producer identity, and fingerprints match.

## Data gates

Stage 4 training is allowed only if all of the following hold:

- exactly 600 unique states and 300 per source policy;
- at least 24 labelled maps and all ten new maps represented;
- preflight failures are zero after excluding and reselecting unstable states;
- repair collection errors are zero;
- every retained candidate has exactly trial indices 0--7 with distinct paired
  seeds and complete post-repair structure;
- every state has pairwise labels and total oriented sample weight one;
- formal MovingAI/OOD map overlap is zero.

Generated artifacts remain ignored under `build/`; configs, code, hashes, gate
reports, and research decisions are committed.
