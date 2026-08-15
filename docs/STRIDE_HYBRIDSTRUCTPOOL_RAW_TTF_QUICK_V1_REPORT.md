# STRIDE HybridStructPool raw-TTF Quick v1 report

## Result

The registered eight-key, three-controller Quick completed all 24 timed
episodes with zero execution errors, process timeouts, invalid actions, or
fingerprint mismatches. All controllers solved all eight episodes. The
reset-inclusive raw-TTF clock was used throughout and no capped TTF value was
analyzed.

| Controller | Success | Mean raw TTF (s) | Median raw TTF (s) | Mean repairs | Mean PP (s) | Mean selection (s) | Normalized wall AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Official Adaptive LNS2 | 8/8 | 13.6621 | 12.0588 | 57.500 | 5.5435 | 0.0042 | 0.67679 |
| Frozen V2 | 8/8 | 11.4024 | 10.0297 | 20.375 | 3.4161 | 0.4896 | 0.74495 |
| Full engineered HybridStructPool | 8/8 | 12.3253 | 9.5570 | 8.750 | 3.5086 | 1.4339 | 0.78003 |

HybridStructPool was 9.7848% faster than official Adaptive LNS2 and faster on
6/8 paired keys. It reduced mean repair decisions by 48.75. Against frozen V2,
however, HybridStructPool was 8.0938% slower and faster on only 3/8 keys, despite
using 11.625 fewer repair decisions.

The overhead explanation is direct. Hybrid spent 1.1791 seconds per episode in
candidate generation and 1.4339 seconds in total neighborhood selection,
compared with 0.1766 and 0.4896 seconds for V2. The extra 0.9443 seconds of
selection time closely matches the 0.9229-second raw-TTF loss against V2. Full
candidate coverage improves the number of repair decisions, but exact online
generation and scoring still consume the saved repair time.

## Map groups

- Maze-300: Hybrid was 0.7076% slower than official Adaptive and 12.8805%
  slower than V2.
- Room-500: Hybrid was 21.2232% faster than official Adaptive, but 2.0621%
  slower than V2.

The Quick therefore passes integrity and the direct official-LNS2 comparison,
but fails the frozen overall promotion gate because it does not beat V2, loses
the V2 paired-faster gate, and exceeds the registered per-group regression
limit. Its normalized wall-clock conflict AUC is also worse than both baselines.

## Decision boundary

This is positive development evidence that the complete engineered pool can
beat official Adaptive LNS2 on the tested high-load cohort. It is not a formal
speed claim: the cohort has only eight existing development keys and is not a
fresh result-blind sample. HybridStructPool is not promoted as the default pool,
and the failed promotion gates stop automatic expansion on this cohort.

Further material speed improvement would require a separately validated
cross-decision incremental causal index or a result-blind routing/selection
mechanism that avoids paying the full Hybrid cost when V2 is sufficient. Simple
candidate deletion remains disallowed because it previously changed Copeland
scores and degraded quality.
