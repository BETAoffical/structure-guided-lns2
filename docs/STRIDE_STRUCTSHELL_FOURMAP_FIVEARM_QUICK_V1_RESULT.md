# StructShell Four-Map Five-Arm Quick Result

## Decision

The 60-second exploratory screen completed all 20 strict-serial episodes with
no execution or identity errors. Dual16 solved all four keys and had the lowest
unweighted mean restricted TTF, but it was not consistently fastest by map.
This is a positive development signal for retaining both structural candidates,
not a promotion result.

## Scope

- Maps: Maze32, Random-high, Room64 and Warehouse w1020a opposite-exchange.
- Load: 200, 400, 600 and 600 agents respectively.
- Solver seed: 23; one paired key per map.
- Arms: Official Adaptive LNS2, V2, Component16, Hotspot16 and Dual16.
- Dual16 retained the complete V2 pool and added at most one Component16 plus
  one Hotspot16 challenger before the frozen Copeland selection.
- TTF was reset-inclusive and right-censored at 60 seconds.

## TTF result

| Map | Official LNS2 | V2 | Component16 | Hotspot16 | Dual16 | Fastest |
|---|---:|---:|---:|---:|---:|---|
| Maze32 | **1.828** | 2.115 | 6.022 | 1.965 | 2.129 | Official |
| Random-high | **12.048** | 15.896 | 28.938 | 25.329 | 23.204 | Official |
| Room64 | 51.758 | 60.000 F | **36.860** | 41.004 | 38.889 | Component16 |
| Warehouse | 11.250 | 9.559 | 11.048 | 9.005 | **8.860** | Dual16 |
| **Mean restricted TTF** | 19.221 | 21.893 | 20.717 | 19.326 | **18.270** | Dual16 |

All arms except V2 solved 4/4 keys; V2 solved 3/4. `F` is valid 60-second
right-censoring. Relative to the four-key means, Dual16 was 16.55% faster than
V2 and 4.95% faster than Official LNS2. With only one seed per map, these values
are exploratory and do not establish statistical superiority.

## Interpretation

- Keeping both candidates is technically viable. Dual16 was faster than both
  single-family arms on Random-high and Warehouse, and it rescued Room where V2
  did not finish.
- The union is not universally beneficial. On Maze it was slightly slower than
  V2 and Hotspot16; on Random-high it was substantially slower than both V2 and
  Official LNS2.
- The overall Dual16 mean advantage is driven mainly by the Room success and a
  smaller Warehouse win. Excluding Room, Dual16 is slower than V2 on the other
  three keys combined.
- Dual16 selected a structural challenger on 25/28 Maze decisions, 248/322
  Random decisions, 32/43 Room decisions and 8/11 Warehouse decisions. Thus the
  two structural candidates materially affected the action sequence; they were
  not merely generated and ignored.
- The combined pool did not simply double runtime. Its mean cumulative
  candidate-generation, selection and PP times were 1.041 s, 2.615 s and
  10.671 s per episode. The dominant map-to-map difference remained the repair
  sequence and PP success, not candidate construction alone.

The practical conclusion is to retain Dual16 as a challenger for further
testing, but not enable it unconditionally as the default. The next minimal
test should repeat the same five arms on a second paired seed, with no new
maps, features, ranker, size study or audit layer. That directly tests whether
the Room/Warehouse benefit and Random slowdown reproduce.

## Identity and runtime

- Source commit: `a5d018a`.
- Output root: `build/stride-structshell-fourmap-fivearm-quick-v1`.
- Config SHA-256:
  `cb74e6a1ef4a808efed54b7c1997ae14851d2f4f9136c02c68fea276a76eebd3`.
- Run fingerprint:
  `d5308137f37deb6f5444f96a8d989b61ba1efd7b85953460558000bf2df18f15`.
- Schedule SHA-256:
  `af118ff456f7af03604a7ec7f247ce9f969a1f5563808bd9977112625472a5d4`.
- Report SHA-256:
  `48095bb6f36a7d350f93777aec0fb44d7d31135a7ec72825e523a1a65ab06bf2`.
- Observed total wall time: approximately 8 minutes 38 seconds, including four
  reset anchors and all 20 episodes.

This was a single-seed development comparison. It carries no bootstrap gate,
formal speed claim or default-replacement permission.
