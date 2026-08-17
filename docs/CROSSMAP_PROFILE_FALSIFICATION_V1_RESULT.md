# Cross-Map StructShell Profile Falsification V1 Result

## Decision

The preregistered Component16-versus-Hotspot16 map-profile hypothesis was
falsified on the second paired key. The run stopped immediately, executed six
timed episodes, and cancelled the remaining six seed-20 episodes.

The registered decision is:

`stop_profile_hypothesis_falsified`

This result does not allow a larger test, Router training, a formal speed
claim, or replacement of V2. V2 remains the default.

## Scope

- Maps/tasks: `maze-32-32-4` random-04 with 200 agents and
  `random-32-32-20` random-01 with 400 agents.
- Solver seeds: 19 and 20 were preregistered; seed 20 was not run after the
  first failed key.
- Arms: V2, Component16, and Hotspot16.
- Each StructShell arm retained the complete V2 pool, added at most one
  size-16 challenger, and used the frozen V2 Copeland selector.
- TTF was reset-inclusive and right-censored at 15 seconds.
- This was a minimal development falsification, not a fresh-map confirmation
  or a cross-map generalization test.

## Result

| Paired key | V2 | Component16 | Hotspot16 | Registered expectation | Gate |
|---|---:|---:|---:|---|---|
| Maze32, seed 19 | 15.000 F | 2.773 S | 4.855 S | Component16 | Pass |
| Random-high, seed 19 | 15.000 F | 14.716 S | 15.000 F | Hotspot16 | Fail |

`S` means feasible within 15 seconds. `F` is valid right-censoring at 15
seconds.

Maze32 reproduced the Component16 advantage. Random-high did not reproduce the
Hotspot16 advantage observed at seed 17: Component16 solved the state, while
Hotspot16 and V2 did not. Hotspot16 therefore failed both registered gates on
the second key: it was not strictly fastest and its success was inferior to
Component16.

The aggregate values over these two diagnostic keys were Component16 2/2 at
8.744 seconds, Hotspot16 1/2 at 9.927 seconds, and V2 0/2 at 15 seconds. These
two-key aggregates are not a promotion result and do not establish that
Component16 is generally better.

## Mechanism from the saved traces

The Random-high reversal was driven by neighborhood repairability and exposure,
not by StructShell generation cost.

| Random-high seed 19 | Repairs | Struct selected | Struct positive | Struct rollback | PP | Selection | Final conflicts |
|---|---:|---:|---:|---:|---:|---:|---:|
| Component16 | 218 | 120 | 41 | 70 | 7.116 s | 4.844 s | 0 |
| Hotspot16 | 186 | 151 | 37 | 106 | 8.301 s | 4.273 s | 10 |
| V2 | 256 | 0 | - | - | 8.775 s | 3.174 s | 2 |

- Hotspot16 selected its structural challenger on 81.2% of decisions, versus
  55.0% for Component16.
- Once conflicts were at most 50, Hotspot16 selected StructShell on 107/109
  decisions; 88/107 rolled back. At conflicts at most 10, all ten decisions
  selected StructShell, nine rolled back, and none reduced conflicts.
- Hotspot16's 35 V2-base actions removed a net 143 conflicts, but repeated
  structural selections starved that effective base pool.
- Almost all structural rollbacks were `conflict_bound_exceeded`; the failed
  Hotspot16 episode had only one `time_limit`. This was not a timeout or
  rollback implementation fault.
- Candidate generation and per-decision selection costs were similar across
  the two families. The decisive difference was repeated selection of
  low-repairability structural neighborhoods and the PP time spent on
  zero-progress repairs.

The seed-17 failure of Component16 had the same tail pattern in the opposite
family: at conflicts at most 50, 85.5% of decisions selected Component16 and
92.2% of those structural actions rolled back. Thus the stable signal is not a
map-level Component-versus-Hotspot label. It is the need to abstain from, or
close, StructShell when its current candidate is unlikely to repair the
remaining conflict state.

Static initial coverage and one-step reduction are insufficient routing
signals. The initial proposal audit favored Component16 in both Random-high
seeds despite different winners, and Hotspot16 had the better first step at
seed 19 despite ultimately failing.

## Integrity and timing

The run finished in approximately 2 minutes 14 seconds. It completed two
reset-only qualifications and six strict-serial timed episodes, then applied
the registered first-failure stop. Six seed-20 schedule entries were cancelled
and no seed-20 episode artifact exists.

- Source commit: `baa86df995839bbd6fbcdb03ad78eb15b22d0a09`.
- Output root: `build/crossmap-profile-falsification-v1-r2`.
- Config SHA-256:
  `50baf0f580723991856a76be30d48d43d3853b3b9e5e01f23021fabe4937952d`.
- Run fingerprint:
  `884ff8b09e327b5d90f33204019a2f5adf65a0bef65d3be220c706068548a1a8`.
- Schedule SHA-256:
  `6304b4ff1e382b72f830a931e8a1d7516ac892fb04ef750ddfde6810dab8e8c0`.
- Native SHA-256:
  `c98ae60b05b1db2f75bccf60c518e409417f3c990ff3fc5437eca47ddbc105a5`.
- Freshness-audit SHA-256:
  `3ac5f0d65ae26caabf2419d10cfc1f32759e8df9edc2513f230b7b311c7b2dd6`.
- Report SHA-256:
  `bc6c7c11bd41a15d6ed72b547e932c6ec5ea251535744b18c9ffdf029cfb37f4`.

All six real traces passed run-configuration, controller-augmentation,
manifest, byte/hash, event-chain, summary, and initial/final fingerprint
validation. There were zero execution, process-timeout, illegal-action,
fingerprint, or identity errors. The complete six-trace checksum list is the
report's `trace_integrity` array; the compact `inputs.trace_sha256_by_episode`
map is not a complete list because controller lanes share episode IDs.

## Next decision

Do not run seed 20 under this identity and do not tune the rejected map-profile
rule. The next admissible hypothesis is a separate, bounded abstention study:
keep V2 as the default and close the single structural challenger when
pre-action evidence predicts low repairability or a low-conflict tail. The
present two seeds are sufficient to motivate that hypothesis, but not to set
or validate its threshold. No additional solver run is authorized by this
result.
