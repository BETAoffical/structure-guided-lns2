# StructShell Cross-Map Quick Screen V1 Result

## Decision

The 30-second development quick screen completed all 12 registered timed
episodes with no execution, timeout-process, action, fingerprint, or identity
errors. No fixed-size StructShell family passed the registered screen, so the
experiment stops without a second seed, size study, router, Official Adaptive
comparison, or promotion claim. V2 remains the default.

The registered decision is:

`stop_no_fixed16_family_passed_quick_screen`

## Scope

- Maps: `room-64-64-16`, `maze-32-32-4`, `den020d`, and
  `random-32-32-20-high-load`.
- Solver seed: `17`.
- Arms: V2, Component16, and Hotspot16.
- Each StructShell arm retained the complete V2 pool and added at most one
  size-16 family challenger. The frozen V2 Copeland ranker made the final
  selection.
- TTF was reset-inclusive and right-censored at 30 seconds.
- This was development triage on previously known maps/tasks, not a fresh
  confirmation or a cross-map generalization test.

## Result

| Map | V2 | Component16 | Hotspot16 |
|---|---:|---:|---:|
| Room | 30.000 F | 30.000 F | 30.000 F |
| Maze32 | 2.911 S | 2.133 S | 4.659 S |
| den020d | 15.022 S | 15.163 S | 14.753 S |
| Random-high | 13.493 S | 30.000 F | 9.921 S |
| **Mean restricted TTF** | **15.357** | **19.324** | **14.833** |

`S` means feasible within 30 seconds; `F` means valid right-censoring at 30
seconds.

- V2: 3/4 successes, mean restricted TTF `15.3565 s`.
- Component16: 2/4 successes, `19.3239 s`; it was 25.84% slower overall and
  failed on Random-high.
- Hotspot16: 3/4 successes, `14.8334 s`; it was 3.41% faster overall and was
  non-worse on 3/4 pairs, but missed the preregistered 5% improvement gate.

The map-level crossover is material: Component16 helped Maze32 but failed on
Random-high, while Hotspot16 helped Random-high but slowed Maze32. One seed per
map is insufficient to turn this crossover into a state router rule.

## Time and integrity

The complete qualification and timed screen took about five minutes of wall
time. It used four reset-only qualifications followed by 12 strict-serial timed
episodes. The run completed 12/12 schedule entries and passed all pairing and
qualification-anchor checks.

- Source commit: `4244170`.
- Output root: `build/stride-structshell-crossmap-quick-screen-v1`.
- Config SHA-256:
  `fd224ea528796eb1cc86703d52e8e86c2dae0bc11cb48e992d705e6071d61df7`.
- Run fingerprint:
  `d768e8d95e9a5d7dde389fbebf4c491fbe935f82e8eb6712d85ecded51fb9ae7`.
- Report SHA-256:
  `ce043d19dce818cc6adea896b64507db25d5117fcf5b650b62e073fa31d43567`.

## Boundary

Do not add a second seed merely to rescue Hotspot16 after the failed registered
gate. If this direction is resumed, it needs a separately frozen experiment
that tests an action-preconditioned Component-versus-Hotspot hypothesis on new
keys. The present result does not justify a global family, a map-name router,
or replacing V2.
