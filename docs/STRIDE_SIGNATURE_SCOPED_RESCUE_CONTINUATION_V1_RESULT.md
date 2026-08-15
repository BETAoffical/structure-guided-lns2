# STRIDE Signature-Scoped Rescue Continuation V1 Result

## Decision

The initial paired screen completed 540/540 episodes with zero execution errors
and zero process timeouts. Integrity passed, but the preregistered initial gate
failed because signature-scoped rescue increased charged repair wall time.
Trial 4--7 extension is therefore forbidden.

This stops the repeated-new-signature rescue branch. The result is mechanism
evidence only and does not authorize runtime integration, training, candidate
pool replacement, OOD claims, or TTF claims.

## Paired initial screen

Each arm contains 180 episodes over 45 frozen first-repeat-stall states and
trial 0--3.

| Arm | Success | normalized fixed AUC | restricted mean decisions | mean repair wall | mean rescue count |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen controller | 46.11% | 0.45410 | 41.09 | 72.29 s | 0.00 |
| One-shot compact+blocker | 47.22% | 0.42304 | 40.77 | 73.09 s | 0.66 |
| Signature-scoped compact+blocker | 47.78% | 0.41906 | 39.74 | 74.30 s | 1.67 |

Relative to one-shot rescue, signature-scoped rescue changed:

- success by +0.56 percentage points;
- normalized fixed AUC by -0.00398;
- restricted mean repair decisions by -1.03;
- mean charged repair wall by +1.21 seconds (+1.66%).

There were 151 interventions after a later distinct platform signature. Thus
the new trigger was active; failure is not caused by an inert implementation.

## Uncertainty and maps

State-cluster paired bootstrap intervals for signature-scoped minus one-shot
were:

- success: -4.44 to +5.56 percentage points;
- normalized fixed AUC: -0.01974 to +0.01073;
- raw repair iterations: -2.69 to +1.07;
- repair wall: -1.36 to +3.88 seconds.

Maze-128-128-1 gained 3.13 success points but paid 3.21 additional repair-wall
seconds. Maze-128-128-2 had zero successes under all arms and generated no
later-signature rescue. Maze-32-32-4 lost 1.47 success points and paid 0.22
additional seconds. No map breached the preregistered five-point success-loss
limit, but the global wall-time gate still failed.

## Interpretation

Repeated rescue can modestly improve conflict trajectory and bounded repair
length, but the improvement is neither statistically stable nor free: each new
signature rescue adds native PP work. The platform-entry rate is identical
because all arms share the same first action at an already frozen
first-repeat-stall checkpoint; this experiment evaluates post-entry recovery,
not prevention.

The evidence does not support repeatedly rescuing every newly formed platform.
Keep the one-shot compact+blocker mechanism as diagnostic evidence only. Any
future recovery work must improve the quality of the first post-failure state
without serially purchasing small gains through additional PP calls.

## Registered artifacts and validation

- Implementation/protocol commit: `6cd6babbce337c4ac4970a7dbfc621c1acee6443`.
- Run fingerprint:
  `e7ed5d8d5c9a4333b4a73928c616fca205a0600e7343abed72ba233551406537`.
- Initial report SHA-256:
  `6927F60438D29BEE1F76C08CA13CB91E20AF54007791421616F20ADCFBE3B92D`.
- Initial run-config SHA-256:
  `BBE98DA89782FBE4F42C6EA759F51DD866FA0094B05AD3E1F776102C56E1F59D`.
- Collection-status SHA-256:
  `31D843410EF7FB4C69E080ECC9956015461FFE017C2A5B79FB95E71FE81C5E21`.
- WSL Python suite with 16 workers: 975 passed, 35 skipped.
- Linux CTest with 16-way scheduling: 13/13 passed.
- Windows native `build/windows/Release/lns2_tests.exe`: passed.
- Repository hygiene: 24/24 evidence entries verified, zero errors.
