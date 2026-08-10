# STRIDE TailSwitch v1 smoke report

## Result

The one-state, four-policy native smoke passed all registered execution
integrity checks. This is an interface qualification only and is not mechanism,
TTF, or generalization evidence.

- qualification resets: 33/33;
- qualification errors/timeouts: 0/0;
- TailSwitch episodes: 4/4;
- successful episodes: 4/4;
- forced first action count: exactly one in every episode;
- invalid actions and state-fingerprint mismatches: 0;
- restored initial conflicts: 39 in every branch;
- report SHA-256:
  `934d09f135c7681b38fb7cf02f94c45de32a794a7701771f34d67e118a7d10fc`;
- qualification-report SHA-256:
  `2a8462c3d28e0e06cb957ff611c357fe82395f5cf10f33d1c22755cdb9322193`.

## Branch separation

| Policy | Repair rounds | Normalized fixed-200 AUC | Structural candidates added after step 1 | SlotPool reductions |
| --- | ---: | ---: | ---: | ---: |
| `v2-then-v2` | 47 | 0.055833 | 0 | 0 |
| `struct-then-v2` | 22 | 0.036346 | 0 | 0 |
| `v2-then-struct` | 78 | 0.069038 | 118 | 7 |
| `struct-then-struct` | 20 | 0.052372 | 78 | 5 |

The zero/nonzero structural counters confirm that the continuation factor is
implemented independently of the forced first action. The different repair
trajectories confirm that all four interventions are behaviorally distinct.
The single state is deliberately not interpreted as a causal result.

## Verification

- WSL Python: 732 passed, 35 skipped;
- Windows native tests: all passed;
- Linux CTest: 11/11 passed;
- repository hygiene: passed, 0 errors.

The next step is the frozen 66-state, 264-episode formal causal diagnostic.
