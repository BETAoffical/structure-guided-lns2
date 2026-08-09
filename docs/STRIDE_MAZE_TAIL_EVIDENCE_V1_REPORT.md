# STRIDE Maze Tail Evidence v1 Reset-Only Report

## Result

The preregistered reset-only qualification completed successfully.  All 192
registered resets completed with zero errors and zero timeouts.  The analysis
passed every outcome-blind cohort gate and did not read controller, candidate
repair, future trajectory, or TTF outcomes.

This milestone establishes a frozen independent Maze cohort for the next
three-controller full-episode block.  It does not establish tail incidence,
TTF improvement, fresh-map generalization, or permission to train a predictor.

## Registered product

- 4 checksum-pinned MovingAI Maze layouts;
- 64 generated static OD tasks;
- task seeds `811` and `853`;
- solver seeds `17`, `29`, and `43`;
- `uniform_random` and `opposite_exchange` task families;
- 192 complete reset jobs;
- zero errors and zero timeouts.

The wide-corridor `maze-128-128-10` layout produced no eligible conflict band
and remains in the evidence as the preregistered negative control.  It was not
replaced after observing the reset results.

## Frozen selected cohort

The deterministic reset-only rule selected 11 tasks from 3 maps, forming 33
solver keys.  Both conflict bands, both task seeds, and both task variants are
represented.

| Map | Variant | Band | Task seed | Agents | Initial conflicts across solver seeds |
| --- | --- | --- | ---: | ---: | --- |
| `maze-128-128-1` | opposite exchange | high | 853 | 120 | 149--208, mean 187.00 |
| `maze-128-128-1` | opposite exchange | moderate | 811 | 80 | 39--79, mean 53.00 |
| `maze-128-128-1` | uniform random | high | 853 | 120 | 160--180, mean 170.67 |
| `maze-128-128-1` | uniform random | moderate | 853 | 80 | 42--51, mean 48.00 |
| `maze-128-128-2` | opposite exchange | high | 853 | 600 | 158--314, mean 246.00 |
| `maze-128-128-2` | opposite exchange | moderate | 853 | 400 | 43--59, mean 49.00 |
| `maze-128-128-2` | uniform random | moderate | 853 | 500 | 38--69, mean 49.00 |
| `maze-32-32-4` | opposite exchange | high | 811 | 160 | 160--180, mean 168.33 |
| `maze-32-32-4` | opposite exchange | moderate | 853 | 120 | 30--45, mean 37.33 |
| `maze-32-32-4` | uniform random | high | 811 | 200 | 147--261, mean 219.00 |
| `maze-32-32-4` | uniform random | moderate | 853 | 160 | 68--76, mean 73.33 |

## Integrity and hashes

- candidate dataset manifest: `d24498bebc6a569996dc310b43721a631be64035ff0c09beb2fba156e67a7053`;
- candidate dataset summary: `b75058948bbe9829e71abd375cebab0432cc10992b69836ff3d132933b7cfe2f`;
- qualification manifest: `8db87c81b069691eacbc889e0e613d5ab55f29002b2ebf906bba64029b3b438d`;
- qualification report: `be42fbf4dce812b60c77959d44e986c95c1687d5fb6fb6299485b223e098dc46`;
- qualification run config: `dfab457a2832b5f0a5fb27c63296b3c95e173d3d38f3cfda8aea8f8d0eca18f6`;
- selected cohort: `555479deba9c85b700379f936191d5ceba2237b26ba3b07156ea5c961e32c436`;
- reset task summaries: `c4f50917fd0bedddb6e027b2548881ed02e0c25a56cf82da83ba4a8d0eaf02b4`;
- preflight report: `e376ed2b54e2d8374a583217d12bcba130dacadc32d62afe3594a7ac06e3f5cf`.

## Next evidence block

The selected 33 keys are frozen before running any controller.  Each key will
run `v2-full`, `v2-plus-structpool`, and `v2-plus-slotpool` to completion under
strict rotating order and deterministic PP replay.  No key may be removed
after observing a controller result.  Speed metrics are descriptive in this
block; the primary purpose is to measure independent harmful-tail incidence
and identify first-divergence actions for the later 16-seed paired replay.
