# STRIDE RepairDependencyPool v1 report

## Outcome

The platform-entry zero-solver reconstruction gate failed, so the experiment stopped before forced continuation, training, runtime integration, or TTF evaluation.

- Registered platform cases: 45.
- Observed escape classes: 36 `same_set_order_escape`, 7 `set_change_escape`, and 2 `right_censored`.
- Causal intervention annotations remain separate: 21 set, 13 order, 2 joint, and 9 residual-PP cases.
- External-blocker-backed set-change cases: 6.
- Cases with a legal RepairDependency candidate covering every required `escape added agent intersect observed external blocker`: 0/6; the frozen gate required at least 5/6.

This is a design rejection, not a solver-performance result. No new PP was run and no claim about platform rate, success, AUC, repair rounds, or TTF is made.

## What was implemented

`stride-repairdependencypool-v1` keeps the complete frozen V2 candidate pool and its V2 anchor. New candidates start from outcome-blind natural cores drawn from topology boundary, spatiotemporal hotspot, and path overlap. A core can produce only:

- `direct-boundary`: current conflict-edge blockers outside the core;
- `temporal-corridor`: direct blockers plus agents with both time overlap and reverse flow in an articulation or degree-at-most-two corridor.

Every added agent carries its pre-action conflict-edge, corridor-cell, overlap, and reverse-queue evidence. The implementation enforces at most 8 added agents, at most 32 total agents, exact-set deduplication, Jaccard at most 0.90 among additions, three-family round-robin retention, and at most 6 new candidates per state. An oversized variant is rejected whole; it is never truncated into a near-global neighborhood. There are no fixed 8/16/24/32 preferred sizes.

The transparent selector uses a new candidate only when it is the unique Pareto choice over dependency coverage, evidence density, smaller actual size, and the frozen V2 quality score. Otherwise it returns the V2 anchor. This selector chose a new candidate in 5/45 states, but that count is not an effectiveness result because continuation was correctly not run.

## Expanded witness audit

The 45 per-case records now preserve the first structural action, platform trigger, repeated candidate and agent set, persistent conflict edges and agents, strict escape additions/removals and repair-order change, plus separate observed-escape and causal-intervention classes. The first-structural and platform-trigger states include state, path, conflict-edge, and blob fingerprints.

Each case also embeds all 16 frozen `selected_native_order` causal PP trials at the platform checkpoint: seed, agent set, repair order, conflicts, rollback, failure reason, failed agent/order position, per-agent new conflict pairs, and internal/external blockers. These are existing causal diagnostics and are explicitly forbidden as online features.

The classification check was exact: all 36 same-set escapes changed repair order; 6 of the 7 set-change escapes added at least one agent observed as an external blocker in the causal audit. The historical 76.5% StructPool Top-3 coverage number is not used as a root-cause proportion or gate.

## Static gate results

All 45 states produced legal candidates, for 125 additions total. Candidate sizes ranged from 3 to 32 agents with mean 14.128. Per-map legal candidate counts were:

| Map | States | New candidates |
|---|---:|---:|
| `maze-128-128-1` | 16 | 33 |
| `maze-128-128-2` | 12 | 43 |
| `maze-32-32-4` | 17 | 49 |

Engineering constraints all passed: every added blocker has recorded evidence, the six-candidate budget and 8/32 caps were respected, all three maps were covered, and no near-global truncation occurred. The causal reconstruction requirement alone failed.

Failure decomposition for the six registered cases:

| Case prefix | Required agents | Best legal coverage | Mechanism |
|---|---:|---:|---|
| `0b91b171` | 4 | 0 | all four escape blockers absent from every natural core and supported blocker relation |
| `0bf72604` | 7 | 1 | full set occurs only inside a rejected 53-agent natural variant |
| `0bfb2e4a` | 4 | 0 | full set occurs only inside a rejected 37-agent natural variant |
| `3a1fd2ce` | 9 | 0 | two required agents absent from all cores/visible blockers; no feasible raw variant covers the set |
| `6f05f238` | 11 | 0 | all eleven required agents absent from every core and visible blocker relation |
| `c177dbfd` | 9 | 3 | four required agents absent from all cores/visible blockers; best legal candidate covers three |

Thus neither the six-candidate pruning nor the deterministic selector caused the zero: no feasible raw variant covered all required agents in any of the six cases. Two full historical escape sets can be recovered only by violating the 32-agent compactness bound, while the other four need dependencies not represented by the registered pre-action relations.

## Interpretation and next boundary

The rejected hypothesis was specific: a compact natural topology core augmented only by current boundary conflicts and low-degree reverse-queue overlap is sufficient to reconstruct most observed set-change escape witnesses. It was not sufficient on this cohort.

The result does not show that a compact entry-safe pool is impossible. It shows that the current evidence vocabulary misses longer-range or higher-order dependencies and that simply expanding closure would recreate the already rejected oversized-neighborhood failure. Per the preregistration, the correct action is to stop before expensive continuation. Any successor must be a new, separately justified design of pre-action dependency evidence—not a relaxed threshold, result-selected subset, larger closure, new ranker, or another generic future-value collection.

Canonical outputs are under `build/stride-repairdependencypool-v1/`:

- `platform_entry_witnesses.jsonl`
- `repairdependency_cohort.jsonl`
- `materialization_report.json`
- `materialization_report.md`
- `status.json`

## Verification

- WSL Python: `885 passed, 35 skipped` with 16 pytest workers.
- Linux native: CTest `11/11` passed with `-j16`.
- Windows native: `build/windows/Release/lns2_tests.exe` passed.
- Repository hygiene: passed with 1,181 tracked files, 0 untracked files in the audit scope, and all 24 retained-evidence entries verified.
- Final materialization used 16 workers, ran zero new solver jobs, and recorded run fingerprint `773b1a0684c2f60f8f501f423541f2a36c13612f71ff64055eee5f5eeaed4300`.
