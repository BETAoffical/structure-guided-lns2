# STRIDE SafeSlot Gate v1 Result

## Outcome

The preregistered offline gate failed and is stopped before fresh-map or
runtime evaluation. All 587 eligible challenger rows, 98 states, and 16 maps
were retained. Four outer whole-map folds and three inner whole-map folds were
used exactly as registered. No model artifact was exported.

The candidate classifier contains a weak cross-map signal:

- weighted ROC-AUC: `0.6650`, above the `0.65` gate;
- average precision: `0.4719`, versus weighted positive prevalence `0.3418`;
- weighted accuracy: `0.6684`;
- majority accuracy: `0.6582`;
- accuracy gain: `0.0102`, below the required `0.03`.

This signal is not sufficiently stable for action selection. None of the four
outer training folds found an inner threshold that simultaneously met the 80%
safe-replacement precision, 10% state coverage, map coverage, and registered
current-step/residual safeguards. Every fold therefore selected the `1.01`
sentinel and abstained to exact V2 on all 98 outer validation states.

Outer held-out candidate ROC-AUC values were `0.7236`, `0.6907`, `0.6920`, and
`0.4294`. The last fold contains `ht_bartrand_n`, `lak526d`, and
`w_encounter3`; its below-random result is direct evidence that the current
representation does not transfer uniformly across maps.

## Threshold diagnostic

The following pooled outer-OOF table is retrospective diagnosis only. It was
not used to select or promote a threshold:

| Threshold | Selected states | Coverage | Selected maps | Safe precision |
|---:|---:|---:|---:|---:|
| 0.70 | 18 | 18.37% | 9 | 61.11% |
| 0.75 | 14 | 14.29% | 7 | 71.43% |
| 0.80 | 5 | 5.10% | 5 | 60.00% |
| 0.85 | 2 | 2.04% | 2 | 50.00% |
| 0.90 | 0 | 0% | 0 | 0% |
| 0.95 | 0 | 0% | 0 | 0% |

No registered threshold approaches 80% precision while retaining the required
coverage. Lowering the precision gate or choosing 0.75 after observing this
table would be outer-fold tuning and is prohibited.

## Interpretation

The failure is not an absence of useful structural candidates: the frozen
label audit found 201 residual-qualified positives over 64 states and all 16
maps. The failure is recognizing those candidates from current pre-action
features on unseen maps.

Likely contributors are:

- only 98 independent state units and 16 independent map units; the 587
  candidate rows are strongly correlated within states;
- substantial map-dependent behavior, exposed by the fourth outer fold;
- the final label is a conjunction of immediate repair advantage, two-seed-
  half stability, no-progress safety, and residual-structure safety;
- the current 124-dimensional aggregate difference plus SlotPool descriptors
  may omit local temporal-conflict motifs and candidate-anchor path
  interactions that determine whether PP leaves an easier state.

This does not justify adding future repair rounds, TTF, or post-repair fields to
runtime input. Those fields remain unavailable before the action.

## Decision

- `v2-full` remains the exact default and fallback.
- `stride-safeslot-gate-v1` is not promoted.
- Fresh-map label confirmation, runtime integration, Maze regression, and TTF
  evaluation are not authorized.
- The same outer labels may be used for failure analysis but not for threshold
  or hyperparameter retuning.
- Any successor requires a new preregistration and either genuinely new
  independent maps/states or a separately motivated pre-action representation.

A sensible successor study would separate current-step benefit prediction from
residual-hazard prediction and add lightweight local conflict-subgraph and
temporal-path interaction features. It must validate on new whole maps and
retain the same exact-V2 abstention rule.

Artifact SHA-256 values:

- training report: `3f4eb46330e542a7ac935b584f380e8961e6e1abc2bbaedc7d1146abf6203af4`;
- training rows: `e89bf671b6cdefc048d4abc553e0952c218690a773f5ac1cd086757da4589012`;
- outer-OOF predictions: `4dabcee0c89aa29a37a13706dc330211b0d3dfa1b3fb279b07cd484e7ecc61a8`;
- outer-OOF policy: `20d5bce56159c8e0a653a4930fd483a649fe79093cc0550f5429def918d21a78`;
- fold diagnostics: `05a23b46fd1701550aef737784257a505c7190c0efacf8a5a4bc350a49ddc81e`.

The final producer identity records CPython 3.12.3, NumPy 1.26.4, and
scikit-learn 1.5.0. A WSL launch attempt stopped before the first fit because
that interpreter lacks scikit-learn; it produced no model outcome.
