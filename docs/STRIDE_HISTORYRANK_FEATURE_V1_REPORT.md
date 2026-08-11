# STRIDE HistoryRank Feature V1 Report

## Status

`stride-historyrank-feature-v1` completed the preregistered 45-checkpoint audit. Integrity passed, but **none of the 13 frozen single-feature selectors passed readiness**. No model was trained, no solver was modified, and no TTF/runtime/future trajectory was read.

## Primary retained-candidate results

| Directed selection feature | Stable improvement | Both halves positive | Mean seed delta | Mean no-progress delta |
|---|---:|---:|---:|---:|
| internal conflict coverage delta | 48.89% | 53.33% | +0.06403 | -0.22500 |
| family novelty | 46.67% | 51.11% | +0.05905 | -0.18611 |
| frozen V2 score margin | 46.67% | 51.11% | +0.05905 | -0.18611 |
| component coverage mean delta | 42.22% | 46.67% | +0.02025 | -0.19861 |
| incident conflict coverage delta | 37.78% | 44.44% | +0.05176 | -0.16389 |
| absolute log size ratio | 31.11% | 37.78% | -0.01004 | -0.39444 |
| lower boundary conflict edges | 26.67% | 35.56% | -0.01115 | -0.21944 |
| changed-feature fraction | 26.67% | 33.33% | -0.01966 | -0.37639 |
| added-agent ratio | 22.22% | 31.11% | -0.02889 | -0.38750 |
| removed-agent ratio | 20.00% | 28.89% | -0.03042 | -0.34861 |
| agent novelty | 17.78% | 26.67% | -0.03178 | -0.34306 |
| lower path overlap | 15.56% | 31.11% | -0.02125 | -0.35833 |
| standardized 124D distance | 13.33% | 24.44% | -0.03442 | -0.51667 |

The readiness gates required at least 60% stable improvement, at least 60% both-half positivity, mean seed gain at least `0.02`, non-increasing no-progress rate, and positive mean gain on every map.

## Interpretation

The data rejects a simple “escape by choosing the most different neighborhood” repair. Maximum agent novelty, feature distance, and family/agent replacement reduce no-progress probability but often reduce useful repair quality; extreme novelty discards relevant conflict structure.

The best single signal was increased internal-conflict coverage. It produced the largest mean gain (`+0.06403`) and stable improvement in `22/45` checkpoints, but still failed the two stability gates. The next-V2 fallback and family novelty chose the same action pattern and remained at `21/45`.

Therefore the repeated long-tail mechanism is not explained by one monotonic feature. A good post-no-op choice must preserve enough conflict/component coverage while changing the failed neighborhood in a controlled way. This is a conditional multi-feature ranking problem, not a global novelty penalty or a fixed family/size rule.

## Decision and next step

No single-feature rule is eligible for runtime or TTF evaluation. The next safe step is to preregister a fresh-map HistoryRank source cohort, collect the same strictly paired one-step labels without result-based filtering, and train a small grouped ranker using map-disjoint nested validation. The repeated Maze outcomes may be used only to define the frozen feature schema and final external regression; they must not be reused as training examples or threshold-tuning data.

## Artifact identity

- Configuration SHA-256: `b50be709330de7df298452b0580196859f29d4d3f1b3b16a569a2b069d07c23c`
- Row manifest SHA-256: `147508161420621bf1beb91e40297b9a5deaee3e5e56ddcd73fc4ea15a325766`
- Report SHA-256: `42b44e71e157b0e5e5fd4835ab4d9f94a232999300046effdca1204ce29b369e`
- Status SHA-256: `c9491b7077106fa49ab88baffc19a3d0032793d3b4ffc1bde493f4f1a9cc2da2`
