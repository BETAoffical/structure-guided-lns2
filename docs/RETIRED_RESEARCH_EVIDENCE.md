# Retired research evidence

The active branch keeps runnable code only for the official Adaptive baseline,
`v2-full`, `mixed-full-v2`, and `v3-s3`. Research chains listed here remain
available through frozen evidence and Git history, but are not controller
implementations and are not eligible for online selection.

## Value labels

- `artifacts/initlns-v3-value-label-pilot-v1`: cost-to-go labels looked
  promising in the initial pilot.
- `artifacts/initlns-v3-value-stability-v1`: label stability was insufficient
  for a value-model pilot.
- `artifacts/initlns-v3-value-uncertainty-audit-v1`: the distributional signal
  was insufficient.

The value chain produced diagnostic evidence, not a promoted controller.

## Receding-Q labels

`artifacts/initlns-receding-q-frozen-v1` preserves the current-schema pilot,
four-seed follow-up, and quality gate. The final decisions were respectively
`fresh_receding_q_labels_insufficient`,
`four_seed_receding_q_labels_insufficient`, and
`quality_stability_insufficient`.

The frozen pilot covers 12 states, 215 candidates, and 430 rollouts. The
quality gate is explicitly diagnostic-only. The implementation is recoverable
from Git tag `backup/selector-cleanup-00-original`.

