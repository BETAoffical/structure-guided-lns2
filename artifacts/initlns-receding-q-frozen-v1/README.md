# Frozen receding-Q evidence

This directory preserves the final receding-Q label pilot, four-seed stability
follow-up, and quality-stability gate used before the executable research chain
was retired from the active branch.

The evidence does **not** contain a runnable controller and does not promote or
replace `v2-full`. The frozen decisions are:

- pilot: `fresh_receding_q_labels_insufficient`;
- four-seed follow-up: `four_seed_receding_q_labels_insufficient`;
- quality gate: `quality_stability_insufficient` (`diagnostic_only=true`).

`evidence_manifest.json` records the source build directories, final decisions,
coverage counts, and SHA-256 digest of every copied file. The Git tag
`backup/selector-cleanup-00-original` preserves the corresponding implementation.

