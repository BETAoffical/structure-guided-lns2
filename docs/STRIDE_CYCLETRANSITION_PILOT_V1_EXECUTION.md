# STRIDE CycleTransition Pilot v1 Execution Record

The result-blind preparation ran from preregistration commit
`38c5fecf1b140ead9d2c2307e3ef97e50b297e4c` with 16 workers. It completed
without errors or timeouts before any new PP outcome was generated.

Frozen preparation identity:

- 18 unique first-structural states: six on each of the three registered Maze
  maps;
- 679 exact-deduplicated candidates: 319 complete V2 base candidates and 360
  raw equal-size structural candidates;
- per-state total candidate range 29-42 and structural range 11-24;
- all 18 historical source-selected structural actions map exactly into the
  regenerated full pool;
- all four nominal sizes 8, 16, 24, and 32 occur after exact-set deduplication;
- no repair outcome or classification field was read during cohort selection
  or preparation.

Artifact hashes:

- `cohort_manifest.jsonl`:
  `5c246607a3ba5dfa9ff166b930e9c36da11e5bf28a8338b32f052ae58b06bec0`;
- `preparation_report.json`:
  `975b547b410a6d6ed7dd4bf3bde0f2760869029691abb96cf6921cc2929d1dff`.

The next bounded step is 5,432 independent state-candidate-seed jobs, using 16
workers and a 300-second hard limit per job. Any native error or timeout stops
the collection. Passing the audit can only authorize a separately
preregistered grouped predictability audit; it cannot authorize training,
runtime integration, TTF testing, or a long-tail avoidance claim.
