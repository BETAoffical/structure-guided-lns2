# StructShell cross-map offline profile v1

## Scope

This is a source-only mechanism diagnostic over the frozen V2 traces from
`stride-structshell-crossmap-quick-screen-v1`. It reconstructs three verified
pre-action states per map (initial, middle and last), then generates only the
Component16 and Hotspot16 candidates. It does not load the native solver, run
PP, reset an environment, execute a controller, train a model or make a speed
claim.

The stored report is
`build/stride-structshell-crossmap-offline-profile-v1/offline_profile_report.json`
with SHA-256
`27fe2c45fd1f2e0c3ada5eb7f5006afe2be8914a432bab591ba2d24bd52aeff9`.

## Result

- Maze is the only informative Component16 point win. Component and Hotspot
  candidates differ at the initial and middle checkpoints; Component16 was
  faster in the source episode.
- Random-high is the only informative Hotspot16 point win. Component16 cuts a
  small, boundary-heavy slice from a very large conflict component and was
  repeatedly selected despite frequent rollback; Hotspot16 was selected more
  selectively and completed the source episode.
- On den020d the two fixed16 families generate the same candidate at all three
  checkpoints, so their 0.409 second source difference is not family evidence.
- Room is not an informative label because both challengers reached the
  30-second censoring limit.
- At the final sampled state all four maps produce the same Component16 and
  Hotspot16 agent set. Any future family routing hypothesis therefore concerns
  difficult early states, not permanent map-level routing.

Only one independent map supports each family. The source-count gate fails;
additional collection is not authorized by this report, Router training is
always disabled, and V2 remains the default.

Middle and last checkpoints are V2 outcome-conditioned retrospective profiles.
They are unlabelled mechanism observations, not counterfactual family labels.
Component/Hotspot outcomes are bound through the frozen quick-screen report;
only the four V2 profile traces are reconstructed and revalidated here.
