# Research repository rules

These rules apply to all work in this repository.

- Treat `papers/current/` and `results/release/` as the current paper and
  evidence release. `papers/current` must point to the highest active version.
- Preserve every binary64-distinct original deviation plus zero in the exact
  threshold set. Never tolerance-cluster exact threshold candidates.
- Never modify raw or dated released evidence in place. A changed algorithm,
  protocol, or timing policy requires a new dated result directory.
- Never hand-edit generated numerical macros, tables, figures, summaries, or
  evidence manifests. Regenerate them from the canonical result directory.
- Never describe a numerical observation as a proof or a solver-tolerance
  certificate as interval arithmetic.
- Never cite bibliographic metadata that has not been resolved through the
  primary publisher or registry. Record semantic support and exact locators in
  `research/EVIDENCE_LEDGER_V4.csv`; keep negative novelty claims qualified.
- Record discovered counterexamples, failed gates, and post-protocol changes.
  Do not weaken a theorem, test, or gate silently.
- Preserve the V2/V3 material under `papers/` as immutable provenance.
- Before declaring a release complete, run `make verify`, build every paper
  variant and the anonymous supplement, scan blind artifacts for identity, and
  render the final PDFs for visual review.
- Do not create a release tag until the exact submitted commit and upload
  package are final.
