# Changelog

## v2.0.2 — 31 October 2026 (prepared locally; not yet published)

Numerical-robustness release. The prefix–suffix algebra, the interval bound,
the clique-LP comparator, the search and the experiment protocol are
unchanged. Every repair below addresses a case in which v2.0.0/v2.0.1 could
report a wrong status or an insufficiently wide numerical enclosure; each comes
with regression tests.

### Fixed (`research/compressed_interval_oracle.py`)
- Nonfinite threshold capacities are classified in exact rational arithmetic
  instead of being reported as infeasible (`5b77dcd`). This is the
  "numerical-mask" fix: an overflowing resource intermediate could previously
  mask a feasible threshold.
- Nonfinite round-off radii are handled without warnings and fail closed
  (`a915ec7`).
- Ragged subnormal rounding radii are corrected, and reported gaps are rounded
  outward (`400e9e8`).
- The accumulated enclosure radius is inflated by a factor whose sufficiency is
  proved in a separate Lean model of binary64 arithmetic (`b8a2900`).
- Ragged score-radius scaling is rounded outward (`26fe281`).
- The outward crossover selection loss is allocated to group radii
  (`6e493f3`).
- Fixed-grid overflow is reported as a numerical-domain failure, not as
  infeasibility (`e6415b3`).
- Each exact-slope fallback cache retains at most eight K-entry payloads, so
  retained slope data is O(K) rather than O(BK) (`9046986`).

### Added
- `results/release/2026-10-31-v2.0.2-release-revision/`: the complete
  serialized campaign rerun with this exact code (validation, kernel, matched
  calls, primary, robustness, stress, pricing-derived and integer
  integration), a frozen held-out primary panel with new seeds 601–605, the
  campaign protocol and a SHA-256 manifest.
- The five intermediate held-out primary confirmations run on pinned repair
  revisions (seeds 101–505), each with protocol, raw timings and a SHA-256
  manifest; the four 2 October panels also include an independent
  reconciliation of every raw timing.
- `scripts/verify_release.py` now also checks the v2.0.2 evidence (manifest,
  row counts, protocol and thread controls, commit ancestry) and prints, but
  does not require, its scientific gate outcomes.

### Not rerun
- The external robust-knapsack coefficient-transfer panel keeps its v2.0.0
  evidence in `results/release/2026-08-09-paper-b-final-r5/`.

### Scope
- Tests: 246 passed (Python 3.12.12, NumPy 2.5.1, SciPy 1.18.0).
- The numerical implementation is tested, not formally verified end to end.

## v2.0.1 — 30 August 2026
Provenance record and deterministic UCI reconstruction; no code or evidence
change.

## v2.0.0 — 28 August 2026
Paper-aligned derived analysis and figures.

## v1.0.0 — 9 August 2026
Reviewer-facing solver and frozen base evidence.
