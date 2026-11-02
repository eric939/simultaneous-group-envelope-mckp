# Changelog

## v2.1.0 — 1 November 2026

Experiment release. The prefix–suffix evaluator, the interval bound, the
clique-LP comparator and the fixed-threshold LP routine are unchanged from
v2.0.2.

### Added
- `research/scale_campaign.py`: two serialized experiments with a frozen
  protocol (`v2.1-scale-20261101-frozen`). `lp-direct` compares the complete
  LP-family search with the batched and the clique-LP interval bound against
  evaluating every fixed-threshold LP with the same LP routine (held-out
  seeds 601–605, the robustness configurations and the UCI pricing panel).
  `integer` solves the exact robust MCKP at n = 360–1440 by threshold-interval
  search with either bound, by direct threshold enumeration with LP-bound
  pruning, and by the compact Bertsimas–Sim MILP with HiGHS and with SCIP;
  every decision is re-checked in exact rational arithmetic.
- `research/fixed_theta_highs.py`: HiGHS fixed-threshold MCKP subproblems and
  the compact MILPs (HiGHS, SCIP).
- `research/integrated_exact_solver.py`: optional HiGHS subproblems and
  fixed-threshold LP bounds on single-threshold intervals (defaults
  unchanged).
- Protocol v2.1.1 replaces v2.1 before any reported result is used: every
  integer method starts cold (the rounding heuristic did not check the time
  limit), each run starts only below a one-minute load of 2, and CPU time,
  host load and search effort are recorded. Amendment v2.1.2 runs the
  integer phase in a Linux container without the gate, recording the host
  load from a sampler outside the container (`--container-host-load`).
- `results/release/2026-11-02-v2.1.1-scale/`: the reported campaign (every
  planned run, including time-outs), protocol note, environments, logs,
  container recipe and host-load samples, two stopped integer attempts, the
  non-reported pilots, and a SHA-256 manifest.
- `results/release/2026-11-01-v2.1-scale/`: the superseded v2.1 campaign,
  run on a busy machine with the heuristic start, kept unchanged.
- `tests/test_scale_campaign.py`; `scripts/verify_release.py` checks both
  campaigns, the load gate of the LP-family phase and the cold starts.

### Findings recorded in the v2.1.1 evidence
- The batched bound is faster than the clique LP on all 60 held-out
  instances (2.27 times in geometric mean); the clique LP is faster on all 9
  pricing instances. The batched root bound is never looser than the
  clique root bound on the 102 instances with a completed scan.
- A direct scan of every fixed-threshold LP is about as fast as the batched
  search with 25–26 thresholds; the batched search is faster on all 15
  instances with thousands of thresholds (45 times in geometric mean).
- In exact integer search both bounds lead to the same number of
  fixed-threshold subproblems on 44 of 45 instances, and these take a median
  of over 90% of the time, so the bound changes the time only by its own
  cost. Direct enumeration is fastest with few thresholds; the compact MILPs
  are fastest on the smallest many-breakpoints instances, and the clique-LP
  search and SCIP on the pricing panel. 178 of 225 runs certify; all certified
  objectives agree.

## v2.0.2 — 31 October 2026

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
