# V4 theorem and implementation audit

Audit date: 1 August 2026.

Reviewers: the primary Codex research orchestrator and an independent Codex
proof-review agent. The independent reviewer inspected the manuscript,
implementation, and tests without editing them, attempted to falsify the
central statements, and reported findings before seeing the fixes described
below. This is a documented independent review, not a Lean formalization or a
substitute for expert peer review.

## Scope

The review covered the complete threshold representation, minimax validity and
singleton exactness, the epigraph-dual dominance mapping, baseline
cancellation, the two-envelope representation, real-arithmetic complexity,
the Lipschitz/bracketing certificate, the adaptive global-bound invariant, and
the exact-integer interval-search integration.

## Adversarial finding and resolution

The first review found that `build_full_theta_candidates` merged deviations
within `1e-10`. A constructed 100-group, `Gamma=1` instance had distinct
breakpoints `1` and `1+9e-11`, with the latter the only feasible threshold.
Clustering removed that breakpoint and could make the exact interval solver
report infeasibility.

The release implementation now preserves every binary64-distinct absolute
deviation. The legacy tolerance argument remains API-compatible but does not
cluster candidates. A regression in `tests/test_integrated_exact_solver.py`
replays the counterexample for both envelope and clique interval searches;
both return the brute-force objective and select the retained upper
breakpoint. The near-repeat oracle test also asserts that all distinct
thresholds are present.

The review additionally identified and resolved two consistency issues:

- the manuscript now states that every option set is finite and nonempty,
  matching the model invariant; and
- the compressed adaptive routine refreshes its fixed-threshold lower bound at
  the same child-processing point as the clique routine, so the comparison
  shares the same search mechanics apart from the interval upper bound.

## Final assessment

After the fixes, the independent reviewer replayed the original failure,
checked the changed control flow, and re-ran both focused and complete test
suites. The original counterexample succeeds for both exact-search variants,
the focused suite reports 98 passes, and the complete repository suite reports
166 passes. No remaining circularity, hidden division, endpoint inconsistency,
or theorem/implementation mismatch was found in the audited results.

Verdict: theorem and implementation sign-off for release under the
manuscript's stated real-arithmetic and solver-tolerance qualifications.
Machine formalization was not performed; independent mathematical peer review
of the statement-to-model correspondence remains recommended.
