# V4 theorem and implementation audit

Initial audit date: 1 August 2026. Adversarial re-audit: 9 August 2026.

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

That August 1 verdict was superseded by a stronger August 9 audit. The later
review found adversarial binary64 cancellation in the feasibility mask,
range sums, fixed-threshold capacities, and returned LP values; tolerance
acceptance of negative original robust certificates; structural tolerance
removal of positive hull segments; rounded-cost dominance and hull-topology
errors; a multiplier-resolution nontermination case; a signed-overflow
enclosure error; and padded/quadratic implementation paths inconsistent with
the general complexity claim.

The release candidate now uses exact-sign predicates for original robust
feasibility, exact scaled-integer fixed-threshold costs and capacities,
exact-cost dominance and hull topology, and exact rational fallback whenever
a numerical enclosure leaves a sign or comparison ambiguous. Vectorized
range accumulation carries an explicit forward-error enclosure; the optimized
rectangular kernel is enabled only when its padded size is at most twice the
ragged option count, while highly unequal menus retain the ragged path.
Adjacent-sorted preprocessing and both query branches therefore realize the
ordinary-path storage and preprocessing claims. Singleton multiplier minimization is exact
piecewise-linear arithmetic, and multiplier resolution returns a valid
coarser gap rather than raising or claiming the requested tolerance. The
public adaptive and integer entry points reject an unrepresentable aggregate
objective range with rescaling guidance. Fixed-threshold LP scalar bounds and
integer incumbent comparisons use exact binary64 rationals, so cancellation
and sub-ULP improvements cannot be turned into zero-gap certificates. Regression tests
preserve every reported counterexample, including cancellation across widely
separated exponents and distinct exact costs that collapse to one binary64
value.

A new dated evidence release is required because these are algorithmic
changes. The R2 campaign was preserved as a failed performance audit after its
first 18 matched-trace cells revealed repeated full-family exact fallbacks.
The ambiguity-set restriction and guarded query kernel retain the same
enclosures and exact comparisons, and all focused and complete tests pass.
R3 then passed validation, kernel, trace validity, and every primary gate
except the precommitted 2x geometric-mean speed target and its 1.5x bootstrap
lower bound. It is preserved as a failed performance audit. The subsequent
implementation uses exact scaled-integer fixed-LP arithmetic, accepts a
floating slope order only after exact adjacent-order verification, and uses a
rectangular exact-comparison kernel only for equal-width menus. The full 187
tests and 41,079 independent exact-enclosure comparisons pass. The final
R4 passes every fixed scientific gate: validation, kernel identity and speed,
primary tolerance/speed/interval/win/dominance/breadth, and robustness
tolerance/configuration breadth. It also preserves and reports the negative
UCI timing result and the exact-integration many-breakpoints boundary. The
complete 187-test suite and release verifier pass against the R4 evidence and
regenerated artifacts. All six manuscript variants compile; the public, blind,
companion, combined, and summary PDFs were rendered and visually reviewed with
no clipping, overlap, missing content, malformed figures, or unresolved-reference
defect. All fonts are embedded and no Type 3 font is present. The blind PDFs and
the anonymous archive pass the strict identity scan. Two independent rebuilds
of each review archive were byte-identical, and the eight-file arXiv archive
compiled from a clean extraction. These checks complete the publication-artifact
audit. The pre-R4 campaigns remain failed audit evidence rather than release
evidence.

Machine formalization was not performed; independent mathematical peer review
of the statement-to-model correspondence remains recommended.
