# Revision History and Scientific Guardrails

This branch records only durable facts for the independent v4 publication
project. The separate v3 manuscript, experiments, and history remain on the
`v3` branch; v4 is neither a revision of v3 nor a replacement for it.

## Certified-accumulation performance gates and R4 rerun

Date: 2026-08-09.

- The first post-correction campaign in
  `results/release/2026-08-09-paper-b-final/` passed its numerical-identity
  gate but reached only 2.7174x geometric-mean total kernel speedup for
  `n >= 360`, below the protocol-fixed 3x gate. The campaign was stopped during
  the primary phase and this incomplete failed run is preserved unchanged.
- No gate was weakened. The certified range accumulator was refactored to use
  vectorized prefix sums with an explicit forward-error enclosure; exact
  rational fallbacks remain responsible for ambiguous feasibility and value
  comparisons.
- The same adversarial review exposed rounded-cost dominance, sub-epsilon hull
  segments, cancellation in fixed-threshold LP objectives, and objective
  improvements smaller than one aggregate binary64 unit. Fixed-threshold
  feasibility, hull topology, LP scalar bounds, and incumbent comparisons now
  use exact scaled integers or exact binary64 rationals; floating LP solutions
  remain branching guides only. Inputs whose aggregate objective range is not
  representable are rejected with rescaling guidance.
- The R2 validation and kernel phases then passed, including an 18.49x
  construction-plus-nine-query kernel speedup. The matched-trace phase exposed
  a different bottleneck: exact ambiguity checks rescanned the full threshold
  family at overlapping multiplier comparisons. The first 18 trace cells were
  all slower than the sparse clique comparator, so the run was stopped and its
  incomplete files remain preserved in
  `results/release/2026-08-09-paper-b-final-r2/`. No negative timing result was
  discarded or relabeled as release evidence.
- The certified ordinary path now retains only thresholds whose numerical
  enclosures can attain the interval maximum, uses exact scaled-integer
  evaluation for that ambiguity set, caches shared multiplier evaluations
  across overlapping intervals, and uses a rectangular vector kernel only
  when its padded size is at most twice the ragged option count. Highly unequal
  menus retain the ragged path, so both branches preserve linear option-count
  work and storage. A fixed 24-cell trace rehearsal retained every certificate
  and bound while reducing representative runtime from 6.27 seconds to 0.06
  seconds; it produced 22/24 wins and a 1.66x geometric-mean speedup before
  timing repetitions.
- R3 passed validation, the kernel gate at 45.39x total and 8.81x query-only
  speedup for `n >= 360`, and the matched trace with 22/24 wins and a 1.63x
  geometric mean. Its complete 60-instance primary phase retained every
  certificate, achieved 60/60 wins and positive family breadth, but failed the
  protocol-fixed 2x geometric-mean gate at 1.52x and the 1.5x bootstrap-lower
  gate at 1.42x. The run was stopped before completing robustness; its files
  remain preserved unchanged in
  `results/release/2026-08-09-paper-b-final-r3/`.
- No gate was weakened. Profiling showed that exact fixed-threshold hull work
  and exact ambiguity comparisons dominated the corrected implementation.
  Fixed-threshold LP evaluation now uses scaled integers, cross-product hull
  tests, a fast proposed slope order accepted only after exact adjacent-order
  verification, and exact fallback otherwise. The certified rectangular path
  uses a standard outward rounding-factor enclosure, and exact threshold
  comparisons use a rectangular scaled-integer kernel when all menu widths
  agree. The ragged and adversarial fallbacks are unchanged.
- The complete suite reports 187 tests, and 41,079 exact rational enclosure
  checks over 2,500 random wide-exponent instances found no escape. A fixed
  one-repeat rehearsal showed materially larger speedups across the difficult
  small, medium, and large regimes; it is diagnostic evidence, not a release
  result. Because the algorithm and timing path changed, the complete campaign
  is rerun in the new canonical directory
  `results/release/2026-08-09-paper-b-final-r4/`.
- R4 passes every protocol-fixed validation, kernel, primary, and robustness
  gate. The kernel records 60.14x construction-plus-nine-query and 9.90x
  query-only speedup for `n >= 360`; matched trace records 24/24 wins at 2.21x;
  and the primary panel records 60/60 wins at 2.37x with a 95% interval of
  [2.27, 2.47]. Stress records 8/8 wins at 5.36x and the published-coefficient
  panel records 9/9 wins at 2.75x.
- The release also preserves two operating boundaries. The sparse-budget
  robustness configuration records 6/12 wins despite a 1.29x geometric mean,
  and the UCI-calibrated panel records 0/9 wins with a 0.27 clique/compressed
  time ratio when only about 0.5% of thresholds are evaluated as fixed LPs.
  The manuscript reports both results directly. In the separate exact audit,
  envelope search, clique search, and enumeration each certify 7/12 cases
  within five seconds, compact SCIP certifies 12/12, and every jointly
  certified objective agrees exactly at recorded precision.
- The final R4 release verifier passes all 187 tests and checks 41 evidence
  files, 61 source files, and ten regenerated text artifacts. All six PDF
  variants compile and were rendered for visual review; every font is embedded,
  no Type 3 font is present, and the blind files pass the identity scan. The
  anonymous review archive and eight-file arXiv source archive rebuild
  deterministically, and the latter compiles from a clean extraction.

## Version 4 — Current Manuscript

Release-candidate update: 2026-08-09 (original protocol freeze: 2026-08-01).

The canonical source is `papers/current/main.tex`. The focused
algorithmic contribution is simultaneous Lagrangian evaluation over the
complete robust-MCKP threshold family using exactly-one group envelopes.

Durable contributions and boundaries:

- The August 9 adversarial audit found that signed range accumulation could
  erase small capacity or envelope terms, that tolerance masks could misclassify
  threshold feasibility, and that the shared integer solver still accepted
  small negative original certificates. The release candidate replaces those
  paths with vectorized range accumulation carrying explicit forward-error
  enclosures, exact rational fallback for ambiguous comparisons, and exact-sign
  incumbent validation. Counterexamples and regressions are preserved in the
  test suite; the changed algorithms use a new dated evidence release.
- The released oracle uses ragged group arrays and a linear/log-linear
  multiplier-scale heuristic. A guarded rectangular query kernel is used only
  when its padded size is at most twice the ragged option count, preserving the
  stated linear storage and query-work bound. Earlier notes below about
  unrestricted padded arrays and a solver-tolerance-only certificate describe
  superseded pre-audit builds.
- The first dated-release rerun passed all 40 validation cases, then stopped in
  the kernel phase because its storage-accounting helper still referenced the
  removed padded eligibility array. The helper was corrected for ragged arrays;
  no partial kernel evidence was accepted as released evidence.
- A subsequent kernel rerun was manually interrupted while timing the dense
  reference oracle so its apparent delay could be inspected; it had not failed
  a scientific gate. The kernel phase was restarted from scratch, and only its
  completed output is eligible for the final manifest.
- That restart was itself discarded when the Lipschitz-constant construction
  was strengthened to an outward-rounded exact bound during the run. The final
  campaign begins only after this last certificate change; earlier partial CSVs
  are overwritten before manifest generation and are not release evidence.

- The cancellation and two-envelope representation yields a prefix/suffix
  range algorithm with `O(B + K log(B + 1))` time and `O(B + K)` theoretical
  working storage for ragged group arrays.
- An epigraph-dual mapping proves that the exact minimax envelope bound is no
  larger than the bounded-threshold group-clique LP.
- Convex geometric bracketing and Lipschitz contraction give the deployed
  oracle an explicit additive minimization certificate; its evaluated bound
  inherits minimax dominance up to that gap.
- Interval-bound validity, feasible-singleton equality with the
  fixed-threshold LP, and the adaptive certificate invariant are proved.
- The interval bounds can be embedded in an exact threshold search; a separate
  12-instance audit checks enumeration and compact SCIP and preserves the
  negative many-breakpoints boundary.
- The classical Bertsimas--Sim reduction, fixed-MCKP LP algorithms, filtering,
  branching, clique strengthening, and universal solver-superiority claims
  are explicitly outside the novelty claim.
- Algebraic validation uses 40 irregular held-out instances. The released
  campaign also contains primary, common-trace, robustness, stress, exact
  integration, UCI-calibrated, and published-coefficient panels.
- The finite-menu pricing specialization is a motivating example and a source
  of application-derived coefficient scales. It is not transaction-level
  validation, causal demand estimation, or evidence of commercial pricing
  performance.

## Paper A Freeze and Lineage Update

Date: 2026-08-09.

- Froze the companion Paper A as *A Certifying MCKP Framework for
  Gamma-Robust Discrete Pricing* (arXiv:2603.18653v2) in the separate public
  `certifying-robust-pricing-mckp` repository.
- Separated Paper A's public software surface from its publication archive:
  the submitted source remains locally preserved at commit `5cf4d20`, while
  the reviewer-facing GitHub repository is a code-only `v1.0.1` release at
  commit `85f87d5`; the patch release adds exact-safe fixed-threshold
  feasibility without changing the frozen manuscript.
- Updated the introduction, pricing specialization, prior-art table,
  bibliography, cover letter, project map, and evidence ledger to state the
  cumulative but nonduplicative relationship: Paper A owns the robust-pricing
  model, full-breakpoint decomposition, fixed-threshold hull/rounding
  certificates, and exact full-family search; Paper B owns the simultaneous
  envelope, interval-minimax, and certified multiplier contributions.
- The first post-edit verification failed on the expected `README.md` source
  hash mismatch. No numerical evidence, algorithm, protocol, or timing policy
  changed. The evidence manifest was regenerated from the unchanged canonical
  `results/release/` directory after all editorial changes, rather than edited
  manually or bypassed.
- The first anonymous-supplement rebuild then rejected the public repository
  URL in `README.md`. The builder now treats that file as public-only package
  metadata and continues to supply its purpose-built anonymous README; the
  strict identity scan remains unchanged for every included source and PDF.
- Renamed the public projects and local worktrees to make their contribution
  boundary visible at a glance: Paper A is `certifying-robust-pricing-mckp` and
  Paper B is `simultaneous-group-envelope-mckp`. The internal Python import
  package remains `robust_mckp` for API compatibility.

## Independent Release Audit and Complete Rerun

Date: 2026-08-01.

- Preserved all binary64-distinct deviations in the exact threshold set after
  an adversarial counterexample showed that tolerance clustering could remove
  the only feasible breakpoint; added exact-search regressions for both bounds.
- Aligned the adaptive lower-bound refresh policy across the envelope and
  clique comparators.
- Serialized the UCI application design and enforced and recorded five
  single-thread environment controls before numerical work.
- Reran the complete eight-phase fixed design and the separate exact-integer
  audit. The certified August evidence is now the canonical `results/release/`;
  the superseded July snapshot remains recoverable from Git history.
- Added a claim-level literature evidence ledger, an independent theorem audit
  record, durable repository rules, and a dated protocol-deviation record.
- Regenerated the figure with embedded TrueType fonts, eliminating Type 3
  fonts from the submission PDFs.

## Certified-Oracle and External-Evidence Audit

Date: 2026-07-21.

- Replaced heuristic scalar refinement in the production oracle by certified
  minimization of the finite convex piecewise-linear multiplier envelope.
- Added exact epigraph-LP enclosure tests and release gates on the scaled
  certificate gap.
- Added a nine-instance panel from a published CC BY 4.0 robust-knapsack
  archive, with archive digest, instance provenance, and an explicit statement
  that transferring source deviations to the uncertain resource is a
  model-compatible coefficient test rather than source-model replication.
- Re-audited the 2025--2026 literature, including the independent v3 preprint
  and chance-constrained MCKP work, without finding a direct collision.
- Added explicit v3 title/arXiv disclosure and non-overlap language to the
  cover letter.

## Mathematical and Release Audit

Date: 2026-07-20.

- Corrected the adaptive certificate statement so exhaustive threshold
  evaluation removes obsolete interval records before asserting
  `UB = LB = M`.
- Made initialization robust when root endpoints and the midpoint are
  infeasible, with regressions for a hidden feasible anchor and a globally
  infeasible threshold family.
- Added the bounded-threshold group-clique validity derivation,
  interval-dominance tests, and explicit distinctions between real-arithmetic
  bounds, solver-tolerance certificates, ragged theoretical storage, and the
  released padded implementation.
- Reconciled the UCI data statement: the source has 541,909 records; calibration
  uses the first 200,000, of which 192,451 retained observations form 2,549 SKU
  aggregates.
- Rebuilt public and blind release artifacts after the numerical, test,
  protocol, row-count, and hash checks passed.

## Non-Negotiable Correctness Rules

- The threshold set is the complete original deviation set plus zero. Reduced
  sets cannot support the complete-family claim.
- Exclude infeasible thresholds using the exact group-baseline capacity test
  before defining the target maximum.
- Every active interval record must upper-bound every unevaluated feasible
  threshold it covers. Discarded bounds remain in the global upper bound until
  dominated or made obsolete by exhaustive evaluation.
- A numerically evaluated multiplier bound must not be called the exact minimax
  value. The released implementation reports a computable additive certificate;
  dominance and singleton equality for the deployed value are stated only up
  to that certificate.
- The released stopping test is a scaled floating-point solver-tolerance
  certificate, not an interval-arithmetic proof.
- The theorem's `O(B + K)` storage bound assumes ragged group arrays; the
  released padded implementation is `O(B + n m_max + K)` in general.
- The implemented theory assumes integer `Gamma` in `{0, ..., n}`. Fractional
  uncertainty budgets require an explicit extension and new validation.

## Remaining Scientific Limits

- Numerical claims are tied to the dated serialized campaign and its recorded
  environment and protocol. Regenerate the manifest after any source or
  evidence change.
- The timing study supports the reported simultaneous-evaluation advantage in
  the tested regime, not universal dominance over every alternative solver.
- The many-breakpoints family remains a documented boundary case for exact
  integration.
- The UCI-calibrated application is semi-synthetic; its
  role is coefficient-scale realism, not external validation of decisions.
