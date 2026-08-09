# Revision History and Scientific Guardrails

This branch records only durable facts for the independent v4 publication
project. The separate v3 manuscript, experiments, and history remain on the
`v3` branch; v4 is neither a revision of v3 nor a replacement for it.

## Version 4 — Current Manuscript

Date: 2026-08-01.

The canonical source is `papers/current/main.tex`. The focused
algorithmic contribution is simultaneous Lagrangian evaluation over the
complete robust-MCKP threshold family using exactly-one group envelopes.

Durable contributions and boundaries:

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
  `robust_pricing_mckp` repository.
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
