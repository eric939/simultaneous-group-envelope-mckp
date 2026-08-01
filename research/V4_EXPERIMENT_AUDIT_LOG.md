# V4 experiment audit log

Initial audit: 21 July 2026; final rerun audit: 1 August 2026. The canonical evidence directory is
`results/release`.

## Certified final design

`v4_publication_campaign.py` serializes the complete design and gates to
`protocol.json` before each phase. Its SHA-256 digest is
`6db8550e9e9bb47f352bac93423d82067ed09e6ba6cbe4608a7544c2e3a7d261`.
Instance families, sizes, seeds, repetitions, timing order, single-thread rule,
tolerance, statistical unit, family--size-cell bootstrap, external archive
digest, and go/no-go thresholds were unchanged during the final run. This is a
reproducible fixed engineering design, not an external preregistration.

An earlier sparse-comparator run produced 60/60 timing wins and a 95% CI above
one but failed the predeclared 2x geometric-mean gate (1.79x observed).
Profiling then identified redundant reconstruction in the shared
fixed-threshold LP routine and repeated evaluation of the compressed oracle's
fixed multiplier grid. Those algorithm-preserving inefficiencies were removed,
new parity tests were added, and a new protocol was serialized before the
complete final rerun. The failed run is documented here but its scratch output
is intentionally omitted from the clean release and is not used in manuscript
tables.

## Tests before the certified run

- Exact trace identity against the dense oracle.
- Direct-formula identity for the cancellation equation.
- Certified enclosure of exact epigraph-LP interval minimax values.
- Validity against complete fixed-threshold LP scans.
- Singleton equality with the fixed-threshold LP.
- Deterministic instance-stratified bootstrap and instance-level aggregation.
- Certified solver retry behavior.
- Retention of tolerance-pruned interval bounds in the final certificate.
- Direct fixed-LP handling of singleton comparator intervals.
- Sparse CSR assembly of both clique-LP constraint matrices.
- Exact endpoint handling for repeated and near-repeated deviations.
- Separation of zero from arbitrarily small positive multipliers.
- Equality of the reusable direct-hull fixed-MCKP LP routine with the independent reference solver.
- Certified multiplier counts and maximum scaled gap propagation through the
  adaptive tree.
- Deterministic family--size-cell bootstrap and repeat-block timing diagnostics.

## Corrections found during the audit

1. **Ambiguous HiGHS status.** One wide-menu comparator LP returned a feasible primal with model status “unknown.” A feasible solution to this maximization LP is not a safe upper bound. The implementation now retries dual simplex and interior point and accepts only optimizer-certified optimality or infeasibility. The unchanged robustness panel was restarted.
2. **Discarded-bound accounting.** Both adaptive wrappers originally removed intervals once they were within tolerance but did not retain their last upper bounds in the reported terminal certificate. Pruning choices and attained fixed-threshold values were unaffected, but some `exact` labels were too strong. Both wrappers now carry the maximum discarded bound until the lower bound dominates it. A regression test constructs the failure mode. Every affected panel was rerun.
3. **Singleton fairness.** The comparator wrapper solved a bounded-threshold LP and a fixed-threshold LP when a child was already a singleton; the compressed wrapper solved only the fixed LP. The redundant comparator solve was removed, a regression test was added, and every end-to-end timing panel was rerun from the beginning.
4. **Dense comparator assembly.** The clique comparator used dense matrices despite a sparse formulation. Both equality and inequality matrices now use CSR storage; all headline experiments were rerun. The earlier dense-baseline speedups are superseded.
5. **Search-policy mismatch.** The wrappers used oracle-specific fourth candidates. Both now evaluate the same endpoints and midpoint, leaving the interval bound as the sole method difference.
6. **Inference and repeat selection.** Bootstrap resampling now preserves every family--size cell. The representative numerical result is paired with the median runtime after all repeated bounds and statuses pass consistency checks. Raw outputs include bounds, work counts, execution order, sparse nonzeros, and CSR bytes.
7. **Fixed-LP overhead.** Both methods now share a direct group-hull fixed-MCKP LP routine validated against the independent reference at every threshold in 40 held-out instances.
8. **Multiplier theorem--implementation gap.** The production oracle no longer relies on a fixed grid plus heuristic scalar refinement. Convex geometric bracketing, golden-section contraction, and an explicit Lipschitz constant return an evaluated valid upper bound and a certified lower enclosure. Directed binary64 rounding is outward on both endpoints, and the reported gap includes both rounding steps.
9. **Publication-line separation.** The benchmark seed namespace is `v4|family|n|m|Gamma|seed`; no v3 implementation token remains in the v4 generator or evidence. The cover letter identifies the independent v3 preprint and states the exact non-overlap.
10. **External coefficient provenance.** Nine cases from the CC BY 4.0 Gersing--Büsing--Koster archive are parsed only after verifying SHA-256 `8571b3e545607415a38a39dc506b21bd891b6a22ce252e42a1622a5a5f451818`. The transfer of source objective deviations to v4 resource deviations is disclosed as an out-of-generator coefficient test, not source-model replication.
11. **Exact threshold preservation.** Independent adversarial review found that tolerance-clustering distinct deviations could remove the only feasible breakpoint. The exact builder now preserves every binary64-distinct deviation plus zero, and a 100-group counterexample is a regression for both interval-search variants.
12. **Recorded thread control.** The July protocol stated one thread while its environment variables were unset. The August runner sets and records `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, and `NUMEXPR_NUM_THREADS` to one; the verifier rejects missing controls.
13. **Complete application protocol.** The UCI application design is now part of the serialized protocol rather than a separate constant outside it.
14. **Comparator-policy alignment.** The compressed and clique adaptive routines now refresh the common fixed-threshold lower bound at the same child-processing point before pruning.

No gate, seed, family, instance size, timing rule, or unsuccessful instance was
changed or excluded during the certified final run. Only that complete run
feeds the manuscript artifacts. All validation, kernel, primary, and robustness
gates pass; all interval bounds in the primary, common-trace, and published-
coefficient panels carry certified multiplier gaps below `1e-8` on the reported
scale. The generated evidence manifest hashes the final executable/test source
snapshot and every released CSV/JSON evidence file.
