# Simultaneous Group-Envelope MCKP

Reference implementation and reproducibility artifact for **“Simultaneous
Group-Envelope Bounds for Γ-Robust Multiple-Choice Knapsack Problems.”**
Manuscript v4 (August 2026) is the canonical working paper. Version 3 is retained
as provenance for a separate paper and research program; v4 neither replaces
nor supersedes it. That Paper A line now has its own repository,
[`eric939/certifying-robust-pricing-mckp`](https://github.com/eric939/certifying-robust-pricing-mckp),
whose public `v1.0.1` release contains only the frozen reviewer-facing solver
and reproducibility code for arXiv:2603.18653. Paper A's submitted manuscript
source is preserved separately in the local publication archive. This
repository is Paper B and will receive a new arXiv identifier.

The v4 contribution is an all-threshold Lagrangian evaluation algorithm for
exactly-one groups. For `K` options and `B` robust-deviation thresholds, it
evaluates a multiplier in `O(B + K log(B + 1))` time and `O(B + K)` working
storage. The paper combines this oracle with a valid adaptive certificate for
the maximum fixed-threshold MCKP LP value and proves that the exact minimax
bound is no weaker than the bounded-threshold group-clique LP. The production
oracle uses convex bracketing and Lipschitz contraction to report an explicit
multiplier-optimization gap, so its deployed dominance and singleton-accuracy
claims are numerically certified. A released exact
interval-search integration validates global gap accounting without claiming a
universally superior integer solver.

The certified binary64 implementation checks that the aggregate objective
range is finite and representable. Instances outside that numerical domain
are rejected with rescaling guidance; they are never labeled exact or
infeasible from saturated floating-point values.

## Repository map

Start with `papers/current/main.tex`. The `current` link always identifies the
latest editable draft; today it points to `papers/v4/`.

See `PROJECTS.md` for the formal Paper A/Paper B contribution boundary,
repository ownership, citation order, and release sequence.

- `research/compressed_interval_oracle.py`: proposed group-envelope oracle.
- `research/benchmark_instances.py`: neutral, deterministic v4 benchmark
  generator with an explicit v4 seed namespace.
- `research/bound_dominance.py`: independently solved epigraph LP used to validate the formal
  dominance theorem.
- `research/integrated_exact_solver.py`: exact integer threshold-interval
  search using either the envelope or clique bound.
- `research/exact_integration_campaign.py`: controlled exact-solver audit.
- `research/v4_publication_campaign.py`: serialized validation and experiment
  protocol, gates, generators, and statistical summaries.
- `research/generate_v4_publication_artifacts.py`: tables, figure, macros, and
  hash manifest used by the manuscript.
- `research/LITERATURE_NOVELTY_AUDIT_V4.md`: source-by-source novelty audit.
- `tests/test_compressed_interval_oracle.py` and
  `tests/test_v4_publication_campaign.py`: v4 algebra and protocol tests.
- `papers/`: one clear home for V1--V4, the current-draft pointer, final PDFs,
  historical source archives, provenance, and SHA-256 checksums.
- `results/release/2026-08-09-paper-b-final-r4/`: canonical released instance-level results,
  raw timing repetitions, protocol, environments, summaries, and public-data
  calibration aggregates.
- `src/robust_mckp/`: shared solver infrastructure used by the experiments.

The canonical Paper B source is now on `main`; the historical `v3` and earlier
paper directories remain immutable provenance.

Submission-ready manuscript PDFs are checked in under `papers/current/pdf/`;
their source files and build instructions are listed in `SUBMISSION.md`. The
complete V3 paper and its provenance live together under `papers/v3/`.

## Install and verify

```bash
uv sync --frozen --extra experiments --extra validation --extra dev
make verify PYTHON=.venv/bin/python
```

`verify` runs the test suite, checks every released evidence hash, regenerates
the manuscript tables/macros/figure in a temporary directory, and compares the
generated text artifacts with the checked-in versions. It does not rerun the
long timing campaign.

Preview or remove ignored build debris and local result runs without touching
the versioned papers, `results/release`, Git branches, the virtual environment,
or the raw-data cache:

```bash
make clean-preview PYTHON=.venv/bin/python
make clean PYTHON=.venv/bin/python
```

To rerun the complete fixed-design campaign and rebuild its paper artifacts:

```bash
make reproduce PYTHON=.venv/bin/python
```

The full command uses the released UCI-derived aggregate calibration, never the
raw transaction file. See `REPRODUCIBILITY.md` for the protocol, run-time scope,
data provenance, and separate manuscript-build command.

## Headline released evidence

- 40/40 algebraic, epigraph-LP, and complete-scan validation cases pass; the
  maximum scaled multiplier-certificate gap is below `1e-8`.
- 60/60 primary instances reach scaled tolerance `1e-6`; the proposed method
  wins 60/60 paired timings with geometric-mean speedup 2.37 (95% stratified
  bootstrap interval [2.27, 2.47]).
- All 36 robustness cases and all eight stress cases reach tolerance.
- A nine-instance panel transformed from a published robust-knapsack archive
  wins 9/9 timings, with geometric-mean speedup 2.75 (95% design-stratified
  bootstrap interval [2.66, 2.84]). This is an out-of-generator coefficient
  test, not a replication of the source paper's uncertainty model.
- The 200,000-record UCI-calibrated panel is application-derived and
  semi-synthetic; it tests coefficient scales rather than causal demand claims.
  Both methods certify all nine cases, but the compressed method wins 0/9 and
  the clique/compressed time ratio is 0.27 because only about 0.5% of thresholds
  are evaluated as fixed LPs.
- In the separate 12-instance exact audit, envelope and clique interval search
  and complete enumeration each certify 7 cases, while compact SCIP certifies
  all 12; jointly certified objectives agree exactly at recorded precision.

These are instance-level paired results on the recorded single-threaded
environment, not universal performance claims.

## Citation and license

Please cite `CITATION.cff` and the accompanying manuscript. Licensed under the
MIT License; see `LICENSE`.
