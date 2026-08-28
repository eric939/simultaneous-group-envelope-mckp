# Batched Lagrangian bounds for robust MCKP — v2 reproducibility code

This reviewer-facing repository contains the algorithms, tests, experiment
drivers, and released computational evidence for the current v2 revision of
*Batched Lagrangian Bounds for Robust Multiple-Choice Knapsack* (the revision
line of [arXiv:2608.08861](https://arxiv.org/abs/2608.08861)).
The paper-aligned code and evidence tag is
[`v2.0.0`](https://github.com/eric939/simultaneous-group-envelope-mckp/tree/v2.0.0).

The manuscript, submission packages, publication notes, literature audits,
and historical paper sources are intentionally not part of this software
repository.

## What is included

- `research/compressed_interval_oracle.py`: the prefix–suffix evaluator,
  simultaneous threshold-family bound, and certified multiplier search.
- `research/integrated_exact_solver.py`: threshold-interval integration with
  exact feasibility and objective checks.
- `research/bound_dominance.py`, `research/interval_baselines.py`, and
  `research/structural_feasibility_study.py`: the matched clique-LP comparator
  and independent LP validation routines.
- `src/robust_mckp/`: the shared resource-robust MCKP model and certifying
  fixed-threshold solvers used by the computational study.
- `research/benchmark_instances.py` and
  `research/v4_publication_campaign.py`: deterministic generators and the
  serialized frozen campaign.
- `scripts/analyze_v2_operating_region.py`: the post-hoc 122-instance
  descriptive analysis added in v2.
- `scripts/plot_v2_evidence.py` and
  `scripts/plot_common_hinge_mechanism.py`: manuscript-neutral PNG figure
  regeneration.
- `tests/`: correctness, adversarial floating-point, and end-to-end regression
  tests.
- `results/release/2026-08-09-paper-b-final-r5/`: the immutable 41-file base
  evidence release, including raw repetitions, summaries, protocol,
  environments, calibration aggregates, and SHA-256 manifest.
- `results/release/2026-08-28-paper-b-v2-analysis/`: the compact v2 derived
  release containing CSV/JSON analysis, PNG visualizations, and a manifest.

The v2 manuscript also proves that the same common-hinge algebra covers
standard objective-coefficient budget uncertainty. This repository does not
claim a native objective-uncertainty implementation or experiment; those
remain future work.

## Install and verify

```bash
uv sync --frozen --extra experiments --extra validation --extra dev
make verify PYTHON=.venv/bin/python
```

This runs the complete test suite, verifies both release manifests, checks all
scientific gates, recomputes the v2 descriptive statistics, and enforces the
public-tree hygiene boundary. It requires neither manuscript source nor a TeX
installation.

To rerun the fixed computational campaign into a new local directory:

```bash
make reproduce PYTHON=.venv/bin/python
```

To regenerate only the v2 derived CSV, JSON, and PNG outputs:

```bash
make derive-v2 PYTHON=.venv/bin/python
```

Released evidence is immutable; fresh runs are written only below
`results/local/`. See `REPRODUCIBILITY.md` for provenance and interpretation.

The positive primary timings and the adverse pricing/integer results are all
retained. The v2 operating-region correlations are explicitly descriptive,
not causal or confirmatory. No manuscript source, manuscript PDF, submission
package, or raw third-party dataset is tracked.

The software is MIT licensed.
