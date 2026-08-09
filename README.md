# Simultaneous Group-Envelope MCKP — reproducibility code

This reviewer-facing repository contains the algorithms, solver integration,
tests, experiment drivers, and released computational evidence for
*Simultaneous Group-Envelope Bounds for Gamma-Robust Multiple-Choice Knapsack
Problems*.

The manuscript, submission packages, publication notes, literature audits,
and historical paper sources are intentionally not part of this software
repository.

## What is included

- `src/robust_mckp/`: the shared robust MCKP model, exact certificate,
  fixed-threshold relaxations, rounding, and exact branch-and-bound solver.
- `research/compressed_interval_oracle.py`: the simultaneous group-envelope
  oracle and certified multiplier search.
- `research/integrated_exact_solver.py`: threshold-interval integration with
  exact feasibility and objective checks.
- `research/bound_dominance.py` and
  `research/structural_feasibility_study.py`: independent comparator and LP
  validation routines.
- `research/benchmark_instances.py` and
  `research/v4_publication_campaign.py`: deterministic instance generators and
  the serialized computational campaign.
- `scripts/`: public benchmark, calibration, campaign, and release-verification
  entry points.
- `tests/`: correctness, adversarial floating-point, and end-to-end regression
  tests.
- `results/release/2026-08-09-paper-b-final-r5/`: the compact canonical
  evidence release, including raw repetitions, summaries, protocol and
  environment records, calibration aggregates, and a SHA-256 manifest.

## Install and verify

```bash
uv sync --frozen --extra experiments --extra validation --extra dev
make verify PYTHON=.venv/bin/python
```

This runs the complete test suite, verifies every released evidence byte, and
checks all scientific release gates. It requires neither manuscript source nor
a TeX installation.

To rerun the fixed computational design into a new local directory:

```bash
make reproduce PYTHON=.venv/bin/python
```

Released evidence is immutable; fresh runs are written only under
`results/local/`. See `REPRODUCIBILITY.md` for the environment, data provenance,
and exact-integration audit.

The software is MIT licensed.
