# Submission and Public Artifact Manifest

**Current-version flag (2026-08-01):** `papers/current/` is canonical and
points to `papers/v4/`.
Version 3 is a separate manuscript and is not part of the v4 submission.

Historical V1--V3 records are separated under `papers/` and by immutable Git
tags. They are provenance archives, not files for the v4 ScholarOne package.

## Journal builds

The main source is `papers/current/main.tex`. The wrappers select public
or blind metadata and main-paper or electronic-companion content:

- `journal.tex`: public main manuscript;
- `journal-blind.tex`: anonymous main manuscript;
- `companion.tex`: public electronic companion;
- `companion-blind.tex`: anonymous electronic companion; and
- `executive-summary.tex`: optional one-page editorial summary.

Build and package all six PDFs with:

```bash
make package
```

Build the identity-scanned anonymous supplement with:

```bash
make anonymous PYTHON=.venv/bin/python
```

The same target also builds `pdf/full.pdf`, the convenient combined
main-plus-appendix version; it is not a separate journal upload.

The current local builds are 20 pages for each OPRE main-paper variant, four
pages for each companion, 19 pages for the combined reading version, and one
page for the executive summary. The summary is cover-letter support and should
be uploaded only if the journal permits it.

The OPRE wrapper is prepared as a Focused Technical submission: the abstract
is text-forward, the introduction contains no equations or mathematical
notation, and all mathematical proofs appear in the main paper. The electronic
companion is limited to comparator implementation, exact-integration audit,
statistical protocol, generators, and reproducibility detail.

For anonymous review, upload only the blind main manuscript, blind electronic
companion, and the generated anonymous supplement if the journal accepts
review artifacts. Inspect the journal-generated merged proof before final
submission. Do not upload the public PDFs, public TeX source, `pyproject.toml`,
or `CITATION.cff` in an anonymous submission.

## Public GitHub artifact

Repository: <https://github.com/eric939/robust_mckp>

The v4 branch/package must expose:

- `research/compressed_interval_oracle.py`;
- `research/bound_dominance.py`, `research/integrated_exact_solver.py`, and
  `research/exact_integration_campaign.py`;
- the complete serialized campaign and artifact generator under `research/`;
- all v4-specific tests;
- `papers/v4/` source, generated TeX inputs, vector figure, and evidence
  manifest;
- `results/release/`, including raw timing repetitions,
  instance-level records, summaries, protocol, environments, and UCI-derived
  aggregates; and
- current `README.md`, `REPRODUCIBILITY.md`, `SUBMISSION.md`, `CITATION.cff`,
  and `REVISION_HISTORY.md`.

Submission-ready article PDFs are checked in under `papers/v4/pdf/` and can also
be attached to a tagged GitHub release or deposited with the journal artifact.
Raw UCI Online Retail transactions are not redistributed.

## Pre-submission checks

```bash
make verify PYTHON=.venv/bin/python
make package
make anonymous PYTHON=.venv/bin/python
```

Then check the compiled files:

```bash
pdftotext papers/current/pdf/paper.pdf /tmp/main_v4_public.txt
pdftotext papers/current/pdf/paper-blind.pdf /tmp/main_v4_blind.txt
rg -ni "TODO|PLACEHOLDER|Version 3|main_v3|results/v3" /tmp/main_v4_public.txt
rg -ni "\\b(Eric|Shao|ershao)\\b|ETH Zürich|github.com/eric939|robust_mckp" /tmp/main_v4_blind.txt
rg -ni "undefined references|undefined citation|multiply defined|missing file" \
  papers/current/*.log
```

All three searches should return no actionable hit. Bibliographic references
to a prior working paper, if retained, must follow the target journal's
double-blind self-citation policy.

Before upload, create an immutable `v4` release tag from the exact submitted
commit and record that tag or archive DOI in the submission form. Branch names
alone are mutable and are not an archival identifier.

## Independence and disclosure rule

The supplied `papers/current/cover-letter.md` presents v4 as Paper B: a
nonduplicative paper that builds on Paper A while retaining its own research
question, novelty claim, theorems, evidence, and submission package. Do not
describe v4 as a revision, replacement, or superseding version of Paper A. The
cover letter must explicitly identify *A Certifying MCKP Framework for
Gamma-Robust Discrete Pricing* (arXiv:2603.18653v2), state what Paper B inherits,
and state the precise non-overlap in model target, theorem, algorithm, and
evidence.
Repeat that disclosure in any journal form asking about related manuscripts.
Do not imply that the classical Bertsimas–Sim threshold reduction or
fixed-MCKP LP geometry is new.

## Data and claim policy

The public UCI panel is application-derived and semi-synthetic. Its calibration
uses a nonrandom 200,000-record prefix to set coefficient scales; it does not
estimate causal demand or validate commercial pricing outcomes. Runtime claims
are tied to the released single-threaded environment and comparator
implementation. The published-coefficient panel transparently transfers the
source deviations from objective to resource uncertainty, so it is an
out-of-generator stress test rather than a direct source-model comparison. The
principal theoretical claim is simultaneous evaluation and valid certification
of the fixed-threshold LP family together with exact-minimax dominance over the
group-clique interval LP; the deployed evaluated bound inherits this result up
to its explicit multiplier-optimization certificate. The exact integration validates
global gap accounting but does not support universal integer-solver
superiority.
