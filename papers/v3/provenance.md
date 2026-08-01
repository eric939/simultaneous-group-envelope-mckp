# V3 manuscript recovery status

## Outcome

The complete public V3 manuscript PDF has been recovered and is preserved at
`original/robust_mckp_v3_full_20260522.pdf`. It is the canonical V3 paper.
The final V3 code and scientific records remain preserved exactly at commit
`2e2829e55243b846e5944965a839eb9b28dd5e9c`.

The recoverable PDF properties are:

- title: *Certifying Gamma-Robust Discrete Pricing via Full-Breakpoint
  Multiple-Choice Knapsack Decomposition*;
- author: Eric Shao;
- length: 39 pages, US letter;
- embedded creation time: 22 May 2026, 21:47:50 CEST;
- SHA-256: `245e2c0f99bd4e05ca485e97bc8aa34e13608ffe1d29cd2e89e2e20ec8a0821e`;
- recovery date: 21 July 2026; and
- recovery source: a surviving local Apple Mail saved-attachment cache named
  `main_v2.pdf`.

The PDF creation time is 44 seconds after the final V3 code commit at 21:47:06
CEST. Its title, contents, 39-page length, public identity, and manuscript
filename agree with the contemporaneous V3 repository records and Codex build
logs. All 39 pages were rendered after recovery and visually checked; fonts
are embedded, figures and tables are intact, and text extraction found no
placeholder or unresolved-reference artifacts.

## Remaining source limitation

The V3 branch's `SUBMISSION.md` records that the ignored local artifact
workspace was expected to contain:

- `paper_versions/v2/main_v2.tex`;
- `paper_versions/v2/main_v2_blind.tex`;
- `submission_ready/main_v2_public.pdf`;
- `submission_ready/main_v2_blind_submission.pdf`;
- generated `auto/`, `tables/`, `figures/`, and result summaries.

Those paths were ignored by the V3 `.gitignore`, so no commit contains them.
They were subsequently removed from the local working tree during cleanup.
The public PDF is now recovered, but its exact TeX source, blind PDF, generated
input directories, and submission bundle have not been recovered as standalone
files. Codex session archives preserve extensive source excerpts and build
records, but they are not represented as an exact source package.

Available exact records are:

1. the recovered complete V3 public PDF under `original/`;
2. the public arXiv v1 PDF and source preserved under `papers/v2/`;
3. the complete V3 code snapshot at `papers/v3/code.tar.gz`;
4. V3's `README.md`, `REVISION_HISTORY.md`, `SUBMISSION.md`, tests, and
   experiment drivers inside that snapshot; and
5. the public `v3` Git branch and immutable V3 code tag.

A provenance-labeled legacy reconstruction was created before recovery at
`reconstructed/robust_mckp_v3_legacy_reconstruction.tex`, with the matching
PDF beside it. The reconstruction states the V3 model, full-breakpoint
decomposition, hull-rounding certificate, global-gap invariant, exact sweep,
implementation record, and evidence limits supported by the surviving source.
Its title page and metadata explicitly say that it is not the original. It is
retained for audit history but is superseded as the reader-facing V3 paper by
the recovered 39-page PDF.
