# Reproducibility Guide

**Canonical paper and evidence:** v4, August 2026. The manuscript source is
`papers/current/`; the released result directory is `results/release/`.

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -e ".[experiments,validation,dev]"
```

The released campaign used Python 3.14.2, NumPy 2.4.4, SciPy 1.17.1 with
HiGHS 1.14.0, and one Apple M4 thread. Phase-specific machine and dependency
records are released as `environment_*.json`. The solver methods in the main
comparison both use the same SciPy/HiGHS fixed-threshold LP routine.

## Verify the released artifact

```bash
make verify PYTHON=.venv/bin/python
```

This bounded check:

1. runs the complete test suite;
2. verifies SHA-256 hashes for every released CSV/JSON and a conservative
   source snapshot covering the package, research drivers, release scripts,
   tests, dependency declaration, and build targets;
3. requires every protocol-fixed validation, kernel, primary, and robustness gate
   to be true; and
4. regenerates all numerical TeX macros and tables in a temporary directory
   and byte-compares them with the checked-in manuscript inputs.

The vector figure is regenerated but is not byte-compared because PDF metadata
can vary across Matplotlib versions. Its plotted values come from the same
hash-verified CSVs.

## Serialized experimental design

The protocol is serialized in `results/release/protocol.json`.
Its SHA-256 is recorded before each phase. The complete certified campaign was
run after the algorithmic and reporting protocol had been finalized; it is a
reproducible fixed design, not a preregistered study. The statistical unit is an instance;
timing repetitions estimate an instance runtime and are not treated as
independent observations. Method order alternates within paired blocks, all
reported timing is end to end, and thread counts are fixed to one.

The released campaign contains:

- 40 irregular algebraic and complete-threshold validation cases;
- a 48-instance kernel ablation over four sizes and four families;
- a 24-instance common-trace comparison on identical dyadic intervals;
- 60 primary instances: four families, three sizes, and five seeds, with five
  balanced repetitions per method;
- 36 robustness instances covering `Gamma=1`, twelve-option menus, and
  `Gamma=floor(0.2n)`;
- eight descriptive stress instances through 5,760 groups;
- nine semi-synthetic pricing portfolios calibrated from public UCI Online
  Retail aggregates; and
- nine published robust-knapsack coefficient sets, transformed transparently
  into two-option groups for an out-of-generator stress test.

A separate exact-integration audit contains twelve 30-group instances across
four families and three seeds, with two balanced repetitions and a five-second
limit. It compares envelope interval search, clique interval search, complete
threshold enumeration, and compact SCIP. This audit validates exact objective
agreement and gap accounting; it is not part of the protocol-fixed LP-certificate
timing claim.

Primary geometric-mean confidence intervals use 10,000 bootstrap draws,
resampling seeds within family-by-size cells. All headline summaries use one
observation per instance. These intervals
summarize variability over generated seeds within the fixed factorial design;
they are not population-sampling claims about all robust MCKP instances.

## Full end-to-end rerun

```bash
make reproduce PYTHON=.venv/bin/python
```

This command runs tests, executes every serialized phase in order, and generates a
fresh set of paper macros, tables, and figures under
`tmp/v4_reproduction_paper/`. Fresh timing results are written to
`results/local/`; released results are never overwritten. Override these
locations with `LOCAL_RESULTS=...` and `LOCAL_PAPER=...`.

The application phase uses the released UCI-derived aggregates in
`results/release/uci_calibration/`. Raw UCI transactions
are not redistributed. To rebuild those aggregates from the source data,
download the UCI Online Retail CSV, place it in a local cache directory, and
run:

```bash
.venv/bin/python scripts/run_pathC_data_calibration.py \
  --source uci_online_retail \
  --max-rows 200000 \
  --cache-dir data_cache/pathC_uci \
  --output-dir results/local/uci_calibration
```

The calibration script records whether public or fallback synthetic data were
used. A reproduction of the reported UCI-calibrated panel must show
`public_data_used: True` in its source report. The public data are used only to
calibrate aggregate price, volume, segment, and uncertainty scales; elasticity
curves and generated choice menus remain modeled. The reported calibration uses
the first 200,000 source rows rather than a random or full-data sample; this is
why the panel is treated as a scoped scale check rather than external decision validation.

The published-coefficient phase uses the CC BY 4.0 benchmark archive associated
with Gersing, Büsing, and Koster (Zenodo DOI `10.5281/zenodo.7419028`). The
runner downloads the 234,397,168-byte archive when absent and requires SHA-256
`8571b3e545607415a38a39dc506b21bd891b6a22ce252e42a1622a5a5f451818`.
The source model places deviations in the objective; the released transformation
retains its profits, nominal weights, capacities, and deviation magnitudes but
uses those magnitudes as resource deviations. Accordingly this is a coefficient
provenance test, not a reproduction of the source algorithm or model.

## Regenerate evidence or compile the paper

Regenerate checked-in evidence from the released records:

```bash
make evidence PYTHON=.venv/bin/python
```

Compile the public/blind main paper, electronic companion, and executive
summary (requires `tectonic`):

```bash
make paper
```

Build the blind PDFs and a deterministic anonymous review supplement, then
scan its filenames, text sources, PDF text, and PDF metadata for identity
tokens and local absolute paths:

```bash
make anonymous PYTHON=.venv/bin/python
```

The resulting archive is
`papers/current/anonymous-supplement.zip`. It intentionally
excludes public manuscript source, citation metadata, repository URLs, and
package metadata that identify the author.

The main source is `papers/current/main.tex`; the small wrapper files
select the public/blind and main/companion variants.

## Interpretation limits

The released evidence supports algebraic correctness, certified minimization
of the deployed finite convex multiplier problem, valid LP-family
certification, formal objective-bound dominance up to the reported numerical
gap over the bounded-threshold group-clique LP, and a paired certificate-time advantage over its sparse
implementation on the tested designs. The separate exact audit does not
establish universal solver dominance or integer-solver superiority. The
published-coefficient panel reduces dependence on the internal generator but
is not a direct algorithm comparison. The UCI
panel is application-derived and semi-synthetic, not a causal demand estimate
or transaction-level pricing validation.
