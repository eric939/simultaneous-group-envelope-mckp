# Reproducibility

## Environment

The checked-in `uv.lock` freezes the dependency graph used by the release.

```bash
uv sync --frozen --extra experiments --extra validation --extra dev
```

The frozen August campaign used Python 3.14.2, NumPy 2.5.1, SciPy 1.18.0
with HiGHS 1.12.0, and one Apple M4 thread. Phase-specific dependency,
machine, and thread-control records are included as `environment_*.json`.
The exact-integration audit additionally records PySCIPOpt and SCIP versions.

The implementation requires aggregate objective minima and maxima to remain
finite and representable in binary64. Inputs outside that domain are rejected
with rescaling guidance.

## Verify the released evidence

```bash
make verify PYTHON=.venv/bin/python
```

The verifier:

1. runs the complete test suite;
2. checks all 41 base-release hashes;
3. checks validation, kernel, primary, robustness, application, external, and
   exact-integration gates and row counts;
4. checks protocol digests, environment records, thread controls, and public
   calibration aggregates;
5. checks the v2 derived-release manifest;
6. recomputes its 122-row post-hoc statistics from the frozen base CSVs; and
7. rejects tracked manuscript, build, or raw-data artifacts.

The check does not need manuscript source, publication metadata, LaTeX, or raw
UCI transactions.

## Regenerate the v2 analysis

```bash
make derive-v2 PYTHON=.venv/bin/python
```

Fresh CSV, JSON, and PNG outputs are written to
`results/local/v2-analysis/`. The analysis combines the primary, robustness,
stress, coefficient-transfer, and pricing-derived panels. It must reproduce
122 rows, the recorded input hashes, the two log correlations, and the two
pricing fixed-LP medians. These quantities are hypothesis-generating and do
not validate a method selector or identify a runtime cause.

## Full end-to-end rerun

```bash
make reproduce PYTHON=.venv/bin/python
```

This reruns every serialized campaign phase and the separate exact-integration
audit into `results/local/paper-b-reproduction/`; dated releases are never
overwritten.

The application phase uses released UCI Online Retail aggregates. Raw
transactions are not redistributed. To rebuild the aggregates, obtain the UCI
source and run:

```bash
.venv/bin/python scripts/run_pathC_data_calibration.py \
  --source uci_online_retail \
  --max-rows 200000 \
  --cache-dir data_cache/pathC_uci \
  --output-dir results/local/uci_calibration
```

The external-coefficient phase uses the CC BY 4.0 benchmark archive associated
with Gersing, Büsing, and Koster (Zenodo DOI `10.5281/zenodo.7419028`). Its
expected size is 234,397,168 bytes and SHA-256 is
`8571b3e545607415a38a39dc506b21bd891b6a22ce252e42a1622a5a5f451818`.
Place it at `data_cache/RobustKnapsack.zip`; the campaign runner can download
it when absent. The released transformation is a coefficient-provenance stress
test, not a replication of the archive's objective-uncertainty model.

## Interpretation boundary

The release supports the prefix–suffix identities, certified multiplier
search, interval bounds, comparator checks, and paired runtimes reported by the
v2 manuscript. Timing results are instance-level observations, not universal
performance claims. The resource-uncertainty implementation does not establish
native objective-uncertainty runtime performance.
