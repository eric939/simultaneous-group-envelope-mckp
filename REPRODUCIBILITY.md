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
6. verifies the separately hashed release-layer, UCI-source, and computational
   development provenance records;
7. recomputes the 122-row post-hoc statistics from the frozen base CSVs; and
8. rejects tracked manuscript, build, or raw-data artifacts.

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

The application phase uses released aggregates from Daqing Chen's UCI Online
Retail dataset (UCI id 352, DOI `10.24432/C5BW33`, CC BY 4.0). Raw transactions
are not redistributed. The following deterministic route downloads the
official UCI archive, verifies the exact bytes used for this reconstruction,
extracts the workbook, and rebuilds the aggregates:

```bash
mkdir -p data_cache/pathC_uci
curl -L --fail \
  'https://archive.ics.uci.edu/static/public/352/online%2Bretail.zip' \
  -o data_cache/pathC_uci/online-retail.zip
echo 'f5385cbb54bbebf7196389109c6b0621faab0c304e3702548165e71c84aede8b  data_cache/pathC_uci/online-retail.zip' \
  | shasum -a 256 -c -
unzip -p data_cache/pathC_uci/online-retail.zip 'Online Retail.xlsx' \
  > 'data_cache/pathC_uci/Online Retail.xlsx'
echo '43465a06f2ccf7c8b5bd2892bc7defb52f97487934fe93b16ae4c3936424676d  data_cache/pathC_uci/Online Retail.xlsx' \
  | shasum -a 256 -c -
make reconstruct-uci PYTHON=.venv/bin/python
```

The verified archive is 23,715,478 bytes; its `Online Retail.xlsx` member is
23,715,344 bytes. The standard-library streaming importer reads the first
200,000 data rows, retains positive quantities and prices, and applies the
eight-observation SKU threshold. It reproduces the frozen SKU and segment CSVs
byte for byte (SHA-256 `fddf028c...b1d30a` and `1ba04343...0b7f6`,
respectively). Fresh `calibration_config.json` and `data_source_report.txt`
files record the resolved raw path, input format, byte count, observed hash,
expected hash, and verification status.

The August frozen campaign itself recorded only the aggregate hashes, not the
raw workbook hash. The pinned workbook was retrieved and checked during the v2
provenance repair; this closes the reconstruction path but does not
retroactively claim that the original campaign logged information it did not.
The current source and release-layer record is
`provenance/RELEASE_PROVENANCE.json`.

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

The final 2.37-fold primary timing result is development evidence. Earlier R2
and R3 failed performance audits—including a same-design 1.79-fold run—guided
algorithm-preserving implementation improvements before the final complete
rerun. The fixed final protocol and all retained negative panels remain
verifiable, but this is not described as an external preregistration or an
untouched confirmation experiment. The full concise chronology is in
`provenance/DEVELOPMENT_HISTORY.md`.
