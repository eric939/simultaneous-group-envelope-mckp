# Reproducibility

## Environment

The checked-in `uv.lock` freezes the dependency graph used by the final
computational release.

```bash
uv sync --frozen --extra experiments --extra validation --extra dev
```

The released campaign used Python 3.14.2, NumPy 2.5.1, SciPy 1.18.0 with
HiGHS 1.12.0, and one Apple M4 thread. Phase-specific dependency, machine, and
thread-control records are included as `environment_*.json` files. The exact
integration audit additionally records PySCIPOpt and SCIP versions.

The implementation requires aggregate objective minima and maxima to remain
finite and representable in binary64. Inputs outside that explicit numerical
domain are rejected with rescaling guidance.

## Verify the released evidence

```bash
make verify PYTHON=.venv/bin/python
```

The verifier:

1. runs the complete test suite;
2. checks the SHA-256 manifest against every released CSV, JSON, and text file;
3. checks all validation, kernel, primary, robustness, and exact-integration
   gates and row counts;
4. checks protocol digests and recorded single-thread controls; and
5. checks the released public-data calibration aggregates.

The check does not need manuscript source, publication metadata, or LaTeX.

## Full end-to-end rerun

```bash
make reproduce PYTHON=.venv/bin/python
```

This runs the tests and every serialized campaign phase, then executes the
separate exact-integration audit. Fresh results are written to
`results/local/paper-b-reproduction/`; the dated release is never overwritten.

The application phase uses the released UCI Online Retail aggregates in the
canonical result directory. Raw transactions are not redistributed. To rebuild
those aggregates, download the UCI Online Retail CSV and run:

```bash
.venv/bin/python scripts/run_pathC_data_calibration.py \
  --source uci_online_retail \
  --max-rows 200000 \
  --cache-dir data_cache/pathC_uci \
  --output-dir results/local/uci_calibration
```

A reproduction of the reported application panel must record
`public_data_used: True`. The data calibrate aggregate coefficient scales;
choice menus and elasticity curves remain modeled.

The external-coefficient phase uses the CC BY 4.0 benchmark archive associated
with Gersing, Büsing, and Koster (Zenodo DOI `10.5281/zenodo.7419028`). The
expected archive size is 234,397,168 bytes and its SHA-256 is
`8571b3e545607415a38a39dc506b21bd891b6a22ce252e42a1622a5a5f451818`.
Place it at `data_cache/RobustKnapsack.zip`; the campaign runner can download it
when absent. The transformation is a coefficient-provenance stress test, not a
replication of the source uncertainty model.

## Interpretation

The release supports the algebraic identities, certified multiplier search,
LP-family bounds, comparator dominance checks, and paired runtimes reported by
the accompanying manuscript. Timing results are instance-level observations,
not universal solver-performance claims. The exact audit checks objective and
gap accounting separately from the LP-bound timing study.
