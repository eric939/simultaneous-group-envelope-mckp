from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from scripts.run_pathC_data_calibration import load_local_retail_xlsx


ROOT = Path(__file__).resolve().parents[1]


def _write_minimal_retail_xlsx(path: Path) -> None:
    shared = """<?xml version="1.0" encoding="UTF-8"?>
<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
 count="5" uniqueCount="5">
  <si><t>StockCode</t></si>
  <si><t>Quantity</t></si>
  <si><t>UnitPrice</t></si>
  <si><t>SKU-A</t></si>
  <si><t>SKU-B</t></si>
</sst>
"""
    worksheet = """<?xml version="1.0" encoding="UTF-8"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData>
    <row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c><c r="C1" t="s"><v>2</v></c></row>
    <row r="2"><c r="A2" t="s"><v>3</v></c><c r="B2"><v>2</v></c><c r="C2"><v>3.5</v></c></row>
    <row r="3"><c r="A3" t="s"><v>3</v></c><c r="B3"><v>4</v></c><c r="C3"><v>4.5</v></c></row>
    <row r="4"><c r="A4" t="s"><v>4</v></c><c r="B4"><v>-1</v></c><c r="C4"><v>2</v></c></row>
  </sheetData>
</worksheet>
"""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("xl/sharedStrings.xml", shared)
        archive.writestr("xl/worksheets/sheet1.xml", worksheet)


def test_xlsx_streaming_aggregation(tmp_path: Path) -> None:
    workbook = tmp_path / "Online Retail.xlsx"
    _write_minimal_retail_xlsx(workbook)

    result = load_local_retail_xlsx(workbook, max_rows=3, min_obs=2)

    assert result["rows_examined"] == 3
    assert result["rows_retained_before_minimum"] == 2
    assert result["source_used"] == "local_public_csv"
    assert len(result["sku_rows"]) == 1
    row = result["sku_rows"][0]
    assert row["sku"] == "SKU-A"
    assert row["observations"] == 2
    assert row["total_quantity"] == pytest.approx(6.0)
    assert row["total_revenue"] == pytest.approx(25.0)


def test_cli_records_raw_input_provenance_and_checks_hash(tmp_path: Path) -> None:
    workbook = tmp_path / "Online Retail.xlsx"
    output = tmp_path / "output"
    cache = tmp_path / "cache"
    _write_minimal_retail_xlsx(workbook)
    expected = hashlib.sha256(workbook.read_bytes()).hexdigest()

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_pathC_data_calibration.py"),
            "--source",
            "uci_online_retail",
            "--input-file",
            str(workbook),
            "--expected-sha256",
            expected,
            "--max-rows",
            "3",
            "--min-sku-observations",
            "2",
            "--cache-dir",
            str(cache),
            "--output-dir",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    config = json.loads((output / "calibration_config.json").read_text())
    report = (output / "data_source_report.txt").read_text()
    assert config["raw_input_path"] == str(workbook.resolve())
    assert config["raw_input_sha256"] == expected
    assert config["raw_input_bytes"] == workbook.stat().st_size
    assert config["raw_input_format"] == "xlsx"
    assert config["raw_hash_verified"] is True
    assert f"Raw input SHA-256: {expected}" in report
    assert "Raw input hash verified: True" in report

    failed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_pathC_data_calibration.py"),
            "--source",
            "uci_online_retail",
            "--input-file",
            str(workbook),
            "--expected-sha256",
            "0" * 64,
            "--cache-dir",
            str(cache),
            "--output-dir",
            str(tmp_path / "rejected"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert failed.returncode != 0
    assert "input SHA-256 mismatch" in failed.stderr
