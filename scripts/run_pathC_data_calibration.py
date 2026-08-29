#!/usr/bin/env python3
"""Path C calibration layer for semi-synthetic robust pricing applications.

The script tries lightweight public-data hooks first, but never requires
credentials or large raw downloads. If public data are unavailable in the local
environment, it writes documented semi-synthetic calibration defaults.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SEGMENT_DEFAULTS = [
    {
        "segment": "staples",
        "share": 0.34,
        "reference_price_median": 3.25,
        "reference_price_log_sd": 0.32,
        "baseline_volume_median": 180.0,
        "baseline_volume_log_sd": 0.45,
        "elasticity_low": 1.15,
        "elasticity_high": 1.95,
        "uncertainty_scale": 0.08,
        "fairness_band": 0.08,
        "weight_multiplier": 1.20,
    },
    {
        "segment": "mainstream",
        "share": 0.30,
        "reference_price_median": 5.50,
        "reference_price_log_sd": 0.36,
        "baseline_volume_median": 85.0,
        "baseline_volume_log_sd": 0.55,
        "elasticity_low": 1.90,
        "elasticity_high": 3.10,
        "uncertainty_scale": 0.14,
        "fairness_band": 0.10,
        "weight_multiplier": 1.00,
    },
    {
        "segment": "premium",
        "share": 0.16,
        "reference_price_median": 10.50,
        "reference_price_log_sd": 0.42,
        "baseline_volume_median": 28.0,
        "baseline_volume_log_sd": 0.65,
        "elasticity_low": 1.35,
        "elasticity_high": 2.45,
        "uncertainty_scale": 0.12,
        "fairness_band": 0.12,
        "weight_multiplier": 0.90,
    },
    {
        "segment": "private_label",
        "share": 0.12,
        "reference_price_median": 3.80,
        "reference_price_log_sd": 0.28,
        "baseline_volume_median": 120.0,
        "baseline_volume_log_sd": 0.50,
        "elasticity_low": 2.70,
        "elasticity_high": 4.60,
        "uncertainty_scale": 0.22,
        "fairness_band": 0.10,
        "weight_multiplier": 1.05,
    },
    {
        "segment": "seasonal",
        "share": 0.08,
        "reference_price_median": 7.25,
        "reference_price_log_sd": 0.55,
        "baseline_volume_median": 40.0,
        "baseline_volume_log_sd": 0.85,
        "elasticity_low": 2.10,
        "elasticity_high": 4.20,
        "uncertainty_scale": 0.30,
        "fairness_band": 0.15,
        "weight_multiplier": 0.80,
    },
]

UCI_ONLINE_RETAIL_DATASET_URL = (
    "https://archive.ics.uci.edu/dataset/352/online%2Bretail"
)
UCI_ONLINE_RETAIL_DOWNLOAD_URL = (
    "https://archive.ics.uci.edu/static/public/352/online%2Bretail.zip"
)
UCI_ONLINE_RETAIL_DOI = "10.24432/C5BW33"
UCI_ONLINE_RETAIL_ZIP_SHA256 = (
    "f5385cbb54bbebf7196389109c6b0621faab0c304e3702548165e71c84aede8b"
)
UCI_ONLINE_RETAIL_XLSX_SHA256 = (
    "43465a06f2ccf7c8b5bd2892bc7defb52f97487934fe93b16ae4c3936424676d"
)
_SPREADSHEET_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def median(vals: Iterable[float], default: float = float("nan")) -> float:
    clean = [float(v) for v in vals if math.isfinite(float(v))]
    return float(statistics.median(clean)) if clean else default


def try_import(name: str):
    try:
        return __import__(name)
    except Exception:
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def find_local_retail_input(cache_dir: Path) -> Optional[Path]:
    """Find a deterministic local input, preferring the official XLSX."""

    preferred = [
        cache_dir / "Online Retail.xlsx",
        cache_dir / "online_retail.xlsx",
        cache_dir / "online_retail_uci.csv",
    ]
    for path in preferred:
        if path.is_file():
            return path

    candidates = []
    for pattern in ["*.xlsx", "*.csv", "*.txt"]:
        candidates.extend(sorted(cache_dir.glob(pattern)))
    for path in candidates:
        lower = path.name.lower()
        if "online" in lower or "retail" in lower or "transaction" in lower:
            return path
    return candidates[0] if candidates else None


def _aggregate_retail_rows(
    rows: Iterable[Dict[str, object]],
    *,
    max_rows: Optional[int],
    min_obs: int,
    source_used: str,
) -> Dict[str, object]:
    sku_stats: Dict[str, Dict[str, object]] = {}
    rows_examined = 0
    rows_retained = 0
    for idx, row in enumerate(rows):
        if max_rows is not None and idx >= max_rows:
            break
        rows_examined += 1
        try:
            sku = str(row["StockCode"]).strip()
            qty = float(row["Quantity"])
            price = float(row["UnitPrice"])
        except Exception:
            continue
        if not sku or qty <= 0 or price <= 0 or price > 10_000:
            continue
        rows_retained += 1
        rec = sku_stats.setdefault(
            sku,
            {
                "sku": sku,
                "obs": 0,
                "quantity": 0.0,
                "revenue": 0.0,
                "prices": [],
            },
        )
        rec["obs"] = int(rec["obs"]) + 1
        rec["quantity"] = float(rec["quantity"]) + qty
        rec["revenue"] = float(rec["revenue"]) + qty * price
        rec["prices"].append(price)

    output_rows: List[Dict[str, object]] = []
    for sku, rec in sku_stats.items():
        prices = [float(p) for p in rec["prices"]]
        if int(rec["obs"]) < min_obs:
            continue
        mean_price = float(rec["revenue"]) / max(float(rec["quantity"]), 1e-12)
        price_med = median(prices)
        price_sd = statistics.pstdev(prices) if len(prices) > 1 else 0.0
        output_rows.append(
            {
                "sku": sku,
                "source": source_used,
                "observations": int(rec["obs"]),
                "total_quantity": float(rec["quantity"]),
                "total_revenue": float(rec["revenue"]),
                "reference_price": price_med,
                "average_unit_price": mean_price,
                "price_cv": price_sd / price_med if price_med > 0 else 0.0,
                "baseline_volume": float(rec["quantity"]) / max(int(rec["obs"]), 1),
            }
        )
    return {
        "sku_rows": output_rows,
        "source_used": source_used,
        "rows_examined": rows_examined,
        "rows_retained_before_minimum": rows_retained,
    }


def load_local_retail_csv(
    path: Path, max_rows: Optional[int], min_obs: int
) -> Dict[str, object]:
    """Aggregate a local retail-like CSV with common transaction columns."""

    with path.open(newline="", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV has no header")
        fields = {name.lower(): name for name in reader.fieldnames}
        sku_col = fields.get("stockcode") or fields.get("sku") or fields.get("product_id") or fields.get("item")
        qty_col = fields.get("quantity") or fields.get("qty")
        price_col = fields.get("unitprice") or fields.get("unit_price") or fields.get("price")
        if not (sku_col and qty_col and price_col):
            raise ValueError(f"CSV lacks SKU/quantity/price columns: {reader.fieldnames}")
        normalized_rows = (
            {
                "StockCode": row.get(sku_col, ""),
                "Quantity": row.get(qty_col, ""),
                "UnitPrice": row.get(price_col, ""),
            }
            for row in reader
        )
        result = _aggregate_retail_rows(
            normalized_rows,
            max_rows=max_rows,
            min_obs=min_obs,
            source_used="local_public_csv",
        )
    result["source_path"] = str(path)
    return result


def _column_name(cell_reference: str) -> str:
    return "".join(char for char in cell_reference if char.isalpha()).upper()


def _shared_strings(archive: zipfile.ZipFile) -> List[str]:
    strings: List[str] = []
    try:
        stream = archive.open("xl/sharedStrings.xml")
    except KeyError:
        return strings
    with stream:
        for _, element in ElementTree.iterparse(stream, events=("end",)):
            if element.tag == f"{_SPREADSHEET_NS}si":
                strings.append(
                    "".join(
                        node.text or ""
                        for node in element.iter(f"{_SPREADSHEET_NS}t")
                    )
                )
                element.clear()
    return strings


def iter_online_retail_xlsx(path: Path) -> Iterator[Dict[str, object]]:
    """Stream the three calibration columns from the official UCI workbook.

    This intentionally uses only the Python standard library so the documented
    reconstruction path does not depend on a spreadsheet engine.
    """

    with zipfile.ZipFile(path) as archive:
        shared = _shared_strings(archive)
        try:
            worksheet = archive.open("xl/worksheets/sheet1.xml")
        except KeyError as exc:
            raise ValueError("XLSX has no xl/worksheets/sheet1.xml") from exc

        headers: Dict[str, str] = {}
        with worksheet:
            for _, row in ElementTree.iterparse(worksheet, events=("end",)):
                if row.tag != f"{_SPREADSHEET_NS}row":
                    continue
                values: Dict[str, object] = {}
                for cell in row.findall(f"{_SPREADSHEET_NS}c"):
                    column = _column_name(cell.attrib.get("r", ""))
                    cell_type = cell.attrib.get("t")
                    value_node = cell.find(f"{_SPREADSHEET_NS}v")
                    raw_value = value_node.text if value_node is not None else ""
                    if cell_type == "s" and raw_value != "":
                        raw_value = shared[int(raw_value)]
                    elif cell_type == "inlineStr":
                        raw_value = "".join(
                            node.text or ""
                            for node in cell.iter(f"{_SPREADSHEET_NS}t")
                        )
                    values[column] = raw_value

                if not headers:
                    headers = {
                        column: str(value).strip() for column, value in values.items()
                    }
                    required = {"StockCode", "Quantity", "UnitPrice"}
                    if not required.issubset(headers.values()):
                        raise ValueError(
                            f"XLSX lacks required columns: {sorted(required)}; "
                            f"found={sorted(headers.values())}"
                        )
                else:
                    record = {
                        header: values.get(column, "")
                        for column, header in headers.items()
                        if header in {"StockCode", "Quantity", "UnitPrice"}
                    }
                    yield record
                row.clear()


def load_local_retail_xlsx(
    path: Path, max_rows: Optional[int], min_obs: int
) -> Dict[str, object]:
    # Keep the historical source token so an XLSX reconstruction reproduces
    # the frozen CSV bytes. The explicit input format and raw hash are recorded
    # separately in calibration_config.json and data_source_report.txt.
    result = _aggregate_retail_rows(
        iter_online_retail_xlsx(path),
        max_rows=max_rows,
        min_obs=min_obs,
        source_used="local_public_csv",
    )
    result["source_path"] = str(path)
    return result


def synthetic_sku_rows(n: int = 200) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    counts = []
    allocated = 0
    for seg in SEGMENT_DEFAULTS[:-1]:
        cnt = int(round(float(seg["share"]) * n))
        counts.append(cnt)
        allocated += cnt
    counts.append(n - allocated)
    idx = 0
    for seg, cnt in zip(SEGMENT_DEFAULTS, counts):
        for _ in range(max(0, cnt)):
            rows.append(
                {
                    "sku": f"synthetic_{idx:04d}",
                    "source": "synthetic_default",
                    "segment": seg["segment"],
                    "reference_price": seg["reference_price_median"],
                    "baseline_volume": seg["baseline_volume_median"],
                    "uncertainty_scale": seg["uncertainty_scale"],
                    "elasticity_mid": 0.5 * (seg["elasticity_low"] + seg["elasticity_high"]),
                    "fairness_band": seg["fairness_band"],
                }
            )
            idx += 1
    return rows


def segment_rows_from_skus(sku_rows: List[Dict[str, object]], source_used: str) -> List[Dict[str, object]]:
    if not sku_rows or source_used == "synthetic_default":
        return [dict(row, source_used=source_used) for row in SEGMENT_DEFAULTS]

    # Public transaction files rarely identify useful category labels. Use quantile
    # proxies based on price and volume to calibrate realistic segment medians.
    prices = [float(r["reference_price"]) for r in sku_rows]
    volumes = [float(r["baseline_volume"]) for r in sku_rows]
    p_med = median(prices, 1.0)
    v_med = median(volumes, 1.0)
    buckets: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in sku_rows:
        p = float(row["reference_price"])
        v = float(row["baseline_volume"])
        if p <= p_med and v >= v_med:
            bucket = "staples"
        elif p > p_med * 1.35:
            bucket = "premium"
        elif p <= p_med * 0.8:
            bucket = "private_label"
        elif v < v_med * 0.55:
            bucket = "seasonal"
        else:
            bucket = "mainstream"
        row["segment"] = bucket
        buckets[bucket].append(row)

    defaults = {r["segment"]: r for r in SEGMENT_DEFAULTS}
    out: List[Dict[str, object]] = []
    total = max(len(sku_rows), 1)
    for seg_name in [r["segment"] for r in SEGMENT_DEFAULTS]:
        group = buckets.get(seg_name, [])
        default = defaults[seg_name]
        out.append(
            {
                "segment": seg_name,
                "source_used": source_used,
                "share": len(group) / total if group else default["share"],
                "reference_price_median": median([r["reference_price"] for r in group], default["reference_price_median"]),
                "reference_price_log_sd": default["reference_price_log_sd"],
                "baseline_volume_median": median([r["baseline_volume"] for r in group], default["baseline_volume_median"]),
                "baseline_volume_log_sd": default["baseline_volume_log_sd"],
                "elasticity_low": default["elasticity_low"],
                "elasticity_high": default["elasticity_high"],
                "uncertainty_scale": max(default["uncertainty_scale"], median([r.get("price_cv", 0.0) for r in group], 0.0)),
                "fairness_band": default["fairness_band"],
                "weight_multiplier": default["weight_multiplier"],
            }
        )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="auto", choices=["auto", "uci_online_retail", "uci_online_retail_ii", "kaggle_retail_price_optimization", "synthetic_only"])
    parser.add_argument("--max-rows", type=int, default=200_000)
    parser.add_argument("--min-sku-observations", type=int, default=8)
    parser.add_argument("--cache-dir", default="data_cache/pathC")
    parser.add_argument("--output-dir", default="results/pathC/calibration")
    parser.add_argument(
        "--input-file",
        default=None,
        help="Explicit CSV or XLSX transaction file; overrides cache discovery.",
    )
    parser.add_argument(
        "--expected-sha256",
        default=None,
        help="Reject the input unless its SHA-256 matches this value.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    report: List[str] = []
    report.append("Path C data calibration report")
    report.append(f"Requested source: {args.source}")

    source_used = "synthetic_default"
    sku_rows: List[Dict[str, object]] = []
    input_record: Dict[str, object] = {}

    if args.source in {"auto", "uci_online_retail", "uci_online_retail_ii"}:
        local_input = Path(args.input_file).expanduser() if args.input_file else find_local_retail_input(cache_dir)
        if local_input is not None:
            try:
                local_input = local_input.resolve()
                if not local_input.is_file():
                    raise ValueError(f"input file does not exist: {local_input}")
                observed_sha256 = sha256_file(local_input)
                if args.expected_sha256 and observed_sha256.lower() != args.expected_sha256.lower():
                    raise ValueError(
                        "input SHA-256 mismatch: "
                        f"expected={args.expected_sha256.lower()} observed={observed_sha256}"
                    )
                if local_input.suffix.lower() == ".xlsx":
                    loaded = load_local_retail_xlsx(
                        local_input, args.max_rows, args.min_sku_observations
                    )
                elif local_input.suffix.lower() in {".csv", ".txt"}:
                    loaded = load_local_retail_csv(
                        local_input, args.max_rows, args.min_sku_observations
                    )
                else:
                    raise ValueError("input must be .xlsx, .csv, or .txt")
                sku_rows = loaded["sku_rows"]
                source_used = str(loaded["source_used"])
                input_record = {
                    "raw_input_format": local_input.suffix.lower().lstrip("."),
                    "raw_input_path": str(local_input),
                    "raw_input_bytes": local_input.stat().st_size,
                    "raw_input_sha256": observed_sha256,
                    "expected_raw_sha256": args.expected_sha256,
                    "raw_hash_verified": bool(args.expected_sha256),
                    "rows_examined": int(loaded["rows_examined"]),
                    "rows_retained_before_minimum": int(
                        loaded["rows_retained_before_minimum"]
                    ),
                }
                report.append(f"Loaded transaction input: {local_input}")
                report.append(f"Raw input bytes: {local_input.stat().st_size}")
                report.append(f"Raw input SHA-256: {observed_sha256}")
                if args.expected_sha256:
                    report.append("Raw input hash verified: True")
                report.append(f"Retained SKU rows: {len(sku_rows)}")
            except Exception as exc:
                report.append(f"Local transaction input was not usable: {local_input}; error={exc}")
                if args.input_file or args.expected_sha256:
                    raise
        else:
            report.append(f"No local cached retail CSV or XLSX found in {cache_dir}.")

    if args.source in {"auto", "kaggle_retail_price_optimization"} and source_used == "synthetic_default":
        kaggle = try_import("kaggle")
        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
        if kaggle is None or not kaggle_json.exists():
            report.append("Kaggle package/credentials unavailable; Kaggle data skipped.")
        else:
            report.append("Kaggle credentials detected, but no download is attempted automatically in this reproducibility-safe pass.")

    if args.source == "synthetic_only" or not sku_rows:
        sku_rows = synthetic_sku_rows()
        source_used = "synthetic_default"
        report.append("Using documented semi-synthetic calibration defaults.")

    segment_rows = segment_rows_from_skus(sku_rows, source_used)
    summary = [
        {
            "source_requested": args.source,
            "source_used": source_used,
            "sku_rows": len(sku_rows),
            "segment_rows": len(segment_rows),
            "public_data_used": source_used != "synthetic_default",
            "notes": "Public data unavailable or not used" if source_used == "synthetic_default" else "Local public-data aggregate used",
        }
    ]

    write_csv(out_dir / "sku_calibration.csv", sku_rows)
    write_csv(out_dir / "segment_calibration.csv", segment_rows)
    write_csv(out_dir / "calibration_summary.csv", summary)
    config_record = dict(vars(args))
    config_record.update(input_record)
    if args.source == "uci_online_retail":
        config_record["uci_dataset_url"] = UCI_ONLINE_RETAIL_DATASET_URL
        config_record["uci_download_url"] = UCI_ONLINE_RETAIL_DOWNLOAD_URL
        config_record["uci_dataset_doi"] = UCI_ONLINE_RETAIL_DOI
        config_record["official_zip_sha256"] = UCI_ONLINE_RETAIL_ZIP_SHA256
        config_record["official_xlsx_sha256"] = UCI_ONLINE_RETAIL_XLSX_SHA256
    (out_dir / "calibration_config.json").write_text(
        json.dumps(config_record, indent=2, sort_keys=True) + "\n"
    )
    report.append("")
    report.append(f"source_used: {source_used}")
    report.append(f"public_data_used: {source_used != 'synthetic_default'}")
    report.append("Generated files:")
    report.append("- calibration_summary.csv")
    report.append("- sku_calibration.csv")
    report.append("- segment_calibration.csv")
    report.append("- calibration_config.json")
    (out_dir / "data_source_report.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
