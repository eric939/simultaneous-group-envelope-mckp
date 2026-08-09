#!/usr/bin/env python3
"""Path C calibration layer for semi-synthetic robust pricing applications.

The script tries lightweight public-data hooks first, but never requires
credentials or large raw downloads. If public data are unavailable in the local
environment, it writes documented semi-synthetic calibration defaults.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

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


def find_local_retail_csv(cache_dir: Path) -> Optional[Path]:
    candidates = []
    for pattern in ["*.csv", "*.txt"]:
        candidates.extend(cache_dir.glob(pattern))
    for path in candidates:
        lower = path.name.lower()
        if "online" in lower or "retail" in lower or "transaction" in lower:
            return path
    return candidates[0] if candidates else None


def load_local_retail_csv(path: Path, max_rows: Optional[int], min_obs: int) -> Dict[str, object]:
    """Aggregate a local retail-like CSV if it has common transaction columns."""

    sku_stats: Dict[str, Dict[str, object]] = {}
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
        for idx, row in enumerate(reader):
            if max_rows is not None and idx >= max_rows:
                break
            try:
                sku = str(row[sku_col]).strip()
                qty = float(row[qty_col])
                price = float(row[price_col])
            except Exception:
                continue
            if not sku or qty <= 0 or price <= 0 or price > 10_000:
                continue
            rec = sku_stats.setdefault(sku, {"sku": sku, "obs": 0, "quantity": 0.0, "revenue": 0.0, "prices": []})
            rec["obs"] = int(rec["obs"]) + 1
            rec["quantity"] = float(rec["quantity"]) + qty
            rec["revenue"] = float(rec["revenue"]) + qty * price
            rec["prices"].append(price)

    rows: List[Dict[str, object]] = []
    for sku, rec in sku_stats.items():
        prices = [float(p) for p in rec["prices"]]
        if int(rec["obs"]) < min_obs:
            continue
        mean_price = float(rec["revenue"]) / max(float(rec["quantity"]), 1e-12)
        price_med = median(prices)
        price_sd = statistics.pstdev(prices) if len(prices) > 1 else 0.0
        rows.append(
            {
                "sku": sku,
                "source": "local_public_csv",
                "observations": int(rec["obs"]),
                "total_quantity": float(rec["quantity"]),
                "total_revenue": float(rec["revenue"]),
                "reference_price": price_med,
                "average_unit_price": mean_price,
                "price_cv": price_sd / price_med if price_med > 0 else 0.0,
                "baseline_volume": float(rec["quantity"]) / max(int(rec["obs"]), 1),
            }
        )
    return {"sku_rows": rows, "source_used": "local_public_csv", "source_path": str(path)}


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

    if args.source in {"auto", "uci_online_retail", "uci_online_retail_ii"}:
        ucimlrepo = try_import("ucimlrepo")
        if ucimlrepo is None:
            report.append("ucimlrepo is not installed; UCI API access skipped.")
        else:
            report.append("ucimlrepo is installed, but this lightweight script does not download raw data automatically.")
        local_csv = find_local_retail_csv(cache_dir)
        if local_csv is not None:
            try:
                loaded = load_local_retail_csv(local_csv, args.max_rows, args.min_sku_observations)
                sku_rows = loaded["sku_rows"]
                source_used = str(loaded["source_used"])
                report.append(f"Loaded local cached retail-like CSV: {local_csv}")
                report.append(f"Retained SKU rows: {len(sku_rows)}")
            except Exception as exc:
                report.append(f"Local cached CSV was present but not usable: {local_csv}; error={exc}")
        else:
            report.append(f"No local cached retail CSV found in {cache_dir}.")

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
    (out_dir / "calibration_config.json").write_text(json.dumps(vars(args), indent=2))
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
