#!/usr/bin/env python3
"""Build the v2 exploratory performance analysis from frozen Paper B rows.

This is a post hoc descriptive analysis.  It is not a selector-training or
confirmation experiment.  The script records the hashes of every input file so
the derived artifact remains tied to the released August 2026 evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PANELS = (
    ("primary.csv", "Primary", "#0072B2", "o"),
    ("robustness.csv", "Robustness", "#009E73", "s"),
    ("stress.csv", "Stress", "#E69F00", "D"),
    ("external_knapsack.csv", "Coefficient transfer", "#CC79A7", "^"),
    ("application.csv", "Pricing-derived", "#D55E00", "X"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(release_root: Path) -> tuple[list[dict[str, object]], dict[str, str]]:
    rows: list[dict[str, object]] = []
    hashes: dict[str, str] = {}
    for filename, label, color, marker in PANELS:
        path = release_root / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        hashes[filename] = sha256(path)
        with path.open(newline="", encoding="utf-8") as handle:
            for raw in csv.DictReader(handle):
                n = int(raw["n"])
                threshold_count = int(raw["theta_count"])
                interval_evaluations = int(raw["clique_interval_lp_evaluations"])
                iterations = int(raw["clique_solver_iterations"])
                speedup = float(raw["adaptive_speedup"])
                compressed_theta_fraction = float(raw["compressed_theta_fraction"])
                clique_theta_fraction = float(raw["clique_theta_fraction"])
                if n <= 0 or threshold_count <= 0 or interval_evaluations <= 0:
                    raise ValueError(f"invalid positive-count field in {raw['instance']}")
                if speedup <= 0 or not math.isfinite(speedup):
                    raise ValueError(f"invalid speedup in {raw['instance']}")
                if not 0 <= compressed_theta_fraction <= 1:
                    raise ValueError(f"invalid compressed fraction in {raw['instance']}")
                if not 0 <= clique_theta_fraction <= 1:
                    raise ValueError(f"invalid clique fraction in {raw['instance']}")
                rows.append(
                    {
                        "panel": label,
                        "color": color,
                        "marker": marker,
                        "instance": raw["instance"],
                        "family": raw["family"],
                        "configuration": raw["configuration"],
                        "n": n,
                        "menu_size": int(raw["menu_size"]),
                        "gamma": int(raw["gamma"]),
                        "threshold_count": threshold_count,
                        "breakpoints_per_group": threshold_count / n,
                        "clique_iterations_per_bound": iterations / interval_evaluations,
                        "adaptive_speedup": speedup,
                        "compressed_theta_fraction": compressed_theta_fraction,
                        "clique_theta_fraction": clique_theta_fraction,
                    }
                )
    return rows, hashes


def correlation(rows: list[dict[str, object]], field: str) -> float:
    x = np.log(np.asarray([float(row[field]) for row in rows], dtype=float))
    y = np.log(np.asarray([float(row["adaptive_speedup"]) for row in rows], dtype=float))
    return float(np.corrcoef(x, y)[0, 1])


def write_derived_csv(rows: list[dict[str, object]], path: Path) -> None:
    fields = (
        "panel",
        "instance",
        "family",
        "configuration",
        "n",
        "menu_size",
        "gamma",
        "threshold_count",
        "breakpoints_per_group",
        "clique_iterations_per_bound",
        "adaptive_speedup",
        "compressed_theta_fraction",
        "clique_theta_fraction",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def make_figure(rows: list[dict[str, object]], output_dir: Path) -> dict[str, float]:
    corr_density = correlation(rows, "breakpoints_per_group")
    corr_iterations = correlation(rows, "clique_iterations_per_bound")

    n_values = np.asarray([int(row["n"]) for row in rows], dtype=float)
    lo, hi = float(np.log10(n_values).min()), float(np.log10(n_values).max())

    def marker_size(n: int) -> float:
        if hi == lo:
            return 36.0
        return 24.0 + 44.0 * (math.log10(n) - lo) / (hi - lo)

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 160,
            "savefig.dpi": 220,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.35), constrained_layout=True)

    for filename, label, color, marker in PANELS:
        del filename
        subset = [row for row in rows if row["panel"] == label]
        sizes = [marker_size(int(row["n"])) for row in subset]
        y = [float(row["adaptive_speedup"]) for row in subset]
        axes[0].scatter(
            [float(row["breakpoints_per_group"]) for row in subset],
            y,
            s=sizes,
            c=color,
            marker=marker,
            alpha=0.78,
            edgecolors="white",
            linewidths=0.45,
            label=label,
        )
        axes[1].scatter(
            [float(row["clique_iterations_per_bound"]) for row in subset],
            y,
            s=sizes,
            c=color,
            marker=marker,
            alpha=0.78,
            edgecolors="white",
            linewidths=0.45,
            label=label,
        )

    labels = (
        (axes[0], "Distinct thresholds per group, $B/n$", corr_density, "a"),
        (
            axes[1],
            "Clique simplex iterations per interval bound",
            corr_iterations,
            "b",
        ),
    )
    for axis, xlabel, corr, panel_letter in labels:
        axis.axhline(1.0, color="#444444", linestyle="--", linewidth=0.9)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.grid(True, which="major", color="#D9D9D9", linewidth=0.6)
        axis.grid(True, which="minor", color="#EEEEEE", linewidth=0.35)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Clique-LP / batched-method time")
        axis.set_title(f"({panel_letter}) Exploratory log-correlation $r={corr:.2f}$")

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="outside lower center",
        ncol=3,
        frameon=False,
    )
    fig.suptitle("Observed performance patterns across released experiments", y=1.02)
    fig.savefig(output_dir / "operating-region.png", bbox_inches="tight")
    plt.close(fig)
    return {
        "log_speedup_vs_log_breakpoints_per_group": corr_density,
        "log_speedup_vs_log_clique_iterations_per_bound": corr_iterations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    release_root = args.release_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, hashes = read_rows(release_root)
    write_derived_csv(rows, output_dir / "operating-region.csv")
    correlations = make_figure(rows, output_dir)
    pricing_rows = [row for row in rows if row["panel"] == "Pricing-derived"]
    pricing_compressed_theta_median = float(
        np.median([float(row["compressed_theta_fraction"]) for row in pricing_rows])
    )
    pricing_clique_theta_median = float(
        np.median([float(row["clique_theta_fraction"]) for row in pricing_rows])
    )
    summary = {
        "analysis": "post_hoc_descriptive",
        "instances": len(rows),
        "input_sha256": hashes,
        "correlations": correlations,
        "pricing_theta_fraction_medians": {
            "compressed": pricing_compressed_theta_median,
            "clique": pricing_clique_theta_median,
        },
        "interpretation": (
            "Hypothesis-generating only. The panels differ jointly in scale, "
            "menu width, breakpoint structure, and solver work."
        ),
    }
    (output_dir / "operating-region-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
