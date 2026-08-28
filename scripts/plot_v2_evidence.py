#!/usr/bin/env python3
"""Regenerate the two v2 evidence figures from the frozen release CSVs.

The script deliberately preserves the submitted figures' observations,
aggregation, geometry, colors, markers, scales, and dimensions.  Only the
reader-facing method names are standardized for Paper B v2.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import math
import statistics
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results" / "local" / "v2-analysis"

EXPECTED_SHA256 = {
    "common_trace.csv": "8dc65a36cf7569fbbe7a210c0b06cf267d3b6616ca864bbefb195657d6a3e370",
    "kernel.csv": "607ad3b31a6a509be1c147fa73032e4ba80ed798340e6aa3afbd21a9a9bf9dd6",
    "primary.csv": "c4633cbddb474ab95249d26ee7e41bd1dd9927d9ce8bad760bc5e2c293047e4b",
    "stress.csv": "36327792b91ffd0590e23ac490eefe2727a07de3db6c0d0c07c6ee98b32ee7f6",
}

FAMILIES = [
    "dense_frontier",
    "correlated_risk",
    "near_tie",
    "many_breakpoints",
]
FAMILY_LABELS = {
    "dense_frontier": "Dense frontier",
    "correlated_risk": "Correlated risk",
    "near_tie": "Near tie",
    "many_breakpoints": "Many breakpoints",
}
FAMILY_COLORS = {
    "dense_frontier": "#0072B2",
    "correlated_risk": "#D55E00",
    "near_tie": "#009E73",
    "many_breakpoints": "#CC79A7",
}
FAMILY_MARKERS = {
    "dense_frontier": "o",
    "correlated_risk": "s",
    "near_tie": "^",
    "many_breakpoints": "D",
}

PREFIX_SUFFIX_LABEL = "Prefix–suffix evaluator"
BATCHED_LABEL = "Batched method"
CLIQUE_LABEL = "Clique-LP comparator"

EXPECTED_COMMON_TRACE_GM = 2.20706031603489
EXPECTED_COMMON_TRACE_FAMILY_GM = {
    "dense_frontier": 2.138224746801722,
    "correlated_risk": 2.989682098640983,
    "near_tie": 2.2161712449691695,
    "many_breakpoints": 1.6748469623925304,
}
EXPECTED_KERNEL_SLOPES = (0.9617990533425016, 2.0302073864256776)
EXPECTED_SIZE_GM = {
    360: 1.6924378962930147,
    720: 2.4158525271782683,
    1440: 3.2433100738717346,
    2880: 4.58060077256377,
    5760: 6.2824225357952255,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_frozen_csv(release_dir: Path, name: str) -> list[dict[str, str]]:
    path = release_dir / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen release file: {path}")
    observed = sha256(path)
    expected = EXPECTED_SHA256[name]
    if observed != expected:
        raise ValueError(
            f"Frozen-input hash mismatch for {name}: expected {expected}, got {observed}"
        )
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def geometric_mean(values: Iterable[float]) -> float:
    sample = list(values)
    if not sample or any(value <= 0 for value in sample):
        raise ValueError("Geometric means require a nonempty, positive sample")
    return math.exp(sum(math.log(value) for value in sample) / len(sample))


def log_log_slope(xs: list[int], ys: list[float]) -> float:
    log_x = [math.log(value) for value in xs]
    log_y = [math.log(value) for value in ys]
    mean_x = statistics.mean(log_x)
    mean_y = statistics.mean(log_y)
    return sum(
        (x_value - mean_x) * (y_value - mean_y)
        for x_value, y_value in zip(log_x, log_y, strict=True)
    ) / sum((x_value - mean_x) ** 2 for x_value in log_x)


def assert_close(name: str, observed: float, expected: float) -> None:
    if not math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"{name}: expected {expected:.16g}, got {observed:.16g}")


def validate_frozen_observations(
    common_trace: list[dict[str, str]],
    kernel: list[dict[str, str]],
    primary: list[dict[str, str]],
    stress: list[dict[str, str]],
) -> dict[str, object]:
    expected_rows = {
        "common_trace.csv": (common_trace, 24),
        "kernel.csv": (kernel, 48),
        "primary.csv": (primary, 60),
        "stress.csv": (stress, 8),
    }
    for name, (rows, count) in expected_rows.items():
        if len(rows) != count:
            raise ValueError(f"{name}: expected {count} rows, got {len(rows)}")

    trace_speeds = [float(row["speedup"]) for row in common_trace]
    trace_gm = geometric_mean(trace_speeds)
    assert_close("matched-trace geometric mean", trace_gm, EXPECTED_COMMON_TRACE_GM)
    if not all(speed > 1.0 for speed in trace_speeds):
        raise ValueError("Matched-trace plot no longer has 24/24 observations above parity")

    trace_family_gm = {}
    for family in FAMILIES:
        sample = [
            float(row["speedup"])
            for row in common_trace
            if row["family"] == family
        ]
        if len(sample) != 6:
            raise ValueError(f"Matched trace has {len(sample)} rows for {family}, expected 6")
        trace_family_gm[family] = geometric_mean(sample)
        assert_close(
            f"matched-trace {family} geometric mean",
            trace_family_gm[family],
            EXPECTED_COMMON_TRACE_FAMILY_GM[family],
        )

    kernel_sample = [row for row in kernel if row["family"] == "many_breakpoints"]
    kernel_sizes = sorted({int(row["options"]) for row in kernel_sample})
    prefix_times = [
        geometric_mean(
            float(row["compressed_total_seconds"])
            for row in kernel_sample
            if int(row["options"]) == size
        )
        for size in kernel_sizes
    ]
    dense_times = [
        geometric_mean(
            float(row["dense_total_seconds"])
            for row in kernel_sample
            if int(row["options"]) == size
        )
        for size in kernel_sizes
    ]
    prefix_slope = log_log_slope(kernel_sizes, prefix_times)
    dense_slope = log_log_slope(kernel_sizes, dense_times)
    assert_close("prefix-suffix log-log slope", prefix_slope, EXPECTED_KERNEL_SLOPES[0])
    assert_close("dense-materialization log-log slope", dense_slope, EXPECTED_KERNEL_SLOPES[1])

    complete_rows = primary + stress
    size_means = {
        size: geometric_mean(
            float(row["adaptive_speedup"])
            for row in complete_rows
            if int(row["n"]) == size
        )
        for size in sorted({int(row["n"]) for row in complete_rows})
    }
    if set(size_means) != set(EXPECTED_SIZE_GM):
        raise ValueError(f"Unexpected size grid: {sorted(size_means)}")
    for size, observed in size_means.items():
        assert_close(f"size-{size} geometric mean", observed, EXPECTED_SIZE_GM[size])

    return {
        "matched_trace": {
            "instances": len(common_trace),
            "wins": sum(speed > 1.0 for speed in trace_speeds),
            "geometric_mean_speedup": trace_gm,
            "family_geometric_means": trace_family_gm,
        },
        "kernel_many_breakpoints": {
            "options": kernel_sizes,
            "prefix_suffix_seconds": prefix_times,
            "dense_materialization_seconds": dense_times,
            "prefix_suffix_slope": prefix_slope,
            "dense_materialization_slope": dense_slope,
        },
        "complete_search": {
            "instances": len(complete_rows),
            "size_geometric_means": size_means,
        },
    }


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Preserve the released layout pass without writing a PDF artifact.  The
    # constrained-layout engine settles a few rotated labels during this first
    # render, after which the PNG is byte-identical to the v2 release.
    fig.savefig(
        io.BytesIO(),
        format="pdf",
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def common_trace_figure(rows: list[dict[str, str]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.9, 2.55), constrained_layout=True)

    for index, family in enumerate(FAMILIES):
        base = len(FAMILIES) - index - 1
        speedups = sorted(
            float(row["speedup"]) for row in rows if row["family"] == family
        )
        offsets = (
            [-0.17 + 0.34 * item / (len(speedups) - 1) for item in range(len(speedups))]
            if len(speedups) > 1
            else [0.0]
        )
        ax.scatter(
            speedups,
            [base + offset for offset in offsets],
            s=34,
            color=FAMILY_COLORS[family],
            edgecolors="0.2",
            linewidths=0.45,
            zorder=3,
        )
        ax.scatter(
            [geometric_mean(speedups)],
            [base],
            s=58,
            marker="D",
            color=FAMILY_COLORS[family],
            edgecolors="black",
            linewidths=0.85,
            zorder=4,
        )

    overall = geometric_mean(float(row["speedup"]) for row in rows)
    ax.axvline(1.0, color="0.45", linewidth=1.0, linestyle=":", zorder=1)
    ax.axvline(overall, color="black", linewidth=1.05, linestyle="--", zorder=1)
    ax.text(1.04, 3.4, "parity", color="0.35", fontsize=7.2, va="center")
    ax.text(
        overall + 0.06,
        3.4,
        f"overall GM {overall:.2f}$\\times$",
        color="black",
        fontsize=7.2,
        va="center",
    )

    ax.set_xlim(0.85, 5.55)
    ax.set_ylim(-0.42, 3.55)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_yticks(range(3, -1, -1), [FAMILY_LABELS[family] for family in FAMILIES])
    ax.set_xlabel(f"{BATCHED_LABEL} speedup over {CLIQUE_LABEL}")
    ax.grid(axis="x", color="0.88", linewidth=0.65)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    instance_handle = ax.scatter(
        [], [], s=32, color="0.55", edgecolors="0.2", linewidths=0.45
    )
    family_mean_handle = ax.scatter(
        [],
        [],
        s=52,
        marker="D",
        color="0.55",
        edgecolors="black",
        linewidths=0.85,
    )
    ax.legend(
        [instance_handle, family_mean_handle],
        ["Instance", "Family geometric mean"],
        frameon=False,
        fontsize=7.2,
        ncol=2,
        loc="lower right",
    )
    save_figure(fig, path)


def speedup_scaling_figure(
    primary: list[dict[str, str]],
    stress: list[dict[str, str]],
    kernel: list[dict[str, str]],
    path: Path,
) -> None:
    complete_rows = primary + stress
    fig, (kernel_ax, search_ax) = plt.subplots(
        1, 2, figsize=(7.3, 3.25), constrained_layout=True
    )

    kernel_sample = [row for row in kernel if row["family"] == "many_breakpoints"]
    kernel_sizes = sorted({int(row["options"]) for row in kernel_sample})
    prefix_times = [
        geometric_mean(
            float(row["compressed_total_seconds"])
            for row in kernel_sample
            if int(row["options"]) == size
        )
        for size in kernel_sizes
    ]
    dense_times = [
        geometric_mean(
            float(row["dense_total_seconds"])
            for row in kernel_sample
            if int(row["options"]) == size
        )
        for size in kernel_sizes
    ]
    prefix_slope = log_log_slope(kernel_sizes, prefix_times)
    dense_slope = log_log_slope(kernel_sizes, dense_times)

    kernel_ax.plot(
        kernel_sizes,
        prefix_times,
        color="#0072B2",
        marker="o",
        linewidth=1.5,
        markersize=4,
        label=f"{PREFIX_SUFFIX_LABEL} (slope {prefix_slope:.2f})",
    )
    kernel_ax.plot(
        kernel_sizes,
        dense_times,
        color="#D55E00",
        marker="s",
        linewidth=1.5,
        markersize=4,
        label=f"Dense materialization (slope {dense_slope:.2f})",
    )
    kernel_ax.set_xscale("log", base=2)
    kernel_ax.set_yscale("log")
    kernel_ax.set_xticks(
        kernel_sizes, [f"{value:,}" for value in kernel_sizes], rotation=25
    )
    kernel_ax.set_xlabel(r"Options $K$ ($B \approx 5K/6$)")
    kernel_ax.set_ylabel("Kernel time (s)")
    kernel_ax.set_title(f"(a) {PREFIX_SUFFIX_LABEL}", loc="left", fontsize=9)
    kernel_ax.grid(axis="y", which="both", color="0.88", linewidth=0.6)
    kernel_ax.legend(frameon=False, fontsize=7, loc="upper left")

    offsets = {name: (index - 1.5) * 0.018 for index, name in enumerate(FAMILIES)}
    for family in FAMILIES:
        sample = [row for row in complete_rows if row["family"] == family]
        x_values = [int(row["n"]) * math.exp(offsets[family]) for row in sample]
        y_values = [float(row["adaptive_speedup"]) for row in sample]
        search_ax.scatter(
            x_values,
            y_values,
            s=20,
            alpha=0.72,
            color=FAMILY_COLORS[family],
            marker=FAMILY_MARKERS[family],
            label=FAMILY_LABELS[family],
            edgecolors="0.2",
            linewidths=0.3,
        )

    sizes = sorted({int(row["n"]) for row in complete_rows})
    means = [
        geometric_mean(
            float(row["adaptive_speedup"])
            for row in complete_rows
            if int(row["n"]) == size
        )
        for size in sizes
    ]
    search_ax.plot(
        sizes,
        means,
        color="black",
        linewidth=1.4,
        marker="D",
        markersize=4,
        label="Geometric mean",
    )
    search_ax.axhline(1.0, color="0.35", linewidth=0.8, linestyle="--")
    search_ax.set_xscale("log", base=2)
    search_ax.set_yscale("log")
    search_ax.set_xticks(sizes, [f"{value:,}" for value in sizes], rotation=25)
    search_ax.set_xlabel(r"Exactly-one groups $n$")
    search_ax.set_ylabel(f"Speedup over {CLIQUE_LABEL}")
    search_ax.set_title(f"(b) {BATCHED_LABEL}", loc="left", fontsize=9)
    search_ax.grid(axis="y", which="both", color="0.88", linewidth=0.6)
    search_ax.legend(ncol=2, frameon=False, fontsize=6.7, loc="upper left")

    save_figure(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    release_dir = args.release_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    common_trace = read_frozen_csv(release_dir, "common_trace.csv")
    kernel = read_frozen_csv(release_dir, "kernel.csv")
    primary = read_frozen_csv(release_dir, "primary.csv")
    stress = read_frozen_csv(release_dir, "stress.csv")

    observations = validate_frozen_observations(common_trace, kernel, primary, stress)
    common_trace_figure(common_trace, output_dir / "common-trace-comparator.png")
    speedup_scaling_figure(
        primary, stress, kernel, output_dir / "speedup-scaling.png"
    )

    trace = observations["matched_trace"]
    kernel_stats = observations["kernel_many_breakpoints"]
    complete = observations["complete_search"]
    print(f"Frozen inputs verified in {release_dir}")
    print(
        "Matched trace: "
        f"{trace['wins']}/{trace['instances']} wins; "
        f"geometric mean {trace['geometric_mean_speedup']:.12f}x"
    )
    print(
        "Many-breakpoints slopes: "
        f"{kernel_stats['prefix_suffix_slope']:.12f} "
        f"and {kernel_stats['dense_materialization_slope']:.12f}"
    )
    size_text = ", ".join(
        f"n={size}: {value:.12f}x"
        for size, value in complete["size_geometric_means"].items()
    )
    print(f"Complete-search geometric means: {size_text}")
    print(f"Wrote {output_dir / 'common-trace-comparator.png'}")
    print(f"Wrote {output_dir / 'speedup-scaling.png'}")


if __name__ == "__main__":
    main()
