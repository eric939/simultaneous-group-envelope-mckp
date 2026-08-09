#!/usr/bin/env python3
"""Generate the v4 manuscript's tables, macros, figure, and evidence manifest."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

import matplotlib.pyplot as plt

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.v4_publication_campaign import geometric_mean  # noqa: E402


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def tex_escape(value: str) -> str:
    return value.replace("_", r"\_").replace("%", r"\%")


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def repository_relative(path: Path) -> str:
    """Return a portable repository-relative label without leaking local paths."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def source_snapshot() -> list[Path]:
    """Conservatively hash executable and validation sources behind the release."""
    paths: set[Path] = {ROOT / "Makefile", ROOT / "pyproject.toml"}
    paths.update((ROOT / "src" / "robust_mckp").glob("*.py"))
    paths.update((ROOT / "tests").glob("*.py"))
    paths.update((ROOT / "papers" / "v4").glob("*.tex"))
    for relative in (
        ".gitignore",
        "AGENTS.md",
        "CITATION.cff",
        "README.md",
        "REPRODUCIBILITY.md",
        "REVISION_HISTORY.md",
        "SUBMISSION.md",
        "papers/README.md",
        "papers/v4/README.md",
        "papers/v4/cover-letter.md",
        "protocol/deviations.md",
        "protocol/release-20260801.md",
        "research/bound_dominance.py",
        "research/benchmark_instances.py",
        "research/compressed_interval_oracle.py",
        "research/EVIDENCE_LEDGER_V4.csv",
        "research/exact_integration_campaign.py",
        "research/generate_v4_publication_artifacts.py",
        "research/integrated_exact_solver.py",
        "research/LITERATURE_NOVELTY_AUDIT_V4.md",
        "research/novelty_go_no_go.py",
        "research/structural_feasibility_study.py",
        "research/THEOREM_AUDIT_V4.md",
        "research/V4_EXPERIMENT_AUDIT_LOG.md",
        "research/v4_publication_campaign.py",
        "scripts/benchmark_solvers.py",
        "scripts/build_v4_anonymous_supplement.py",
        "scripts/run_pathC_data_calibration.py",
        "scripts/run_pathC_semisynthetic_application.py",
        "scripts/run_v4_publication_campaign.py",
        "scripts/verify_v4_release.py",
    ):
        paths.add(ROOT / relative)
    return sorted(path for path in paths if path.is_file())


def grouped(rows: list[dict[str, str]], field: str) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row[field], []).append(row)
    return groups


def build_macros(results: Path) -> str:
    validation = read_json(results / "validation_summary.json")
    kernel = read_json(results / "kernel_summary.json")
    common_trace = read_json(results / "common_trace_summary.json")
    primary = read_json(results / "primary_summary.json")
    robustness = read_json(results / "robustness_summary.json")
    stress = read_json(results / "stress_summary.json")
    application = read_json(results / "application_summary.json")
    external = read_json(results / "external_knapsack_summary.json")
    kernel_rows = read_csv(results / "kernel.csv")
    primary_rows = read_csv(results / "primary.csv")
    many_breakpoints = [row for row in kernel_rows if row["family"] == "many_breakpoints"]
    kernel_sizes = sorted({int(row["options"]) for row in many_breakpoints})
    compressed_kernel_times = [
        geometric_mean(
            float(row["compressed_total_seconds"])
            for row in many_breakpoints
            if int(row["options"]) == size
        )
        for size in kernel_sizes
    ]
    dense_kernel_times = [
        geometric_mean(
            float(row["dense_total_seconds"])
            for row in many_breakpoints
            if int(row["options"]) == size
        )
        for size in kernel_sizes
    ]
    timing = primary["timing_stability"]
    return "\n".join(
        [
            r"\newcommand{\PubValidationCases}{%d}" % validation["instances"],
            r"\newcommand{\PubValidationMaxError}{\num{%.2e}}" % validation["maximum_absolute_error"],
            r"\newcommand{\PubValidationMinSlack}{\num{%.2e}}" % validation["minimum_validity_slack"],
            r"\newcommand{\PubValidationCertViolation}{\num{%.2e}}" % validation["maximum_certificate_violation"],
            r"\newcommand{\PubValidationCertGap}{\num{%.2e}}" % validation["maximum_scaled_certificate_gap"],
            r"\newcommand{\PubKernelCases}{%d}" % kernel["instances"],
            r"\newcommand{\PubKernelLargeSpeedup}{%.2f}" % kernel["geomean_total_speedup_n_ge_360"],
            r"\newcommand{\PubKernelMaxError}{\num{%.2e}}" % kernel["maximum_identity_error"],
            r"\newcommand{\PubKernelCompressedSlope}{%.2f}" % log_log_slope(kernel_sizes, compressed_kernel_times),
            r"\newcommand{\PubKernelDenseSlope}{%.2f}" % log_log_slope(kernel_sizes, dense_kernel_times),
            r"\newcommand{\PubTraceCases}{%d}" % common_trace["instances"],
            r"\newcommand{\PubTraceIntervals}{%d}" % common_trace["interval_evaluations"],
            r"\newcommand{\PubTraceGeoSpeedup}{%.2f}" % common_trace["geomean_speedup"],
            r"\newcommand{\PubTraceWins}{%d}" % common_trace["wins"],
            r"\newcommand{\PubTraceDominancePct}{%.1f\%%}" % (100 * common_trace["bound_dominance_rate"]),
            r"\newcommand{\PubTraceCertGap}{\num{%.2e}}" % common_trace["maximum_scaled_certificate_gap"],
            r"\newcommand{\PubPrimaryCases}{%d}" % primary["instances"],
            r"\newcommand{\PubPrimaryGeoSpeedup}{%.2f}" % primary["geomean_speedup"],
            r"\newcommand{\PubPrimaryCILow}{%.2f}" % primary["geomean_speedup_ci95"][0],
            r"\newcommand{\PubPrimaryCIHigh}{%.2f}" % primary["geomean_speedup_ci95"][1],
            r"\newcommand{\PubPrimaryWins}{%d}" % primary["wins"],
            r"\newcommand{\PubPrimaryMaxDifference}{\num{%.2e}}" % primary["maximum_final_lower_bound_relative_difference"],
            r"\newcommand{\PubPrimaryCertGap}{\num{%.2e}}" % primary["maximum_scaled_oracle_optimality_gap"],
            r"\newcommand{\PubPrimaryMedianThetaPct}{%.1f\%%}" % (100 * primary["median_compressed_theta_fraction"]),
            r"\newcommand{\PubPrimarySignP}{\num{%.2e}}" % primary["sign_test_pvalue"],
            r"\newcommand{\PubPrimaryNLowSpeedup}{%.2f}" % primary["size_geomean_speedup"]["360"],
            r"\newcommand{\PubPrimaryNHighSpeedup}{%.2f}" % primary["size_geomean_speedup"]["1440"],
            r"\newcommand{\PubRepeatBlocks}{%d}" % timing["repeat_blocks"],
            r"\newcommand{\PubRepeatWins}{%d}" % timing["repeat_block_wins"],
            r"\newcommand{\PubRepeatMinSpeedup}{%.2f}" % timing["minimum_repeat_block_speedup"],
            r"\newcommand{\PubCompressedMedianCV}{%.2f\%%}" % (100 * timing["median_within_instance_cv"]["compressed"]),
            r"\newcommand{\PubCliqueMedianCV}{%.2f\%%}" % (100 * timing["median_within_instance_cv"]["clique"]),
            r"\newcommand{\PubCliqueMedianNnz}{\num{%.0f}}" % statistics.median(float(row["clique_interval_matrix_nnz"]) for row in primary_rows),
            r"\newcommand{\PubCliqueMedianCSRMiB}{%.2f}" % (statistics.median(float(row["clique_interval_matrix_storage_bytes"]) for row in primary_rows) / (2**20)),
            r"\newcommand{\PubFamilyCorrelated}{%.2f}" % primary["family_median_speedup"]["correlated_risk"],
            r"\newcommand{\PubFamilyDense}{%.2f}" % primary["family_median_speedup"]["dense_frontier"],
            r"\newcommand{\PubFamilyBreakpoints}{%.2f}" % primary["family_median_speedup"]["many_breakpoints"],
            r"\newcommand{\PubFamilyNearTie}{%.2f}" % primary["family_median_speedup"]["near_tie"],
            r"\newcommand{\PubRobustnessCases}{%d}" % robustness["instances"],
            r"\newcommand{\PubStressCases}{%d}" % stress["instances"],
            r"\newcommand{\PubStressGeoSpeedup}{%.2f}" % stress["geomean_speedup"],
            r"\newcommand{\PubStressNHighSpeedup}{%.2f}" % stress["size_geomean_speedup"]["5760"],
            r"\newcommand{\PubApplicationCases}{%d}" % application["instances"],
            r"\newcommand{\PubApplicationWins}{%d}" % application["wins"],
            r"\newcommand{\PubApplicationGeoSpeedup}{%.2f}" % application["geomean_speedup"],
            r"\newcommand{\PubApplicationCILow}{%.2f}" % application["geomean_speedup_ci95"][0],
            r"\newcommand{\PubApplicationCIHigh}{%.2f}" % application["geomean_speedup_ci95"][1],
            r"\newcommand{\PubApplicationNHighSpeedup}{%.2f}" % application["size_geomean_speedup"]["1440"],
            r"\newcommand{\PubExternalCases}{%d}" % external["instances"],
            r"\newcommand{\PubExternalWins}{%d}" % external["wins"],
            r"\newcommand{\PubExternalGeoSpeedup}{%.2f}" % external["geomean_speedup"],
            r"\newcommand{\PubExternalCILow}{%.2f}" % external["geomean_speedup_ci95"][0],
            r"\newcommand{\PubExternalCIHigh}{%.2f}" % external["geomean_speedup_ci95"][1],
            r"\newcommand{\PubExternalNHighSpeedup}{%.2f}" % external["size_geomean_speedup"]["10000"],
            r"\newcommand{\PubExternalCertGap}{\num{%.2e}}" % external["maximum_scaled_oracle_optimality_gap"],
        ]
    )


def primary_table(rows: list[dict[str, str]]) -> str:
    body = []
    for n, sample in sorted(grouped(rows, "n").items(), key=lambda item: int(item[0])):
        speedups = [float(row["adaptive_speedup"]) for row in sample]
        body.append(
            f"{n} & {len(sample)} & "
            f"{statistics.median(float(row['compressed_seconds']) for row in sample):.3f} & "
            f"{statistics.median(float(row['clique_seconds']) for row in sample):.3f} & "
            f"{geometric_mean(speedups):.2f} & "
            f"{sum(value > 1 for value in speedups)}/{len(sample)} & "
            f"{statistics.median(float(row['compressed_theta_evaluations']) for row in sample):.0f}/"
            f"{statistics.median(float(row['clique_theta_evaluations']) for row in sample):.0f} & "
            f"{statistics.median(float(row['clique_interval_lp_evaluations']) for row in sample):.0f} \\\\"
        )
    return "\n".join(
        [
            r"\begin{tabular}{rrrrrrrr}",
            r"\toprule",
            r"Groups & Cases & Compressed (s) & Clique (s) & Geometric speedup & Wins & \shortstack{Fixed-threshold LP\\evaluations (C / Q)} & \shortstack{Clique interval LP\\evaluations} \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )


def robustness_table(results: Path, rows: list[dict[str, str]]) -> str:
    summary = read_json(results / "robustness_summary.json")["by_configuration"]
    labels = {
        "sparse_budget": r"Sparse budget ($m=6,\Gamma=1$)",
        "wide_menu": r"Wide menu ($m=12,\Gamma=\lfloor\sqrt n\rfloor$)",
        "large_budget": r"Large budget ($m=6,\Gamma=0.2n$)",
    }
    body = []
    for name in ("sparse_budget", "wide_menu", "large_budget"):
        result = summary[name]
        sample = [row for row in rows if row["configuration"] == name]
        body.append(
            f"{labels[name]} & {result['geomean_speedup']:.2f} "
            f"[{result['geomean_speedup_ci95'][0]:.2f}, {result['geomean_speedup_ci95'][1]:.2f}] & "
            f"{result['wins']}/{result['instances']} & "
            f"{100 * statistics.median(float(row['compressed_theta_fraction']) for row in sample):.1f}\\% \\\\"
        )
    return "\n".join(
        [
            r"\begin{tabular}{lrrr}",
            r"\toprule",
            r"Configuration & Geometric speedup [95\% interval] & Wins & Thresholds evaluated \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )


def kernel_table(rows: list[dict[str, str]]) -> str:
    body = []
    for n, sample in sorted(grouped(rows, "n").items(), key=lambda item: int(item[0])):
        storage = [float(row["storage_ratio"]) for row in sample]
        body.append(
            f"{n} & {len(sample)} & "
            f"{geometric_mean(float(row['query_speedup']) for row in sample):.2f} & "
            f"{geometric_mean(float(row['total_speedup']) for row in sample):.2f} & "
            f"{statistics.median(storage):.2f} ({max(storage):.1f}) & "
            f"{max(float(row['identity_error']) for row in sample):.1e} \\\\"
        )
    return "\n".join(
        [
            r"\begin{tabular}{rrrrrr}",
            r"\toprule",
            r"Groups & Cases & Query speedup & Total speedup & Storage ratio & Maximum error \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )


def stress_table(rows: list[dict[str, str]]) -> str:
    body = []
    for n, sample in sorted(grouped(rows, "n").items(), key=lambda item: int(item[0])):
        speedups = [float(row["adaptive_speedup"]) for row in sample]
        body.append(
            f"{n} & {len(sample)} & "
            f"{statistics.median(float(row['compressed_seconds']) for row in sample):.3f} & "
            f"{statistics.median(float(row['clique_seconds']) for row in sample):.3f} & "
            f"{geometric_mean(speedups):.2f} & {sum(value > 1 for value in speedups)}/{len(sample)} \\\\"
        )
    return "\n".join(
        [
            r"\begin{tabular}{rrrrrr}",
            r"\toprule",
            r"Groups & Cases & Compressed (s) & Clique (s) & Geometric speedup & Wins \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )


def external_table(rows: list[dict[str, str]]) -> str:
    body = []
    for n, sample in sorted(grouped(rows, "n").items(), key=lambda item: int(item[0])):
        speedups = [float(row["adaptive_speedup"]) for row in sample]
        body.append(
            f"{int(n):,} & {len(sample)} & "
            f"{statistics.median(float(row['compressed_seconds']) for row in sample):.3f} & "
            f"{statistics.median(float(row['clique_seconds']) for row in sample):.3f} & "
            f"{geometric_mean(speedups):.2f} & {sum(value > 1 for value in speedups)}/{len(sample)} \\\\"
        )
    return "\n".join(
        [
            r"\begin{tabular}{rrrrrr}",
            r"\toprule",
            r"Binary items & Cases & Envelope (s) & Clique (s) & Geometric speedup & Wins \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )


def log_log_slope(x: list[float], y: list[float]) -> float:
    """Return the least-squares slope after taking natural logarithms."""
    lx = [math.log(value) for value in x]
    ly = [math.log(value) for value in y]
    mean_x = statistics.mean(lx)
    mean_y = statistics.mean(ly)
    return sum((a - mean_x) * (b - mean_y) for a, b in zip(lx, ly)) / sum(
        (a - mean_x) ** 2 for a in lx
    )


def speedup_figure(
    primary: list[dict[str, str]],
    stress: list[dict[str, str]],
    kernel: list[dict[str, str]],
    path: Path,
) -> None:
    rows = primary + stress
    families = ["dense_frontier", "correlated_risk", "near_tie", "many_breakpoints"]
    labels = {
        "dense_frontier": "Dense frontier",
        "correlated_risk": "Correlated risk",
        "near_tie": "Near tie",
        "many_breakpoints": "Many breakpoints",
    }
    colors = {
        "dense_frontier": "#0072B2",
        "correlated_risk": "#D55E00",
        "near_tie": "#009E73",
        "many_breakpoints": "#CC79A7",
    }
    markers = {
        "dense_frontier": "o",
        "correlated_risk": "s",
        "near_tie": "^",
        "many_breakpoints": "D",
    }
    fig, (kernel_ax, end_ax) = plt.subplots(1, 2, figsize=(7.3, 3.25), constrained_layout=True)

    many_breakpoints = [row for row in kernel if row["family"] == "many_breakpoints"]
    kernel_sizes = sorted({int(row["options"]) for row in many_breakpoints})
    compressed_times = [
        geometric_mean(
            float(row["compressed_total_seconds"])
            for row in many_breakpoints
            if int(row["options"]) == size
        )
        for size in kernel_sizes
    ]
    dense_times = [
        geometric_mean(
            float(row["dense_total_seconds"])
            for row in many_breakpoints
            if int(row["options"]) == size
        )
        for size in kernel_sizes
    ]
    compressed_slope = log_log_slope(kernel_sizes, compressed_times)
    dense_slope = log_log_slope(kernel_sizes, dense_times)
    kernel_ax.plot(
        kernel_sizes,
        compressed_times,
        color="#0072B2",
        marker="o",
        linewidth=1.5,
        markersize=4,
        label=rf"Compressed (slope {compressed_slope:.2f})",
    )
    kernel_ax.plot(
        kernel_sizes,
        dense_times,
        color="#D55E00",
        marker="s",
        linewidth=1.5,
        markersize=4,
        label=rf"Dense (slope {dense_slope:.2f})",
    )
    kernel_ax.set_xscale("log", base=2)
    kernel_ax.set_yscale("log")
    kernel_ax.set_xticks(kernel_sizes, [f"{value:,}" for value in kernel_sizes], rotation=25)
    kernel_ax.set_xlabel("Options $K$ ($B \\approx 5K/6$)")
    kernel_ax.set_ylabel("Kernel time (s)")
    kernel_ax.set_title("(a) Simultaneous evaluator", loc="left", fontsize=9)
    kernel_ax.grid(axis="y", which="both", color="0.88", linewidth=0.6)
    kernel_ax.legend(frameon=False, fontsize=7, loc="upper left")

    offsets = {name: (index - 1.5) * 0.018 for index, name in enumerate(families)}
    for family in families:
        sample = [row for row in rows if row["family"] == family]
        x = [int(row["n"]) * math.exp(offsets[family]) for row in sample]
        y = [float(row["adaptive_speedup"]) for row in sample]
        end_ax.scatter(
            x,
            y,
            s=20,
            alpha=0.72,
            color=colors[family],
            marker=markers[family],
            label=labels[family],
            edgecolors="0.2",
            linewidths=0.3,
        )
    sizes = sorted({int(row["n"]) for row in rows})
    means = [geometric_mean(float(row["adaptive_speedup"]) for row in rows if int(row["n"]) == n) for n in sizes]
    end_ax.plot(sizes, means, color="black", linewidth=1.4, marker="D", markersize=4, label="Geometric mean")
    end_ax.axhline(1.0, color="0.35", linewidth=0.8, linestyle="--")
    end_ax.set_xscale("log", base=2)
    end_ax.set_yscale("log")
    end_ax.set_xticks(sizes, [f"{value:,}" for value in sizes], rotation=25)
    end_ax.set_xlabel("Exactly-one groups $n$")
    end_ax.set_ylabel("Speedup over clique LP")
    end_ax.set_title("(b) Adaptive certificate", loc="left", fontsize=9)
    end_ax.grid(axis="y", which="both", color="0.88", linewidth=0.6)
    end_ax.legend(ncol=2, frameon=False, fontsize=6.7, loc="upper left")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def common_trace_figure(rows: list[dict[str, str]], path: Path) -> None:
    """Plot every matched-trace oracle speedup without adaptive-search effects."""
    families = ["dense_frontier", "correlated_risk", "near_tie", "many_breakpoints"]
    labels = {
        "dense_frontier": "Dense frontier",
        "correlated_risk": "Correlated risk",
        "near_tie": "Near tie",
        "many_breakpoints": "Many breakpoints",
    }
    colors = {
        "dense_frontier": "#0072B2",
        "correlated_risk": "#D55E00",
        "near_tie": "#009E73",
        "many_breakpoints": "#CC79A7",
    }
    fig, ax = plt.subplots(figsize=(6.9, 2.55), constrained_layout=True)

    for index, family in enumerate(families):
        base = len(families) - index - 1
        speedups = sorted(float(row["speedup"]) for row in rows if row["family"] == family)
        offsets = (
            [-0.17 + 0.34 * item / (len(speedups) - 1) for item in range(len(speedups))]
            if len(speedups) > 1
            else [0.0]
        )
        ax.scatter(
            speedups,
            [base + offset for offset in offsets],
            s=34,
            color=colors[family],
            edgecolors="0.2",
            linewidths=0.45,
            zorder=3,
        )
        ax.scatter(
            [geometric_mean(speedups)],
            [base],
            s=58,
            marker="D",
            color=colors[family],
            edgecolors="black",
            linewidths=0.85,
            zorder=4,
        )

    overall = geometric_mean(float(row["speedup"]) for row in rows)
    ax.axvline(1.0, color="0.45", linewidth=1.0, linestyle=":", zorder=1)
    ax.axvline(overall, color="black", linewidth=1.05, linestyle="--", zorder=1)
    ax.text(1.04, 3.40, "parity", color="0.35", fontsize=7.2, va="center")
    ax.text(
        overall + 0.06,
        3.40,
        rf"overall GM {overall:.2f}$\times$",
        color="black",
        fontsize=7.2,
        va="center",
    )
    ax.set_xlim(0.85, 5.55)
    ax.set_ylim(-0.42, 3.55)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_yticks(range(3, -1, -1), [labels[family] for family in families])
    ax.set_xlabel("Envelope-oracle speedup over clique LP")
    ax.grid(axis="x", color="0.88", linewidth=0.65)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    instance_handle = ax.scatter(
        [], [], s=32, color="0.55", edgecolors="0.2", linewidths=0.45
    )
    family_mean_handle = ax.scatter(
        [], [], s=52, marker="D", color="0.55", edgecolors="black", linewidths=0.85
    )
    ax.legend(
        [instance_handle, family_mean_handle],
        ["Instance", "Family geometric mean"],
        frameon=False,
        fontsize=7.2,
        ncol=2,
        loc="lower right",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "release")
    parser.add_argument("--paper", type=Path, default=ROOT / "papers" / "v4")
    args = parser.parse_args()
    results = args.results.resolve()
    paper = args.paper.resolve()

    primary = read_csv(results / "primary.csv")
    robustness = read_csv(results / "robustness.csv")
    kernel = read_csv(results / "kernel.csv")
    common_trace = read_csv(results / "common_trace.csv")
    stress = read_csv(results / "stress.csv")
    external = read_csv(results / "external_knapsack.csv")
    generated = paper / "generated"
    write(generated / "numbers.tex", build_macros(results))
    write(generated / "table-primary.tex", primary_table(primary))
    write(generated / "table-robustness.tex", robustness_table(results, robustness))
    write(generated / "table-kernel.tex", kernel_table(kernel))
    write(generated / "table-stress.tex", stress_table(stress))
    write(generated / "table-external.tex", external_table(external))
    speedup_figure(primary, stress, kernel, generated / "speedup-scaling.pdf")
    common_trace_figure(common_trace, generated / "common-trace-comparator.pdf")

    evidence_files = sorted(results.rglob("*.csv")) + sorted(results.rglob("*.json"))
    manifest = {
        "results_directory": repository_relative(results),
        "files": {str(path.relative_to(results)): sha256(path) for path in evidence_files},
        "source_files": {
            path.relative_to(ROOT).as_posix(): sha256(path) for path in source_snapshot()
        },
        "generated": [
            "generated/numbers.tex",
            "generated/table-primary.tex",
            "generated/table-robustness.tex",
            "generated/table-kernel.tex",
            "generated/table-stress.tex",
            "generated/table-external.tex",
            "generated/speedup-scaling.pdf",
            "generated/speedup-scaling.png",
            "generated/common-trace-comparator.pdf",
            "generated/common-trace-comparator.png",
        ],
    }
    write(generated / "evidence-manifest.json", json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
