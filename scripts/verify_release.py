#!/usr/bin/env python3
"""Verify the immutable reviewer-facing Paper B evidence release."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE = ROOT / "results" / "release" / "2026-08-09-paper-b-final-r5"
DEFAULT_V2_ANALYSIS = (
    ROOT / "results" / "release" / "2026-08-28-paper-b-v2-analysis"
)
FORBIDDEN_SUFFIXES = {
    ".aux",
    ".bbl",
    ".blg",
    ".docx",
    ".fdb_latexmk",
    ".fls",
    ".latex",
    ".pdf",
    ".synctex",
    ".tex",
}
FORBIDDEN_PARTS = {"build", "manuscript", "papers", "submission_packages"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"RELEASE VERIFY: FAIL: {message}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split(maxsplit=1)
        relative = relative.lstrip("* ")
        require(len(digest) == 64, f"invalid manifest digest for {relative}")
        require(relative not in entries, f"duplicate manifest entry {relative}")
        entries[relative] = digest
    return entries


def verify_manifest(release: Path) -> dict[str, str]:
    manifest_path = release / "MANIFEST.sha256"
    require(manifest_path.is_file(), f"missing SHA-256 manifest in {release}")
    manifest = read_manifest(manifest_path)
    actual_files = {
        path.relative_to(release).as_posix()
        for path in release.rglob("*")
        if path.is_file() and path != manifest_path
    }
    require(
        set(manifest) == actual_files,
        f"manifest does not cover every released file in {release}",
    )
    for relative, expected in manifest.items():
        require(sha256(release / relative) == expected, f"hash mismatch for {relative}")
    return manifest


def correlation(x: list[float], y: list[float]) -> float:
    require(len(x) == len(y) and len(x) > 1, "invalid correlation sample")
    mean_x = statistics.fmean(x)
    mean_y = statistics.fmean(y)
    numerator = math.fsum(
        (left - mean_x) * (right - mean_y) for left, right in zip(x, y, strict=True)
    )
    denominator = math.sqrt(
        math.fsum((value - mean_x) ** 2 for value in x)
        * math.fsum((value - mean_y) ** 2 for value in y)
    )
    require(denominator > 0.0, "degenerate correlation sample")
    return numerator / denominator


def verify_v2_analysis(base_release: Path, analysis_release: Path) -> int:
    verify_manifest(analysis_release)
    rows = csv_rows(analysis_release / "operating-region.csv")
    summary = json.loads(
        (analysis_release / "operating-region-summary.json").read_text(encoding="utf-8")
    )
    require(len(rows) == 122, "v2 operating-region analysis must contain 122 rows")
    require(int(summary["instances"]) == 122, "v2 analysis summary count mismatch")
    require(summary["analysis"] == "post_hoc_descriptive", "v2 analysis role changed")

    expected_panels = {
        "Primary": 60,
        "Robustness": 36,
        "Stress": 8,
        "Coefficient transfer": 9,
        "Pricing-derived": 9,
    }
    for panel, count in expected_panels.items():
        require(
            sum(row["panel"] == panel for row in rows) == count,
            f"unexpected v2 row count for {panel}",
        )

    input_hashes = summary["input_sha256"]
    for filename, expected in input_hashes.items():
        require(
            sha256(base_release / filename) == expected,
            f"v2 analysis input hash mismatch for {filename}",
        )

    log_speedup = [math.log(float(row["adaptive_speedup"])) for row in rows]
    density_correlation = correlation(
        [math.log(float(row["breakpoints_per_group"])) for row in rows],
        log_speedup,
    )
    iteration_correlation = correlation(
        [math.log(float(row["clique_iterations_per_bound"])) for row in rows],
        log_speedup,
    )
    correlations = summary["correlations"]
    require(
        math.isclose(
            density_correlation,
            float(correlations["log_speedup_vs_log_breakpoints_per_group"]),
            rel_tol=1e-14,
            abs_tol=1e-14,
        ),
        "v2 breakpoint-density correlation mismatch",
    )
    require(
        math.isclose(
            iteration_correlation,
            float(correlations["log_speedup_vs_log_clique_iterations_per_bound"]),
            rel_tol=1e-14,
            abs_tol=1e-14,
        ),
        "v2 clique-iteration correlation mismatch",
    )

    pricing = [row for row in rows if row["panel"] == "Pricing-derived"]
    observed_compressed = statistics.median(
        float(row["compressed_theta_fraction"]) for row in pricing
    )
    observed_clique = statistics.median(
        float(row["clique_theta_fraction"]) for row in pricing
    )
    medians = summary["pricing_theta_fraction_medians"]
    require(
        math.isclose(observed_compressed, float(medians["compressed"]), abs_tol=1e-15),
        "v2 pricing compressed fixed-LP median mismatch",
    )
    require(
        math.isclose(observed_clique, float(medians["clique"]), abs_tol=1e-15),
        "v2 pricing clique fixed-LP median mismatch",
    )
    return len(rows)


def verify_public_tree() -> None:
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=ROOT
    ).decode("utf-8").split("\0")
    for relative in filter(None, tracked):
        path = Path(relative)
        require(not (set(path.parts) & FORBIDDEN_PARTS), f"tracked publication path: {relative}")
        suffixes = {suffix.lower() for suffix in path.suffixes}
        require(not (suffixes & FORBIDDEN_SUFFIXES), f"tracked manuscript/build file: {relative}")
        require("data/raw" not in path.as_posix().lower(), f"tracked raw-data path: {relative}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument(
        "--v2-analysis-dir", type=Path, default=DEFAULT_V2_ANALYSIS
    )
    args = parser.parse_args()
    release = args.release_dir.resolve()
    analysis_release = args.v2_analysis_dir.resolve()
    require(release.is_dir(), f"missing release directory {release}")
    require(
        analysis_release.is_dir(),
        f"missing v2 analysis release directory {analysis_release}",
    )

    manifest = verify_manifest(release)

    expected_counts = {
        "validation": 40,
        "kernel": 48,
        "common_trace": 24,
        "primary": 60,
        "robustness": 36,
        "stress": 8,
        "application": 9,
        "external_knapsack": 9,
    }
    for stem, expected in expected_counts.items():
        rows = csv_rows(release / f"{stem}.csv")
        summary = json.loads((release / f"{stem}_summary.json").read_text(encoding="utf-8"))
        require(len(rows) == expected, f"unexpected row count in {stem}.csv")
        require(int(summary["instances"]) == expected, f"summary count mismatch for {stem}")

    for name in (
        "validation_summary.json",
        "kernel_summary.json",
        "primary_summary.json",
        "robustness_summary.json",
    ):
        summary = json.loads((release / name).read_text(encoding="utf-8"))
        failed = [key for key, passed in summary.get("gates", {}).items() if not passed]
        require(not failed, f"failed gates in {name}: {', '.join(failed)}")

    exact_dir = release / "exact_integration"
    exact_rows = csv_rows(exact_dir / "exact_integration.csv")
    exact_summary = json.loads(
        (exact_dir / "exact_integration_summary.json").read_text(encoding="utf-8")
    )
    require(len(exact_rows) == 12, "unexpected exact-integration row count")
    require(int(exact_summary["instances"]) == 12, "exact-integration summary count mismatch")
    require(
        float(exact_summary["maximum_certified_objective_difference"]) <= 2e-7,
        "certified objectives disagree in the exact-integration audit",
    )

    protocol_path = release / "protocol.json"
    protocol_digest = sha256(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    thread_controls = protocol.get("thread_environment", {})
    require(
        thread_controls and all(value == "1" for value in thread_controls.values()),
        "protocol does not require single-thread controls",
    )
    for environment in release.glob("environment_*.json"):
        record = json.loads(environment.read_text(encoding="utf-8"))
        require(record.get("protocol_sha256") == protocol_digest, f"protocol mismatch in {environment.name}")
        observed = record.get("thread_environment", {})
        require(
            all(observed.get(key) == value for key, value in thread_controls.items()),
            f"thread-control mismatch in {environment.name}",
        )

    exact_environment = json.loads(
        (exact_dir / "environment_exact_integration.json").read_text(encoding="utf-8")
    )
    require(exact_environment.get("protocol_sha256") == protocol_digest, "exact-audit protocol mismatch")
    for key in ("timestamp_utc", "numpy", "scipy", "pyscipopt", "scip", "machine"):
        require(bool(exact_environment.get(key)), f"missing {key} in exact-audit environment")
    observed_exact = exact_environment.get("thread_environment", {})
    require(
        all(observed_exact.get(key) == value for key, value in thread_controls.items()),
        "thread-control mismatch in exact-audit environment",
    )

    calibration = release / "uci_calibration"
    config = json.loads((calibration / "calibration_config.json").read_text(encoding="utf-8"))
    sku_rows = csv_rows(calibration / "sku_calibration.csv")
    require(int(config["max_rows"]) == 200_000, "unexpected UCI calibration row cap")
    require(len(sku_rows) == 2_549, "unexpected retained UCI SKU count")
    require(
        sum(int(float(row["observations"])) for row in sku_rows) == 192_451,
        "unexpected contributing UCI observation count",
    )

    v2_rows = verify_v2_analysis(release, analysis_release)
    verify_public_tree()

    print(
        "RELEASE VERIFY: PASS "
        f"({len(manifest)} evidence files, {sum(expected_counts.values())} campaign rows, "
        f"{len(exact_rows)} exact-integration rows, {v2_rows} v2 analysis rows)"
    )


if __name__ == "__main__":
    main()
