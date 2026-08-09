#!/usr/bin/env python3
"""Verify the immutable reviewer-facing Paper B evidence release."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE = ROOT / "results" / "release" / "2026-08-09-paper-b-final-r5"
FORBIDDEN_PUBLIC_PATHS = (
    "papers",
    "submission_packages",
    "PROJECTS.md",
    "REVISION_HISTORY.md",
    "SUBMISSION.md",
)


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE)
    args = parser.parse_args()
    release = args.release_dir.resolve()
    require(release.is_dir(), f"missing release directory {release}")

    for relative in FORBIDDEN_PUBLIC_PATHS:
        require(not (ROOT / relative).exists(), f"publication-only path is public: {relative}")

    manifest_path = release / "MANIFEST.sha256"
    require(manifest_path.is_file(), "missing SHA-256 manifest")
    manifest = read_manifest(manifest_path)
    actual_files = {
        path.relative_to(release).as_posix()
        for path in release.rglob("*")
        if path.is_file() and path != manifest_path
    }
    require(set(manifest) == actual_files, "manifest does not cover every released evidence file")
    for relative, expected in manifest.items():
        require(sha256(release / relative) == expected, f"hash mismatch for {relative}")

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

    print(
        "RELEASE VERIFY: PASS "
        f"({len(manifest)} evidence files, {sum(expected_counts.values())} campaign rows, "
        f"{len(exact_rows)} exact-integration rows)"
    )


if __name__ == "__main__":
    main()
