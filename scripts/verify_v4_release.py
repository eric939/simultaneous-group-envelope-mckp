#!/usr/bin/env python3
"""Verify the released v4 evidence and regenerate its textual paper artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results" / "release"
DEFAULT_PAPER = ROOT / "papers" / "v4"
TEXT_ARTIFACTS = (
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
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"V4 RELEASE VERIFY: FAIL: {message}")


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--paper", type=Path, default=DEFAULT_PAPER)
    args = parser.parse_args()
    results = args.results.resolve()
    paper = args.paper.resolve()
    manifest_path = paper / "generated" / "evidence-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    for relative, expected in manifest["files"].items():
        path = results / relative
        require(path.is_file(), f"missing evidence file {relative}")
        require(sha256(path) == expected, f"hash mismatch for {relative}")

    for relative, expected in manifest["source_files"].items():
        path = ROOT / relative
        require(path.is_file(), f"missing source file {relative}")
        require(sha256(path) == expected, f"source hash mismatch for {relative}")

    for name in (
        "validation_summary.json",
        "kernel_summary.json",
        "primary_summary.json",
        "robustness_summary.json",
    ):
        summary = json.loads((results / name).read_text(encoding="utf-8"))
        failed = [key for key, passed in summary.get("gates", {}).items() if not passed]
        require(not failed, f"failed gates in {name}: {', '.join(failed)}")

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
        rows = csv_rows(results / f"{stem}.csv")
        require(len(rows) == expected, f"unexpected row count in {stem}.csv")
        summary = json.loads((results / f"{stem}_summary.json").read_text(encoding="utf-8"))
        require(int(summary["instances"]) == expected, f"summary count mismatch for {stem}")

    exact_dir = results / "exact_integration"
    exact_rows = csv_rows(exact_dir / "exact_integration.csv")
    exact_summary = json.loads(
        (exact_dir / "exact_integration_summary.json").read_text(encoding="utf-8")
    )
    require(len(exact_rows) == 12, "unexpected row count in exact integration audit")
    require(
        int(exact_summary["instances"]) == 12,
        "summary count mismatch for exact integration audit",
    )
    require(
        float(exact_summary["maximum_certified_objective_difference"]) <= 2e-7,
        "certified objectives disagree in exact integration audit",
    )

    protocol_digest = sha256(results / "protocol.json")
    protocol = json.loads((results / "protocol.json").read_text(encoding="utf-8"))
    require("application" in protocol, "application design missing from protocol")
    required_thread_environment = protocol.get("thread_environment", {})
    require(
        required_thread_environment
        and all(value == "1" for value in required_thread_environment.values()),
        "protocol does not require single-thread environment controls",
    )
    for environment in results.glob("environment_*.json"):
        record = json.loads(environment.read_text(encoding="utf-8"))
        require(
            record["protocol_sha256"] == protocol_digest,
            f"protocol digest mismatch in {environment.name}",
        )
        observed_threads = record.get("thread_environment", {})
        require(
            all(
                observed_threads.get(key) == value
                for key, value in required_thread_environment.items()
            ),
            f"thread controls were not recorded in {environment.name}",
        )

    exact_environment = json.loads(
        (exact_dir / "environment_exact_integration.json").read_text(encoding="utf-8")
    )
    observed_exact_threads = exact_environment.get("thread_environment", {})
    require(
        exact_environment.get("protocol_sha256") == protocol_digest,
        "protocol digest mismatch in exact integration audit",
    )
    for key in ("timestamp_utc", "numpy", "scipy", "pyscipopt", "scip", "machine"):
        require(bool(exact_environment.get(key)), f"missing {key} in exact integration environment")
    require(
        all(
            observed_exact_threads.get(key) == value
            for key, value in required_thread_environment.items()
        ),
        "thread controls were not recorded in exact integration audit",
    )

    calibration = results / "uci_calibration"
    config = json.loads((calibration / "calibration_config.json").read_text(encoding="utf-8"))
    sku_rows = csv_rows(calibration / "sku_calibration.csv")
    require(int(config["max_rows"]) == 200_000, "unexpected UCI calibration row cap")
    require(len(sku_rows) == 2_549, "unexpected retained UCI SKU count")
    require(
        sum(int(float(row["observations"])) for row in sku_rows) == 192_451,
        "unexpected contributing UCI observation count",
    )

    with tempfile.TemporaryDirectory(prefix="robust_mckp_v4_verify_") as tmp:
        regenerated = Path(tmp) / "paper"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "research" / "generate_v4_publication_artifacts.py"),
                "--results",
                str(results),
                "--paper",
                str(regenerated),
            ],
            check=True,
            cwd=ROOT,
        )
        for relative in TEXT_ARTIFACTS:
            expected = (paper / relative).read_bytes()
            actual = (regenerated / relative).read_bytes()
            require(actual == expected, f"generated artifact differs: {relative}")

    print(
        "V4 RELEASE VERIFY: PASS "
        f"({len(manifest['files'])} evidence files, "
        f"{len(manifest['source_files'])} source files, "
        f"{len(TEXT_ARTIFACTS)} regenerated text artifacts)"
    )


if __name__ == "__main__":
    main()
