#!/usr/bin/env python3
"""Build and identity-scan the deterministic v4 anonymous review supplement."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = "robust_mckp_v4_anonymous"
ZIP_TIMESTAMP = (2026, 7, 21, 0, 0, 0)
IDENTITY_TERMS = (
    rb"\bEri" + rb"c\b",
    rb"\bSha" + rb"o\b",
    b"er" + b"sh" + b"ao",
    rb"ET" + rb"H\s+Z(?:urich|rich|\\\"urich)?",
    rb"github\.com/" + b"eric939",
    b"/" + b"Users/",
)
IDENTITY = re.compile(rb"(?:" + b"|".join(IDENTITY_TERMS) + rb")", re.IGNORECASE)
TEXT_SUFFIXES = {".csv", ".json", ".md", ".py", ".tex", ".txt"}
PUBLIC_ONLY_SOURCE_FILES = {
    "CITATION.cff",
    "SUBMISSION.md",
    "papers/v4/cover-letter.md",
    "papers/v4/main.tex",
    "pyproject.toml",
}


README = """# Anonymous v4 review supplement

This identity-scanned archive contains the implementation, tests, serialized
protocol, raw result records, environment records, generated numerical paper
inputs, vector figure, and blind manuscript PDFs.

Create an environment and verify the released evidence with:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-anonymous.txt
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python scripts/verify_v4_release.py \
  --results results/release \
  --paper papers/v4
```

The archive-specific evidence manifest excludes only public package metadata
that would identify the author. Its remaining source hashes and all evidence
hashes are verified by the command above. Raw UCI transactions are not
redistributed; the released aggregate calibration is included.
"""

REQUIREMENTS = """numpy==2.4.4
scipy==1.17.1
matplotlib>=3.8
pytest>=8
tqdm>=4.66
"""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def add_file(payloads: dict[str, bytes], path: Path, archive_name: str | None = None) -> None:
    relative = archive_name or path.relative_to(ROOT).as_posix()
    payloads[relative] = path.read_bytes()


def pdf_searchable_bytes(path: Path) -> bytes:
    parts: list[bytes] = []
    for command in (["pdftotext", str(path), "-"], ["pdfinfo", str(path)]):
        result = subprocess.run(command, check=True, capture_output=True)
        parts.extend((result.stdout, result.stderr))
    return b"\n".join(parts)


def scan_payload(name: str, data: bytes) -> None:
    match = IDENTITY.search(name.encode("utf-8") + b"\n" + data)
    if match:
        token = match.group(0).decode("utf-8", errors="replace")
        raise SystemExit(f"ANONYMOUS PACKAGE: FAIL: identity token {token!r} in {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=ROOT / "results" / "release",
    )
    parser.add_argument("--paper", type=Path, default=ROOT / "papers" / "v4")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "papers" / "v4" / "anonymous-supplement.zip",
    )
    args = parser.parse_args()
    results = args.results.resolve()
    paper = args.paper.resolve()
    output = args.output.resolve()

    manifest_path = paper / "generated" / "evidence-manifest.json"
    public_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payloads: dict[str, bytes] = {}

    for relative in sorted(public_manifest["source_files"]):
        if relative in PUBLIC_ONLY_SOURCE_FILES:
            continue
        add_file(payloads, ROOT / relative)
    add_file(payloads, ROOT / "LICENSE")

    for path in sorted(results.rglob("*")):
        if path.is_file():
            relative = Path("results") / results.name / path.relative_to(results)
            add_file(payloads, path, relative.as_posix())

    for relative in public_manifest["generated"]:
        add_file(payloads, paper / relative, (Path("papers/v4") / relative).as_posix())

    for name, archive_name in (
        ("journal-blind.pdf", "papers/v4/pdf/paper-blind.pdf"),
        ("companion-blind.pdf", "papers/v4/pdf/companion-blind.pdf"),
    ):
        path = paper / name
        if not path.is_file():
            raise SystemExit(f"ANONYMOUS PACKAGE: FAIL: missing blind PDF {path}")
        add_file(payloads, path, archive_name)
        scan_payload(name, pdf_searchable_bytes(path))

    anonymous_manifest = dict(public_manifest)
    anonymous_manifest["anonymous_package"] = True
    anonymous_manifest["results_directory"] = f"results/{results.name}"
    anonymous_manifest["source_files"] = {
        name: digest
        for name, digest in public_manifest["source_files"].items()
        if name not in PUBLIC_ONLY_SOURCE_FILES
    }
    manifest_name = "papers/v4/generated/evidence-manifest.json"
    payloads[manifest_name] = (
        json.dumps(anonymous_manifest, indent=2, sort_keys=True).rstrip() + "\n"
    ).encode("utf-8")
    payloads["README_ANONYMOUS.md"] = README.encode("utf-8")
    payloads["requirements-anonymous.txt"] = REQUIREMENTS.encode("utf-8")

    for name, data in payloads.items():
        suffix = Path(name).suffix.lower()
        if suffix in TEXT_SUFFIXES or Path(name).name in {"LICENSE", "Makefile"}:
            scan_payload(name, data)

    package_manifest = {
        "archive_root": ARCHIVE_ROOT,
        "files": {name: sha256(data) for name, data in sorted(payloads.items())},
    }
    package_manifest_bytes = (
        json.dumps(package_manifest, indent=2, sort_keys=True).rstrip() + "\n"
    ).encode("utf-8")
    scan_payload("ANONYMOUS_PACKAGE_MANIFEST.json", package_manifest_bytes)
    payloads["ANONYMOUS_PACKAGE_MANIFEST.json"] = package_manifest_bytes

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".zip", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for name, data in sorted(payloads.items()):
                info = zipfile.ZipInfo(f"{ARCHIVE_ROOT}/{name}", ZIP_TIMESTAMP)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, data)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)

    print(
        "ANONYMOUS PACKAGE: PASS "
        f"({len(payloads)} files, sha256={sha256(output.read_bytes())}, output={output})"
    )


if __name__ == "__main__":
    main()
