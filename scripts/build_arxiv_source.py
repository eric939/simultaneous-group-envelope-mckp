#!/usr/bin/env python3
"""Build the deterministic source-only arXiv package for Paper B."""
from __future__ import annotations

import argparse
import hashlib
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "papers" / "v4"
ZIP_TIMESTAMP = (2026, 8, 9, 0, 0, 0)
MEMBERS = (
    "main.tex",
    "generated/numbers.tex",
    "generated/table-primary.tex",
    "generated/table-robustness.tex",
    "generated/table-kernel.tex",
    "generated/table-external.tex",
    "generated/speedup-scaling.pdf",
    "generated/common-trace-comparator.pdf",
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PAPER / "arxiv-paper-b-source.zip",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    payloads = {name: (PAPER / name).read_bytes() for name in MEMBERS}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output.parent, suffix=".zip", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for name, data in sorted(payloads.items()):
                info = zipfile.ZipInfo(name, ZIP_TIMESTAMP)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, data)
        temporary.replace(output)
        output.chmod(0o644)
    finally:
        temporary.unlink(missing_ok=True)
    digest = sha256(output.read_bytes())
    (output.parent / "arxiv-paper-b-source.sha256").write_text(
        f"{digest}  {output.name}\n", encoding="utf-8"
    )
    print(
        f"ARXIV SOURCE: PASS ({len(payloads)} files, sha256={digest}, output={output})"
    )


if __name__ == "__main__":
    main()
