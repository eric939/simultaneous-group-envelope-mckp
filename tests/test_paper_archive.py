from __future__ import annotations

import hashlib
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_archive_manifest_payloads_match() -> None:
    manifest = ROOT / "papers" / "archive-manifest.sha256"
    for line in manifest.read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        payload = ROOT / relative.strip()
        assert payload.is_file(), relative
        assert hashlib.sha256(payload.read_bytes()).hexdigest() == expected


def test_code_archives_are_complete_git_exports() -> None:
    expected = {
        ROOT / "papers/v2/code.tar.gz":
            "robust_mckp-v2-code-baddc9e/README.md",
        ROOT / "papers/v3/code.tar.gz":
            "robust_mckp-v3-code-2e2829e/REVISION_HISTORY.md",
    }
    for archive, required_member in expected.items():
        with tarfile.open(archive, "r:gz") as bundle:
            assert required_member in bundle.getnames()


def test_cleanup_never_targets_versioned_papers() -> None:
    cleanup = (ROOT / "scripts/clean_workspace.py").read_text()
    assert 'ROOT / "papers" / "v1"' not in cleanup
    assert 'ROOT / "papers" / "v2"' not in cleanup
    assert 'ROOT / "papers" / "v3"' not in cleanup
    assert 'ROOT / "papers" / "v4" / "pdf"' not in cleanup
