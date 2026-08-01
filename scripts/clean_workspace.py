#!/usr/bin/env python3
"""Preview or remove ignored, reproducible workspace debris.

The complete ``papers/`` history and ``results/release`` are never cleanup
candidates.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKIPPED_ROOTS = {ROOT / ".git", ROOT / ".venv", ROOT / "data_cache"}
TRANSIENT_DIRECTORIES = (
    ROOT / ".pytest_cache",
    ROOT / "tmp",
    ROOT / "src" / "robust_mckp.egg-info",
    ROOT / "papers" / "v4" / "build",
)
LATEX_SUFFIXES = (".aux", ".log", ".out", ".toc", ".xdv", ".synctex.gz")


def tracked_files() -> set[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return {
        (ROOT / item.decode("utf-8")).resolve()
        for item in result.stdout.split(b"\0")
        if item
    }


def skipped(path: Path) -> bool:
    resolved = path.resolve()
    return any(resolved == root or root in resolved.parents for root in SKIPPED_ROOTS)


def candidates(tracked: set[Path]) -> list[Path]:
    found: set[Path] = set()
    results_root = ROOT / "results"
    result_directories = (
        tuple(
            path
            for path in results_root.iterdir()
            if path.is_dir()
            and path.name != "release"
        )
        if results_root.is_dir()
        else ()
    )
    for path in (*TRANSIENT_DIRECTORIES, *result_directories):
        resolved = path.resolve()
        contains_tracked_file = any(
            item == resolved or resolved in item.parents for item in tracked
        )
        if path.exists() and not contains_tracked_file:
            found.add(path)
    for path in ROOT.rglob(".DS_Store"):
        if not skipped(path) and path.resolve() not in tracked:
            found.add(path)
    for path in ROOT.rglob("__pycache__"):
        if not skipped(path) and not any(path.resolve() in item.parents for item in tracked):
            found.add(path)
    paper = ROOT / "papers" / "v4"
    if paper.is_dir():
        for path in paper.iterdir():
            if not path.is_file() or path.resolve() in tracked:
                continue
            if path.suffix == ".pdf" or any(path.name.endswith(suffix) for suffix in LATEX_SUFFIXES):
                found.add(path)
    return sorted(found, key=lambda path: (len(path.parts), path.as_posix()), reverse=True)


def size(path: Path) -> int:
    if path.is_file() or path.is_symlink():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="perform the displayed removals")
    args = parser.parse_args()
    paths = candidates(tracked_files())
    total = sum(size(path) for path in paths)
    action = "REMOVE" if args.apply else "WOULD REMOVE"
    for path in paths:
        print(f"{action}: {path.relative_to(ROOT)}")
    print(f"{action}: {len(paths)} paths, {total / (1024**2):.1f} MiB")
    if args.apply:
        for path in paths:
            if path.exists():
                remove(path)


if __name__ == "__main__":
    main()
