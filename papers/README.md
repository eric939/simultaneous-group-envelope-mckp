# Papers

All manuscript versions live here. Start in `current/`, which points to the
latest working draft in `v4/`.

| Folder | Status | Main file |
| --- | --- | --- |
| `current/` | Latest working draft (points to v4) | `main.tex` |
| `v1/` | Historical record; no distinct recoverable manuscript | `README.md` |
| `v2/` | March 2026 public arXiv paper and exact source archive | `paper.pdf` |
| `v3/` | May 2026 independent paper line | `paper.pdf` |
| `v4/` | Current August 2026 submission | `main.tex` |

The editable continuation of the V2/V3 pricing line now lives in the separate
Paper A repository, `eric939/robust_pricing_mckp`. The V2 and V3 directories
here remain immutable provenance. This repository's only editable manuscript
is Paper B under `v4/`.

V2 and V3 payloads are immutable. Their checksums are recorded in
`archive-manifest.sha256`. V4 is the only editable manuscript folder.

Use the root commands rather than entering version-specific build folders:

```bash
make paper
make package
make verify PYTHON=.venv/bin/python
```
