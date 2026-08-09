# Version 4 — current paper

This is the only editable manuscript version. `papers/current` points here.

## What to edit

- `main.tex`: canonical manuscript source.
- `journal.tex` and `journal-blind.tex`: public and anonymous journal builds.
- `companion.tex` and `companion-blind.tex`: electronic companion builds.
- `executive-summary.tex`: one-page summary.
- `cover-letter.md`: submission cover letter.

Do not hand-edit `generated/`; it is rebuilt from `results/release`.

## Ready-to-use files

Final PDFs are in `pdf/`. The identity-scanned review archive is
`anonymous-supplement.zip`.

From the repository root:

```bash
make evidence PYTHON=.venv/bin/python
make verify PYTHON=.venv/bin/python
make package
make anonymous PYTHON=.venv/bin/python
```
