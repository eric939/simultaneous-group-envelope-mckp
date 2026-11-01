# Paper B v2.1 scale experiments — frozen protocol

Frozen on 1 November 2026, before any reported run. Code: branch
`experiments/v2.1` of the Paper B release worktree (commit recorded in each
phase's `environment_*.json`; the runner refuses a dirty tracked tree).
Driver: `research/scale_campaign.py`; its `PROTOCOL` dictionary, serialized as
`protocol.json` in every output directory, is the authoritative design.

Pilots were run on separate seeds (991, 992, 993, 99) and are not reported.
They led to two changes before freezing: (i) both threshold searches bound a
single-threshold interval by its fixed-threshold LP, as the LP-family search
does (the batched minimization is slow on a single threshold); (ii) the
integer time limit is 120 s, and a run certifies only if it finishes within
the limit plus one second.

Environment: CPython 3.12.12, NumPy 2.5.1, SciPy 1.18.0 (HiGHS via
`scipy.optimize.milp`), PySCIPOpt 6.2.1, from the release `uv.lock`, at
`/Volumes/LaCie/Archives/Codex/Projects/robust-mckp/Environments/paper-b-v2.1-20261101`;
all BLAS/OpenMP thread variables 1; phases run serially, one at a time.

## Phases (all reported regardless of outcome)

1. `lp-direct`: complete LP-family search with the batched bound and the
   clique-LP bound, and direct evaluation of every fixed-threshold LP with the
   same LP routine. Held-out seeds 601–605 (60 instances, 5 repetitions),
   the robustness configurations (36 instances, 3 repetitions) and the UCI
   pricing panel (9 instances, 3 repetitions); 60 s limit per call.
2. `integer`: exact robust MCKP. Threshold-interval search with each bound and
   HiGHS fixed-threshold subproblems, direct threshold enumeration with LP
   pruning, compact MILP with HiGHS and with SCIP. Structured families,
   n ∈ {360, 720, 1440}, seeds 701–703 (36 instances), and the UCI pricing
   panel (9 instances); one run each, 120 s limit.

Outputs: `/Volumes/LaCie/Archives/Codex/Projects/robust-mckp/Artifacts/paper-b-v2.1-scale/{lp-direct,integer}`.
Every failure, time-out and adverse ratio is reported.
