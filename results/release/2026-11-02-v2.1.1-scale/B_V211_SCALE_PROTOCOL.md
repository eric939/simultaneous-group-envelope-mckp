# Paper B v2.1.1 scale experiments — frozen protocol and amendment

This campaign replaces the v2.1 campaign in `../2026-11-01-v2.1-scale`, which
is kept unchanged as a superseded record. The v2.1 runs were made while
unrelated jobs loaded the machine (one-minute load 44 at the start of the
LP-family phase, 12 at the start of the integer phase), and the integer
methods started from a rounding heuristic that did not check the time limit.
27 integer runs exceeded the limit by more than one second: 18 runs of the
threshold-based methods (199–331 s, consistent with the heuristic) and 9
cold-started compact MILP runs (up to 123 s). Instances, seeds, methods and limits
are unchanged; the driver is `research/scale_campaign.py`, whose `PROTOCOL`
dictionary, serialized as `protocol.json` in every output directory, is the
authoritative design.

## Protocol v2.1.1 (frozen 1 November 2026, commit dc7a635)

- Every integer method starts cold, without a heuristic decision.
- An LP-family instance, or an integer run, starts only when the one-minute
  load average is below 2 (polled every 30 s).
- CPU time of the process, the load at the start, and search effort (interval
  bounds, multiplier evaluations, root bounds) are recorded.

Pilots on separate seeds (991, 992, 993, 99) are in `pilots-not-reported/`.

The `lp-direct` phase ran under v2.1.1 to completion (105 instances, every
instance started below the gate): `lp-direct/`, `lp-direct.log`.

The `integer` phase was started under v2.1.1 but unrelated jobs kept the load
above the gate for about ten hours after 19 of 225 runs; that attempt was
stopped and is kept in `stopped-attempts/integer-gated-v2.1.1/`.

## Amendment v2.1.2 (2 November 2026, commits dada982 and 0885e82)

The integer phase was rerun from the start in a Linux container (Docker
Desktop, linux/aarch64) on the same laptop, without the load gate, while the
unrelated jobs ran. The amendment changes only the execution environment:
instances, seeds, methods, limits and cold starts are those of v2.1.1, and
the container uses the same CPython 3.12.12, NumPy 2.5.1, SciPy 1.18.0 and
PySCIPOpt 6.2.1 from the release `uv.lock`. Because a container's load
average is that of its virtual machine, the host's one-minute load is sampled
outside the container every 5 s (`container/host_load_sampler.sh`, samples in
`container/host_load_samples.log`) and recorded at the start of every run.
Integer times are therefore measured under contention and are compared only
between methods on the same instance.

A first container attempt stopped after one run because reading the sample
file raced with its replacement; commit 0885e82 adds a retry, and the phase
was restarted from the start. That attempt is kept in
`stopped-attempts/integer-container-attempt-1/`; a one-instance smoke test on
a pilot seed is in `pilots-not-reported/pilot-container-smoke/`.

Reported integer evidence: `integer/`, `integer.log` (225 runs). The two
phases ran at different commits (dc7a635 for `lp-direct`, 0885e82 for
`integer`); they differ only in the container mode of
`research/scale_campaign.py`, so that file's hash differs between the two
`environment_*.json` files while the hashes of the method code agree. Build recipe:
`container/Dockerfile` (a clone of this repository at commit 0885e82) and
`container/run_v212_docker.sh`.

Every failure, time-out and adverse ratio is reported.
