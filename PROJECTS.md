# Two-paper project boundary

This repository is **Paper B**, *Simultaneous Group-Envelope Bounds for
Gamma-Robust Multiple-Choice Knapsack Problems*. It will be submitted to arXiv
under a new identifier.

The separate repository
[`eric939/robust_pricing_mckp`](https://github.com/eric939/robust_pricing_mckp)
is **Paper A**, the continuation of arXiv:2603.18653. Paper A is being revised
and frozen as version 2 before Paper B is submitted.

## Relationship

Paper A establishes the robust finite-menu pricing model, its full-breakpoint
MCKP decomposition, fixed-threshold LP and rounding structure, and exact
certification machinery. Paper B takes the resulting family of fixed-threshold
MCKP relaxations as its starting point and asks a new question: how can their
largest LP value be bounded without solving or materializing every relaxation?

Paper B answers that question with the exactly-one baseline cancellation,
simultaneous group-envelope evaluation, interval minimax-dominance theorem,
certified multiplier search, and adaptive LP-family certificate. Those results
do not belong to Paper A.

## Repository ownership

| Asset | Paper A: `robust_pricing_mckp` | Paper B: `robust_mckp` |
| --- | --- | --- |
| arXiv record | 2603.18653v2 | new identifier |
| Primary application | finite-menu robust pricing | robust MCKP LP-family certification |
| Threshold role | exact full-breakpoint decomposition | simultaneous interval/all-threshold bound |
| Integer method | HullRound and exact full-breakpoint search | separately scoped integration audit |
| Main evidence | pricing, rounding, exact-search and sweep evidence | envelope, minimax, scaling and comparator evidence |
| Canonical code | pricing/full-breakpoint solver modules | compressed interval oracle and V4 campaign |

Shared low-level MCKP utilities may exist independently in both repositories.
Neither repository may depend on an unpublished path or generated artifact in
the other. Cross-project reuse must be attributed and frozen to a commit or
release.

## Release sequence

1. Finish and verify Paper A in `robust_pricing_mckp`.
2. Submit and freeze arXiv:2603.18653v2.
3. Update Paper B's citation and relationship wording to the frozen Paper A
   metadata.
4. Verify every Paper B variant, anonymous supplement, and release manifest.
5. Submit Paper B under a new arXiv identifier.

The current Paper B manuscript already contains the substantive overlap
disclosure in its pricing-specialization section. Its bibliography metadata
must be updated only after Paper A's v2 title and version are final.
