# Two-paper project boundary

This repository is **Paper B**, *Simultaneous Group-Envelope Bounds for
Gamma-Robust Multiple-Choice Knapsack Problems*. It will be submitted to arXiv
under a new identifier.

The separate repository
[`eric939/certifying-robust-pricing-mckp`](https://github.com/eric939/certifying-robust-pricing-mckp)
is **Paper A**, the continuation of arXiv:2603.18653. Paper A has been revised
and frozen as version 2 before Paper B is submitted. Its definitive title is
*A Certifying MCKP Framework for Gamma-Robust Discrete Pricing*, and its public
repository is frozen on `main` at merge commit
`924fed52f5c8a2cd995e7102c3f42655c91e06f6`.

## Relationship

Paper A establishes the robust finite-menu pricing model, its exact
full-breakpoint MCKP decomposition, fixed-threshold LP hull and one-item
rounding certificate, and exact full-family search. Paper B cites that result
and takes the resulting family of fixed-threshold MCKP relaxations as its
starting point. It asks a different question: how can the largest LP value over
that family be certified without solving or materializing every relaxation?

Paper B answers that question with the exactly-one baseline cancellation,
simultaneous group-envelope evaluation, interval minimax-dominance theorem,
certified multiplier search, and adaptive LP-family certificate. Those results
do not belong to Paper A.

## Repository ownership

| Asset | Paper A: `certifying-robust-pricing-mckp` | Paper B: `simultaneous-group-envelope-mckp` |
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

1. Finish and verify Paper A in `certifying-robust-pricing-mckp`. **Complete.**
2. Submit and freeze arXiv:2603.18653v2. **Verified source package ready; arXiv
   account submission pending.**
3. Update Paper B's citation and relationship wording to the frozen Paper A
   metadata. **Complete in the current working branch.**
4. Verify every Paper B variant, anonymous supplement, and release manifest.
5. Submit Paper B under a new arXiv identifier.

The current Paper B manuscript contains the substantive overlap disclosure in
its introduction and pricing-specialization section, and its bibliography uses
Paper A's frozen v2 title and version.
