Dear Editors:

Please consider the manuscript “Simultaneous Group-Envelope Bounds for
Γ-Robust Multiple-Choice Knapsack Problems” as a Focused Technical submission
to *Operations Research*.

The paper studies a documented computational bottleneck in robust discrete
optimization. Under cardinality-budget uncertainty, a binary model reduces to
a family of nominal problems indexed by deviation thresholds. Büsing, Gersing,
and Koster (2023) report that, in their large-scale study of bounded-threshold
robust relaxations, LP solves consume 33.1% of runtime on average among solved
instances and 93.17% at one million items, where the root LP alone averages
1,062 seconds. For a multiple-choice knapsack problem with exactly-one groups,
we show that
Lagrangian dualization cancels the threshold-dependent group baselines. This
exposes two simple group envelopes and permits simultaneous evaluation of one
multiplier across the complete threshold family. The resulting minimax
interval certificate is valid, exact on feasible singleton intervals, and
provably no weaker than the bounded-threshold group-clique relaxation. The
paper also gives a convex bracketing algorithm with an explicit minimization
gap, so the implemented bound inherits the exact dominance theorem up to its
stated tolerance, and an adaptive algorithm with explicit global-bound
invariants. The primary deliverable is a faster complete threshold-disjunctive
LP certificate that can serve as a root certificate or interval bound in an
outer robust search.

The closest published algorithmic system, DnC+, is not a scientifically
matched runtime comparator for this claim: it combines threshold filtering,
estimators, cuts, incumbent improvement, and early termination in a Java/Gurobi
solver for objective uncertainty. We instead isolate the relevant
bounded-threshold clique formulation. Both certificates use the same adaptive
search, fixed-threshold LP solver, candidate rule, stopping tolerance, and
timing policy; only the interval upper bound differs. A common-trace experiment
also evaluates both oracles on identical intervals. The comparison therefore
tests the proposed component directly without presenting it as a reproduction
of or whole-solver contest with DnC+.

The computational study is designed around the paper’s mechanism rather than
one aggregate benchmark. Separate panels test algebraic identity, kernel
scaling, identical-interval oracle behavior, end-to-end certification,
robustness to altered budgets and menu widths, stress scaling through 5,760
groups, and two separately scoped external-coefficient panels. In the fixed-design
60-instance primary comparison, both methods reach the same prescribed
tolerance; every proposed interval bound carries a numerical minimization
certificate, the method wins all paired timings, and the geometric-mean
speedup is 2.33-fold with a design-stratified 95% interval of 2.16 to 2.51.
On nine coefficient sets transferred from a published robust-knapsack archive,
it again wins every instance-level median comparison with a 2.31-fold geometric-mean speedup. A separate
exact-integration audit is deliberately reported as a scope boundary: integer
subproblem work can dominate, and the paper makes no claim of universal
superiority over compact mixed-integer optimization.

For transparency, the author also has a companion preprint, “A Certifying MCKP
Framework for Γ-Robust Discrete Pricing” (arXiv:2603.18653v2). That Paper A
derives the robust finite-menu pricing model, its exact full-breakpoint MCKP
decomposition, fixed-threshold hull and one-item rounding certificates, and
exact full-family search. The submitted manuscript cites Paper A and takes its
threshold family as a motivating specialization. It asks a different
algorithmic question: how to certify the largest LP value over the family
without solving or materializing every relaxation. Its baseline cancellation,
simultaneous two-envelope evaluator, interval minimax-dominance theorem,
certified multiplier search, adaptive LP-family certificate, and corresponding
evidence package do not appear in Paper A. The relationship is therefore
cumulative but nonduplicative: Paper A supplies the model and fixed-threshold
certification foundation, while this Paper B supplies the all-threshold
LP-family accelerator. The manuscript does not present the Bertsimas–Sim
threshold reduction or fixed-MCKP LP geometry as new.

The manuscript and electronic companion include data-and-code statements. A
public reproducibility repository contains the serialized protocol, raw timing
records, environment records, tests, source hashes, and generators for every
reported numerical artifact. A separate identity-scanned archive is available
for anonymous review.

The manuscript is not under review elsewhere, has not appeared in archival
journal form, and presents original work. The author declares no relevant
financial conflict of interest.

Thank you for your consideration.

Sincerely,

Eric Shao

Department of Mathematics, ETH Zürich

ershao@student.ethz.ch
