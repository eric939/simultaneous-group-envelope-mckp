Dear Editors:

Please consider the manuscript “Simultaneous Group-Envelope Bounds for
Γ-Robust Multiple-Choice Knapsack Problems” as a Focused Technical submission
to *Operations Research*.

The paper studies a specific computational bottleneck in robust discrete
optimization. Under cardinality-budget uncertainty, a binary model reduces to
a family of nominal problems indexed by deviation thresholds. For a
multiple-choice knapsack problem with exactly-one groups, we show that
Lagrangian dualization cancels the threshold-dependent group baselines. This
exposes two simple group envelopes and permits simultaneous evaluation of one
multiplier across the complete threshold family. The resulting minimax
interval certificate is valid, exact on feasible singleton intervals, and
provably no weaker than the bounded-threshold group-clique relaxation. The
paper also gives a convex bracketing algorithm with an explicit minimization
gap, so the implemented bound inherits the exact dominance theorem up to its
stated tolerance, and an adaptive algorithm with explicit global-bound
invariants.

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
exact-integration audit is deliberately reported as a
scope boundary: integer subproblem work can dominate, and the paper makes no
claim of universal superiority over compact mixed-integer optimization.

For transparency, the author also has a public preprint, “Robust Discrete
Pricing Optimization via Multiple-Choice Knapsack Reductions”
(arXiv:2603.18653). That manuscript studies fixed-threshold MCKP hull geometry,
rounding gaps, and a pricing application. It does not contain or anticipate the
present paper's simultaneous all-threshold envelope evaluator, certified
minimax algorithm, dominance theorem, adaptive interval certificate, or
computational study. The present manuscript is therefore an independent
strategic pivot with its own research question, theorem set, protocol, and
evidence package; the overlap is limited to classical robust-MCKP background.
The paper does not present the Bertsimas–Sim threshold reduction or a fixed
MCKP relaxation as new.

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
