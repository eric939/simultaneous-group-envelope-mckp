# Computational development history

This record separates algorithm-development evidence from untouched
confirmation evidence for *Batched Lagrangian Bounds for Robust
Multiple-Choice Knapsack*. It is part of the public verification layer and is
not an immutable campaign output.

## What the final timing result means

The released primary panel reports a 2.37-fold instance-level geometric-mean
speedup and 60 wins in 60 paired instance comparisons. The protocol for that
complete final rerun was serialized before execution, and no seed, family,
instance size, timing rule, gate, or unsuccessful instance was changed or
excluded during the run.

The 2.37-fold result is nevertheless **development evidence**, not an external
preregistration or untouched confirmation result. Earlier runs of the same
scientific design were inspected, and their failures guided
algorithm-preserving implementation work before the final rerun.

## Preserved failed-audit chronology

- **R2:** the campaign was stopped after its first 18 matched-trace cells
  exposed repeated full-family exact fallbacks. The ambiguity-set restriction
  and a guarded query kernel were then implemented without changing the
  mathematical enclosures or exact comparisons. R2 remains a failed
  performance audit and does not feed the released tables.
- **R3:** validation, kernel, and trace-validity checks passed, but the campaign
  failed the precommitted 2-fold geometric-mean target and the 1.5-fold
  bootstrap-lower-bound target. The contemporaneous experiment audit records
  60/60 timing wins and a 95% confidence interval above one, but only a
  1.79-fold geometric-mean speedup. Profiling identified redundant
  reconstruction in the shared fixed-threshold LP routine and repeated
  evaluation of the compressed oracle's fixed multiplier grid. Those
  algorithm-preserving inefficiencies were removed and parity tests were
  added. R3 is not used in the released tables.
- **R4:** after a new protocol serialization, the complete campaign passed all
  fixed scientific gates and produced the released 2.37-fold primary result.
  It also retained the adverse pricing-derived and exact-integer findings.
- **R5:** changed acknowledgments and matching submission disclosure only. Its
  evidence bytes are identical to R4; no algorithm, protocol, result, or claim
  changed.

Scratch outputs from R2 and R3 are not presented as release evidence. Their
omission from the compact public evidence directory does not convert the final
campaign into confirmation evidence; this chronology is the durable disclosure
of their role in method development.

## Interpretation boundary

The released timing panels support a reproducible engineering comparison for
the tested implementation and instance families. They do not establish
universal runtime dominance. A future, separately frozen campaign on new
instances or a native objective-uncertainty implementation would be the proper
place for confirmation evidence.
