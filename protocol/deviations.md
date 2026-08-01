# Protocol deviations and corrections

## 1 August 2026 release rerun

Independent review found that the July 21 protocol stated one thread but the
environment records showed the relevant controls unset, and that the UCI
application design lived outside the serialized protocol object. It also found
that a tolerance-based threshold deduplication could omit a distinct feasible
breakpoint and that the two adaptive comparators refreshed their shared lower
bound at slightly different points.

No July evidence was overwritten. The corrections were made before a complete
new fixed-design rerun. The August protocol serializes the application design,
requires and records the thread controls, preserves every distinct threshold,
and aligns the search mechanics. The manuscript's numerical inputs and release
manifest are regenerated only from the new dated result directory.

The factorial designs and statistical gates were not relaxed after observing
the new results.
