import sys, time
sys.path.insert(0, ".")
from research import scale_campaign as sc
from research.integrated_exact_solver import build_full_theta_candidates  # noqa
import research.integrated_exact_solver as ies
want = {"many_breakpoints_n1440_m6_g37_s701", "uci_online_retail_n1440_s61"}
for panel, family, n, seed, inst in sc.integer_instances():
    if inst.name not in want: continue
    t=time.perf_counter(); th=ies.build_full_theta_candidates(inst); t1=time.perf_counter()-t
    t=time.perf_counter(); sc._hullround_incumbent(inst); t2=time.perf_counter()-t
    t=time.perf_counter(); env=ies.CompressedThetaIntervalOracle(inst); t3=time.perf_counter()-t
    print(inst.name, "B", len(th), "candidates %.1fs hullround %.1fs envelope-oracle %.1fs" % (t1,t2,t3), flush=True)
