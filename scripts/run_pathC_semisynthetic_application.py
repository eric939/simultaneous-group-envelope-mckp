#!/usr/bin/env python3
"""Run the Path C semi-synthetic robust category-pricing application."""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_mckp import GlobalThetaBNBConfig, PricingInstance, from_pricing_data, solve, solve_global_theta_bnb  # noqa: E402
from robust_mckp.certificate import compute_certificate  # noqa: E402


@dataclass
class SegmentCal:
    segment: str
    share: float
    reference_price_median: float
    reference_price_log_sd: float
    baseline_volume_median: float
    baseline_volume_log_sd: float
    elasticity_low: float
    elasticity_high: float
    uncertainty_scale: float
    fairness_band: float
    weight_multiplier: float


@dataclass
class Portfolio:
    reference_prices: np.ndarray
    weights: np.ndarray
    price_menus: List[np.ndarray]
    demands: List[np.ndarray]
    uncertainties: List[np.ndarray]
    margin_target: float
    tolerances: np.ndarray
    segments: List[str]
    segment_ids: np.ndarray
    elasticities: np.ndarray
    baseline_volumes: np.ndarray
    uncertainty_scales: np.ndarray

    @property
    def n(self) -> int:
        return int(self.reference_prices.size)

    def subset(self, n: int, m: Optional[int] = None) -> "Portfolio":
        n = min(int(n), self.n)
        price_menus = [x[:m].copy() if m else x.copy() for x in self.price_menus[:n]]
        demands = [x[: len(price_menus[i])].copy() for i, x in enumerate(self.demands[:n])]
        uncertainties = [x[: len(price_menus[i])].copy() for i, x in enumerate(self.uncertainties[:n])]
        return Portfolio(
            reference_prices=self.reference_prices[:n].copy(),
            weights=self.weights[:n].copy(),
            price_menus=price_menus,
            demands=demands,
            uncertainties=uncertainties,
            margin_target=self.margin_target,
            tolerances=self.tolerances[:n].copy(),
            segments=list(self.segments[:n]),
            segment_ids=self.segment_ids[:n].copy(),
            elasticities=self.elasticities[:n].copy(),
            baseline_volumes=self.baseline_volumes[:n].copy(),
            uncertainty_scales=self.uncertainty_scales[:n].copy(),
        )

    def to_instance(self, gamma: int) -> PricingInstance:
        return from_pricing_data(
            reference_prices=self.reference_prices,
            weights=self.weights,
            price_menus=self.price_menus,
            demands=self.demands,
            uncertainties=self.uncertainties,
            margin_target=self.margin_target,
            tolerances=self.tolerances,
            gamma=int(gamma),
        )


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_segment_calibration(calibration_dir: Path) -> Tuple[List[SegmentCal], str]:
    path = calibration_dir / "segment_calibration.csv"
    source_used = "synthetic_default"
    summary = calibration_dir / "calibration_summary.csv"
    if summary.exists():
        with summary.open(newline="") as f:
            rows = list(csv.DictReader(f))
            if rows:
                source_used = rows[0].get("source_used", source_used)
    rows: List[SegmentCal] = []
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            rows.append(
                SegmentCal(
                    segment=r["segment"],
                    share=float(r["share"]),
                    reference_price_median=float(r["reference_price_median"]),
                    reference_price_log_sd=float(r["reference_price_log_sd"]),
                    baseline_volume_median=float(r["baseline_volume_median"]),
                    baseline_volume_log_sd=float(r["baseline_volume_log_sd"]),
                    elasticity_low=float(r["elasticity_low"]),
                    elasticity_high=float(r["elasticity_high"]),
                    uncertainty_scale=float(r["uncertainty_scale"]),
                    fairness_band=float(r["fairness_band"]),
                    weight_multiplier=float(r["weight_multiplier"]),
                )
            )
    return rows, source_used


def psychological_round(target: float, lower: float, upper: float) -> float:
    step, offset = 0.10, 0.09
    lo = int(math.ceil((lower - offset) / step - 1e-12))
    hi = int(math.floor((upper - offset) / step + 1e-12))
    if lo > hi:
        return float(np.round(np.clip(target, lower, upper), 2))
    k = int(round((target - offset) / step))
    return float(np.round(offset + step * min(max(k, lo), hi), 2))


def price_menu(a: float, m: int, band: float) -> np.ndarray:
    lower, upper = (1.0 - band) * a, (1.0 + band) * a
    grid = np.linspace(lower, upper, m)
    rounded = np.unique(np.array([psychological_round(float(x), lower, upper) for x in grid]))
    rounded = rounded[(rounded >= lower - 1e-9) & (rounded <= upper + 1e-9)]
    if rounded.size == 0:
        rounded = np.array([psychological_round(a, lower, upper)])
    return np.round(rounded, 2)


def build_portfolio(cal: List[SegmentCal], *, seed: int, n: int, m: int, default_band: Optional[float] = None) -> Portfolio:
    rng = np.random.default_rng(seed)
    shares = np.array([max(0.0, c.share) for c in cal], dtype=float)
    shares = shares / shares.sum()
    counts = np.floor(shares * n).astype(int)
    while counts.sum() < n:
        counts[int(np.argmax(shares * n - counts))] += 1
    while counts.sum() > n:
        counts[int(np.argmax(counts))] -= 1

    refs: List[float] = []
    vols: List[float] = []
    etas: List[float] = []
    bands: List[float] = []
    alphas: List[float] = []
    weights: List[float] = []
    segs: List[str] = []
    seg_ids: List[int] = []
    for sid, (seg, count) in enumerate(zip(cal, counts)):
        for _ in range(int(count)):
            refs.append(float(rng.lognormal(math.log(seg.reference_price_median), seg.reference_price_log_sd)))
            vols.append(float(rng.lognormal(math.log(seg.baseline_volume_median), seg.baseline_volume_log_sd)))
            etas.append(float(rng.uniform(seg.elasticity_low, seg.elasticity_high)))
            bands.append(float(default_band if default_band is not None else seg.fairness_band))
            alphas.append(float(seg.uncertainty_scale))
            weights.append(float(seg.weight_multiplier))
            segs.append(seg.segment)
            seg_ids.append(sid)

    a = np.array(refs, dtype=float)
    d0 = np.array(vols, dtype=float)
    eta = np.array(etas, dtype=float)
    band_arr = np.array(bands, dtype=float)
    alpha = np.array(alphas, dtype=float)
    w = np.array(weights, dtype=float) * d0 / max(float(np.mean(d0)), 1e-12)

    menus: List[np.ndarray] = []
    demands: List[np.ndarray] = []
    uncertainties: List[np.ndarray] = []
    baseline_x = np.zeros(n)
    baseline_g = np.zeros(n)
    for i in range(n):
        x = price_menu(a[i], m, band_arr[i])
        g = d0[i] * (x / a[i]) ** (-eta[i])
        menus.append(x)
        demands.append(g)
        uncertainties.append(alpha[i] * g)
        j0 = int(np.argmin(np.abs(x - a[i])))
        baseline_x[i] = x[j0]
        baseline_g[i] = g[j0]

    # Calibrate margin target to make nominal reference portfolio feasible with slack.
    num = float(np.sum(w * baseline_x * baseline_g))
    den = float(np.sum(w * a * baseline_g))
    delta = 0.985
    margin_target = delta * num / max(den, 1e-12)

    return Portfolio(a, w, menus, demands, uncertainties, margin_target, band_arr, segs, np.array(seg_ids), eta, d0, alpha)


def objective(instance: PricingInstance, sel: Sequence[int]) -> float:
    return float(sum(instance.items[i][int(j)].value for i, j in enumerate(sel)))


def selected_arrays(portfolio: Portfolio, sel: Sequence[int]) -> Dict[str, np.ndarray]:
    n = portfolio.n
    x = np.zeros(n)
    g = np.zeros(n)
    d = np.zeros(n)
    for i, j in enumerate(sel):
        x[i] = portfolio.price_menus[i][int(j)]
        g[i] = portfolio.demands[i][int(j)]
        d[i] = portfolio.uncertainties[i][int(j)]
    k = portfolio.weights * (x - portfolio.margin_target * portfolio.reference_prices)
    return {"x": x, "g": g, "delta": d, "k": k, "s": k * g, "t": k * d}


def stress_metrics(portfolio: Portfolio, sel: Sequence[int], protocol: str, scenarios: int, rng: np.random.Generator, batch: int = 1000) -> Tuple[float, float, float]:
    arr = selected_arrays(portfolio, sel)
    g, d, k = arr["g"], arr["delta"], arr["k"]
    n = g.size
    margins: List[np.ndarray] = []
    seg_ids = portfolio.segment_ids
    unique_seg = sorted(set(int(x) for x in seg_ids))
    for start in range(0, scenarios, batch):
        b = min(batch, scenarios - start)
        if protocol == "iid":
            xi = rng.uniform(-1.0, 1.0, size=(b, n))
        elif protocol == "common_factor":
            common = rng.uniform(-1.0, 1.0, size=(b, 1))
            idem = rng.uniform(-0.35, 0.35, size=(b, n))
            xi = np.clip(common + idem, -1.0, 1.0)
        elif protocol == "segment_block":
            xi = np.zeros((b, n))
            for sid in unique_seg:
                shock = rng.uniform(-1.0, 1.0, size=(b, 1))
                xi[:, seg_ids == sid] = shock
        elif protocol == "heavy_tail":
            xi = np.clip(rng.standard_t(df=3, size=(b, n)) / 1.5, -2.0, 2.0)
        elif protocol == "undercalibrated":
            xi = rng.uniform(-1.5, 1.5, size=(b, n))
        elif protocol == "cross_price_substitution":
            # Out-of-model sensitivity: demand shifts toward relatively
            # cheaper products within a segment. The substitution loading is
            # randomized by scenario; it is not an estimated cross-price
            # demand model.
            xi = rng.uniform(-1.0, 1.0, size=(b, n))
        else:
            raise ValueError(protocol)
        realized = np.maximum(0.0, g[None, :] + xi * d[None, :])
        if protocol == "cross_price_substitution":
            rel_price = arr["x"] / np.maximum(portfolio.reference_prices, 1e-12) - 1.0
            relative_position = np.zeros(n, dtype=float)
            for sid in unique_seg:
                mask = seg_ids == sid
                relative_position[mask] = float(np.mean(rel_price[mask])) - rel_price[mask]
            strength = rng.uniform(0.25, 1.0, size=(b, 1))
            realized = np.maximum(0.0, realized + strength * relative_position[None, :] * g[None, :])
        margins.append(realized @ k)
    values = np.concatenate(margins)
    shortfall = np.maximum(0.0, -values)
    return float(np.mean(values < -1e-8)), float(np.mean(shortfall)), float(np.quantile(values, 0.05))


def gini_positive(values: np.ndarray) -> float:
    x = np.sort(np.maximum(values.astype(float), 0.0))
    if x.size == 0 or x.sum() <= 0:
        return 0.0
    idx = np.arange(1, x.size + 1)
    return float((2.0 * np.sum(idx * x) / (x.size * np.sum(x))) - (x.size + 1) / x.size)


def top_share(values: np.ndarray, k: int = 10) -> float:
    x = np.sort(np.maximum(values.astype(float), 0.0))[::-1]
    total = float(x.sum())
    return float(x[:k].sum() / total) if total > 0 else 0.0


def gamma_grid(n: int, raw: Optional[str]) -> List[int]:
    if raw:
        vals = []
        for part in raw.split(","):
            token = part.strip()
            if token == "sqrt":
                vals.append(int(math.floor(math.sqrt(n))))
            elif token == "n":
                vals.append(n)
            elif token.endswith("n"):
                vals.append(int(float(token[:-1]) * n))
            else:
                vals.append(int(token))
    else:
        vals = [0, 1, 5, 10, int(math.floor(math.sqrt(n))), int(0.05 * n), int(0.1 * n), int(0.25 * n), int(0.5 * n), n]
    return sorted(set(max(0, min(n, int(v))) for v in vals))


def solve_policy(portfolio: Portfolio, gamma: int) -> Tuple[PricingInstance, object]:
    inst = portfolio.to_instance(gamma)
    return inst, solve(inst)


def run_exact_subset(portfolio: Portfolio, seed: int, args: argparse.Namespace) -> List[Dict[str, object]]:
    subset = portfolio.subset(args.exact_small_subset_n, min(args.m, 10))
    out: List[Dict[str, object]] = []
    gammas = sorted(set([int(math.floor(math.sqrt(subset.n))), int(math.floor(0.1 * subset.n))]))
    for gamma in gammas:
        inst = subset.to_instance(gamma)
        t0 = time.perf_counter()
        hr = solve(inst)
        hr_time = time.perf_counter() - t0
        cfg = GlobalThetaBNBConfig(
            time_limit_seconds=args.exact_time_limit,
            node_limit=args.exact_node_limit,
            theta_order="lp_bound_desc",
            use_caches=True,
            use_objective_cutoff=True,
            use_fast_residual_lp_bound=True,
        )
        t1 = time.perf_counter()
        exact = solve_global_theta_bnb(inst, cfg)
        exact_time = time.perf_counter() - t1
        exact_obj = float(exact.objective_value) if exact.objective_value is not None else float("nan")
        hr_gap = (exact_obj - hr.objective) / max(abs(exact_obj), 1e-12) if exact.status == "optimal" else float("nan")
        out.append(
            {
                "seed": seed,
                "n": subset.n,
                "m": min(args.m, 10),
                "gamma": gamma,
                "method": "HullRound",
                "status": "feasible" if hr.is_feasible else "infeasible",
                "objective": float(hr.objective),
                "robust_certificate": float(hr.certificate_value),
                "runtime_seconds": hr_time,
                "exact_reference_objective": exact_obj,
                "gap_to_exact": hr_gap,
            }
        )
        out.append(
            {
                "seed": seed,
                "n": subset.n,
                "m": min(args.m, 10),
                "gamma": gamma,
                "method": "Global theta B&B",
                "status": exact.status,
                "objective": exact_obj,
                "robust_certificate": float(exact.robust_certificate) if exact.robust_certificate is not None else float("nan"),
                "runtime_seconds": exact_time,
                "nodes_explored": exact.total_nodes_explored,
                "absolute_gap": exact.absolute_gap,
                "relative_gap": exact.relative_gap,
                "exact_reference_objective": exact_obj,
                "gap_to_exact": 0.0 if exact.status == "optimal" else float("nan"),
            }
        )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-dir", default="results/pathC/calibration")
    parser.add_argument("--output-dir", default="results/pathC/semisynthetic_application")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--m", type=int, default=20)
    parser.add_argument("--gamma-grid", default=None)
    parser.add_argument("--stress-scenarios", type=int, default=5000)
    parser.add_argument("--segments", default=None)
    parser.add_argument("--fairness-band", type=float, default=None)
    parser.add_argument("--exact-small-subset-n", type=int, default=50)
    parser.add_argument("--run-exact-small-subset", action="store_true")
    parser.add_argument("--run-scip-if-available", action="store_true")
    parser.add_argument("--exact-time-limit", type=float, default=10.0)
    parser.add_argument("--exact-node-limit", type=int, default=100000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cal, source_used = read_segment_calibration(Path(args.calibration_dir))
    if args.segments:
        keep = {s.strip() for s in args.segments.split(",") if s.strip()}
        cal = [c for c in cal if c.segment in keep]

    policy_rows: List[Dict[str, object]] = []
    stress_rows: List[Dict[str, object]] = []
    segment_rows: List[Dict[str, object]] = []
    exact_rows: List[Dict[str, object]] = []
    config = vars(args).copy()
    config["calibration_source_used"] = source_used
    config["resolved_seeds"] = list(range(10_000, 10_000 + args.seeds))
    config["resolved_gamma_values"] = gamma_grid(args.n, args.gamma_grid)
    config["stress_randomization"] = "common random numbers by portfolio, evaluation budget, and protocol"
    config["python_version"] = platform.python_version()
    config["platform"] = platform.platform()
    config["package_versions"] = {}
    for package in ("numpy", "scipy", "highspy", "pyscipopt"):
        try:
            config["package_versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            config["package_versions"][package] = "unavailable"
    (out_dir / "pathC_config.json").write_text(json.dumps(config, indent=2))

    protocols = ["iid", "common_factor", "segment_block", "heavy_tail", "undercalibrated", "cross_price_substitution"]
    config["stress_protocols"] = protocols
    config["cross_price_protocol_scope"] = "stylized out-of-model sensitivity; not estimated cross-elasticities"
    (out_dir / "pathC_config.json").write_text(json.dumps(config, indent=2))
    for seed_idx in range(args.seeds):
        seed = 10_000 + seed_idx
        portfolio = build_portfolio(cal, seed=seed, n=args.n, m=args.m, default_band=args.fairness_band)
        gammas = gamma_grid(portfolio.n, args.gamma_grid)
        nominal_inst, nominal_sol = solve_policy(portfolio, 0)
        box_inst, box_sol = solve_policy(portfolio, portfolio.n)
        nominal_sel = list(nominal_sol.selections)
        box_sel = list(box_sol.selections)
        nominal_obj = objective(nominal_inst, nominal_sel)
        nominal_prices = selected_arrays(portfolio, nominal_sel)["x"]

        policies: Dict[Tuple[str, int], Sequence[int]] = {("nominal", 0): nominal_sel, ("box", portfolio.n): box_sel}
        solve_times: Dict[Tuple[str, int], float] = {("nominal", 0): nominal_sol.elapsed, ("box", portfolio.n): box_sol.elapsed}
        for gamma in gammas:
            inst, sol = solve_policy(portfolio, gamma)
            policies[("HullRound", gamma)] = list(sol.selections)
            solve_times[("HullRound", gamma)] = sol.elapsed

        for (method, policy_gamma), sel in policies.items():
            eval_gammas = gammas if method in {"nominal", "box"} else [policy_gamma]
            for gamma in eval_gammas:
                inst = portfolio.to_instance(gamma)
                obj = objective(inst, sel)
                cert = compute_certificate(inst, sel)
                arr = selected_arrays(portfolio, sel)
                prices = arr["x"]
                margin = arr["s"]
                changed = np.abs(prices - nominal_prices) > 1e-9
                policy_rows.append(
                    {
                        "seed": seed_idx,
                        "source_used": source_used,
                        "method": method,
                        "policy_gamma": policy_gamma,
                        "eval_gamma": gamma,
                        "n": portfolio.n,
                        "m": args.m,
                        "objective": obj,
                        "revenue_ratio": obj / max(abs(nominal_obj), 1e-12),
                        "robust_certificate": cert,
                        "runtime_seconds": solve_times.get((method, policy_gamma), float("nan")),
                        "share_changed_vs_nominal": float(np.mean(changed)),
                        "avg_abs_price_change": float(np.mean(np.abs(prices - nominal_prices))),
                        "top10_margin_share": top_share(margin, 10),
                        "positive_margin_gini": gini_positive(margin),
                    }
                )
                for p_idx, protocol in enumerate(protocols):
                    violation, mean_shortfall, q05 = stress_metrics(
                        portfolio,
                        sel,
                        protocol,
                        args.stress_scenarios,
                        # Common random numbers make stress comparisons paired
                        # across policies at the same evaluation budget.
                        np.random.default_rng(np.random.SeedSequence([seed, gamma, p_idx, 20260714])),
                    )
                    stress_rows.append(
                        {
                            "seed": seed_idx,
                            "method": method,
                            "policy_gamma": policy_gamma,
                            "eval_gamma": gamma,
                            "protocol": protocol,
                            "violation_probability": violation,
                            "mean_margin_shortfall": mean_shortfall,
                            "q05_margin": q05,
                        }
                    )
                for seg in sorted(set(portfolio.segments)):
                    mask = np.array([s == seg for s in portfolio.segments])
                    segment_rows.append(
                        {
                            "seed": seed_idx,
                            "method": method,
                            "policy_gamma": policy_gamma,
                            "eval_gamma": gamma,
                            "segment": seg,
                            "share_changed": float(np.mean(changed[mask])),
                            "median_signed_price_change": float(np.median(prices[mask] - nominal_prices[mask])),
                            "avg_abs_price_change": float(np.mean(np.abs(prices[mask] - nominal_prices[mask]))),
                            "median_uncertainty_scale": float(np.median(portfolio.uncertainty_scales[mask])),
                            "margin_contribution_change": float(np.sum(margin[mask] - selected_arrays(portfolio, nominal_sel)["s"][mask])),
                        }
                    )
        if args.run_exact_small_subset:
            exact_rows.extend(run_exact_subset(portfolio, seed_idx, args))

    write_csv(out_dir / "pathC_policy_results.csv", policy_rows)
    write_csv(out_dir / "pathC_stress_results.csv", stress_rows)
    write_csv(out_dir / "pathC_segment_diagnostics.csv", segment_rows)
    write_csv(out_dir / "pathC_exact_subset_results.csv", exact_rows)

    # Compact text summary.
    hr = [r for r in policy_rows if r["method"] == "HullRound"]
    sqrt_gamma = int(math.floor(math.sqrt(args.n)))
    hr_sqrt = [r for r in hr if int(r["eval_gamma"]) == sqrt_gamma]
    iid_sqrt = [
        r for r in stress_rows
        if r["method"] == "HullRound" and int(r["eval_gamma"]) == sqrt_gamma and r["protocol"] == "iid"
    ]
    block_sqrt = [
        r for r in stress_rows
        if r["method"] == "HullRound" and int(r["eval_gamma"]) == sqrt_gamma and r["protocol"] == "segment_block"
    ]
    lines = [
        "Path C semi-synthetic application summary",
        f"calibration_source_used: {source_used}",
        f"seeds: {args.seeds}",
        f"n: {args.n}",
        f"m: {args.m}",
        f"stress_scenarios: {args.stress_scenarios}",
        f"policy_rows: {len(policy_rows)}",
        f"stress_rows: {len(stress_rows)}",
        f"segment_rows: {len(segment_rows)}",
        f"exact_subset_rows: {len(exact_rows)}",
    ]
    if hr_sqrt:
        lines.append(f"sqrt_gamma: {sqrt_gamma}")
        lines.append(f"median_revenue_ratio_sqrt_gamma: {statistics.median(float(r['revenue_ratio']) for r in hr_sqrt):.6f}")
        lines.append(f"median_share_changed_sqrt_gamma: {statistics.median(float(r['share_changed_vs_nominal']) for r in hr_sqrt):.6f}")
    if iid_sqrt:
        lines.append(f"median_iid_violation_sqrt_gamma: {statistics.median(float(r['violation_probability']) for r in iid_sqrt):.6f}")
    if block_sqrt:
        lines.append(f"median_segment_block_violation_sqrt_gamma: {statistics.median(float(r['violation_probability']) for r in block_sqrt):.6f}")
    if exact_rows:
        exact_opt = [r for r in exact_rows if r["method"] == "Global theta B&B" and r["status"] == "optimal"]
        hr_gap = [float(r["gap_to_exact"]) for r in exact_rows if r["method"] == "HullRound" and math.isfinite(float(r["gap_to_exact"]))]
        lines.append(f"exact_subset_certified_rows: {len(exact_opt)}")
        if hr_gap:
            lines.append(f"exact_subset_hr_median_gap: {statistics.median(hr_gap):.8f}")
            lines.append(f"exact_subset_hr_max_gap: {max(hr_gap):.8f}")
    (out_dir / "pathC_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
