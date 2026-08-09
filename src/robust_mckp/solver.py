"""Main solver for the robust MCKP pricing problem."""
from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .certificate import certificate_is_feasible, compute_certificate
from .greedy import greedy_lp
from .hull import Hull, build_upper_hull
from .model import PricingInstance, Solution
from .rounding import round_lp_solution
from .utils import EPS


@dataclass
class CandidateResult:
    theta: float
    selections: List[int]
    objective: float
    lp_value: float
    certificate_value: float
    diagnostics: Optional[Dict[str, float]] = None


@dataclass
class _ItemSweepState:
    """Per-item state for exact incremental theta sweep."""

    v: np.ndarray
    s: np.ndarray
    abs_t: np.ndarray
    base: np.ndarray
    option_indices: np.ndarray
    active_mask: np.ndarray
    active_heap: List[Tuple[float, int]]


def _extract_arrays(instance: PricingInstance) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    v_list: List[np.ndarray] = []
    s_list: List[np.ndarray] = []
    t_list: List[np.ndarray] = []
    for group in instance.items:
        v_list.append(np.array([opt.value for opt in group], dtype=float))
        s_list.append(np.array([opt.margin for opt in group], dtype=float))
        t_list.append(np.array([opt.uncertainty for opt in group], dtype=float))
    return v_list, s_list, t_list


def _universal_undominated_mask(s: np.ndarray, v: np.ndarray, abs_t: np.ndarray) -> np.ndarray:
    """Return mask of options not strongly dominated in (s, v, |t|).

    An option j is removed only if there exists j' such that
        s[j'] >= s[j], v[j'] >= v[j], |t[j']| <= |t[j]|
    and at least one inequality is strict.
    This condition is theta-independent and preserves correctness.
    """

    m = s.size
    keep = np.ones(m, dtype=bool)
    for j in range(m):
        ge_s = s >= s[j] - EPS
        ge_v = v >= v[j] - EPS
        le_t = abs_t <= abs_t[j] + EPS
        strict = (s > s[j] + EPS) | (v > v[j] + EPS) | (abs_t < abs_t[j] - EPS)
        dominated = ge_s & ge_v & le_t & strict
        dominated[j] = False
        if np.any(dominated):
            keep[j] = False
    return keep


def _build_reduced_candidates(
    v_list: List[np.ndarray],
    s_list: List[np.ndarray],
    t_list: List[np.ndarray],
) -> Tuple[np.ndarray, int, int, List[np.ndarray]]:
    """Build exact reduced theta candidate set using strong universal dominance."""

    abs_t_list = [np.abs(t) for t in t_list]
    raw_candidates = np.unique(np.concatenate([np.array([0.0], dtype=float), *abs_t_list]))
    raw_count = int(raw_candidates.size)

    reduced_values: List[np.ndarray] = [np.array([0.0], dtype=float)]
    keep_masks: List[np.ndarray] = []
    for v, s, abs_t in zip(v_list, s_list, abs_t_list):
        keep = _universal_undominated_mask(s, v, abs_t)
        keep_masks.append(keep)
        reduced_values.append(abs_t[keep])

    reduced_candidates = np.unique(np.concatenate(reduced_values))
    reduced_count = int(reduced_candidates.size)
    return reduced_candidates, raw_count, reduced_count, abs_t_list


def _refresh_active_best_for_item(
    state: _ItemSweepState,
    item_idx: int,
    active_best_bases: np.ndarray,
    active_best_indices: np.ndarray,
) -> None:
    """Refresh active-group maximum (by base=s-|t|) after deletions."""

    while state.active_heap and not state.active_mask[state.active_heap[0][1]]:
        heapq.heappop(state.active_heap)

    if state.active_heap:
        neg_base, opt_idx = state.active_heap[0]
        active_best_bases[item_idx] = -neg_base
        active_best_indices[item_idx] = opt_idx
    else:
        active_best_bases[item_idx] = -np.inf
        active_best_indices[item_idx] = np.iinfo(np.int64).max


def _compute_group_max(values: np.ndarray, indices: np.ndarray) -> Tuple[float, int]:
    """Compute max value with tie-break to smallest original index."""

    if values.size == 0:
        return -np.inf, int(np.iinfo(np.int64).max)
    best_val = float(values[0])
    best_idx = int(indices[0])
    for val, idx in zip(values[1:], indices[1:]):
        v = float(val)
        j = int(idx)
        if v > best_val or (v == best_val and j < best_idx):
            best_val = v
            best_idx = j
    return best_val, best_idx


def _round_down_value(lp_sol, hulls: List[Hull]) -> float:
    """Objective value of the formal round-down solution for certificate diagnostics."""

    value = 0.0
    for pos, hull in zip(lp_sol.positions, hulls):
        vertex_idx = pos.upper_vertex if pos.lambda_ >= 1.0 else pos.lower_vertex
        value += float(hull.values[int(vertex_idx)])
    return float(value)


def _delta_v_max(hulls: List[Hull]) -> float:
    """Largest adjacent upper-hull value jump used in the one-item certificate."""

    max_jump = 0.0
    for hull in hulls:
        if hull.values.size >= 2:
            max_jump = max(max_jump, float(np.max(np.diff(hull.values))))
    return float(max(0.0, max_jump))


def _initialize_sweep_states(
    v_list: List[np.ndarray],
    s_list: List[np.ndarray],
    abs_t_list: List[np.ndarray],
) -> Tuple[
    List[_ItemSweepState],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Initialize exact incremental sweep state at theta = 0."""

    n = len(v_list)
    inf_idx = int(np.iinfo(np.int64).max)

    states: List[_ItemSweepState] = []
    inactive_best_vals = np.full(n, -np.inf, dtype=float)
    inactive_best_indices = np.full(n, inf_idx, dtype=np.int64)
    active_best_bases = np.full(n, -np.inf, dtype=float)
    active_best_indices = np.full(n, inf_idx, dtype=np.int64)

    for i, (v, s, abs_t) in enumerate(zip(v_list, s_list, abs_t_list)):
        base = s - abs_t
        option_indices = np.arange(v.size, dtype=np.int64)
        active_mask = abs_t > 0.0

        inactive_vals = s[~active_mask]
        inactive_idxs = option_indices[~active_mask]
        in_val, in_idx = _compute_group_max(inactive_vals, inactive_idxs)
        inactive_best_vals[i] = in_val
        inactive_best_indices[i] = in_idx

        # Build a heap over all options; active_mask controls lazy deletions.
        active_heap: List[Tuple[float, int]] = [(-float(base_j), int(j)) for j, base_j in enumerate(base)]
        heapq.heapify(active_heap)

        state = _ItemSweepState(
            v=v,
            s=s,
            abs_t=abs_t,
            base=base,
            option_indices=option_indices,
            active_mask=active_mask,
            active_heap=active_heap,
        )
        _refresh_active_best_for_item(state, i, active_best_bases, active_best_indices)
        states.append(state)

    return (
        states,
        inactive_best_vals,
        inactive_best_indices,
        active_best_bases,
        active_best_indices,
        np.empty(n, dtype=np.int64),  # j_star scratch
        np.empty(n, dtype=float),  # s_star scratch
        np.zeros(n, dtype=bool),  # touched scratch
    )


def _update_inactive_best_for_event(
    s_val: float,
    opt_idx: int,
    item_idx: int,
    inactive_best_vals: np.ndarray,
    inactive_best_indices: np.ndarray,
) -> None:
    """Update inactive-group best with exact argmax tie-breaking."""

    cur_val = float(inactive_best_vals[item_idx])
    cur_idx = int(inactive_best_indices[item_idx])
    if s_val > cur_val or (s_val == cur_val and opt_idx < cur_idx):
        inactive_best_vals[item_idx] = s_val
        inactive_best_indices[item_idx] = opt_idx


def _compute_item_baselines(
    theta: float,
    inactive_best_vals: np.ndarray,
    inactive_best_indices: np.ndarray,
    active_best_bases: np.ndarray,
    active_best_indices: np.ndarray,
    j_star_out: np.ndarray,
    s_star_out: np.ndarray,
) -> None:
    """Compute exact per-item baseline maximizers at theta from group maxima."""

    active_scores = active_best_bases + theta
    inactive_valid = np.isfinite(inactive_best_vals)
    active_valid = np.isfinite(active_best_bases)

    choose_active = np.zeros(active_best_bases.size, dtype=bool)
    choose_active |= active_valid & ~inactive_valid
    choose_active |= active_valid & inactive_valid & (active_scores > inactive_best_vals)
    ties = active_valid & inactive_valid & (active_scores == inactive_best_vals)
    choose_active |= ties & (active_best_indices < inactive_best_indices)

    j_star_out[:] = np.where(choose_active, active_best_indices, inactive_best_indices)
    s_star_out[:] = np.where(choose_active, active_scores, inactive_best_vals)


def _make_event_arrays(abs_t_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten option breakpoints into sorted event arrays for incremental updates."""

    event_values = np.concatenate(abs_t_list)
    event_items = np.concatenate(
        [np.full(abs_t.size, i, dtype=np.int64) for i, abs_t in enumerate(abs_t_list)]
    )
    event_options = np.concatenate(
        [np.arange(abs_t.size, dtype=np.int64) for abs_t in abs_t_list]
    )
    order = np.argsort(event_values, kind="mergesort")
    return event_values[order], event_items[order], event_options[order]


def _advance_events_to_theta(
    theta: float,
    event_values: np.ndarray,
    event_items: np.ndarray,
    event_options: np.ndarray,
    event_ptr: int,
    states: List[_ItemSweepState],
    inactive_best_vals: np.ndarray,
    inactive_best_indices: np.ndarray,
    active_best_bases: np.ndarray,
    active_best_indices: np.ndarray,
    touched_mask: np.ndarray,
) -> Tuple[int, np.ndarray]:
    """Apply all option-status transitions with |t| <= theta."""

    touched_mask[:] = False
    n = len(states)
    touched_items: List[int] = []

    while event_ptr < event_values.size and event_values[event_ptr] <= theta:
        i = int(event_items[event_ptr])
        j = int(event_options[event_ptr])
        state = states[i]
        if state.active_mask[j]:
            state.active_mask[j] = False
            _update_inactive_best_for_event(
                s_val=float(state.s[j]),
                opt_idx=j,
                item_idx=i,
                inactive_best_vals=inactive_best_vals,
                inactive_best_indices=inactive_best_indices,
            )
            if not touched_mask[i]:
                touched_mask[i] = True
                touched_items.append(i)
        event_ptr += 1

    for i in touched_items:
        _refresh_active_best_for_item(states[i], i, active_best_bases, active_best_indices)

    if not touched_items:
        return event_ptr, np.empty(0, dtype=np.int64)
    return event_ptr, np.array(touched_items, dtype=np.int64)


def _build_hulls_exact(
    states: List[_ItemSweepState],
    theta: float,
    s_star_vals: np.ndarray,
) -> List[Hull]:
    """Build per-item hulls exactly at a fixed theta."""

    hulls: List[Hull] = []
    for i, state in enumerate(states):
        s_theta = np.where(state.active_mask, state.base + theta, state.s)
        costs = np.maximum(float(s_star_vals[i]) - s_theta, 0.0)
        hulls.append(build_upper_hull(costs, state.v, state.option_indices))
    return hulls


def _finalize_solution(
    instance: PricingInstance,
    best: Optional[CandidateResult],
    start: float,
    instrumentation: dict,
) -> Solution:
    """Create the public Solution object with instrumentation metadata."""

    elapsed = time.perf_counter() - start
    if best is None:
        return Solution(
            selections=[-1 for _ in range(instance.n_items)],
            objective=float("-inf"),
            lp_value=float("-inf"),
            gap_to_lp=float("inf"),
            certificate_value=float("-inf"),
            theta=float("nan"),
            elapsed=elapsed,
            is_feasible=False,
            metadata={"message": "No feasible solution found", "instrumentation": instrumentation},
        )

    if abs(best.lp_value) <= EPS:
        gap = 0.0 if abs(best.objective) <= EPS else float("inf")
    else:
        gap = (best.lp_value - best.objective) / best.lp_value
    if best.diagnostics:
        instrumentation.update(best.diagnostics)

    return Solution(
        selections=best.selections,
        objective=best.objective,
        lp_value=best.lp_value,
        gap_to_lp=float(gap),
        certificate_value=best.certificate_value,
        theta=best.theta,
        elapsed=elapsed,
        is_feasible=True,
        metadata={"instrumentation": instrumentation},
    )


def solve(instance: PricingInstance, *, upgrade_completion: bool = True) -> Solution:
    """Solve the robust pricing MCKP via theta-enumeration and hull-greedy.

    The robust constraint is enforced via:
        Sigma_i s_i^theta(j(i)) >= Gamma theta,  where
        s_i^theta = s_i - max(0, |t_i| - theta).
    The baseline-slack transform yields a knapsack with capacity
        C^theta = Sigma_i max_j s_i^theta(j) - Gamma theta.

    This implementation applies two exact speedups:
    1. Reduce theta candidates using strong universal dominance in (s, v, |t|).
    2. Incrementally sweep theta to update per-item baseline maxima and capacity.

    Args:
        instance: PricingInstance with per-option (v, s, t).
        upgrade_completion: If True, apply optional upgrade-completion step.

    Returns:
        Solution with best feasible discrete selection found.
    """

    start = time.perf_counter()
    v_list, s_list, t_list = _extract_arrays(instance)
    candidates, raw_candidate_count, reduced_candidate_count, abs_t_list = _build_reduced_candidates(
        v_list, s_list, t_list
    )

    (
        states,
        inactive_best_vals,
        inactive_best_indices,
        active_best_bases,
        active_best_indices,
        j_star,
        s_star_vals,
        touched_mask,
    ) = _initialize_sweep_states(v_list, s_list, abs_t_list)

    # Build event stream using ALL options, not only reduced-candidate options.
    event_values, event_items, event_options = _make_event_arrays(abs_t_list)
    event_ptr = 0
    while event_ptr < event_values.size and event_values[event_ptr] <= 0.0 + EPS:
        event_ptr += 1

    _compute_item_baselines(
        theta=0.0,
        inactive_best_vals=inactive_best_vals,
        inactive_best_indices=inactive_best_indices,
        active_best_bases=active_best_bases,
        active_best_indices=active_best_indices,
        j_star_out=j_star,
        s_star_out=s_star_vals,
    )
    prev_j_star = j_star.copy()

    best: Optional[CandidateResult] = None
    theta_skipped_capacity = 0
    theta_cert_infeasible = 0
    theta_rounding_failed = 0
    hull_rebuilds_per_theta: List[int] = []
    baseline_changes_per_theta: List[int] = []

    for step_idx, theta in enumerate(candidates):
        if step_idx > 0:
            event_ptr, _ = _advance_events_to_theta(
                theta=float(theta),
                event_values=event_values,
                event_items=event_items,
                event_options=event_options,
                event_ptr=event_ptr,
                states=states,
                inactive_best_vals=inactive_best_vals,
                inactive_best_indices=inactive_best_indices,
                active_best_bases=active_best_bases,
                active_best_indices=active_best_indices,
                touched_mask=touched_mask,
            )

            _compute_item_baselines(
                theta=float(theta),
                inactive_best_vals=inactive_best_vals,
                inactive_best_indices=inactive_best_indices,
                active_best_bases=active_best_bases,
                active_best_indices=active_best_indices,
                j_star_out=j_star,
                s_star_out=s_star_vals,
            )

        changed_baselines = int(np.count_nonzero(j_star != prev_j_star))
        baseline_changes_per_theta.append(changed_baselines)
        prev_j_star[:] = j_star

        s_star_sum = float(np.sum(s_star_vals))
        capacity = s_star_sum - instance.gamma * float(theta)
        if capacity < -EPS:
            theta_skipped_capacity += 1
            hull_rebuilds_per_theta.append(0)
            continue

        # NOTE: We rebuild all hulls exactly. Reusing hulls only when j* changes is not exact,
        # because c^theta shifts with theta even if the baseline option is unchanged.
        hulls = _build_hulls_exact(states=states, theta=float(theta), s_star_vals=s_star_vals)
        hull_rebuilds_per_theta.append(instance.n_items)

        lp_sol = greedy_lp(hulls, capacity)
        discrete = round_lp_solution(lp_sol, hulls, capacity, upgrade_completion=upgrade_completion)
        if discrete is None:
            theta_rounding_failed += 1
            continue

        selections = [int(hulls[i].option_indices[idx]) for i, idx in enumerate(discrete.vertices)]

        cert = compute_certificate(instance, selections)
        if not certificate_is_feasible(instance, selections):
            theta_cert_infeasible += 1
            continue

        objective = float(sum(v_list[i][sel] for i, sel in enumerate(selections)))
        lp_value = float(lp_sol.lp_value)
        round_down_obj = _round_down_value(lp_sol, hulls)
        delta_bound = _delta_v_max(hulls)
        lp_minus_round_down = float(lp_value - round_down_obj)
        lp_scale = max(abs(lp_value), 1.0)
        candidate_diagnostics = {
            "selected_theta_lp_upper_bound": float(lp_value),
            "selected_theta_round_down_objective": float(round_down_obj),
            "selected_theta_delta_v_max": float(delta_bound),
            "selected_theta_lp_minus_round_down": float(lp_minus_round_down),
            "selected_theta_delta_v_max_over_lp_upper_bound": float(delta_bound / lp_scale),
            "selected_theta_lp_minus_round_down_over_lp_upper_bound": float(lp_minus_round_down / lp_scale),
            "selected_theta_final_objective_gap_to_lp_upper_bound": float(lp_value - objective),
            "selected_theta_final_objective_gap_to_lp_upper_bound_ratio": float((lp_value - objective) / lp_scale),
        }

        if best is None or objective > best.objective + EPS:
            best = CandidateResult(
                theta=float(theta),
                selections=selections,
                objective=objective,
                lp_value=lp_value,
                certificate_value=float(cert),
                diagnostics=candidate_diagnostics,
            )

    instrumentation = {
        "candidate_count_raw": raw_candidate_count,
        "candidate_count_reduced": reduced_candidate_count,
        "candidate_reduction_ratio": (
            (reduced_candidate_count / raw_candidate_count) if raw_candidate_count > 0 else 1.0
        ),
        "theta_evaluated_count": int(candidates.size),
        "theta_skipped_capacity_count": theta_skipped_capacity,
        "theta_rounding_failed_count": theta_rounding_failed,
        "theta_certificate_infeasible_count": theta_cert_infeasible,
        "hull_rebuild_policy": "exact_full_rebuild_per_evaluated_theta",
        "hull_rebuilds_per_theta": hull_rebuilds_per_theta,
        "baseline_changes_per_theta": baseline_changes_per_theta,
        "total_hull_rebuilds": int(sum(hull_rebuilds_per_theta)),
    }
    return _finalize_solution(instance=instance, best=best, start=start, instrumentation=instrumentation)


def _solve_naive_reference(instance: PricingInstance, *, upgrade_completion: bool = True) -> Solution:
    """Reference implementation of naive theta enumeration for testing."""

    start = time.perf_counter()
    v_list, s_list, t_list = _extract_arrays(instance)

    abs_t = np.concatenate([np.abs(t) for t in t_list])
    candidates = np.unique(np.concatenate([np.array([0.0], dtype=float), abs_t]))
    candidates.sort()

    best: Optional[CandidateResult] = None

    for theta in candidates:
        hulls: List[Hull] = []
        s_star_sum = 0.0
        for v, s, t in zip(v_list, s_list, t_list):
            penalty = np.maximum(0.0, np.abs(t) - theta)
            s_theta = s - penalty
            j_star = int(np.argmax(s_theta))
            s_star = float(s_theta[j_star])
            s_star_sum += s_star
            costs = np.maximum(s_star - s_theta, 0.0)
            hulls.append(build_upper_hull(costs, v, np.arange(len(v), dtype=int)))

        capacity = s_star_sum - instance.gamma * float(theta)
        if capacity < -EPS:
            continue

        lp_sol = greedy_lp(hulls, capacity)
        discrete = round_lp_solution(lp_sol, hulls, capacity, upgrade_completion=upgrade_completion)
        if discrete is None:
            continue

        selections = [int(hulls[i].option_indices[idx]) for i, idx in enumerate(discrete.vertices)]
        cert = compute_certificate(instance, selections)
        if not certificate_is_feasible(instance, selections):
            continue

        objective = float(sum(v_list[i][sel] for i, sel in enumerate(selections)))
        if best is None or objective > best.objective + EPS:
            best = CandidateResult(
                theta=float(theta),
                selections=selections,
                objective=objective,
                lp_value=float(lp_sol.lp_value),
                certificate_value=float(cert),
            )

    return _finalize_solution(
        instance=instance,
        best=best,
        start=start,
        instrumentation={"mode": "naive_reference"},
    )
