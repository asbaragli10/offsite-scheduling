"""Small day-by-jig staffing MILP with exact single-cell replay coefficients.

The linear objective is a local surrogate. Interactions between changed cells,
machine waiting and trolley returns are resolved only by full replay afterward.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from scheduling_engine.model import InfeasibleSchedule, RetainedJigSchedule, Solution, evaluate_solution
from scheduling_engine.model.calendar import SECONDS_PER_DAY, is_weekend


@dataclass(frozen=True)
class Window:
    days: tuple[int, ...]
    score: float


@dataclass(frozen=True)
class Proposal:
    plan: np.ndarray | None
    status: int
    message: str
    model_rows: int
    model_columns: int
    build_seconds: float
    solver_seconds: float
    proxy_tardiness_delta: float | None
    proxy_labour_delta: float | None
    node_count: int | None
    relative_gap: float | None
    dual_bound: float | None
    sensitivities: tuple[dict, ...]


def rank_windows(solution, evaluation, data, *, mode: str, size: int,
                 jig_daily_work_seconds: float) -> list[Window]:
    """Rank short weekday blocks by late-job pressure or removable labour."""
    if mode not in ("tardiness", "labour") or size < 1 or jig_daily_work_seconds <= 0:
        raise ValueError("Invalid mode or window size")
    plan = solution.staffing_plan
    last_day = min(plan.shape[0] - 1, int(max(
        evaluation.jig_completion_seconds.max(initial=0),
        evaluation.trolley_worker_return_seconds.max(initial=0)) // SECONDS_PER_DAY) + 1)
    weekdays = [day for day in range(last_day + 1)
                if not is_weekend(data.start_date, day)]
    if len(weekdays) < size:
        return []
    completion_day = np.minimum(
        last_day, (evaluation.jig_completion_seconds // SECONDS_PER_DAY).astype(int))
    pressure = np.bincount(completion_day, weights=evaluation.tardiness_seconds,
                           minlength=plan.shape[0])
    windows = []
    for start in range(len(weekdays) - size + 1):
        days = tuple(weekdays[start:start + size])
        if not np.any(plan[list(days)] > 0):
            continue
        if mode == "tardiness":
            # A previous shift can clear work before the late products arrive.
            score = sum(pressure[day] + 0.7 * pressure[min(day + 1, last_day)]
                        + 0.4 * pressure[min(day + 2, last_day)] for day in days)
        else:
            active = plan[list(days)] > 0
            busy = evaluation.station_day_busy_seconds[list(days)]
            workers = plan[list(days)]
            spare = float(np.sum(active * (workers > 1) *
                                 np.maximum(0.0, 1.0 - busy / jig_daily_work_seconds)))
            # Protect the shifts that currently finish late products. Cost
            # experiments first explore spare capacity elsewhere.
            nearby_tardiness = sum(pressure[min(day + offset, last_day)]
                                   for day in days for offset in (0, 1, 2))
            score = spare - nearby_tardiness / 10000.0
        windows.append(Window(days, float(score)))
    return sorted(windows, key=lambda item: (-item.score, item.days))


def solve_window(solution: Solution, evaluation, data, problem, retained: RetainedJigSchedule,
                 window: Window, *, mode: str, labour_cap: float, tardiness_cap: float,
                 max_changes: int, time_limit: float) -> Proposal:
    """Choose staffing changes with a budgeted MILP and exact marginal scores."""
    if mode not in ("tardiness", "labour") or max_changes < 1 or time_limit <= 0:
        raise ValueError("Invalid MILP parameters")
    started = perf_counter()
    plan = solution.staffing_plan
    baseline_t, _, baseline_c = map(float, evaluation.objectives)
    options: list[dict] = []
    retained.validate(number_of_products=data.number_of_products,
                      number_of_jigs=problem.jig.number_of_jigs)
    for day in window.days:
        for station in range(problem.jig.number_of_jigs):
            original = int(plan[day, station])
            for new in (original - 1, original + 1):
                if not 0 <= new <= problem.jig.max_workers_per_jig:
                    continue
                # Opening a jig on an unstaffed day needs a different dispatch
                # model. Existing active jig cells are the local scope here.
                if original == 0:
                    continue
                if perf_counter() - started > time_limit * 0.65:
                    break
                trial_plan = plan.copy()
                trial_plan[day, station] = new
                row = {"day": day, "jig": station + 1, "old_workers": original,
                       "new_workers": new, "feasible": False}
                try:
                    trial = evaluate_solution(
                        Solution(solution.machine_sequence, trial_plan), data, problem,
                        fixed_jig_schedule=retained)
                except InfeasibleSchedule as exc:
                    row["reason"] = str(exc)
                else:
                    row.update(feasible=True,
                               tardiness_delta_seconds=float(trial.objectives[0] - baseline_t),
                               labour_delta=float(trial.objectives[2] - baseline_c),
                               setup_delta_seconds=float(trial.objectives[1] - evaluation.objectives[1]))
                options.append(row)
    build_seconds = perf_counter() - started
    valid = [row for row in options if row["feasible"] and
             abs(row["setup_delta_seconds"]) <= 1e-6]
    if not valid:
        return Proposal(None, 2, "No feasible single-cell staffing alternatives", 0, 0,
                        build_seconds, 0.0, None, None, None, None, None, tuple(options))
    n = len(valid)
    change = np.asarray([r["new_workers"] - r["old_workers"] for r in valid], float)
    dt = np.asarray([r["tardiness_delta_seconds"] for r in valid], float)
    dc = np.asarray([r["labour_delta"] for r in valid], float)
    # Each chosen marginal came from a full replay. Their sum is only a local
    # linear approximation; the combined plan is replayed exactly below.
    scale_t = max(1.0, abs(baseline_t) / 1000)
    scale_c = max(1.0, abs(baseline_c) / 1000)
    objective = (dt / scale_t + 1e-5 * dc / scale_c if mode == "tardiness"
                 else dc / scale_c + 1e-5 * dt / scale_t)
    rows: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []
    for day in window.days:
        for station in range(problem.jig.number_of_jigs):
            mask = np.asarray([r["day"] == day and r["jig"] == station + 1
                               for r in valid], float)
            if np.any(mask):
                rows.append(mask)
                lower.append(0.0)
                upper.append(1.0)
    rows.extend([np.ones(n), dc])
    lower.extend([1.0, -np.inf])
    upper.extend([float(max_changes), labour_cap - baseline_c])
    if mode == "labour":
        rows.append(dt)
        lower.append(-np.inf)
        upper.append(tardiness_cap - baseline_t)
    constraint = LinearConstraint(np.vstack(rows), np.asarray(lower), np.asarray(upper))
    solve_started = perf_counter()
    result = milp(objective, integrality=np.ones(n), bounds=Bounds(np.zeros(n), np.ones(n)),
                  constraints=constraint,
                  options={"time_limit": max(0.1, time_limit - build_seconds),
                           "mip_rel_gap": 0.001})
    solver_seconds = perf_counter() - solve_started
    chosen = np.zeros(n) if result.x is None else np.rint(result.x)
    if (result.x is None or np.sum(chosen) < 1 or
            np.max(np.abs(np.vstack(rows) @ chosen - np.clip(
                np.vstack(rows) @ chosen, lower, upper))) > 1e-5):
        candidate = None
        predicted_t = predicted_c = None
    else:
        candidate = plan.copy()
        for index in np.flatnonzero(chosen):
            row = valid[int(index)]
            candidate[row["day"], row["jig"] - 1] = row["new_workers"]
        predicted_t = float(dt @ chosen)
        predicted_c = float(dc @ chosen)
    return Proposal(candidate, int(result.status), str(result.message), len(rows), n,
                    build_seconds, solver_seconds, predicted_t, predicted_c,
                    getattr(result, "mip_node_count", None),
                    getattr(result, "mip_gap", None),
                    getattr(result, "mip_dual_bound", None), tuple(options))
