"""Time-limited assignment MILP for a same-coil machine-sequence window.

The MILP uses a local approximation of downstream jig tardiness. The coupled
machine/trolley/jig simulator, not this model, decides whether a proposal is
feasible and improves the real objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from scheduling_engine.data import ProductData
from scheduling_engine.model import ScheduleEvaluation, Solution


@dataclass(frozen=True)
class Window:
    start: int
    size: int
    coil: int
    tardiness_seconds: float


@dataclass(frozen=True)
class MilpProposal:
    segment: np.ndarray | None
    status: int
    message: str
    model_rows: int
    model_columns: int
    binary_variables: int
    build_seconds: float
    solver_seconds: float
    proxy_objective_hours: float | None
    proxy_baseline_hours: float
    dual_bound_hours: float | None
    relative_gap: float | None
    node_count: int | None


def rank_windows(solution: Solution, evaluation: ScheduleEvaluation,
                 data: ProductData, *, size: int, stride: int) -> list[Window]:
    """Rank same-coil windows by their current contribution to tardiness."""
    if size < 2 or stride < 1:
        raise ValueError("Window size must be at least 2 and stride positive")
    sequence = solution.machine_sequence
    coils = data.coil_sizes[sequence]
    boundaries = np.r_[0, np.flatnonzero(coils[1:] != coils[:-1]) + 1, len(sequence)]
    windows: list[Window] = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        if right - left < size:
            continue
        starts = list(range(int(left), int(right - size + 1), stride))
        last = int(right - size)
        if last not in starts:
            starts.append(last)
        for start in starts:
            products = sequence[start:start + size]
            windows.append(Window(start, size, int(coils[start]),
                                  float(evaluation.tardiness_seconds[products].sum())))
    return sorted(windows, key=lambda item: (-item.tardiness_seconds, item.start))


def solve_window(solution: Solution, evaluation: ScheduleEvaluation, data: ProductData,
                 window: Window, *, scale: float, max_displacement: int,
                 solver_time_limit: float) -> MilpProposal:
    """Minimise a local jig-tardiness proxy through machine position assignment.

    A position permutation keeps the same set of products inside a contiguous
    single-coil window, preserving the exact setup sequence outside and inside
    that window. Processing durations affect cumulative machine completion.
    The proxy moves each job's saved jig completion by the estimated change in
    its machine completion. Exact downstream rescheduling follows separately.
    """
    started = perf_counter()
    sequence = solution.machine_sequence
    products = np.asarray(sequence[window.start:window.start + window.size], dtype=int)
    n = len(products)
    if n != window.size or np.unique(data.coil_sizes[products]).size != 1:
        raise ValueError("MILP window must contain the requested number of same-coil products")
    if solver_time_limit <= 0 or scale <= 0 or max_displacement < 1:
        raise ValueError("MILP scale, time limit and displacement must be positive")

    processing = data.machine_work_seconds[products] / 3600.0
    original_completion = np.cumsum(processing)
    lateness = (evaluation.jig_completion_seconds[products]
                - data.deadline_seconds[products]) / 3600.0
    baseline_proxy = float(np.maximum(0.0, lateness).sum())
    total_work = float(processing.sum())

    # x[i,k] means product i is put in position k. C[k] is cumulative machine
    # work at that position, and T[i] bounds the proxy tardiness of product i.
    x_count = n * n
    completion_offset = x_count
    tardiness_offset = x_count + n
    columns_count = x_count + 2 * n
    objective = np.zeros(columns_count)
    objective[tardiness_offset:] = 1.0
    lower = np.zeros(columns_count)
    upper = np.full(columns_count, np.inf)
    upper[:x_count] = 1.0
    upper[completion_offset:tardiness_offset] = total_work
    integrality = np.zeros(columns_count, dtype=np.int8)
    integrality[:x_count] = 1
    for i in range(n):
        for k in range(n):
            if abs(i - k) > max_displacement:
                upper[i * n + k] = 0.0

    row_indices: list[int] = []
    col_indices: list[int] = []
    coefficients: list[float] = []
    row_lower: list[float] = []
    row_upper: list[float] = []

    def add_row(terms: list[tuple[int, float]], minimum: float, maximum: float) -> None:
        row = len(row_lower)
        for column, coefficient in terms:
            row_indices.append(row)
            col_indices.append(column)
            coefficients.append(float(coefficient))
        row_lower.append(float(minimum))
        row_upper.append(float(maximum))

    for i in range(n):
        add_row([(i * n + k, 1.0) for k in range(n)], 1.0, 1.0)
    for k in range(n):
        add_row([(i * n + k, 1.0) for i in range(n)], 1.0, 1.0)
    for k in range(n):
        terms = [(completion_offset + k, 1.0)]
        if k:
            terms.append((completion_offset + k - 1, -1.0))
        terms.extend((i * n + k, -processing[i]) for i in range(n))
        add_row(terms, 0.0, 0.0)
    for i in range(n):
        # Big-M only deactivates the lateness bound at unassigned positions.
        big_m = max(0.0, float(lateness[i] + scale * (total_work - original_completion[i]))) + 1.0
        for k in range(n):
            rhs = float(lateness[i] - scale * original_completion[i] - big_m)
            add_row([(tardiness_offset + i, 1.0),
                     (completion_offset + k, -scale),
                     (i * n + k, -big_m)], rhs, np.inf)
    # Exclude the original permutation so a solver call proposes a real move.
    add_row([(i * n + i, 1.0) for i in range(n)], -np.inf, float(n - 2))
    matrix = coo_matrix((coefficients, (row_indices, col_indices)),
                        shape=(len(row_lower), columns_count)).tocsr()
    built = perf_counter()
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(matrix, np.asarray(row_lower), np.asarray(row_upper)),
        options={"time_limit": solver_time_limit, "mip_rel_gap": 0.02, "disp": False},
    )
    ended = perf_counter()
    segment = None
    if result.x is not None:
        assignment = np.asarray(result.x[:x_count]).reshape(n, n)
        rounded = np.rint(assignment)
        if (np.max(np.abs(assignment - rounded)) <= 1e-5
                and np.all(rounded.sum(axis=0) == 1)
                and np.all(rounded.sum(axis=1) == 1)):
            segment = products[np.argmax(rounded, axis=0)]
            if np.array_equal(segment, products):
                segment = None

    def finite_or_none(value: object) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if np.isfinite(number) else None

    nodes = getattr(result, "mip_node_count", None)
    return MilpProposal(
        segment=segment,
        status=int(result.status),
        message=str(result.message),
        model_rows=matrix.shape[0],
        model_columns=matrix.shape[1],
        binary_variables=x_count,
        build_seconds=built - started,
        solver_seconds=ended - built,
        proxy_objective_hours=finite_or_none(result.fun),
        proxy_baseline_hours=baseline_proxy,
        dual_bound_hours=finite_or_none(getattr(result, "mip_dual_bound", None)),
        relative_gap=finite_or_none(getattr(result, "mip_gap", None)),
        node_count=int(nodes) if nodes is not None else None,
    )
