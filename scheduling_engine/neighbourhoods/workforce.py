"""Context-aware jig staffing moves with 70/15/15 family defaults."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scheduling_engine.config import JigConfig, NeighbourhoodConfig
from scheduling_engine.data import ProductData
from scheduling_engine.model.calendar import SECONDS_PER_DAY, is_weekend
from scheduling_engine.model.simulator import ScheduleEvaluation
from scheduling_engine.model.types import Solution


REDISTRIBUTION_NAMES = (
    "rebalance_day_active",
    "shift_same_jig",
    "single_cell_active",
    "jig_block_active",
)
REDISTRIBUTION_PROBABILITIES = np.asarray((0.38, 0.30, 0.22, 0.10))
OPENING_NAMES = ("open_jig_day", "open_jig_block")
OPENING_PROBABILITIES = np.asarray((0.35, 0.65))
CLOSING_NAMES = ("close_jig_day", "close_jig_block")
CLOSING_PROBABILITIES = np.asarray((0.85, 0.15))


def _normalise(values: tuple[float, ...] | list[float] | np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    return result / result.sum()


def _weighted_choice(
    rng: np.random.Generator, rows: list[int], weights: np.ndarray
) -> int | None:
    if not rows:
        return None
    probabilities = np.asarray([max(0.0, weights[row]) for row in rows], dtype=float)
    if probabilities.sum() <= 1e-12:
        return int(rng.choice(rows))
    return int(rng.choice(rows, p=probabilities / probabilities.sum()))


def _weekday_rows(data: ProductData, number_of_days: int, last_used_day: int) -> list[int]:
    last = min(number_of_days - 1, last_used_day + 3)
    return [day for day in range(last + 1) if not is_weekend(data.start_date, day)]


def _weekday_blocks(rows: list[int], *, minimum: int = 2, maximum: int = 3) -> list[list[int]]:
    blocks: list[list[int]] = []
    for length in range(minimum, maximum + 1):
        blocks.extend([rows[index : index + length] for index in range(len(rows) - length + 1)])
    return blocks


def _redistribute(
    plan: np.ndarray,
    name: str,
    rows: list[int],
    pressure: np.ndarray,
    slack: np.ndarray,
    utilization: np.ndarray,
    maximum: int,
    rng: np.random.Generator,
) -> bool:
    if name == "rebalance_day_active":
        eligible = [
            day
            for day in rows
            if np.count_nonzero(plan[day]) >= 2
            and np.any(plan[day] > 1)
            and np.any((plan[day] > 0) & (plan[day] < maximum))
        ]
        day = _weighted_choice(rng, eligible, pressure + 1e-6)
        if day is None:
            return False
        donors = np.flatnonzero(plan[day] > 1)
        receivers = np.flatnonzero((plan[day] > 0) & (plan[day] < maximum))
        donor = int(donors[np.argmin(utilization[day, donors])])
        receivers = receivers[receivers != donor]
        if not receivers.size:
            return False
        receiver = int(receivers[np.argmax(utilization[day, receivers])])
        plan[day, donor] -= 1
        plan[day, receiver] += 1
        return True

    if name == "shift_same_jig":
        station = int(rng.integers(plan.shape[1]))
        # Redistribution changes worker counts but never the open/closed state.
        donors = [day for day in rows if plan[day, station] > 1]
        receivers = [
            day for day in rows if 0 < plan[day, station] < maximum
        ]
        if not donors or not receivers:
            return False
        donor = _weighted_choice(rng, donors, slack + 1.0 / (pressure + 1.0))
        receiver = _weighted_choice(rng, receivers, pressure + 1e-6)
        if donor is None or receiver is None or donor == receiver:
            return False
        plan[donor, station] -= 1
        plan[receiver, station] += 1
        return True

    if name == "single_cell_active":
        day = _weighted_choice(rng, rows, pressure + 1e-6)
        if day is None:
            return False
        active = np.flatnonzero(plan[day] > 0)
        if not active.size:
            return False
        station = int(rng.choice(active))
        increase = pressure[day] >= np.median(pressure[rows]) or rng.random() < 0.5
        if increase and plan[day, station] < maximum:
            plan[day, station] += 1
            return True
        if plan[day, station] > 1:
            plan[day, station] -= 1
            return True
        return False

    station = int(rng.integers(plan.shape[1]))
    blocks = [block for block in _weekday_blocks(rows) if all(plan[day, station] > 0 for day in block)]
    if not blocks:
        return False
    block = blocks[int(rng.integers(len(blocks)))]
    increase = float(np.mean(pressure[block])) >= float(np.median(pressure[rows]))
    delta = 1 if increase else -1
    if delta > 0 and any(plan[day, station] >= maximum for day in block):
        delta = -1
    if delta < 0 and any(plan[day, station] <= 1 for day in block):
        return False
    for day in block:
        plan[day, station] += delta
    return True


def _open(
    plan: np.ndarray,
    name: str,
    rows: list[int],
    pressure: np.ndarray,
    maximum: int,
    rng: np.random.Generator,
) -> bool:
    if name == "open_jig_day":
        eligible = [day for day in rows if np.any(plan[day] == 0)]
        day = _weighted_choice(rng, eligible, pressure + 1e-6)
        if day is None:
            return False
        closed = np.flatnonzero(plan[day] == 0)
        globally_closed = closed[np.all(plan[:, closed] == 0, axis=0)] if closed.size else closed
        station = int(rng.choice(globally_closed if globally_closed.size else closed))
        relative_pressure = pressure[day] / max(float(pressure.max()), 1.0)
        plan[day, station] = max(1, min(maximum, 1 + int(round(relative_pressure * (maximum - 1)))))
        return True

    blocks = [
        block
        for block in _weekday_blocks(rows)
        if any(np.all(plan[block, station] == 0) for station in range(plan.shape[1]))
    ]
    if not blocks:
        return False
    block_weights = np.asarray([np.mean(pressure[block]) for block in blocks], dtype=float)
    block_index = _weighted_choice(rng, list(range(len(blocks))), block_weights + 1e-6)
    block = blocks[int(block_index)]
    stations = [station for station in range(plan.shape[1]) if np.all(plan[block, station] == 0)]
    station = int(rng.choice(stations))
    workers = max(1, min(maximum, 1 + int(np.mean(pressure[block]) > 0)))
    plan[block, station] = workers
    return True


def _close(
    plan: np.ndarray,
    name: str,
    rows: list[int],
    pressure: np.ndarray,
    utilization: np.ndarray,
    rng: np.random.Generator,
) -> bool:
    if name == "close_jig_day":
        eligible = [day for day in rows if np.count_nonzero(plan[day]) >= 2]
        day = _weighted_choice(rng, eligible, 1.0 / (pressure + 1.0))
        if day is None:
            return False
        active = np.flatnonzero(plan[day] > 0)
        station = int(active[np.argmin(utilization[day, active])])
        plan[day, station] = 0
        return True

    candidates: list[tuple[list[int], int]] = []
    for block in _weekday_blocks(rows):
        for station in range(plan.shape[1]):
            if all(plan[day, station] > 0 and np.count_nonzero(plan[day]) >= 2 for day in block):
                candidates.append((block, station))
    if not candidates:
        return False
    scores = np.asarray(
        [np.mean(pressure[block]) + np.mean(utilization[block, station]) for block, station in candidates]
    )
    block, station = candidates[int(np.argmin(scores + rng.random(len(scores)) * 1e-9))]
    plan[block, station] = 0
    return True


def propose_staffing_plan(
    solution: Solution,
    evaluation: ScheduleEvaluation,
    data: ProductData,
    jig: JigConfig,
    neighbourhood: NeighbourhoodConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, str]:
    """Draw one family and retry its operators without biasing the family mix."""

    plan = solution.staffing_plan
    last_used_day = int(evaluation.jig_completion_seconds.max(initial=0.0) // SECONDS_PER_DAY)
    rows = _weekday_rows(data, plan.shape[0], last_used_day)
    if not rows:
        return plan.copy(), "workforce_noop"

    tardy = evaluation.tardiness_seconds
    pressure = np.zeros(plan.shape[0], dtype=float)
    slack = np.zeros(plan.shape[0], dtype=float)
    for product in range(data.number_of_products):
        day = min(plan.shape[0] - 1, int(evaluation.jig_completion_seconds[product] // SECONDS_PER_DAY))
        if tardy[product] > 0:
            pressure[day] += tardy[product]
        else:
            slack[day] += max(0.0, data.deadline_seconds[product] - evaluation.jig_completion_seconds[product])
    # Preceding days inherit decayed pressure because they can relieve a downstream bottleneck.
    for day in reversed(rows):
        for offset, decay in ((1, 0.8), (2, 0.64), (3, 0.512)):
            previous = day - offset
            if previous >= 0:
                pressure[previous] += pressure[day] * decay
    utilization = evaluation.station_day_busy_seconds / jig.daily_work_seconds

    family_names = ("redistribute", "open", "close")
    family_probs = _normalise(
        (
            neighbourhood.workforce_redistribute_weight,
            neighbourhood.workforce_open_weight,
            neighbourhood.workforce_close_weight,
        )
    )
    family = str(rng.choice(family_names, p=family_probs))
    for _ in range(12):
        candidate = plan.copy()
        if family == "redistribute":
            name = str(rng.choice(REDISTRIBUTION_NAMES, p=REDISTRIBUTION_PROBABILITIES))
            changed = _redistribute(
                candidate, name, rows, pressure, slack, utilization, jig.max_workers_per_jig, rng
            )
        elif family == "open":
            name = str(rng.choice(OPENING_NAMES, p=OPENING_PROBABILITIES))
            changed = _open(candidate, name, rows, pressure, jig.max_workers_per_jig, rng)
        else:
            name = str(rng.choice(CLOSING_NAMES, p=CLOSING_PROBABILITIES))
            changed = _close(candidate, name, rows, pressure, utilization, rng)
        if changed and not np.array_equal(candidate, plan):
            return candidate, name
    return plan.copy(), f"{family}_jig_noop" if family != "redistribute" else "redistribute_noop"
