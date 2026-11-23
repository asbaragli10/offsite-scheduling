"""Solution representation shared by neighbourhood and optimisation modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .simulator import ScheduleEvaluation


@dataclass
class Solution:
    """A machine permutation and day-by-jig integer staffing plan."""

    machine_sequence: np.ndarray
    staffing_plan: np.ndarray

    def copy(self) -> "Solution":
        return Solution(self.machine_sequence.copy(), self.staffing_plan.copy())


@dataclass(frozen=True)
class RetainedJigSchedule:
    """Minimal replay state for an already decoded jig schedule.

    Product-to-jig assignments and the explicit processing order on every jig
    are authoritative. Start times are retained for auditability and output,
    but replay uses the explicit order so equal-start/zero-work products remain
    deterministic.
    """

    jig_by_product: np.ndarray
    jig_order_by_station: tuple[np.ndarray, ...]
    jig_start_seconds: np.ndarray

    @classmethod
    def from_evaluation(
        cls, evaluation: "ScheduleEvaluation"
    ) -> "RetainedJigSchedule":
        return cls(
            jig_by_product=np.asarray(
                evaluation.jig_by_product, dtype=int
            ).copy(),
            jig_order_by_station=tuple(
                np.asarray(order, dtype=int).copy()
                for order in evaluation.jig_order_by_station
            ),
            jig_start_seconds=np.asarray(
                evaluation.jig_start_seconds, dtype=float
            ).copy(),
        )

    def copy(self) -> "RetainedJigSchedule":
        return RetainedJigSchedule(
            jig_by_product=np.asarray(self.jig_by_product, dtype=int).copy(),
            jig_order_by_station=tuple(
                np.asarray(order, dtype=int).copy()
                for order in self.jig_order_by_station
            ),
            jig_start_seconds=np.asarray(
                self.jig_start_seconds, dtype=float
            ).copy(),
        )

    def validate(self, *, number_of_products: int, number_of_jigs: int) -> None:
        assignments = np.asarray(self.jig_by_product, dtype=int)
        starts = np.asarray(self.jig_start_seconds, dtype=float)
        if assignments.shape != (number_of_products,):
            raise ValueError("retained jig assignments do not match the product count")
        if starts.shape != (number_of_products,) or not np.all(np.isfinite(starts)):
            raise ValueError("retained jig start times do not match the product count")
        if len(self.jig_order_by_station) != number_of_jigs:
            raise ValueError("retained jig order does not match the jig count")

        orders = [np.asarray(order, dtype=int) for order in self.jig_order_by_station]
        if any(order.ndim != 1 for order in orders):
            raise ValueError("each retained per-jig processing order must be one-dimensional")
        flattened = (
            np.concatenate(orders) if orders else np.asarray([], dtype=int)
        )
        if (
            flattened.shape != (number_of_products,)
            or set(flattened.tolist()) != set(range(number_of_products))
        ):
            raise ValueError(
                "retained per-jig processing orders must contain every product exactly once"
            )
        for station, order in enumerate(orders):
            if order.size and np.any(assignments[order] != station):
                raise ValueError(
                    "retained product assignments disagree with the per-jig processing order"
                )


def validate_solution(
    solution: Solution,
    *,
    number_of_products: int,
    number_of_jigs: int,
    max_workers_per_jig: int,
) -> None:
    sequence = np.asarray(solution.machine_sequence, dtype=int)
    plan = np.asarray(solution.staffing_plan, dtype=int)
    if sequence.ndim != 1 or len(sequence) != number_of_products:
        raise ValueError("machine_sequence must contain every product exactly once")
    if set(sequence.tolist()) != set(range(number_of_products)):
        raise ValueError("machine_sequence must be a permutation of product positions")
    if plan.ndim != 2 or plan.shape[1] != number_of_jigs:
        raise ValueError(f"staffing_plan must have shape (calendar_days, {number_of_jigs})")
    if plan.shape[0] == 0:
        raise ValueError("staffing_plan must contain at least one calendar day")
    if np.any(plan < 0) or np.any(plan > max_workers_per_jig):
        raise ValueError(f"staffing entries must be in [0, {max_workers_per_jig}]")
