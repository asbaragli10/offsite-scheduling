"""Feasible Earliest-Due-Date initial solution."""

from __future__ import annotations

import math

import numpy as np

from scheduling_engine.config import ProblemConfig
from scheduling_engine.data import ProductData

from .calendar import calendar_days_for_working_days, is_weekend
from .types import Solution


def build_initial_solution(
    data: ProductData,
    problem: ProblemConfig,
    *,
    active_jigs: int = 2,
    workers_per_active_jig: int = 2,
    horizon_buffer_working_days: int = 20,
) -> Solution:
    """Build an EDD machine sequence and a conservative feasible staffing horizon."""

    active_jigs = min(active_jigs, problem.jig.number_of_jigs)
    workers_per_active_jig = min(workers_per_active_jig, problem.jig.max_workers_per_jig)
    if active_jigs <= 0 or workers_per_active_jig <= 0:
        raise ValueError("initial staffing requires at least one active jig and one worker")
    if horizon_buffer_working_days < 0:
        raise ValueError("horizon_buffer_working_days cannot be negative")

    sequence = np.argsort(data.deadline_seconds, kind="stable").astype(int)
    number_of_trolleys = math.ceil(data.number_of_products / problem.trolley.capacity)
    worst_machine_work = (
        float(data.machine_work_seconds.sum())
        + max(0, data.number_of_products - 1) * problem.machine.setup_seconds
        + number_of_trolleys * problem.trolley.transfer_seconds
    )
    machine_days = math.ceil(worst_machine_work / problem.machine.daily_work_seconds)

    base_jig_work = float(
        (data.screws * problem.jig.assembly_seconds_per_screw).sum()
        + (data.studs * problem.jig.handling_seconds_per_stud).sum()
    )
    # A single large product cannot be divided across jigs. A serial bound
    # guarantees a feasible horizon even for highly uneven product workloads.
    serial_speed = workers_per_active_jig ** problem.jig.worker_efficiency_exponent
    # Return travel uses exactly one worker and therefore receives no
    # multiworker speed-up in the conservative horizon.
    jig_days = math.ceil(
        base_jig_work / (serial_speed * problem.jig.daily_work_seconds)
        + number_of_trolleys * problem.trolley.empty_return_seconds / problem.jig.daily_work_seconds
    )
    required_working_days = max(1, machine_days + jig_days + horizon_buffer_working_days)
    calendar_days = calendar_days_for_working_days(data.start_date, required_working_days)

    plan = np.zeros((calendar_days, problem.jig.number_of_jigs), dtype=int)
    for day in range(calendar_days):
        if not is_weekend(data.start_date, day):
            plan[day, :active_jigs] = workers_per_active_jig
    return Solution(sequence, plan)
