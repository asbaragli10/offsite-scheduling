"""Coupled machine, finite-trolley, and staffed-jig schedule decoder."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import heapq
import math

import numpy as np

from jssp_merit.config import ProblemConfig
from jssp_merit.data import ProductData

from .calendar import MachineSegment, MachineTimeline, SECONDS_PER_DAY, is_weekend
from .types import RetainedJigSchedule, Solution, validate_solution


class InfeasibleSchedule(ValueError):
    """Raised when the finite fleet or staffing horizon cannot finish a schedule."""


@dataclass(frozen=True)
class ProcessingSegment:
    day_index: int
    station: int
    start_seconds: float
    end_seconds: float
    workers: int
    product_index: int = -1
    phase: str = "assembly"


@dataclass(frozen=True)
class ScheduleEvaluation:
    objectives: np.ndarray
    machine_start_seconds: np.ndarray
    machine_completion_seconds: np.ndarray
    machine_setup_by_product: np.ndarray
    trolley_by_product: np.ndarray  # trip index, distinct from physical trolley ID
    trolley_transfer_start_seconds: np.ndarray
    trolley_ready_seconds: np.ndarray  # loaded trolley arrival at jig queue
    physical_trolley_by_product: np.ndarray
    trolley_operator_return_seconds: np.ndarray
    trolley_empty_trigger_seconds: np.ndarray  # one value per trip
    trolley_empty_return_start_seconds: np.ndarray
    trolley_empty_arrival_seconds: np.ndarray
    trolley_worker_return_seconds: np.ndarray
    trolley_return_jig: np.ndarray  # zero-based, one value per trip
    trolley_wait_start_seconds: np.ndarray
    trolley_wait_end_seconds: np.ndarray
    jig_start_seconds: np.ndarray
    jig_handling_start_seconds: np.ndarray
    jig_handling_completion_seconds: np.ndarray
    jig_assembly_start_seconds: np.ndarray
    jig_assembly_completion_seconds: np.ndarray
    jig_completion_seconds: np.ndarray
    jig_by_product: np.ndarray
    jig_order_by_station: tuple[np.ndarray, ...]
    tardiness_seconds: np.ndarray
    jig_base_work_seconds: np.ndarray
    jig_handling_base_work_seconds: np.ndarray
    jig_assembly_base_work_seconds: np.ndarray
    station_day_busy_seconds: np.ndarray
    station_day_worker_busy_seconds: np.ndarray
    machine_paid_worker_seconds: float
    machine_busy_worker_seconds: float
    machine_labour_cost: float
    jig_paid_worker_seconds_by_station: np.ndarray
    jig_busy_worker_seconds_by_station: np.ndarray
    jig_labour_cost: float
    processing_segments: tuple[ProcessingSegment, ...]
    machine_setup_start_seconds: np.ndarray
    machine_setup_completion_seconds: np.ndarray
    machine_segments: tuple[MachineSegment, ...]

    @property
    def total_tardiness_seconds(self) -> float:
        return float(self.objectives[0])

    @property
    def machine_setup_seconds(self) -> float:
        return float(self.objectives[1])

    @property
    def labour_cost(self) -> float:
        return float(self.objectives[2])

    @property
    def average_tardiness_seconds(self) -> float:
        late = self.tardiness_seconds[self.tardiness_seconds > 0]
        return float(late.mean()) if late.size else 0.0


def _next_staffed_time(*, station: int, earliest_seconds: float,
                       staffed_mask: np.ndarray, next_staffed_day: np.ndarray,
                       daily_work_seconds: float) -> float:
    """Return the next jig shift instant with at least one staffed worker."""
    current = max(0.0, float(earliest_seconds))
    day = int(current // SECONDS_PER_DAY)
    if day >= staffed_mask.shape[0]:
        return np.inf
    within = current - day * SECONDS_PER_DAY
    if staffed_mask[day, station] and within < daily_work_seconds - 1e-9:
        return current
    following_day = int(next_staffed_day[day + 1, station])
    return (float(following_day * SECONDS_PER_DAY)
            if following_day < staffed_mask.shape[0] else np.inf)


def _next_return_start(*, station: int, earliest_seconds: float,
                       staffing_plan: np.ndarray, data: ProductData,
                       problem: ProblemConfig) -> float:
    """Fit the complete 84-second worker trip within one staffed jig shift."""
    duration = problem.trolley.empty_return_seconds
    if duration > problem.jig.daily_work_seconds:
        raise ValueError("empty trolley round trip exceeds the jig shift")
    current = max(0.0, float(earliest_seconds))
    while True:
        day = int(current // SECONDS_PER_DAY)
        if day >= staffing_plan.shape[0]:
            return np.inf
        within = current - day * SECONDS_PER_DAY
        if (not is_weekend(data.start_date, day)
                and staffing_plan[day, station] > 0
                and within + duration <= problem.jig.daily_work_seconds + 1e-9):
            return current
        current = (day + 1) * SECONDS_PER_DAY


def _simulate_jig_operation(*, station: int, ready_seconds: float,
                            base_work_seconds: float, staffing_plan: np.ndarray,
                            data: ProductData, problem: ProblemConfig,
                            absences: list[tuple[float, float]], collect_segments: bool,
                            product_index: int = -1, phase: str = "assembly",
                            ) -> tuple[float, float, tuple[ProcessingSegment, ...]]:
    """Process work at workers**0.9, with one worker absent on return trips."""
    if base_work_seconds <= 1e-12:
        return ready_seconds, ready_seconds, ()
    current = float(ready_seconds)
    remaining = float(base_work_seconds)
    first_start = np.inf
    segments: list[ProcessingSegment] = []
    capacity = float(problem.jig.daily_work_seconds)
    while remaining > 1e-9:
        day = int(current // SECONDS_PER_DAY)
        if day >= staffing_plan.shape[0]:
            return np.inf, np.inf, ()
        within = current - day * SECONDS_PER_DAY
        staffed = int(staffing_plan[day, station])
        if is_weekend(data.start_date, day) or staffed <= 0 or within >= capacity - 1e-9:
            current = (day + 1) * SECONDS_PER_DAY
            continue
        position = bisect_right(absences, (current, math.inf)) - 1
        on_return = position >= 0 and current < absences[position][1] - 1e-9
        next_boundary = (absences[position][1] if on_return else
                         absences[position + 1][0] if position + 1 < len(absences) else math.inf)
        workers = staffed - int(on_return)
        if workers <= 0:
            current = next_boundary
            continue
        speed = workers ** problem.jig.worker_efficiency_exponent
        wall = min(capacity - within, next_boundary - current, remaining / speed)
        if wall <= 1e-9:
            if not np.isfinite(next_boundary):
                # A sub-nanosecond remainder at an otherwise unbounded interval
                # must not advance the clock to infinity. The previous branch
                # did so, and int(inf // SECONDS_PER_DAY) then raised on NaN.
                if remaining / speed <= 1e-9:
                    if not np.isfinite(first_start):
                        first_start = current
                    remaining = 0.0
                    break
                current = (day + 1) * SECONDS_PER_DAY
            else:
                current = max(current + 1e-9, next_boundary)
            continue
        if not np.isfinite(first_start):
            first_start = current
        if collect_segments:
            segments.append(ProcessingSegment(day, station, current, current + wall,
                                              workers, product_index, phase))
        current += wall
        remaining -= wall * speed
    return float(first_start), float(current), tuple(segments)


def evaluate_solution(solution: Solution, data: ProductData, problem: ProblemConfig,
                      *, collect_segments: bool = False,
                      fixed_jig_schedule: ScheduleEvaluation | RetainedJigSchedule | None = None,
                      ) -> ScheduleEvaluation:
    """Decode trolley trips and jig work together so empty returns can block production.

    A redistribution retains product-to-jig assignments and per-jig order, but
    still recomputes every machine and trolley time because jig handling controls
    when a physical trolley becomes available at the machine.
    """
    validate_solution(solution, number_of_products=data.number_of_products,
                      number_of_jigs=problem.jig.number_of_jigs,
                      max_workers_per_jig=problem.jig.max_workers_per_jig)
    sequence = np.asarray(solution.machine_sequence, dtype=int)
    staffing = np.asarray(solution.staffing_plan, dtype=int)
    n = data.number_of_products
    handling_base = data.studs.astype(float) * problem.jig.handling_seconds_per_stud
    assembly_base = data.screws.astype(float) * problem.jig.assembly_seconds_per_screw
    staffed_mask = staffing > 0
    for day in range(staffing.shape[0]):
        if is_weekend(data.start_date, day):
            staffed_mask[day] = False
    next_staffed_day = np.full((staffing.shape[0] + 1, staffing.shape[1]),
                               staffing.shape[0], dtype=int)
    for day in range(staffing.shape[0] - 1, -1, -1):
        next_staffed_day[day] = np.where(staffed_mask[day], day, next_staffed_day[day + 1])

    machine = MachineTimeline(data, problem.machine, problem.trolley,
                              collect_segments=collect_segments)
    fleet = [(0.0, trolley_id) for trolley_id in range(problem.trolley.fleet_size)]
    heapq.heapify(fleet)
    trips = math.ceil(n / problem.trolley.capacity)
    empty_trigger = np.zeros(trips)
    empty_start = np.zeros(trips)
    empty_arrival = np.zeros(trips)
    worker_return = np.zeros(trips)
    return_jig = np.full(trips, -1, dtype=int)
    wait_start = np.zeros(trips)
    wait_end = np.zeros(trips)

    jig_start = np.zeros(n)
    handling_start = np.zeros(n)
    handling_completion = np.zeros(n)
    assembly_start = np.zeros(n)
    assembly_completion = np.zeros(n)
    jig_completion = np.zeros(n)
    assignment = np.full(n, -1, dtype=int)
    jig_order: list[list[int]] = [[] for _ in range(problem.jig.number_of_jigs)]
    segments_by_product: list[tuple[ProcessingSegment, ...]] = [() for _ in range(n)]
    return_segments: list[ProcessingSegment] = []
    station_available = np.zeros(problem.jig.number_of_jigs)
    return_busy_until = np.zeros(problem.jig.number_of_jigs)
    absences: list[list[tuple[float, float]]] = [[] for _ in range(problem.jig.number_of_jigs)]

    retained: RetainedJigSchedule | None = None
    retained_rank = np.zeros(n, dtype=int)
    last_rank = np.full(problem.jig.number_of_jigs, -1, dtype=int)
    if fixed_jig_schedule is not None:
        retained = (fixed_jig_schedule if isinstance(fixed_jig_schedule, RetainedJigSchedule)
                    else RetainedJigSchedule.from_evaluation(fixed_jig_schedule))
        retained.validate(number_of_products=n, number_of_jigs=problem.jig.number_of_jigs)
        for station, order in enumerate(retained.jig_order_by_station):
            retained_rank[order] = np.arange(len(order))

    def phases(product: int, station: int, ready: float,
               *, detail: bool) -> tuple[float, float, float, float, tuple[ProcessingSegment, ...]]:
        first, handled, h_segments = _simulate_jig_operation(
            station=station, ready_seconds=ready,
            base_work_seconds=float(handling_base[product]), staffing_plan=staffing,
            data=data, problem=problem, absences=absences[station],
            collect_segments=detail, product_index=product, phase="handling")
        if not np.isfinite(handled):
            raise InfeasibleSchedule("Staffing horizon ends before material handling")
        assembly_first, finished, a_segments = _simulate_jig_operation(
            station=station, ready_seconds=handled,
            base_work_seconds=float(assembly_base[product]), staffing_plan=staffing,
            data=data, problem=problem, absences=absences[station],
            collect_segments=detail, product_index=product, phase="assembly")
        if not np.isfinite(finished):
            raise InfeasibleSchedule("Staffing horizon ends before assembly")
        return first, handled, assembly_first, finished, h_segments + a_segments

    def commit(product: int, station: int, result: tuple) -> None:
        first, handled, assembly_first, finished, segments = result
        jig_start[product] = first
        handling_start[product] = first
        handling_completion[product] = handled
        assembly_start[product] = assembly_first
        assembly_completion[product] = finished
        jig_completion[product] = finished
        assignment[product] = station
        jig_order[station].append(product)
        segments_by_product[product] = segments
        station_available[station] = finished

    for trip, start_position in enumerate(range(0, n, problem.trolley.capacity)):
        batch = [int(p) for p in sequence[start_position:start_position + problem.trolley.capacity]]
        trolley_available, physical_id = heapq.heappop(fleet)
        wait_start[trip] = machine.time
        machine.wait_until(trolley_available)
        wait_end[trip] = machine.time
        for product in batch:
            machine.process(product, trip, physical_id)
        _, release, _ = machine.transfer(batch, trip)

        if retained is None:
            machine_position = {product: pos for pos, product in enumerate(batch)}
            products_to_schedule = sorted(batch, key=lambda p: (data.deadline_seconds[p],
                                                                 machine_position[p]))
        else:
            products_to_schedule = sorted(batch, key=lambda p: (int(retained.jig_by_product[p]),
                                                                 int(retained_rank[p])))
        for product in products_to_schedule:
            if retained is None:
                best_station = -1
                best_completion = np.inf
                for station in range(problem.jig.number_of_jigs):
                    ready = _next_staffed_time(
                        station=station,
                        earliest_seconds=max(float(station_available[station]), release),
                        staffed_mask=staffed_mask, next_staffed_day=next_staffed_day,
                        daily_work_seconds=problem.jig.daily_work_seconds)
                    if not np.isfinite(ready):
                        continue
                    try:
                        trial = phases(product, station, ready, detail=False)
                    except InfeasibleSchedule:
                        continue
                    if trial[3] < best_completion:
                        best_station, best_completion = station, trial[3]
                if best_station < 0:
                    raise InfeasibleSchedule("No jig can finish the released product")
                station = best_station
            else:
                station = int(retained.jig_by_product[product])
                rank = int(retained_rank[product])
                if rank <= last_rank[station]:
                    raise InfeasibleSchedule("Retained jig order conflicts with trolley release order")
                last_rank[station] = rank
            ready = _next_staffed_time(
                station=station, earliest_seconds=max(float(station_available[station]), release),
                staffed_mask=staffed_mask, next_staffed_day=next_staffed_day,
                daily_work_seconds=problem.jig.daily_work_seconds)
            if not np.isfinite(ready):
                raise InfeasibleSchedule("Assigned jig has no remaining staffed shift")
            # Retain committed intervals even in fast evaluations: move
            # selection and utilisation use the same worker-time ledger.
            commit(product, station, phases(product, station, ready, detail=True))

        # All handling for this trolley is now committed. The last handler's
        # jig supplies one worker for the 54+30-second empty round trip.
        last_product = min(batch, key=lambda p: (-handling_completion[p], p))
        station = int(assignment[last_product])
        trigger = float(handling_completion[last_product])
        start = _next_return_start(
            station=station, earliest_seconds=max(trigger, float(return_busy_until[station])),
            staffing_plan=staffing, data=data, problem=problem)
        if not np.isfinite(start):
            raise InfeasibleSchedule("No staffed shift remains to return an empty trolley")
        arrival = start + problem.trolley.empty_arrival_seconds
        finish = start + problem.trolley.empty_return_seconds
        empty_trigger[trip], empty_start[trip] = trigger, start
        empty_arrival[trip], worker_return[trip] = arrival, finish
        return_jig[trip] = station
        return_busy_until[station] = finish
        absences[station].append((start, finish))
        day = int(start // SECONDS_PER_DAY)
        if arrival > start:
            return_segments.append(ProcessingSegment(day, station, start, arrival, 1,
                                                     last_product, "empty_trolley_delivery"))
        if finish > arrival:
            return_segments.append(ProcessingSegment(day, station, arrival, finish, 1,
                                                     last_product, "worker_return_to_jig"))

        # This product is the last handled on its jig among the trolley's
        # products. Its assembly can overlap the worker's return journey.
        assembly_first, finished, a_segments = _simulate_jig_operation(
            station=station, ready_seconds=trigger,
            base_work_seconds=float(assembly_base[last_product]), staffing_plan=staffing,
            data=data, problem=problem, absences=absences[station],
            collect_segments=True, product_index=last_product, phase="assembly")
        if not np.isfinite(finished):
            raise InfeasibleSchedule("Return trip leaves too little staffed time for assembly")
        assembly_start[last_product] = assembly_first
        assembly_completion[last_product] = finished
        jig_completion[last_product] = finished
        station_available[station] = finished
        segments_by_product[last_product] = (
            tuple(segment for segment in segments_by_product[last_product]
                  if segment.phase == "handling") + a_segments
        )
        heapq.heappush(fleet, (arrival, physical_id))

    machine_result = machine.result()
    all_segments = (tuple(segment for parts in segments_by_product for segment in parts)
                    + tuple(return_segments))
    processing_segments = all_segments if collect_segments else ()
    station_day_busy = np.zeros_like(staffing, dtype=float)
    station_day_worker_busy = np.zeros_like(staffing, dtype=float)
    for segment in all_segments:
        duration = segment.end_seconds - segment.start_seconds
        station_day_worker_busy[segment.day_index, segment.station] += duration * segment.workers
        if segment.phase in ("handling", "assembly"):
            station_day_busy[segment.day_index, segment.station] += duration

    tardiness = np.maximum(0.0, jig_completion - data.deadline_seconds)
    total_setup = float(machine_result.setup_seconds.sum())
    last_jig_work = max(float(jig_completion.max(initial=0.0)),
                        float(worker_return.max(initial=0.0)))
    last_used_day = min(staffing.shape[0] - 1, int(last_jig_work // SECONDS_PER_DAY))
    jig_paid_by_station = np.zeros(problem.jig.number_of_jigs)
    for day in range(last_used_day + 1):
        if not is_weekend(data.start_date, day):
            jig_paid_by_station += staffing[day].astype(float) * problem.jig.daily_work_seconds
    jig_labour_cost = float(jig_paid_by_station.sum()) / 3600 * problem.jig.worker_hourly_cost

    machine_last_day = int(machine_result.trolley_operator_return_seconds.max(initial=0.0)
                           // SECONDS_PER_DAY)
    machine_staffed_days = sum(not is_weekend(data.start_date, day)
                               for day in range(machine_last_day + 1))
    machine_paid = float(machine_staffed_days * problem.machine.daily_work_seconds
                         * problem.machine.operator_count)
    machine_busy = float((data.machine_work_seconds.sum() + total_setup
                          + trips * problem.trolley.transfer_seconds)
                         * problem.machine.operator_count)
    machine_labour_cost = machine_paid / 3600 * problem.machine.operator_hourly_cost
    objectives = np.asarray([float(tardiness.sum()), total_setup,
                             machine_labour_cost + jig_labour_cost], dtype=float)
    return ScheduleEvaluation(
        objectives=objectives,
        machine_start_seconds=machine_result.start_seconds,
        machine_completion_seconds=machine_result.completion_seconds,
        machine_setup_by_product=machine_result.setup_seconds,
        trolley_by_product=machine_result.trolley_by_product,
        trolley_transfer_start_seconds=machine_result.trolley_transfer_start_seconds,
        trolley_ready_seconds=machine_result.trolley_ready_seconds,
        physical_trolley_by_product=machine_result.physical_trolley_by_product,
        trolley_operator_return_seconds=machine_result.trolley_operator_return_seconds,
        trolley_empty_trigger_seconds=empty_trigger,
        trolley_empty_return_start_seconds=empty_start,
        trolley_empty_arrival_seconds=empty_arrival,
        trolley_worker_return_seconds=worker_return,
        trolley_return_jig=return_jig,
        trolley_wait_start_seconds=wait_start,
        trolley_wait_end_seconds=wait_end,
        jig_start_seconds=jig_start,
        jig_handling_start_seconds=handling_start,
        jig_handling_completion_seconds=handling_completion,
        jig_assembly_start_seconds=assembly_start,
        jig_assembly_completion_seconds=assembly_completion,
        jig_completion_seconds=jig_completion,
        jig_by_product=assignment,
        jig_order_by_station=tuple(np.asarray(order, dtype=int) for order in jig_order),
        tardiness_seconds=tardiness,
        jig_base_work_seconds=handling_base + assembly_base,
        jig_handling_base_work_seconds=handling_base,
        jig_assembly_base_work_seconds=assembly_base,
        station_day_busy_seconds=station_day_busy,
        station_day_worker_busy_seconds=station_day_worker_busy,
        machine_paid_worker_seconds=machine_paid,
        machine_busy_worker_seconds=machine_busy,
        machine_labour_cost=machine_labour_cost,
        jig_paid_worker_seconds_by_station=jig_paid_by_station,
        jig_busy_worker_seconds_by_station=station_day_worker_busy.sum(axis=0),
        jig_labour_cost=jig_labour_cost,
        processing_segments=processing_segments,
        machine_setup_start_seconds=machine_result.setup_start_seconds,
        machine_setup_completion_seconds=machine_result.setup_completion_seconds,
        machine_segments=machine_result.segments,
    )
