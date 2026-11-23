"""Weekday production-calendar helpers and the single-machine simulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from scheduling_engine.config import MachineConfig, TrolleyConfig
from scheduling_engine.data import ProductData


SECONDS_PER_DAY = 24 * 60 * 60


@dataclass(frozen=True)
class MachineSegment:
    product_index: int
    trolley_index: int
    phase: str
    day_index: int
    start_seconds: float
    end_seconds: float
    workers: int


@dataclass(frozen=True)
class MachineSchedule:
    start_seconds: np.ndarray
    completion_seconds: np.ndarray
    setup_seconds: np.ndarray
    trolley_by_product: np.ndarray
    trolley_transfer_start_seconds: np.ndarray
    trolley_ready_seconds: np.ndarray
    physical_trolley_by_product: np.ndarray
    trolley_operator_return_seconds: np.ndarray
    setup_start_seconds: np.ndarray
    setup_completion_seconds: np.ndarray
    segments: tuple[MachineSegment, ...] = ()


def is_weekend(start_date: pd.Timestamp, day_index: int) -> bool:
    return (start_date.weekday() + int(day_index)) % 7 >= 5


def calendar_days_for_working_days(start_date: pd.Timestamp, working_days: int) -> int:
    """Return the number of calendar rows needed to include N weekdays."""

    if working_days <= 0:
        return 1
    day = 0
    seen = 0
    while seen < working_days:
        if not is_weekend(start_date, day):
            seen += 1
        day += 1
    return day


class MachineTimeline:
    """Incremental machine calendar, driven one physical trolley trip at a time."""

    def __init__(self, data: ProductData, config: MachineConfig,
                 trolley: TrolleyConfig, *, collect_segments: bool):
        self.data, self.config, self.trolley = data, config, trolley
        self.collect_segments = collect_segments
        self.time = 0.0
        self.previous_coil: int | None = None
        n = data.number_of_products
        self.starts = np.zeros(n)
        self.completions = np.zeros(n)
        self.setups = np.zeros(n)
        self.setup_starts = np.full(n, np.nan)
        self.setup_completions = np.full(n, np.nan)
        self.trolley_by_product = np.empty(n, dtype=int)
        self.physical_trolley_by_product = np.empty(n, dtype=int)
        self.transfer_starts = np.empty(n)
        self.trolley_ready = np.empty(n)
        self.operator_returns = np.empty(n)
        self.segments: list[MachineSegment] = []

    def wait_until(self, earliest: float) -> None:
        self.time = max(self.time, float(earliest))

    def _next_work_time(self) -> None:
        while True:
            day = int(self.time // SECONDS_PER_DAY)
            within = self.time - day * SECONDS_PER_DAY
            if not is_weekend(self.data.start_date, day) and within < self.config.daily_work_seconds - 1e-9:
                return
            self.time = (day + 1) * SECONDS_PER_DAY

    def _consume(self, duration: float, phase: str, product: int, trip: int) -> None:
        remaining = float(duration)
        while remaining > 1e-9:
            self._next_work_time()
            day = int(self.time // SECONDS_PER_DAY)
            within = self.time - day * SECONDS_PER_DAY
            amount = min(remaining, self.config.daily_work_seconds - within)
            if self.collect_segments:
                self.segments.append(MachineSegment(
                    product, trip, phase, day, self.time, self.time + amount,
                    self.config.operator_count,
                ))
            self.time += amount
            remaining -= amount

    def process(self, product: int, trip: int, physical_trolley: int) -> None:
        coil = int(self.data.coil_sizes[product])
        setup = 0.0 if self.previous_coil is None or coil == self.previous_coil else float(self.config.setup_seconds)
        self.setups[product] = setup
        if setup > 0:
            self._next_work_time()
            self.setup_starts[product] = self.time
            self._consume(setup, "setup", product, trip)
            self.setup_completions[product] = self.time
        self._next_work_time()
        self.starts[product] = self.time
        self._consume(float(self.data.machine_work_seconds[product]), "processing", product, trip)
        self.completions[product] = self.time
        self.trolley_by_product[product] = trip
        self.physical_trolley_by_product[product] = physical_trolley
        self.previous_coil = coil

    def transfer(self, products: list[int], trip: int) -> tuple[float, float, float]:
        """Return departure, jig-queue arrival, and machine-operator return."""
        duration = self.trolley.transfer_seconds
        if duration > self.config.daily_work_seconds:
            raise ValueError("loaded trolley round trip exceeds the machine shift")
        while True:
            self._next_work_time()
            day = int(self.time // SECONDS_PER_DAY)
            if self.time - day * SECONDS_PER_DAY + duration <= self.config.daily_work_seconds + 1e-9:
                break
            self.time = (day + 1) * SECONDS_PER_DAY
        departure = self.time
        arrival = departure + self.trolley.loaded_arrival_seconds
        finish = departure + duration
        if self.collect_segments:
            product = products[-1]
            if arrival > departure:
                self.segments.append(MachineSegment(product, trip, "loaded_trolley_delivery", day,
                                                    departure, arrival, self.config.operator_count))
            if finish > arrival:
                self.segments.append(MachineSegment(product, trip, "operator_return_to_machine", day,
                                                    arrival, finish, self.config.operator_count))
        self.time = finish
        self.transfer_starts[products] = departure
        self.trolley_ready[products] = arrival
        self.operator_returns[products] = finish
        return departure, arrival, finish

    def result(self) -> MachineSchedule:
        return MachineSchedule(
            self.starts, self.completions, self.setups, self.trolley_by_product,
            self.transfer_starts, self.trolley_ready,
            self.physical_trolley_by_product, self.operator_returns,
            self.setup_starts, self.setup_completions, tuple(self.segments),
        )
