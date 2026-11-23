"""Typed configuration for the production model and optimisation search."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MachineConfig:
    """Physical constraints of the stand-alone machine."""

    daily_work_seconds: float = 6 * 60 * 60
    setup_seconds: float = 60 * 60
    operator_count: int = 1
    operator_hourly_cost: float = 20.0

    def __post_init__(self) -> None:
        if self.daily_work_seconds <= 0 or self.daily_work_seconds > 24 * 60 * 60:
            raise ValueError("machine daily_work_seconds must be in (0, 86400]")
        if self.setup_seconds < 0:
            raise ValueError("machine setup_seconds cannot be negative")
        if self.operator_count <= 0:
            raise ValueError("machine operator_count must be positive")
        if self.operator_hourly_cost < 0:
            raise ValueError("machine operator_hourly_cost cannot be negative")


@dataclass(frozen=True)
class JigConfig:
    """Assembly-jig and workforce constraints."""

    daily_work_seconds: float = 8 * 60 * 60
    number_of_jigs: int = 4
    max_workers_per_jig: int = 4
    worker_efficiency_exponent: float = 0.9
    worker_hourly_cost: float = 20.0
    assembly_seconds_per_screw: float = 60.0
    handling_seconds_per_stud: float = 10.0

    def __post_init__(self) -> None:
        if self.daily_work_seconds <= 0 or self.daily_work_seconds > 24 * 60 * 60:
            raise ValueError("jig daily_work_seconds must be in (0, 86400]")
        if self.number_of_jigs <= 0:
            raise ValueError("number_of_jigs must be positive")
        if self.max_workers_per_jig <= 0:
            raise ValueError("max_workers_per_jig must be positive")
        if not 0 < self.worker_efficiency_exponent <= 1:
            raise ValueError("worker_efficiency_exponent must be in (0, 1]")
        if self.worker_hourly_cost < 0:
            raise ValueError("worker_hourly_cost cannot be negative")
        if self.assembly_seconds_per_screw < 0 or self.handling_seconds_per_stud < 0:
            raise ValueError("assembly and handling rates cannot be negative")


@dataclass(frozen=True)
class TrolleyConfig:
    """Finite trolley fleet and two-leg operator round trips."""

    capacity: int = 6
    transfer_seconds: float = 2 * 60
    fleet_size: int = 30
    loaded_arrival_seconds: float = 90.0
    empty_return_seconds: float = 84.0
    empty_arrival_seconds: float = 54.0

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("trolley capacity must be positive")
        if self.transfer_seconds < 0:
            raise ValueError("trolley transfer_seconds cannot be negative")
        if self.fleet_size <= 0:
            raise ValueError("trolley fleet_size must be positive")
        if not 0 <= self.loaded_arrival_seconds <= self.transfer_seconds:
            raise ValueError("loaded_arrival_seconds must be within the loaded round trip")
        if self.empty_return_seconds < 0:
            raise ValueError("empty_return_seconds cannot be negative")
        if not 0 <= self.empty_arrival_seconds <= self.empty_return_seconds:
            raise ValueError("empty_arrival_seconds must be within the empty round trip")


@dataclass(frozen=True)
class NeighbourhoodConfig:
    """Weights for the integrated and resource-specific move families."""

    machine_probability: float = 0.5
    workforce_probability: float = 0.5
    machine_block_deadline_weight: float = 0.5
    machine_block_global_weight: float = 0.5
    workforce_redistribute_weight: float = 0.70
    workforce_open_weight: float = 0.15
    workforce_close_weight: float = 0.15

    def __post_init__(self) -> None:
        groups = {
            "integrated move": (self.machine_probability, self.workforce_probability),
            "machine move": (
                self.machine_block_deadline_weight,
                self.machine_block_global_weight,
            ),
            "workforce move": (
                self.workforce_redistribute_weight,
                self.workforce_open_weight,
                self.workforce_close_weight,
            ),
        }
        for name, values in groups.items():
            if any(value < 0 for value in values) or sum(values) <= 0:
                raise ValueError(f"{name} weights must be non-negative with a positive sum")


@dataclass(frozen=True)
class AnnealingConfig:
    """Parallel Pareto simulated-annealing parameters."""

    temperature_steps: int = 200
    neighbours_per_step: int = 17_960
    cooling_rate: float = 0.988
    initial_temperature: float | None = None
    initial_acceptance_probability: float = 0.8
    temperature_samples: int = 100
    workers: int = 1
    batch_multiplier: int = 8
    seed: int = 42
    archive_limit: int = 500
    early_stopping_patience: int = 10
    early_stopping_min_steps: int = 60
    early_stopping_hypervolume_relative_gain: float = 0.015
    early_stopping_hypervolume_reference_margin: float = 1.0
    early_stopping_tardiness_seconds: float = 600.0
    early_stopping_setup_seconds: float = 3_600.0
    early_stopping_labour_cost: float = 120.0

    def __post_init__(self) -> None:
        if self.temperature_steps <= 0 or self.neighbours_per_step <= 0:
            raise ValueError("annealing step and neighbour counts must be positive")
        if not 0 < self.cooling_rate < 1:
            raise ValueError("cooling_rate must be between zero and one")
        if self.initial_temperature is not None and self.initial_temperature <= 0:
            raise ValueError("initial_temperature must be positive")
        if not 0 < self.initial_acceptance_probability < 1:
            raise ValueError("initial_acceptance_probability must be between zero and one")
        if self.temperature_samples <= 0:
            raise ValueError("temperature_samples must be positive")
        if self.workers < 0:
            raise ValueError("workers cannot be negative; use zero for all CPU cores")
        if self.batch_multiplier <= 0 or self.archive_limit <= 0:
            raise ValueError("batch_multiplier and archive_limit must be positive")
        if self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience cannot be negative")
        if self.early_stopping_min_steps < 0:
            raise ValueError("early_stopping_min_steps cannot be negative")
        if self.early_stopping_hypervolume_relative_gain <= 0:
            raise ValueError(
                "early_stopping_hypervolume_relative_gain must be positive"
            )
        if self.early_stopping_hypervolume_reference_margin <= 0:
            raise ValueError(
                "early_stopping_hypervolume_reference_margin must be positive"
            )
        tolerances = (
            self.early_stopping_tardiness_seconds,
            self.early_stopping_setup_seconds,
            self.early_stopping_labour_cost,
        )
        if any(tolerance <= 0 for tolerance in tolerances):
            raise ValueError("early-stopping objective tolerances must be positive")

    @property
    def early_stopping_tolerances(self) -> tuple[float, float, float]:
        """Minimum material change for each objective, in objective order."""

        return (
            self.early_stopping_tardiness_seconds,
            self.early_stopping_setup_seconds,
            self.early_stopping_labour_cost,
        )


@dataclass(frozen=True)
class ProblemConfig:
    """Complete immutable model configuration passed to evaluators."""

    machine: MachineConfig = field(default_factory=MachineConfig)
    jig: JigConfig = field(default_factory=JigConfig)
    trolley: TrolleyConfig = field(default_factory=TrolleyConfig)
