"""Robust local calibration for multi-objective annealing temperature."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from jssp_merit.config import NeighbourhoodConfig, ProblemConfig
from jssp_merit.data import ProductData
from jssp_merit.model import (
    InfeasibleSchedule,
    ScheduleEvaluation,
    Solution,
    evaluate_solution,
    pareto_deterioration,
)
from jssp_merit.neighbourhoods import (
    build_move_context,
    is_redistribution_move,
    propose_solution,
)
from jssp_merit.neighbourhoods.selector import retains_jig_schedule


@dataclass(frozen=True)
class TemperatureCalibration:
    temperature: float
    objective_scales: np.ndarray
    median_positive_deterioration: float
    feasible_samples: int


def estimate_initial_temperature(
    initial_solution: Solution,
    data: ProductData,
    problem: ProblemConfig,
    neighbourhood: NeighbourhoodConfig,
    *,
    samples: int = 100,
    acceptance_probability: float = 0.8,
    seed: int = 42,
    initial_evaluation: ScheduleEvaluation | None = None,
) -> TemperatureCalibration:
    """Calibrate T so a typical normalised worsening has the requested acceptance."""

    if samples <= 0:
        raise ValueError("samples must be positive")
    if not 0 < acceptance_probability < 1:
        raise ValueError("acceptance_probability must be between zero and one")
    rng = np.random.default_rng(seed)
    if initial_evaluation is None:
        initial_evaluation = evaluate_solution(
            initial_solution, data, problem, collect_segments=True
        )
    context = build_move_context(initial_evaluation)
    sampled: list[np.ndarray] = []
    for _ in range(samples):
        candidate, move = propose_solution(
            initial_solution, context, data, problem, neighbourhood, rng
        )
        try:
            sampled.append(
                evaluate_solution(
                    candidate,
                    data,
                    problem,
                    fixed_jig_schedule=(
                        initial_evaluation
                        if retains_jig_schedule(move)
                        else None
                    ),
                ).objectives
            )
        except InfeasibleSchedule:
            continue
    if not sampled:
        raise ValueError("Temperature calibration could not generate a feasible neighbour")
    sampled_values = np.asarray(sampled, dtype=float)
    changes = np.abs(sampled_values - initial_evaluation.objectives)
    scales = np.median(changes, axis=0)
    fallbacks = np.maximum(np.abs(initial_evaluation.objectives), 1.0)
    scales = np.where(scales > 1e-12, scales, fallbacks)
    deteriorations = np.asarray(
        [
            pareto_deterioration(initial_evaluation.objectives, objectives, scales)
            for objectives in sampled_values
        ],
        dtype=float,
    )
    positive = deteriorations[deteriorations > 1e-12]
    typical = float(np.median(positive)) if positive.size else 1.0
    temperature = max(1e-12, -typical / np.log(acceptance_probability))
    return TemperatureCalibration(temperature, scales, typical, len(sampled))
