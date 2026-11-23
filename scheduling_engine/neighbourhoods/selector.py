"""Integrated resource selector for machine and workforce moves."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scheduling_engine.config import NeighbourhoodConfig, ProblemConfig
from scheduling_engine.data import ProductData
from scheduling_engine.model.simulator import ScheduleEvaluation
from scheduling_engine.model.types import Solution

from .machine import reinsert_block_within_deadline, swap_global_blocks
from .workforce import REDISTRIBUTION_NAMES, propose_staffing_plan


MACHINE_MOVE_NAMES = frozenset(("block_deadline", "block_global"))
REDISTRIBUTION_MOVE_NAMES = frozenset(REDISTRIBUTION_NAMES)


def is_machine_move(move: str) -> bool:
    return move in MACHINE_MOVE_NAMES


def is_redistribution_move(move: str) -> bool:
    return move in REDISTRIBUTION_MOVE_NAMES or move == "redistribute_noop"


def is_noop_move(move: str) -> bool:
    return move == "noop" or move.endswith("_noop")


def retains_jig_schedule(move: str) -> bool:
    return is_redistribution_move(move) or is_noop_move(move)


def move_family(move: str) -> str:
    if move.startswith("block_"):
        return "machine"
    if is_redistribution_move(move):
        return "redistribution"
    if move.startswith("open"):
        return "opening"
    if move.startswith("close"):
        return "closing"
    return "noop"


@dataclass(frozen=True)
class MoveContext:
    """Current decoded schedule used by context-aware staffing moves."""

    evaluation: ScheduleEvaluation


def build_move_context(evaluation: ScheduleEvaluation) -> MoveContext:
    return MoveContext(evaluation=evaluation)


def propose_solution(
    solution: Solution,
    context: MoveContext,
    data: ProductData,
    problem: ProblemConfig,
    neighbourhood: NeighbourhoodConfig,
    rng: np.random.Generator,
) -> tuple[Solution, str]:
    """Propose a changed candidate, retaining each validated resource operator mix."""

    resource_probabilities = np.asarray(
        [neighbourhood.machine_probability, neighbourhood.workforce_probability], dtype=float
    )
    resource_probabilities /= resource_probabilities.sum()
    machine_probabilities = np.asarray(
        [
            neighbourhood.machine_block_deadline_weight,
            neighbourhood.machine_block_global_weight,
        ],
        dtype=float,
    )
    machine_probabilities /= machine_probabilities.sum()

    # Draw the family once. Resampling it after a failed move biases the mix.
    resource = str(rng.choice(("machine", "workforce"), p=resource_probabilities))
    if resource == "workforce":
        plan, move = propose_staffing_plan(
            solution, context.evaluation, data, problem.jig, neighbourhood, rng
        )
        return Solution(solution.machine_sequence.copy(), plan), move
    move = str(rng.choice(("block_deadline", "block_global"), p=machine_probabilities))
    for _ in range(12):
        if resource == "machine":
            if move == "block_deadline":
                sequence = reinsert_block_within_deadline(solution.machine_sequence, data, rng)
            else:
                sequence = swap_global_blocks(solution.machine_sequence, data, rng)
            candidate = Solution(sequence, solution.staffing_plan.copy())
        if not (
            np.array_equal(candidate.machine_sequence, solution.machine_sequence)
            and np.array_equal(candidate.staffing_plan, solution.staffing_plan)
        ):
            return candidate, move
    return solution.copy(), f"{move}_noop"
