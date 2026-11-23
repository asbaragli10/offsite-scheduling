"""Parallel evaluation with one random Pareto/Metropolis proposal per batch."""
from __future__ import annotations

import os
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from time import perf_counter
import numpy as np

from scheduling_engine.config import AnnealingConfig, NeighbourhoodConfig, ProblemConfig
from scheduling_engine.data import ProductData
from scheduling_engine.model import (
    OBJECTIVE_NAMES, InfeasibleSchedule, RetainedJigSchedule, ScheduleEvaluation,
    Solution, dominates, evaluate_solution, pareto_deterioration,
)
from scheduling_engine.neighbourhoods import MoveContext, build_move_context, propose_solution
from scheduling_engine.neighbourhoods.selector import is_noop_move, move_family, retains_jig_schedule
from .archive import ParetoEntry, archive_accepts, lexicographic_entry, prune_archive, update_archive
from .hypervolume import dominated_hypervolume
from .stopping import ProgressMonitor
from .milp_integration import MilpConfig, MilpAttempt, MilpSearchState, run_checkpoint


@dataclass
class CandidateResult:
    solution: Solution
    objectives: np.ndarray
    move: str


@dataclass
class AnnealingResult:
    selected_solution: Solution
    selected_evaluation: ScheduleEvaluation
    final_solution: Solution
    final_evaluation: ScheduleEvaluation
    archive: list[ParetoEntry]
    history: list[dict[str, float | int]]
    initial_temperature: float
    objective_scales: np.ndarray
    stopped_early: bool
    stop_reason: str | None
    elapsed_seconds: float
    checkpoints: list[dict[str, object]]
    hypervolume_reference: np.ndarray
    move_statistics: dict[str, dict[str, int]]
    milp_attempts: list[MilpAttempt]


_WORKER_DATA = None
_WORKER_PROBLEM = None
_WORKER_NEIGHBOURHOOD = None


def _initialise_worker(data, problem, neighbourhood) -> None:
    global _WORKER_DATA, _WORKER_PROBLEM, _WORKER_NEIGHBOURHOOD
    _WORKER_DATA, _WORKER_PROBLEM, _WORKER_NEIGHBOURHOOD = data, problem, neighbourhood


def _evaluate_candidate(candidate, move, current, evaluation, data, problem, *, detail=False):
    return evaluate_solution(
        candidate, data, problem, collect_segments=detail,
        fixed_jig_schedule=evaluation if retains_jig_schedule(move) else None,
    )


def _local_candidates(current, context, seeds, data, problem, neighbourhood) -> list[CandidateResult]:
    results = []
    for seed in seeds:
        candidate, move = propose_solution(
            current, context, data, problem, neighbourhood, np.random.default_rng(int(seed))
        )
        try:
            objectives = _evaluate_candidate(
                candidate, move, current, context.evaluation, data, problem
            ).objectives
        except InfeasibleSchedule:
            objectives = np.full(len(OBJECTIVE_NAMES), np.inf)
        results.append(CandidateResult(candidate, objectives, move))
    return results


def _worker_candidate(task) -> CandidateResult:
    current, context, seed = task
    return _worker_candidate_batch((current, context, np.asarray([seed])))[0]


def _worker_candidate_batch(task) -> list[CandidateResult]:
    if _WORKER_DATA is None:
        raise RuntimeError("annealing worker was not initialised")
    current, context, seeds = task
    return _local_candidates(current, context, seeds, _WORKER_DATA, _WORKER_PROBLEM, _WORKER_NEIGHBOURHOOD)


def select_batch_candidate(candidates, current_objectives, scales, temperature, rng):
    """Draw one feasible candidate uniformly, then accept/reject without retry."""
    feasible = [i for i, candidate in enumerate(candidates) if np.all(np.isfinite(candidate.objectives))]
    if not feasible:
        return None, False
    index = int(rng.choice(feasible))
    proposal = candidates[index].objectives
    if not dominates(current_objectives, proposal):
        return index, True
    delta = pareto_deterioration(current_objectives, proposal, scales)
    return index, bool(rng.random() < np.exp(-delta / temperature))


def run_parallel_pareto_annealing(
    initial_solution: Solution, data: ProductData, problem: ProblemConfig,
    neighbourhood: NeighbourhoodConfig, annealing: AnnealingConfig, *,
    initial_temperature: float, objective_scales: np.ndarray,
    initial_evaluation: ScheduleEvaluation | None = None,
    progress: Callable[[dict[str, float | int]], None] | None = None,
    checkpoint: Callable[[dict[str, object]], None] | None = None,
    milp_progress: Callable[[dict[str, object]], None] | None = None,
    milp_config: MilpConfig | None = None,
    max_search_seconds: float | None = None,
) -> AnnealingResult:
    if not np.isfinite(initial_temperature) or initial_temperature <= 0:
        raise ValueError("initial_temperature must be finite and positive")
    scales = np.asarray(objective_scales, dtype=float)
    if scales.shape != (len(OBJECTIVE_NAMES),) or not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("objective_scales must contain three finite positive values")
    if max_search_seconds is not None and max_search_seconds <= 0:
        raise ValueError("max_search_seconds must be positive")
    milp_config = milp_config or MilpConfig()
    milp_search_state = MilpSearchState()
    current = initial_solution.copy()
    evaluation = initial_evaluation or evaluate_solution(current, data, problem, collect_segments=True)
    archive = update_archive([], current, evaluation.objectives,
        retained_jig_schedule=RetainedJigSchedule.from_evaluation(evaluation), limit=annealing.archive_limit)
    workers = annealing.workers or (os.cpu_count() or 1)
    batch_size = max(1, workers * annealing.batch_multiplier)
    rng = np.random.default_rng(annealing.seed)
    temperature = float(initial_temperature)
    reference = evaluation.objectives / scales + annealing.early_stopping_hypervolume_reference_margin
    values = np.asarray([entry.objectives for entry in archive])
    hv = dominated_hypervolume(values / scales, reference)
    monitor = ProgressMonitor(values, scales, annealing)
    history, checkpoints, milp_attempts = [], [], []
    statistics: dict[str, Counter] = {}
    started = perf_counter()

    def save_checkpoint(step, elapsed, volume):
        item = {"temperature_step": step, "elapsed_seconds": elapsed, "hypervolume": volume,
                "objectives": [entry.objectives.tolist() for entry in archive]}
        checkpoints.append(item)
        if checkpoint:
            checkpoint(item)

    save_checkpoint(0, 0.0, hv)
    executor = None
    stopped_early = False
    time_limit_reached = False
    try:
        if workers > 1:
            executor = ProcessPoolExecutor(max_workers=workers, initializer=_initialise_worker,
                initargs=(data, problem, neighbourhood))
        for step in range(1, annealing.temperature_steps + 1):
            attempts, accepted, feasible_count, noops = 0, 0, 0, 0
            family_counts = Counter()
            remaining = annealing.neighbours_per_step
            while remaining:
                if max_search_seconds is not None and perf_counter() - started >= max_search_seconds:
                    time_limit_reached = True
                    break
                count = min(batch_size, remaining)
                remaining -= count
                context = build_move_context(evaluation)
                seeds = rng.integers(0, np.iinfo(np.uint32).max, size=count, dtype=np.uint32)
                if executor is None:
                    candidates = _local_candidates(current, context, seeds, data, problem, neighbourhood)
                else:
                    chunks = [chunk for chunk in np.array_split(seeds, min(workers, count)) if chunk.size]
                    candidates = [candidate for batch in executor.map(
                        _worker_candidate_batch, [(current, context, chunk) for chunk in chunks]
                    ) for candidate in batch]
                for candidate in candidates:
                    stats = statistics.setdefault(candidate.move, Counter(
                        attempted=0, feasible=0, proposed=0, accepted=0, archive_contributing=0, noops=0))
                    stats["attempted"] += 1
                    attempts += 1
                    if is_noop_move(candidate.move):
                        stats["noops"] += 1
                        noops += 1
                    else:
                        family_counts[move_family(candidate.move)] += 1
                    if not np.all(np.isfinite(candidate.objectives)):
                        continue
                    stats["feasible"] += 1
                    feasible_count += 1
                    if archive_accepts(archive, candidate.objectives):
                        replay = _evaluate_candidate(candidate.solution, candidate.move, current,
                            evaluation, data, problem)
                        archive = update_archive(archive, candidate.solution, replay.objectives,
                            retained_jig_schedule=RetainedJigSchedule.from_evaluation(replay), limit=None)
                        stats["archive_contributing"] += 1
                index, accept = select_batch_candidate(candidates, evaluation.objectives, scales, temperature, rng)
                if index is not None:
                    chosen = candidates[index]
                    statistics[chosen.move]["proposed"] += 1
                    if accept:
                        new_evaluation = _evaluate_candidate(chosen.solution, chosen.move, current,
                            evaluation, data, problem, detail=True)
                        current, evaluation = chosen.solution.copy(), new_evaluation
                        statistics[chosen.move]["accepted"] += 1
                        accepted += 1
                archive = prune_archive(archive, annealing.archive_limit)
            if step in milp_config.steps and milp_config.mode != "none":
                time_left = (None if max_search_seconds is None else
                             max(0.0, max_search_seconds - (perf_counter() - started)))
                archive, milp_current, milp_evaluation, new_attempts = run_checkpoint(
                    step, archive, data, problem, milp_config, remaining_seconds=time_left,
                    search_state=milp_search_state)
                milp_attempts.extend(new_attempts)
                if milp_progress:
                    for attempt in new_attempts:
                        milp_progress(attempt.record)
                if milp_current is not None:
                    current, evaluation = milp_current, milp_evaluation
                archive = prune_archive(archive, annealing.archive_limit)
            values = np.asarray([entry.objectives for entry in archive])
            hv = dominated_hypervolume(values / scales, reference)
            state = monitor.update(step, values)
            selected = lexicographic_entry(archive)
            snapshot = {
                "temperature_step": step, "temperature": temperature,
                "elapsed_seconds": perf_counter() - started, "attempts": attempts,
                "feasible_candidates": feasible_count, "infeasible_candidates": attempts - feasible_count,
                "accepted_batches": accepted, "archive_size": len(archive),
                "machine_moves": family_counts["machine"], "redistribution_moves": family_counts["redistribution"],
                "opening_moves": family_counts["opening"], "closing_moves": family_counts["closing"],
                "noop_moves": noops, "normalised_hypervolume": hv,
                "milp_calls": sum("side" in item.record and item.record["decision"] != "skipped"
                                  for item in milp_attempts if item.record["temperature_step"] == step),
                "milp_accepted": sum(item.record["decision"] == "accepted"
                                     for item in milp_attempts if item.record["temperature_step"] == step),
                "partial_step": int(remaining > 0),
                "selected_tardiness_seconds": float(selected.objectives[0]),
                "selected_setup_seconds": float(selected.objectives[1]),
                "selected_labour_cost": float(selected.objectives[2]), **state,
            }
            history.append(snapshot)
            save_checkpoint(step, snapshot["elapsed_seconds"], hv)
            if progress:
                progress(snapshot)
            if state["early_stop"]:
                stopped_early = True
                break
            if time_limit_reached or (max_search_seconds is not None and
                                      perf_counter() - started >= max_search_seconds):
                time_limit_reached = True
                break
            temperature *= annealing.cooling_rate
    finally:
        if executor:
            executor.shutdown(wait=True, cancel_futures=True)
    elapsed = perf_counter() - started
    stopped_early = stopped_early or time_limit_reached
    save_checkpoint(len(history), elapsed, hv)
    selected = lexicographic_entry(archive)
    selected_evaluation = evaluate_solution(selected.solution, data, problem,
        collect_segments=True, fixed_jig_schedule=selected.retained_jig_schedule)
    reason = ("matched_search_time_limit" if time_limit_reached else
              (f"No meaningful objective or hypervolume progress for {annealing.early_stopping_patience} "
               f"steps after {annealing.early_stopping_min_steps} minimum steps")
              if stopped_early else "maximum_temperature_steps")
    return AnnealingResult(selected.solution, selected_evaluation, current, evaluation, archive,
        history, initial_temperature, scales, stopped_early, reason, elapsed, checkpoints,
        reference, {name: dict(counts) for name, counts in statistics.items()}, milp_attempts)
