"""Bounded machine and jig MILP checkpoints within an SA search."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from time import perf_counter

import numpy as np

from scheduling_engine.model import InfeasibleSchedule, RetainedJigSchedule, Solution, evaluate_solution
from .archive import ParetoEntry, archive_accepts, update_archive


@dataclass(frozen=True)
class MilpConfig:
    mode: str = "none"
    steps: tuple[int, ...] = (25, 45, 60)
    setup_min_seconds: float = 3600.0
    setup_max_seconds: float = 14400.0
    labour_min: float = 39000.0
    labour_cap: float = 42000.0
    machine_call_seconds: float = 30.0
    jig_call_seconds: float = 8.0
    machine_window_size: int = 24
    machine_stride: int = 6
    max_displacement: int = 12
    jig_window_weekdays: int = 3
    max_changed_cells: int = 3

    def __post_init__(self):
        if self.mode not in ("none", "machine", "jigs", "both"):
            raise ValueError("milp mode must be none, machine, jigs or both")
        if not self.steps or any(step < 1 for step in self.steps) or len(set(self.steps)) != len(self.steps):
            raise ValueError("MILP checkpoint steps must be distinct positive integers")
        if any(not np.isfinite(value) or value <= 0 for value in (
                self.machine_call_seconds, self.jig_call_seconds, self.setup_max_seconds,
                self.labour_cap)):
            raise ValueError("MILP budgets and objective constraints must be positive")
        if (not np.isfinite(self.setup_min_seconds) or self.setup_min_seconds < 0
                or self.setup_min_seconds > self.setup_max_seconds
                or not np.isfinite(self.labour_min) or self.labour_min < 0
                or self.labour_min > self.labour_cap):
            raise ValueError("MILP selection bounds must be finite, non-negative and ordered")
        if (self.machine_window_size < 2 or self.machine_stride < 1 or
                self.max_displacement < 1 or self.jig_window_weekdays < 1 or
                self.max_changed_cells < 1):
            raise ValueError("MILP window sizes, stride, displacement and changes must be positive")


@dataclass
class MilpAttempt:
    record: dict
    parent: Solution | None = None
    parent_evaluation: object | None = None
    candidate: Solution | None = None
    candidate_evaluation: object | None = None
    sensitivities: tuple[dict, ...] = ()


@dataclass
class MilpSearchState:
    machine_windows: set[tuple[str, int]] = field(default_factory=set)
    jig_windows: set[tuple[str, str, tuple[int, ...]]] = field(default_factory=set)


def select_target(archive: list[ParetoEntry], config: MilpConfig) -> ParetoEntry | None:
    """Select minimum tardiness within inclusive setup and labour ranges."""
    eligible = [entry for entry in archive
                if config.setup_min_seconds - 1e-6 <= float(entry.objectives[1])
                <= config.setup_max_seconds + 1e-6
                and config.labour_min - 1e-6 <= float(entry.objectives[2])
                <= config.labour_cap + 1e-6]
    return min(eligible, key=lambda entry: (float(entry.objectives[0]),
                                           float(entry.objectives[2]),
                                           float(entry.objectives[1]))) if eligible else None


def _fingerprint(values, dtype) -> str:
    return hashlib.sha256(np.asarray(values, dtype=dtype).tobytes()).hexdigest()


def run_checkpoint(step, archive, data, problem, config: MilpConfig, *,
                   remaining_seconds: float | None = None,
                   search_state: MilpSearchState | None = None):
    """Return archive, incumbent, evaluation and auditable attempts for one step."""
    if config.mode == "none" or step not in config.steps:
        return archive, None, None, []
    search_state = search_state or MilpSearchState()
    target = select_target(archive, config)
    selection_bounds = {"setup_min_seconds": config.setup_min_seconds,
                        "setup_max_seconds": config.setup_max_seconds,
                        "labour_min": config.labour_min, "labour_cap": config.labour_cap}
    if target is None:
        return archive, None, None, [MilpAttempt({"temperature_step": step,
            "decision": "skipped", "reason": "no archive solution within setup and labour limits",
            **selection_bounds})]
    setup_target_seconds = float(target.objectives[1])
    parent = target.solution.copy()
    target_index = next(index for index, entry in enumerate(archive) if entry is target)
    parent_eval = evaluate_solution(parent, data, problem, collect_segments=True,
                                    fixed_jig_schedule=target.retained_jig_schedule)
    if not np.allclose(parent_eval.objectives, target.objectives, rtol=1e-12, atol=1e-7):
        raise ValueError("MILP target cannot be replayed exactly")
    incumbent = incumbent_eval = None
    attempts = []
    parent_origin = "selected_archive_target"
    sides = (("machine",) if config.mode == "machine" else
             ("jigs",) if config.mode == "jigs" else ("machine", "jigs"))
    for side in sides:
        call_started = perf_counter()
        allowance = config.machine_call_seconds if side == "machine" else config.jig_call_seconds
        if remaining_seconds is not None:
            allowance = min(allowance, remaining_seconds)
        stage = "labour_polish" if side == "jigs" and step == max(config.steps) else "tardiness"
        record = {"temperature_step": step, "side": side, "stage": stage,
                  "source_rule": "minimum tardiness within setup and labour limits; ties by labour then setup",
                  "target_archive_index_at_checkpoint": target_index,
                  "parent_origin": parent_origin,
                  "setup_target_seconds": setup_target_seconds,
                  **selection_bounds,
                  "parent_objectives": parent_eval.objectives.tolist(),
                  "parent_sequence_sha256": _fingerprint(parent.machine_sequence, np.int32),
                  "parent_staffing_sha256": _fingerprint(parent.staffing_plan, np.int16),
                  "call_allowance_seconds": allowance,
                  "decision": "skipped"}
        attempt = MilpAttempt(record, parent.copy(), parent_eval)
        attempts.append(attempt)
        if allowance < (3.0 if side == "jigs" else 2.0):
            record["reason"] = "insufficient remaining search time"
            continue
        try:
            if side == "machine":
                from MILP_post_processing.machine.milp_neighborhood import rank_windows, solve_window
                windows = rank_windows(parent, parent_eval, data,
                                       size=config.machine_window_size, stride=config.machine_stride)
                sequence_hash = record["parent_sequence_sha256"]
                ranked = [(rank, window) for rank, window in enumerate(windows, 1)
                          if (sequence_hash, window.start) not in search_state.machine_windows]
                if not ranked:
                    record["reason"] = "no unseen same-coil window"
                    continue
                window_rank, window = ranked[0]
                search_state.machine_windows.add((sequence_hash, window.start))
                window_products = parent.machine_sequence[window.start:window.start + window.size]
                record.update(window_start=window.start, window_size=window.size,
                              window_rank=window_rank,
                              window=f"{window.start}:{window.start + window.size - 1}",
                              coil=window.coil, window_tardiness_seconds=window.tardiness_seconds,
                              proxy_calendar_hours_per_machine_hour=(
                                  86400 / problem.machine.daily_work_seconds),
                              max_displacement=config.max_displacement,
                              window_product_indices=[int(product) for product in window_products],
                              window_product_ids=[data.product_ids[int(product)]
                                                  for product in window_products])
                solver_limit = max(0.1, allowance - 5.0)
                proposal = solve_window(parent, parent_eval, data, window,
                    scale=86400 / problem.machine.daily_work_seconds,
                    max_displacement=config.max_displacement,
                    solver_time_limit=solver_limit)
                record["decision"] = "rejected"
                record.update(solver_time_limit_seconds=solver_limit, solver_status=proposal.status,
                    solver_message=proposal.message, model_rows=proposal.model_rows,
                    model_columns=proposal.model_columns, binary_variables=proposal.binary_variables,
                    build_seconds=proposal.build_seconds, solver_seconds=proposal.solver_seconds,
                    proxy_objective_hours=proposal.proxy_objective_hours,
                    proxy_baseline_hours=proposal.proxy_baseline_hours,
                    solver_bound=proposal.dual_bound_hours, solver_gap=proposal.relative_gap,
                    solver_nodes=proposal.node_count)
                if proposal.segment is None:
                    record["reason"] = "MILP returned no changed machine permutation"
                    continue
                sequence = parent.machine_sequence.copy()
                sequence[window.start:window.start + window.size] = proposal.segment
                candidate = Solution(sequence, parent.staffing_plan.copy())
                retained = None
            else:
                from MILP_post_processing.jigs.milp_neighborhood import rank_windows, solve_window
                mode = "labour" if stage == "labour_polish" else "tardiness"
                windows = rank_windows(parent, parent_eval, data, mode=mode,
                    size=config.jig_window_weekdays,
                    jig_daily_work_seconds=problem.jig.daily_work_seconds)
                plan_key = record["parent_sequence_sha256"] + record["parent_staffing_sha256"]
                ranked = [(rank, window) for rank, window in enumerate(windows, 1)
                          if (plan_key, mode, window.days) not in search_state.jig_windows]
                if not ranked:
                    record["reason"] = "no unseen staffed weekday window"
                    continue
                window_rank, window = ranked[0]
                search_state.jig_windows.add((plan_key, mode, window.days))
                record.update(window_days=list(window.days),
                              window_rank=window_rank,
                              window=",".join(map(str, window.days)),
                              window_score=window.score,
                              max_changed_cells=config.max_changed_cells)
                retained = RetainedJigSchedule.from_evaluation(parent_eval)
                model_limit = max(0.1, allowance - 1.0)
                proposal = solve_window(parent, parent_eval, data, problem, retained, window,
                    mode=mode, labour_cap=min(config.labour_cap, float(parent_eval.objectives[2])),
                    tardiness_cap=float(parent_eval.objectives[0]),
                    max_changes=config.max_changed_cells, time_limit=model_limit)
                record["decision"] = "rejected"
                attempt.sensitivities = proposal.sensitivities
                record.update(solver_time_limit_seconds=model_limit, solver_status=proposal.status,
                    solver_message=proposal.message, model_rows=proposal.model_rows,
                    model_columns=proposal.model_columns, build_seconds=proposal.build_seconds,
                    solver_seconds=proposal.solver_seconds,
                    proxy_tardiness_delta_seconds=proposal.proxy_tardiness_delta,
                    proxy_labour_delta=proposal.proxy_labour_delta,
                    solver_bound=proposal.dual_bound, solver_gap=proposal.relative_gap,
                    solver_nodes=proposal.node_count)
                if proposal.plan is None:
                    record["reason"] = "MILP returned no changed staffing plan"
                    continue
                candidate = Solution(parent.machine_sequence.copy(), proposal.plan)
            attempt.candidate = candidate
            record["candidate_sequence_sha256"] = _fingerprint(candidate.machine_sequence, np.int32)
            record["candidate_staffing_sha256"] = _fingerprint(candidate.staffing_plan, np.int16)
            record["changed_machine_positions"] = int(np.count_nonzero(
                candidate.machine_sequence != parent.machine_sequence))
            record["changed_staffing_cells"] = int(np.count_nonzero(
                candidate.staffing_plan != parent.staffing_plan))
            candidate_eval = evaluate_solution(candidate, data, problem, collect_segments=True,
                                               fixed_jig_schedule=retained)
            attempt.candidate_evaluation = candidate_eval
            record["candidate_objectives"] = candidate_eval.objectives.tolist()
            obj = candidate_eval.objectives
            old = parent_eval.objectives
            if abs(float(obj[1]) - setup_target_seconds) > 1e-6:
                reason = "selected schedule setup changed"
            elif obj[2] > min(config.labour_cap, float(old[2])) + 1e-6:
                reason = "labour budget increased"
            elif stage == "labour_polish" and obj[0] > old[0] + 1e-6:
                reason = "tardiness increased during labour polish"
            elif stage == "labour_polish" and obj[2] >= old[2] - 1e-6:
                reason = "no exact labour improvement"
            elif stage != "labour_polish" and obj[0] >= old[0] - 1e-6:
                reason = "no exact tardiness improvement"
            elif not archive_accepts(archive, obj):
                reason = "candidate dominated by current archive"
            else:
                reason = "exact improvement accepted into archive"
                archive = update_archive(archive, candidate, obj,
                    retained_jig_schedule=RetainedJigSchedule.from_evaluation(candidate_eval),
                    limit=None)
                parent, parent_eval = candidate, candidate_eval
                parent_origin = f"accepted_{side}_proposal_at_same_checkpoint"
                incumbent, incumbent_eval = candidate, candidate_eval
            record["decision"] = "accepted" if reason.startswith("exact improvement") else "rejected"
            record["reason"] = reason
        except InfeasibleSchedule as exc:
            record.update(decision="rejected", reason=f"full replay infeasible: {exc}")
        finally:
            record["call_elapsed_seconds"] = perf_counter() - call_started
            if remaining_seconds is not None:
                remaining_seconds -= record["call_elapsed_seconds"]
    return archive, incumbent, incumbent_eval, attempts
