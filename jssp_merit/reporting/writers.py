"""Human-readable schedule, staffing, history, and Pareto-front outputs."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from jssp_merit.config import AnnealingConfig, NeighbourhoodConfig, ProblemConfig
from jssp_merit.data import ProductData
from jssp_merit.model import OBJECTIVE_NAMES, ScheduleEvaluation, Solution, evaluate_solution
from jssp_merit.optimisation import AnnealingResult
from jssp_merit.neighbourhoods.selector import move_family

from .archive_state import retained_jig_schedule_payload
from .kpis import write_kpi_bundle
from .details import product_activity, write_details


def _timestamp(data: ProductData, seconds: float) -> pd.Timestamp:
    if not np.isfinite(seconds):
        return pd.NaT
    return data.start_date + pd.Timedelta(seconds=float(seconds))


def schedule_frame(
    solution: Solution, evaluation: ScheduleEvaluation, data: ProductData
) -> pd.DataFrame:
    """Return one row per product in machine/queue order."""

    rows: list[dict[str, object]] = []
    activity = product_activity(evaluation, data.number_of_products)
    jig_position = np.zeros(data.number_of_products, dtype=int)
    for order in evaluation.jig_order_by_station:
        jig_position[order] = np.arange(1, len(order) + 1)
    assembly_workers = {}
    for segment in evaluation.processing_segments:
        if segment.phase == "assembly" and segment.product_index not in assembly_workers:
            assembly_workers[segment.product_index] = segment.workers
    for position, product in enumerate(solution.machine_sequence):
        product = int(product)
        jig = int(evaluation.jig_by_product[product])
        assembly_start = float(evaluation.jig_assembly_start_seconds[product])
        assembly_day = int(assembly_start // (24 * 3600))
        assembly_workers_at_start = (
            int(solution.staffing_plan[assembly_day, jig])
            if 0 <= assembly_day < solution.staffing_plan.shape[0]
            else 0
        )
        assembly_workers_at_start = assembly_workers.get(product, assembly_workers_at_start)
        rows.append(
            {
                "Product Index": product,
                "Sequence Position": position + 1,
                "Source CSV Row": product + 2,
                "Unique Identifier": data.product_ids[product],
                "Coil Size": int(data.coil_sizes[product]),
                "Machine Setup Seconds": evaluation.machine_setup_by_product[product],
                "Machine Work Seconds": float(data.machine_work_seconds[product]),
                "Machine Start": _timestamp(data, evaluation.machine_start_seconds[product]),
                "Machine Completion": _timestamp(
                    data, evaluation.machine_completion_seconds[product]
                ),
                "Trolley": int(evaluation.trolley_by_product[product]) + 1,
                "Physical Trolley": int(evaluation.physical_trolley_by_product[product]) + 1,
                "Trolley Transfer Start": _timestamp(
                    data, evaluation.trolley_transfer_start_seconds[product]
                ),
                "Trolley Jig-Queue Arrival": _timestamp(
                    data, evaluation.trolley_ready_seconds[product]
                ),
                "Machine Operator Return": _timestamp(
                    data, evaluation.trolley_operator_return_seconds[product]
                ),
                "Screws": int(data.screws[product]),
                "Studs": int(data.studs[product]),
                "Jig Base Work Seconds": evaluation.jig_base_work_seconds[product],
                "Material Handling Base Seconds": (
                    evaluation.jig_handling_base_work_seconds[product]
                ),
                "Assembly Base Seconds": (
                    evaluation.jig_assembly_base_work_seconds[product]
                ),
                "Jig": int(evaluation.jig_by_product[product]) + 1,
                "Jig Start": _timestamp(data, evaluation.jig_start_seconds[product]),
                "Material Handling Start": _timestamp(
                    data, evaluation.jig_handling_start_seconds[product]
                ),
                "Material Handling Completion": _timestamp(
                    data, evaluation.jig_handling_completion_seconds[product]
                ),
                "Assembly Start": _timestamp(
                    data, evaluation.jig_assembly_start_seconds[product]
                ),
                "Assembly Workers at Start": assembly_workers_at_start,
                "Assembly Completion": _timestamp(
                    data, evaluation.jig_assembly_completion_seconds[product]
                ),
                "Jig Completion": _timestamp(data, evaluation.jig_completion_seconds[product]),
                "Production Deadline": data.deadlines[product],
                "Tardiness Seconds": evaluation.tardiness_seconds[product],
                "Deadline Seconds": data.deadline_seconds[product],
                "Machine Start Seconds": evaluation.machine_start_seconds[product],
                "Machine Completion Seconds": evaluation.machine_completion_seconds[product],
                "Setup Start Seconds": evaluation.machine_setup_start_seconds[product],
                "Setup Completion Seconds": evaluation.machine_setup_completion_seconds[product],
                "Setup Start": _timestamp(data, evaluation.machine_setup_start_seconds[product]),
                "Setup Completion": _timestamp(data, evaluation.machine_setup_completion_seconds[product]),
                "Trolley Transfer Start Seconds": evaluation.trolley_transfer_start_seconds[product],
                "Trolley Release Seconds": evaluation.trolley_ready_seconds[product],
                "Machine Operator Return Seconds": evaluation.trolley_operator_return_seconds[product],
                "Empty Trolley Trigger Seconds": evaluation.trolley_empty_trigger_seconds[
                    evaluation.trolley_by_product[product]],
                "Empty Trolley Arrival at Machine Seconds": evaluation.trolley_empty_arrival_seconds[
                    evaluation.trolley_by_product[product]],
                "Jig Worker Return Seconds": evaluation.trolley_worker_return_seconds[
                    evaluation.trolley_by_product[product]],
                "Machine Wait for Empty Trolley Seconds": (
                    evaluation.trolley_wait_end_seconds[evaluation.trolley_by_product[product]]
                    - evaluation.trolley_wait_start_seconds[evaluation.trolley_by_product[product]]),
                "Machine Wait for Transfer Calendar Seconds": (
                    evaluation.trolley_transfer_start_seconds[product] - evaluation.machine_completion_seconds[product]),
                "Jig Queue Wait Calendar Seconds": (
                    evaluation.jig_start_seconds[product] - evaluation.trolley_ready_seconds[product]),
                "Jig Processing Position": int(jig_position[product]),
                "Handling Start Seconds": evaluation.jig_handling_start_seconds[product],
                "Handling Completion Seconds": evaluation.jig_handling_completion_seconds[product],
                "Assembly Start Seconds": evaluation.jig_assembly_start_seconds[product],
                "Assembly Completion Seconds": evaluation.jig_assembly_completion_seconds[product],
                "Handling Active Seconds": activity["handling_active_seconds"][product],
                "Assembly Active Seconds": activity["assembly_active_seconds"][product],
                "Jig Worker Seconds": activity["jig_worker_seconds"][product],
                "Jig Paused Calendar Seconds": max(0.0,
                    evaluation.jig_completion_seconds[product] - evaluation.jig_start_seconds[product]
                    - activity["handling_active_seconds"][product] - activity["assembly_active_seconds"][product]),
            }
        )
    return pd.DataFrame(rows)


def _staffing_frame(
    solution: Solution,
    data: ProductData,
    evaluation: ScheduleEvaluation | None = None,
) -> pd.DataFrame:
    plan = solution.staffing_plan
    if evaluation is not None:
        last_day = int(max(evaluation.jig_completion_seconds.max(initial=0.0),
                           evaluation.trolley_worker_return_seconds.max(initial=0.0)) // (24 * 3600))
        plan = plan[: last_day + 1]
    frame = pd.DataFrame(
        plan, columns=[f"Jig {station + 1} Workers" for station in range(plan.shape[1])]
    )
    frame.insert(
        0,
        "Date",
        [data.start_date.date() + pd.Timedelta(days=day) for day in range(plan.shape[0])],
    )
    frame.insert(1, "Weekday", [pd.Timestamp(value).day_name() for value in frame["Date"]])
    return frame


def _jsonable(value: object) -> object:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_outputs(
    output_dir: str | Path,
    *,
    initial_solution: Solution,
    initial_evaluation: ScheduleEvaluation,
    result: AnnealingResult,
    data: ProductData,
    problem: ProblemConfig,
    neighbourhood: NeighbourhoodConfig,
    annealing: AnnealingConfig,
    input_file: str | Path,
    machine_work_time_unit: str,
    machine_work_time_column: str | None = None,
) -> Path:
    """Write all reproducible optimisation artefacts."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    input_source = Path(input_file).resolve()
    portable_input = destination / "input_products.csv"
    if input_source != portable_input.resolve():
        shutil.copy2(input_source, portable_input)
    input_sha256 = _sha256(portable_input)
    write_schedule_bundle(destination / "initial", initial_solution, initial_evaluation, data, problem)
    write_schedule_bundle(destination / "final_incumbent", result.final_solution, result.final_evaluation, data, problem)
    schedule_frame(initial_solution, initial_evaluation, data).to_csv(
        destination / "initial_schedule.csv", index=False
    )
    _staffing_frame(initial_solution, data, initial_evaluation).to_csv(
        destination / "initial_staffing_plan.csv", index=False
    )
    initial_kpis = write_kpi_bundle(
        destination,
        prefix="initial",
        solution=initial_solution,
        evaluation=initial_evaluation,
        data=data,
        problem=problem,
    )
    schedule_frame(result.selected_solution, result.selected_evaluation, data).to_csv(
        destination / "lexicographic_preview_schedule.csv", index=False
    )
    _staffing_frame(result.selected_solution, data, result.selected_evaluation).to_csv(
        destination / "lexicographic_preview_staffing_plan.csv", index=False
    )
    pd.DataFrame(result.history).to_csv(destination / "annealing_history.csv", index=False)

    sorted_archive = sorted(
        result.archive, key=lambda entry: tuple(float(value) for value in entry.objectives)
    )
    frontier_rows = []
    archive_solutions = []
    for solution_index, entry in enumerate(sorted_archive):
        if entry.retained_jig_schedule is None:
            raise ValueError(
                "Pareto entry is missing retained jig schedule state; exact replay "
                "cannot be written"
            )
        evaluation = evaluate_solution(entry.solution, data, problem, collect_segments=True,
                                       fixed_jig_schedule=entry.retained_jig_schedule)
        if not np.allclose(evaluation.objectives, entry.objectives, rtol=1e-12, atol=1e-7):
            raise ValueError(f"Pareto solution {solution_index} cannot be replayed exactly")
        write_schedule_bundle(destination / "solutions" / f"solution_{solution_index:04d}",
                              entry.solution, evaluation, data, problem)
        frontier_rows.append(
            {
                "Solution Index": solution_index,
                **{
                    name: float(entry.objectives[index])
                    for index, name in enumerate(OBJECTIVE_NAMES)
                },
            }
        )
        archive_solutions.append({
            "solution_index": solution_index,
            "objectives": {
                name: float(entry.objectives[index])
                for index, name in enumerate(OBJECTIVE_NAMES)
            },
            "machine_sequence_indices": [
                int(product) for product in entry.solution.machine_sequence
            ],
            "machine_sequence_product_ids": [
                data.product_ids[int(product)] for product in entry.solution.machine_sequence
            ],
            "staffing_plan": entry.solution.staffing_plan.tolist(),
            "retained_jig_schedule": retained_jig_schedule_payload(
                entry.retained_jig_schedule,
                product_ids=data.product_ids,
            ),
        })
    pd.DataFrame(frontier_rows).to_csv(destination / "pareto_frontier.csv", index=False)
    archive_payload = {
        "schema_version": 4,
        "objective_order": list(OBJECTIVE_NAMES),
        "solutions": archive_solutions,
    }
    with (destination / "pareto_solutions.json").open("w", encoding="utf-8") as stream:
        json.dump(archive_payload, stream, indent=2)

    summary = {
        "input_file": str(input_source),
        "portable_input_file": portable_input.name,
        "input_sha256": input_sha256,
        "machine_work_time_unit": machine_work_time_unit,
        "machine_work_time_column": machine_work_time_column,
        "production_start": data.start_date.isoformat(),
        "number_of_products": data.number_of_products,
        "objective_order": list(OBJECTIVE_NAMES),
        "preview_selection_rule": "lexicographic preview only: tardiness, setup duration, labour cost",
        "initial_objectives": dict(zip(OBJECTIVE_NAMES, initial_evaluation.objectives.tolist())),
        "lexicographic_preview_objectives": dict(
            zip(OBJECTIVE_NAMES, result.selected_evaluation.objectives.tolist())
        ),
        "initial_aggregate_kpis": initial_kpis,
        "pareto_archive_size": len(result.archive),
        "pareto_solution_schema_version": 4,
        "pareto_retained_jig_schedule": (
            "product-to-jig assignment, explicit processing order per jig, "
            "and evaluated jig start times"
        ),
        "completed_temperature_decreases": len(result.history),
        "stopped_early": result.stopped_early,
        "stop_reason": result.stop_reason,
        "initial_temperature": result.initial_temperature,
        "objective_scales": result.objective_scales.tolist(),
        "hypervolume_reference": result.hypervolume_reference.tolist(),
        "search_elapsed_seconds": result.elapsed_seconds,
        "live_reference_hypervolume": result.checkpoints[-1]["hypervolume"],
        "problem_config": asdict(problem),
        "neighbourhood_config": asdict(neighbourhood),
        "annealing_config": asdict(annealing),
    }
    with (destination / "run_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(_jsonable(summary), stream, indent=2)
    pd.DataFrame([{k: v for k, v in item.items() if k != "objectives"}
                  for item in result.checkpoints]).to_csv(destination / "hypervolume_history.csv", index=False)
    with (destination / "archive_checkpoints.jsonl").open("w", encoding="utf-8") as stream:
        for item in result.checkpoints:
            stream.write(json.dumps(item) + "\n")
    move_rows = [{"move": name, "family": move_family(name), **counts}
                 for name, counts in result.move_statistics.items()]
    pd.DataFrame(move_rows).to_csv(
        destination / "move_statistics.csv", index=False)
    families = pd.DataFrame(move_rows).drop(columns="move").groupby("family").sum().reindex(
        ["machine", "redistribution", "opening", "closing"], fill_value=0)
    families["attempted_share_all"] = families.attempted / families.attempted.sum()
    jig_attempts = families.loc[["redistribution", "opening", "closing"], "attempted"].sum()
    families["attempted_share_within_jigs"] = [np.nan] + [
        float(families.loc[name, "attempted"] / jig_attempts) if jig_attempts else 0.0
        for name in ("redistribution", "opening", "closing")]
    families.to_csv(destination / "move_family_statistics.csv", index_label="family")
    return destination


def write_schedule_bundle(destination, solution, evaluation, data, problem):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    schedule_frame(solution, evaluation, data).to_csv(destination / "schedule.csv", index=False)
    _staffing_frame(solution, data).to_csv(destination / "full_staffing_plan.csv", index=False)
    write_details(destination, solution, evaluation, data, problem)
    return write_kpi_bundle(destination, prefix="schedule", solution=solution, evaluation=evaluation,
                            data=data, problem=problem)


def write_milp_attempts(destination, attempts, data, problem):
    """Export exact schedules and solver evidence for every in-loop MILP call."""
    root = Path(destination) / "milp_checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for number, attempt in enumerate(attempts, 1):
        record = attempt.record
        step = record["temperature_step"]
        side = record.get("side", "selection")
        folder = root / f"step_{step:04d}" / f"attempt_{number:04d}_{side}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "attempt.json").write_text(json.dumps(_jsonable(record), indent=2), encoding="utf-8")
        (folder / "solver.log").write_text("\n".join(
            f"{key}: {record.get(key)}" for key in (
                "side", "stage", "decision", "reason", "solver_status", "solver_message",
                "model_rows", "model_columns", "solver_nodes", "solver_gap",
                "solver_bound", "build_seconds", "solver_seconds", "call_elapsed_seconds")) + "\n",
            encoding="utf-8")
        if side == "jigs":
            pd.DataFrame(attempt.sensitivities, columns=(
                "day", "jig", "old_workers", "new_workers", "feasible",
                "tardiness_delta_seconds", "labour_delta", "setup_delta_seconds",
                "reason")).to_csv(folder / "single_cell_sensitivities.csv", index=False)
        if attempt.parent is not None and attempt.parent_evaluation is not None:
            write_schedule_bundle(folder / "parent", attempt.parent,
                                  attempt.parent_evaluation, data, problem)
        if attempt.parent is not None and attempt.candidate is not None and attempt.candidate_evaluation is not None:
            candidate = attempt.candidate
            evaluation = attempt.candidate_evaluation
            candidate_folder = folder / "candidate_0001"
            write_schedule_bundle(candidate_folder, candidate, evaluation, data, problem)
            before = attempt.parent.machine_sequence
            after = candidate.machine_sequence
            machine_rows = [{"Machine Position": int(position),
                "Parent Product Index": int(before[position]),
                "Parent Product ID": data.product_ids[int(before[position])],
                "Candidate Product Index": int(after[position]),
                "Candidate Product ID": data.product_ids[int(after[position])]}
                for position in np.flatnonzero(before != after)]
            pd.DataFrame(machine_rows, columns=("Machine Position", "Parent Product Index",
                "Parent Product ID", "Candidate Product Index", "Candidate Product ID")).to_csv(
                candidate_folder / "machine_sequence_delta.csv", index=False)
            old_plan, new_plan = attempt.parent.staffing_plan, candidate.staffing_plan
            staffing_rows = [{"Day Index": int(day), "Jig": int(jig + 1),
                "Parent Workers": int(old_plan[day, jig]), "Candidate Workers": int(new_plan[day, jig]),
                "Change": int(new_plan[day, jig] - old_plan[day, jig])}
                for day, jig in np.argwhere(old_plan != new_plan)]
            pd.DataFrame(staffing_rows, columns=("Day Index", "Jig", "Parent Workers",
                "Candidate Workers", "Change")).to_csv(candidate_folder / "staffing_delta.csv", index=False)
            previous = attempt.parent_evaluation
            if previous is not None:
                pd.DataFrame({
                    "Product Index": np.arange(data.number_of_products),
                    "Product ID": list(data.product_ids),
                    "Deadline Seconds": data.deadline_seconds,
                    "Parent Machine Completion Seconds": previous.machine_completion_seconds,
                    "Candidate Machine Completion Seconds": evaluation.machine_completion_seconds,
                    "Machine Completion Change Seconds": evaluation.machine_completion_seconds - previous.machine_completion_seconds,
                    "Parent Trolley": previous.trolley_by_product + 1,
                    "Candidate Trolley": evaluation.trolley_by_product + 1,
                    "Parent Jig": previous.jig_by_product + 1,
                    "Candidate Jig": evaluation.jig_by_product + 1,
                    "Parent Jig Completion Seconds": previous.jig_completion_seconds,
                    "Candidate Jig Completion Seconds": evaluation.jig_completion_seconds,
                    "Jig Completion Change Seconds": evaluation.jig_completion_seconds - previous.jig_completion_seconds,
                    "Parent Tardiness Seconds": previous.tardiness_seconds,
                    "Candidate Tardiness Seconds": evaluation.tardiness_seconds,
                    "Tardiness Change Seconds": evaluation.tardiness_seconds - previous.tardiness_seconds,
                }).to_csv(candidate_folder / "product_delta.csv", index=False)
        rows.append({"attempt": number, **{key: value for key, value in record.items()
                    if not isinstance(value, (list, dict))}})
    pd.DataFrame(rows).to_csv(root / "milp_attempts.csv", index=False)
    return root


def write_manual_evaluation(
    output_dir: str | Path,
    *,
    solution_index: int,
    solution: Solution,
    evaluation: ScheduleEvaluation,
    data: ProductData,
    problem: ProblemConfig,
) -> Path:
    """Write the detailed second-stage evaluation of a manually chosen point."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    write_schedule_bundle(destination, solution, evaluation, data, problem)
    schedule_frame(solution, evaluation, data).to_csv(
        destination / "selected_schedule.csv", index=False
    )
    _staffing_frame(solution, data, evaluation).to_csv(
        destination / "selected_staffing_plan.csv", index=False
    )
    aggregate = write_kpi_bundle(
        destination,
        prefix="selected",
        solution=solution,
        evaluation=evaluation,
        data=data,
        problem=problem,
    )
    with (destination / "selection_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "solution_index": solution_index,
                "objectives": dict(zip(OBJECTIVE_NAMES, evaluation.objectives.tolist())),
                "aggregate_kpis": aggregate,
            },
            stream,
            indent=2,
        )
    return destination
