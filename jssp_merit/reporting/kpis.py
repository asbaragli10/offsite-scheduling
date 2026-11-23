"""Fine-grained and aggregate production KPI calculations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from jssp_merit.config import ProblemConfig
from jssp_merit.data import ProductData
from jssp_merit.model.calendar import SECONDS_PER_DAY, is_weekend
from jssp_merit.model.simulator import ScheduleEvaluation
from jssp_merit.model.types import Solution


DELAY_BUCKETS = (
    ("On time", 0.0, 0.0),
    ("Late: >0 to 6 hours", 0.0, 6 * 3600.0),
    ("Late: >6 to 12 hours", 6 * 3600.0, 12 * 3600.0),
    ("Late: >12 to 24 hours", 12 * 3600.0, 24 * 3600.0),
    ("Late: >1 to 3 days", 24 * 3600.0, 3 * 24 * 3600.0),
    ("Late: >3 to 7 days", 3 * 24 * 3600.0, 7 * 24 * 3600.0),
    ("Late: >7 days", 7 * 24 * 3600.0, np.inf),
)


def delay_distribution_frame(evaluation: ScheduleEvaluation) -> pd.DataFrame:
    """Return mutually exclusive delay categories as counts and percentages."""

    delays = evaluation.tardiness_seconds
    total = len(delays)
    rows: list[dict[str, object]] = []
    for index, (label, lower, upper) in enumerate(DELAY_BUCKETS):
        if index == 0:
            mask = delays <= 0
        elif np.isinf(upper):
            mask = delays > lower
        else:
            mask = (delays > lower) & (delays <= upper)
        count = int(mask.sum())
        rows.append(
            {
                "Delay Category": label,
                "Lower Bound Seconds (exclusive)": lower if index else 0.0,
                "Upper Bound Seconds (inclusive)": upper,
                "Product Count": count,
                "Percentage": count / total if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def jig_kpis_frame(
    solution: Solution,
    evaluation: ScheduleEvaluation,
    data: ProductData,
    problem: ProblemConfig,
) -> pd.DataFrame:
    """Return utilisation and job throughput for each assembly jig."""

    rows: list[dict[str, object]] = []
    total_jobs = data.number_of_products
    last_paid_day = int(max(evaluation.jig_completion_seconds.max(initial=0.0),
                            evaluation.trolley_worker_return_seconds.max(initial=0.0)) // SECONDS_PER_DAY)
    for station in range(problem.jig.number_of_jigs):
        products = np.flatnonzero(evaluation.jig_by_product == station)
        paid_seconds = float(evaluation.jig_paid_worker_seconds_by_station[station])
        busy_worker_seconds = float(evaluation.jig_busy_worker_seconds_by_station[station])
        busy_wall_seconds = float(evaluation.station_day_busy_seconds[:, station].sum())
        staffed_days = int(
            sum(
                solution.staffing_plan[day, station] > 0
                and not is_weekend(data.start_date, day)
                for day in range(
                    min(solution.staffing_plan.shape[0], last_paid_day + 1)
                )
            )
        )
        rows.append(
            {
                "Jig": station + 1,
                "Jobs Processed": int(len(products)),
                "Jobs Percentage": len(products) / total_jobs if total_jobs else 0.0,
                "Staffed Weekdays": staffed_days,
                "Busy Wall Hours": busy_wall_seconds / 3600.0,
                "Busy Worker Hours": busy_worker_seconds / 3600.0,
                "Paid Worker Hours": paid_seconds / 3600.0,
                "Workforce Utilization": (
                    busy_worker_seconds / paid_seconds if paid_seconds > 0 else 0.0
                ),
                "First Job Start": (
                    data.start_date
                    + pd.Timedelta(seconds=float(evaluation.jig_start_seconds[products].min()))
                    if products.size
                    else pd.NaT
                ),
                "Last Job Completion": (
                    data.start_date
                    + pd.Timedelta(seconds=float(evaluation.jig_completion_seconds[products].max()))
                    if products.size
                    else pd.NaT
                ),
            }
        )
    return pd.DataFrame(rows)


def daily_jig_kpis_frame(
    solution: Solution,
    evaluation: ScheduleEvaluation,
    data: ProductData,
    problem: ProblemConfig,
) -> pd.DataFrame:
    """Return day/jig staffing, utilisation, starts, and completions."""

    last_day = int(max(evaluation.jig_completion_seconds.max(initial=0.0),
                       evaluation.trolley_worker_return_seconds.max(initial=0.0)) // SECONDS_PER_DAY)
    rows: list[dict[str, object]] = []
    for day in range(min(last_day + 1, solution.staffing_plan.shape[0])):
        date = data.start_date + pd.Timedelta(days=day)
        for station in range(problem.jig.number_of_jigs):
            workers = int(solution.staffing_plan[day, station])
            busy_wall = float(evaluation.station_day_busy_seconds[day, station])
            busy_worker = float(evaluation.station_day_worker_busy_seconds[day, station])
            paid_worker = (
                workers * problem.jig.daily_work_seconds
                if not is_weekend(data.start_date, day)
                else 0.0
            )
            started = int(
                np.sum(
                    (evaluation.jig_by_product == station)
                    & ((evaluation.jig_start_seconds // SECONDS_PER_DAY).astype(int) == day)
                )
            )
            completed = int(
                np.sum(
                    (evaluation.jig_by_product == station)
                    & ((evaluation.jig_completion_seconds // SECONDS_PER_DAY).astype(int) == day)
                )
            )
            rows.append(
                {
                    "Date": date.date(),
                    "Weekday": date.day_name(),
                    "Jig": station + 1,
                    "Workers": workers,
                    "Jobs Started": started,
                    "Jobs Completed": completed,
                    "Busy Wall Hours": busy_wall / 3600.0,
                    "Busy Worker Hours": busy_worker / 3600.0,
                    "Paid Worker Hours": paid_worker / 3600.0,
                    "Workforce Utilization": busy_worker / paid_worker if paid_worker else 0.0,
                }
            )
    return pd.DataFrame(rows)


def aggregate_kpis(
    solution: Solution,
    evaluation: ScheduleEvaluation,
    data: ProductData,
    problem: ProblemConfig,
) -> dict[str, object]:
    """Calculate the requested end-to-end headline KPIs."""

    delays = evaluation.tardiness_seconds
    late = delays[delays > 0]
    final_completion = float(evaluation.jig_completion_seconds.max(initial=0.0))
    jig_paid = float(evaluation.jig_paid_worker_seconds_by_station.sum())
    jig_busy = float(evaluation.jig_busy_worker_seconds_by_station.sum())
    total_paid = evaluation.machine_paid_worker_seconds + jig_paid
    total_busy = evaluation.machine_busy_worker_seconds + jig_busy
    distribution = delay_distribution_frame(evaluation)
    category_percentages = {
        str(row["Delay Category"]): float(row["Percentage"])
        for _, row in distribution.iterrows()
    }
    return {
        "production_start": data.start_date.isoformat(),
        "final_jig_completion": (
            data.start_date + pd.Timedelta(seconds=final_completion)
        ).isoformat(),
        "products": data.number_of_products,
        "total_labour_cost": evaluation.labour_cost,
        "machine_labour_cost": evaluation.machine_labour_cost,
        "assembly_jig_labour_cost": evaluation.jig_labour_cost,
        "machine_operator_count": problem.machine.operator_count,
        "machine_workforce_utilization": (
            evaluation.machine_busy_worker_seconds / evaluation.machine_paid_worker_seconds
            if evaluation.machine_paid_worker_seconds
            else 0.0
        ),
        "assembly_workforce_utilization": jig_busy / jig_paid if jig_paid else 0.0,
        "overall_workforce_utilization": total_busy / total_paid if total_paid else 0.0,
        "on_time_products": int(np.sum(delays <= 0)),
        "late_products": int(np.sum(delays > 0)),
        "on_time_percentage": float(np.mean(delays <= 0)) if len(delays) else 0.0,
        "late_percentage": float(np.mean(delays > 0)) if len(delays) else 0.0,
        "delay_category_percentages": category_percentages,
        "mean_delay_all_products_seconds": float(delays.mean()) if len(delays) else 0.0,
        "median_delay_all_products_seconds": float(np.median(delays)) if len(delays) else 0.0,
        "mean_delay_late_products_seconds": float(late.mean()) if late.size else 0.0,
        "median_delay_late_products_seconds": float(np.median(late)) if late.size else 0.0,
        "total_tardiness_seconds": evaluation.total_tardiness_seconds,
        "maximum_tardiness_seconds": float(delays.max(initial=0)),
        "p95_tardiness_seconds": float(np.percentile(delays, 95)) if len(delays) else 0.0,
        "machine_setup_seconds": evaluation.machine_setup_seconds,
        "machine_setup_count": int(np.count_nonzero(evaluation.machine_setup_by_product)),
        "mean_jig_queue_wait_calendar_seconds": float(np.mean(
            evaluation.jig_start_seconds - evaluation.trolley_ready_seconds)),
        "makespan_calendar_seconds": final_completion,
        "makespan_calendar_hours": final_completion / 3600.0,
        "makespan_calendar_days": final_completion / SECONDS_PER_DAY,
    }


def write_kpi_bundle(
    output_dir: str | Path,
    *,
    prefix: str,
    solution: Solution,
    evaluation: ScheduleEvaluation,
    data: ProductData,
    problem: ProblemConfig,
) -> dict[str, object]:
    """Write aggregate and fine-grained KPI files with a shared prefix."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    aggregate = aggregate_kpis(solution, evaluation, data, problem)
    with (destination / f"{prefix}_aggregate_kpis.json").open("w", encoding="utf-8") as stream:
        json.dump(aggregate, stream, indent=2)
    flat_aggregate = {
        key: value
        for key, value in aggregate.items()
        if not isinstance(value, dict)
    }
    pd.DataFrame([flat_aggregate]).to_csv(
        destination / f"{prefix}_aggregate_kpis.csv", index=False
    )
    delay_distribution_frame(evaluation).to_csv(
        destination / f"{prefix}_delay_distribution.csv", index=False
    )
    jig_kpis_frame(solution, evaluation, data, problem).to_csv(
        destination / f"{prefix}_jig_kpis.csv", index=False
    )
    daily_jig_kpis_frame(solution, evaluation, data, problem).to_csv(
        destination / f"{prefix}_daily_jig_kpis.csv", index=False
    )
    return aggregate
