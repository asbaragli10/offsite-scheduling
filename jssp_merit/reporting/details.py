"""Lossless numeric schedule state and resource activity exports."""
from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path
import numpy as np
import pandas as pd

from jssp_merit.model import RetainedJigSchedule
from jssp_merit.model.calendar import SECONDS_PER_DAY, is_weekend
from .archive_state import retained_jig_schedule_payload


def product_activity(evaluation, count):
    totals = {name: np.zeros(count) for name in (
        "handling_active_seconds", "assembly_active_seconds", "jig_worker_seconds"
    )}
    for segment in evaluation.processing_segments:
        duration = segment.end_seconds - segment.start_seconds
        if segment.phase in ("handling", "assembly"):
            totals[f"{segment.phase}_active_seconds"][segment.product_index] += duration
            totals["jig_worker_seconds"][segment.product_index] += duration * segment.workers
    return totals


def write_details(directory, solution, evaluation, data, problem):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {field.name: getattr(evaluation, field.name) for field in fields(evaluation)
              if isinstance(getattr(evaluation, field.name), np.ndarray)}
    arrays.update(machine_sequence=solution.machine_sequence, staffing_plan=solution.staffing_plan,
                  deadline_seconds=data.deadline_seconds, machine_work_seconds=data.machine_work_seconds,
                  screws=data.screws, studs=data.studs, coil_sizes=data.coil_sizes)
    np.savez_compressed(directory / "schedule_arrays.npz", **arrays)
    state = {
        "schema_version": 4, "machine_sequence_indices": solution.machine_sequence.tolist(),
        "staffing_plan": solution.staffing_plan.tolist(),
        "retained_jig_schedule": retained_jig_schedule_payload(
            RetainedJigSchedule.from_evaluation(evaluation), product_ids=data.product_ids),
    }
    (directory / "solution_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")

    columns = ["product_index", "source_csv_row", "product_id", "resource", "phase", "day_index",
               "start_seconds", "end_seconds", "start", "end", "workers", "active_seconds",
               "worker_seconds", "processing_rate", "base_work_seconds", "trolley",
               "physical_trolley"]
    machine_rows, jig_rows = [], []
    for resource, segments, rows in (
        ("machine", evaluation.machine_segments, machine_rows),
        ("jig", evaluation.processing_segments, jig_rows),
    ):
        for segment in segments:
            duration = segment.end_seconds - segment.start_seconds
            productive = segment.phase in ("setup", "processing", "handling", "assembly")
            rate = (1.0 if resource == "machine" else
                    segment.workers ** problem.jig.worker_efficiency_exponent) if productive else 0.0
            product = segment.product_index
            rows.append({
                "product_index": product, "source_csv_row": product + 2,
                "product_id": data.product_ids[product],
                "resource": "machine" if resource == "machine" else f"jig_{segment.station + 1}",
                "phase": segment.phase, "day_index": segment.day_index,
                "start_seconds": segment.start_seconds, "end_seconds": segment.end_seconds,
                "start": data.start_date + pd.Timedelta(seconds=segment.start_seconds),
                "end": data.start_date + pd.Timedelta(seconds=segment.end_seconds),
                "workers": segment.workers, "active_seconds": duration,
                "worker_seconds": duration * segment.workers, "processing_rate": rate,
                "base_work_seconds": duration * rate,
                "trolley": int(evaluation.trolley_by_product[product]) + 1,
                "physical_trolley": int(evaluation.physical_trolley_by_product[product]) + 1,
            })
    pd.DataFrame(machine_rows, columns=columns).to_csv(directory / "machine_segments.csv", index=False)
    pd.DataFrame(jig_rows, columns=columns).to_csv(directory / "jig_segments.csv", index=False)

    last_machine_day = int(evaluation.trolley_operator_return_seconds.max() // SECONDS_PER_DAY)
    last_jig_day = int(max(evaluation.jig_completion_seconds.max(),
                           evaluation.trolley_worker_return_seconds.max()) // SECONDS_PER_DAY)
    machine_busy = np.zeros(last_machine_day + 1)
    for segment in evaluation.machine_segments:
        machine_busy[segment.day_index] += segment.end_seconds - segment.start_seconds
    rows = []
    for day in range(max(last_machine_day, last_jig_day) + 1):
        weekday = not is_weekend(data.start_date, day)
        for station in range(-1, problem.jig.number_of_jigs):
            is_machine = station == -1
            config = problem.machine if is_machine else problem.jig
            workers = (problem.machine.operator_count if day <= last_machine_day else 0) if is_machine else (
                int(solution.staffing_plan[day, station]) if day <= last_jig_day else 0)
            workers = workers if weekday else 0
            busy = float(machine_busy[day]) if is_machine and day <= last_machine_day else (
                float(evaluation.station_day_busy_seconds[day, station])
                if not is_machine and day <= last_jig_day else 0.0)
            paid = workers * config.daily_work_seconds
            busy_worker = (busy * workers if is_machine else
                           float(evaluation.station_day_worker_busy_seconds[day, station]))
            hourly_rate = problem.machine.operator_hourly_cost if is_machine else problem.jig.worker_hourly_cost
            rows.append({
                "day_index": day, "date": (data.start_date + pd.Timedelta(days=day)).date(),
                "resource": "machine" if is_machine else f"jig_{station + 1}",
                "shift_start_seconds": day * SECONDS_PER_DAY,
                "shift_end_seconds": day * SECONDS_PER_DAY + config.daily_work_seconds,
                "workers": workers, "busy_seconds": busy, "paid_worker_seconds": paid,
                "busy_worker_seconds": busy_worker, "idle_paid_worker_seconds": max(0.0, paid - busy_worker),
                "hourly_cost": hourly_rate, "paid_labour_cost": paid / 3600 * hourly_rate,
            })
    pd.DataFrame(rows).to_csv(directory / "resource_days.csv", index=False)
    trolley_rows = []
    for trolley in np.unique(evaluation.trolley_by_product):
        products = np.flatnonzero(evaluation.trolley_by_product == trolley)
        p = int(products[0])
        trolley_rows.append({"trolley": int(trolley) + 1, "products": len(products),
            "physical_trolley": int(evaluation.physical_trolley_by_product[p]) + 1,
            "product_indices": json.dumps(products.tolist()),
            "last_product_machine_completion_seconds": float(evaluation.machine_completion_seconds[products].max()),
            "transfer_start_seconds": float(evaluation.trolley_transfer_start_seconds[p]),
            "jig_queue_arrival_seconds": float(evaluation.trolley_ready_seconds[p]),
            "machine_operator_return_seconds": float(evaluation.trolley_operator_return_seconds[p]),
            "empty_trigger_seconds": float(evaluation.trolley_empty_trigger_seconds[trolley]),
            "empty_return_start_seconds": float(evaluation.trolley_empty_return_start_seconds[trolley]),
            "empty_arrival_machine_seconds": float(evaluation.trolley_empty_arrival_seconds[trolley]),
            "jig_worker_return_seconds": float(evaluation.trolley_worker_return_seconds[trolley]),
            "return_jig": int(evaluation.trolley_return_jig[trolley]) + 1,
            "machine_wait_for_empty_start_seconds": float(evaluation.trolley_wait_start_seconds[trolley]),
            "machine_wait_for_empty_end_seconds": float(evaluation.trolley_wait_end_seconds[trolley]),
            "machine_wait_for_empty_seconds": float(evaluation.trolley_wait_end_seconds[trolley]
                                                    - evaluation.trolley_wait_start_seconds[trolley]),
            "loaded_round_trip_seconds": problem.trolley.transfer_seconds,
            "empty_round_trip_seconds": problem.trolley.empty_return_seconds})
    pd.DataFrame(trolley_rows).to_csv(directory / "trolleys.csv", index=False)
