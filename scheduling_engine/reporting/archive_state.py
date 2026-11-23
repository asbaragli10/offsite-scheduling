"""Portable serialisation for retained Pareto jig-decoder state."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from scheduling_engine.model import RetainedJigSchedule


def retained_jig_schedule_payload(
    schedule: RetainedJigSchedule,
    *,
    product_ids: Sequence[str],
) -> dict[str, object]:
    """Return the version-2 JSON payload for exact jig-schedule replay."""

    number_of_products = len(product_ids)
    schedule.validate(
        number_of_products=number_of_products,
        number_of_jigs=len(schedule.jig_order_by_station),
    )
    assignments = np.asarray(schedule.jig_by_product, dtype=int)
    orders = [
        np.asarray(order, dtype=int) for order in schedule.jig_order_by_station
    ]
    return {
        "jig_assignment_indices": assignments.tolist(),
        "jig_assignment_numbers": (assignments + 1).tolist(),
        "jig_start_seconds": np.asarray(
            schedule.jig_start_seconds, dtype=float
        ).tolist(),
        "processing_order_indices_by_jig": [order.tolist() for order in orders],
        "processing_order_product_ids_by_jig": [
            [str(product_ids[int(product)]) for product in order]
            for order in orders
        ],
    }


def retained_jig_schedule_from_payload(
    payload: object,
) -> RetainedJigSchedule | None:
    """Load retained replay state; return ``None`` for a legacy archive entry."""

    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise ValueError("retained_jig_schedule must be a JSON object")
    try:
        assignments = np.asarray(payload["jig_assignment_indices"], dtype=int)
        starts = np.asarray(payload["jig_start_seconds"], dtype=float)
        orders = tuple(
            np.asarray(order, dtype=int)
            for order in payload["processing_order_indices_by_jig"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("retained_jig_schedule uses an unsupported schema") from exc
    schedule = RetainedJigSchedule(
        jig_by_product=assignments,
        jig_order_by_station=orders,
        jig_start_seconds=starts,
    )
    schedule.validate(
        number_of_products=len(assignments),
        number_of_jigs=len(orders),
    )
    return schedule
