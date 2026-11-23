"""Deterministic release-event order for committed jig queues."""

from __future__ import annotations

import numpy as np


def released_product_order(
    sequence: np.ndarray, release_seconds: np.ndarray, deadline_seconds: np.ndarray
) -> list[int]:
    """EDD at each release event; later releases cannot displace committed jobs.

    All currently released jobs are assigned immediately, even if they must
    wait on their chosen jig. Therefore chronological release, deadline, then
    machine position is equivalent to draining the EDD queue at each event.
    """
    rank = np.empty(len(sequence), dtype=int)
    rank[sequence] = np.arange(len(sequence))
    return sorted(
        (int(product) for product in sequence),
        key=lambda product: (
            float(release_seconds[product]), float(deadline_seconds[product]), int(rank[product])
        ),
    )
