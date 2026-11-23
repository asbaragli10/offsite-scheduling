"""Dependency-free dominated hypervolume for small multi-objective archives."""

from __future__ import annotations

import numpy as np


def dominated_hypervolume(points: np.ndarray, reference: np.ndarray) -> float:
    """Return the exact minimisation hypervolume dominated up to ``reference``.

    Points that are not strictly better than the fixed reference in every
    objective contribute no volume.  The annealer calls this only once per
    temperature decrease, so an exact recursive sweep is inexpensive for the
    bounded three-objective archive.
    """

    values = np.asarray(points, dtype=float)
    bound = np.asarray(reference, dtype=float)
    if values.ndim != 2:
        raise ValueError("hypervolume points must be a two-dimensional array")
    if bound.ndim != 1 or values.shape[1] != bound.size:
        raise ValueError("hypervolume reference dimension does not match points")
    if bound.size == 0 or not np.all(np.isfinite(bound)):
        raise ValueError("hypervolume reference must contain finite values")
    if values.size == 0:
        return 0.0
    finite = np.all(np.isfinite(values), axis=1)
    contributing = finite & np.all(values < bound, axis=1)
    return _sweep_hypervolume(values[contributing], bound)


def _sweep_hypervolume(points: np.ndarray, reference: np.ndarray) -> float:
    if points.shape[0] == 0:
        return 0.0
    dimensions = reference.size
    if dimensions == 1:
        return max(0.0, float(reference[0] - points[:, 0].min()))
    if dimensions == 2:
        order = np.argsort(points[:, 0], kind="stable")
        best_second = float(reference[1])
        area = 0.0
        for index in order:
            first = float(points[index, 0])
            second = float(points[index, 1])
            if second < best_second:
                area += (float(reference[0]) - first) * (best_second - second)
                best_second = second
        return max(0.0, area)

    order = np.argsort(points[:, -1], kind="stable")
    ordered = points[order]
    volume = 0.0
    start = 0
    number = ordered.shape[0]
    while start < number:
        coordinate = float(ordered[start, -1])
        end = start + 1
        while end < number and abs(float(ordered[end, -1]) - coordinate) <= 1e-12:
            end += 1
        next_coordinate = (
            float(ordered[end, -1]) if end < number else float(reference[-1])
        )
        width = next_coordinate - coordinate
        if width > 0:
            volume += width * _sweep_hypervolume(
                ordered[:end, :-1], reference[:-1]
            )
        start = end
    return max(0.0, float(volume))
