"""Three-objective Pareto utilities; average tardiness is a reporting KPI."""

from __future__ import annotations

import numpy as np


OBJECTIVE_NAMES = (
    "total_tardiness_seconds",
    "machine_setup_seconds",
    "labour_cost",
)


def dominates(left: np.ndarray, right: np.ndarray, *, tolerance: float = 1e-9) -> bool:
    """Return whether ``left`` Pareto-dominates ``right`` for minimisation."""

    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    return bool(np.all(left <= right + tolerance) and np.any(left < right - tolerance))


def pareto_deterioration(current: np.ndarray, neighbour: np.ndarray, scales: np.ndarray) -> float:
    """Sum only normalised worsened components; this is not a weighted objective."""

    safe_scales = np.where(np.asarray(scales, dtype=float) > 1e-12, scales, 1.0)
    return float(np.maximum(0.0, (np.asarray(neighbour) - np.asarray(current)) / safe_scales).sum())
