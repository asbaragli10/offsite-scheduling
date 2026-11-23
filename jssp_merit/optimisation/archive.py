"""Non-dominated archive with optional crowding-distance size control."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from jssp_merit.model import RetainedJigSchedule, Solution


@dataclass
class ParetoEntry:
    solution: Solution
    objectives: np.ndarray
    retained_jig_schedule: RetainedJigSchedule | None = None


def archive_accepts(archive: list[ParetoEntry], objectives: np.ndarray) -> bool:
    """Return whether an objective vector would enter the current archive."""

    objectives = np.asarray(objectives, dtype=float)
    if not np.all(np.isfinite(objectives)):
        return False
    if not archive:
        return True
    values = np.asarray([entry.objectives for entry in archive], dtype=float)
    duplicates = np.all(np.abs(values - objectives) <= 1e-9, axis=1)
    archive_dominates = np.all(values <= objectives + 1e-9, axis=1) & np.any(
        values < objectives - 1e-9,
        axis=1,
    )
    return not bool(np.any(duplicates) or np.any(archive_dominates))


def _crowding_distances(entries: list[ParetoEntry]) -> np.ndarray:
    number = len(entries)
    if number <= 2:
        return np.full(number, np.inf)
    values = np.asarray([entry.objectives for entry in entries], dtype=float)
    distances = np.zeros(number, dtype=float)
    for objective in range(values.shape[1]):
        order = np.argsort(values[:, objective], kind="stable")
        distances[order[0]] = np.inf
        distances[order[-1]] = np.inf
        span = values[order[-1], objective] - values[order[0], objective]
        if span <= 1e-12:
            continue
        for position in range(1, number - 1):
            distances[order[position]] += (
                values[order[position + 1], objective]
                - values[order[position - 1], objective]
            ) / span
    return distances


def update_archive(
    archive: list[ParetoEntry],
    solution: Solution,
    objectives: np.ndarray,
    *,
    retained_jig_schedule: RetainedJigSchedule | None = None,
    limit: int | None,
) -> list[ParetoEntry]:
    """Insert a unique non-dominated point and retain a diverse bounded frontier."""

    objectives = np.asarray(objectives, dtype=float)
    if not archive_accepts(archive, objectives):
        return archive
    if archive:
        values = np.asarray([entry.objectives for entry in archive], dtype=float)
        candidate_dominates = np.all(objectives <= values + 1e-9, axis=1) & np.any(
            objectives < values - 1e-9,
            axis=1,
        )
        updated = [
            entry
            for index, entry in enumerate(archive)
            if not candidate_dominates[index]
        ]
    else:
        updated = []
    updated.append(
        ParetoEntry(
            solution.copy(),
            objectives.copy(),
            retained_jig_schedule.copy()
            if retained_jig_schedule is not None
            else None,
        )
    )
    return prune_archive(updated, limit) if limit is not None else updated


def prune_archive(archive: list[ParetoEntry], limit: int) -> list[ParetoEntry]:
    """Apply crowding-distance truncation once after a complete candidate batch."""

    if limit <= 0:
        raise ValueError("archive limit must be positive")
    if len(archive) <= limit:
        return archive
    distances = _crowding_distances(archive)
    keep = np.argsort(-distances, kind="stable")[:limit]
    keep.sort()
    return [archive[int(index)] for index in keep]


def lexicographic_entry(archive: list[ParetoEntry]) -> ParetoEntry:
    """Deterministic preview only: tardiness, setup, then labour cost."""

    if not archive:
        raise ValueError("Pareto archive is empty")
    return min(archive, key=lambda entry: tuple(float(value) for value in entry.objectives))
