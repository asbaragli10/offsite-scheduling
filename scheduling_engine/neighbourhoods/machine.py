"""The validated 50/50 block hybrid for the stand-alone machine."""

from __future__ import annotations

import numpy as np

from scheduling_engine.data import ProductData


def _coil_runs(sequence: np.ndarray, coils: np.ndarray) -> list[tuple[int, int]]:
    if len(sequence) == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = 0
    previous = coils[sequence[0]]
    for position in range(1, len(sequence)):
        current = coils[sequence[position]]
        if current != previous:
            runs.append((start, position - 1))
            start = position
            previous = current
    runs.append((start, len(sequence) - 1))
    return runs


def swap_global_blocks(
    sequence: np.ndarray, data: ProductData, rng: np.random.Generator
) -> np.ndarray:
    """Swap two different-coil contiguous blocks globally (BSG)."""

    runs = _coil_runs(sequence, data.coil_sizes)
    if len(runs) < 2:
        return sequence.copy()
    run_coils = np.asarray([data.coil_sizes[sequence[start]] for start, _ in runs])
    first = int(rng.integers(len(runs)))
    alternatives = np.flatnonzero(run_coils != run_coils[first])
    if not alternatives.size:
        return sequence.copy()
    second = int(rng.choice(alternatives))
    if first > second:
        first, second = second, first
    first_start, first_end = runs[first]
    second_start, second_end = runs[second]
    return np.concatenate(
        (
            sequence[:first_start],
            sequence[second_start : second_end + 1],
            sequence[first_end + 1 : second_start],
            sequence[first_start : first_end + 1],
            sequence[second_end + 1 :],
        )
    )


def reinsert_block_within_deadline(
    sequence: np.ndarray, data: ProductData, rng: np.random.Generator
) -> np.ndarray:
    """Remove and reinsert a same-coil run among products with one deadline (BSwD)."""

    counts = np.unique(data.deadline_seconds, return_counts=True)
    valid_deadlines = counts[0][counts[1] >= 2]
    if not valid_deadlines.size:
        return sequence.copy()
    deadline = float(rng.choice(valid_deadlines))
    positions = np.flatnonzero(data.deadline_seconds[sequence] == deadline)
    if positions.size < 2:
        return sequence.copy()
    anchor = int(rng.choice(positions))
    coil = data.coil_sizes[sequence[anchor]]
    start = anchor
    while (
        start > 0
        and data.deadline_seconds[sequence[start - 1]] == deadline
        and data.coil_sizes[sequence[start - 1]] == coil
    ):
        start -= 1
    end = anchor
    while (
        end < len(sequence) - 1
        and data.deadline_seconds[sequence[end + 1]] == deadline
        and data.coil_sizes[sequence[end + 1]] == coil
    ):
        end += 1
    block = sequence[start : end + 1]
    remaining = np.concatenate((sequence[:start], sequence[end + 1 :]))
    targets = np.flatnonzero(data.deadline_seconds[remaining] == deadline)
    if not targets.size:
        return sequence.copy()
    same_coil = targets[data.coil_sizes[remaining[targets]] == coil]
    target = int(rng.choice(same_coil if same_coil.size else targets))
    insert_at = target + int(rng.random() >= 0.5)
    return np.insert(remaining, insert_at, block)

