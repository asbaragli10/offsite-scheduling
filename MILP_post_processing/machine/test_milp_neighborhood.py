"""Focused checks for the machine MILP's sequencing and setup invariants."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from scheduling_engine.data import ProductData
from scheduling_engine.model import Solution

from .milp_neighborhood import Window, rank_windows, solve_window


class MachineNeighborhoodTests(unittest.TestCase):
    def test_window_never_crosses_a_coil_transition(self) -> None:
        start = pd.Timestamp("2025-05-05")
        data = ProductData(("a", "b", "c", "d"),
                           np.array([100, 100, 150, 150]),
                           np.full(4, 3600.0), np.zeros(4, dtype=int),
                           np.zeros(4, dtype=int), (start,) * 4,
                           np.full(4, 3600.0), start)
        solution = Solution(np.arange(4), np.ones((1, 1), dtype=int))
        evaluation = SimpleNamespace(tardiness_seconds=np.array([3., 2., 1., 0.]))
        windows = rank_windows(solution, evaluation, data, size=2, stride=1)
        self.assertEqual({item.start for item in windows}, {0, 2})

    def test_milp_moves_an_urgent_job_forward(self) -> None:
        start = pd.Timestamp("2025-05-05")
        data = ProductData(("flex_a", "flex_b", "urgent"),
                           np.full(3, 100), np.full(3, 3600.0),
                           np.zeros(3, dtype=int), np.zeros(3, dtype=int),
                           (start,) * 3, np.array([360000., 360000., 3600.]), start)
        solution = Solution(np.arange(3), np.ones((1, 1), dtype=int))
        evaluation = SimpleNamespace(
            jig_completion_seconds=np.array([3600., 7200., 10800.]),
            tardiness_seconds=np.array([0., 0., 7200.]),
        )
        proposal = solve_window(solution, evaluation, data, Window(0, 3, 100, 7200.),
                                scale=1.0, max_displacement=2, solver_time_limit=5.0)
        self.assertIsNotNone(proposal.segment)
        self.assertCountEqual(proposal.segment.tolist(), [0, 1, 2])
        self.assertEqual(int(proposal.segment[0]), 2)
        self.assertLess(proposal.proxy_objective_hours, proposal.proxy_baseline_hours)


if __name__ == "__main__":
    unittest.main()
