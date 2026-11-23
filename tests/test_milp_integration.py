"""Archive eligibility and exact setup preservation at hybrid checkpoints."""
from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scheduling_engine.cli import build_parser
from scheduling_engine.model import Solution
from scheduling_engine.optimisation.archive import ParetoEntry
from scheduling_engine.optimisation.milp_integration import MilpConfig, run_checkpoint, select_target


def entry(tardiness=100., setup=10800., labour=41000.):
    solution = Solution(np.arange(4), np.full((3, 1), 2, dtype=int))
    return ParetoEntry(solution, np.asarray([tardiness, setup, labour]))


def evaluation(objectives):
    return SimpleNamespace(objectives=np.asarray(objectives, dtype=float),
                           jig_by_product=np.zeros(4, dtype=int),
                           jig_order_by_station=(np.arange(4),),
                           jig_start_seconds=np.arange(4, dtype=float))


class TargetSelectionTests(unittest.TestCase):
    def test_inclusive_bounds_and_out_of_range_exclusion(self):
        config = MilpConfig()
        outside = [entry(1., 3599., 40000.), entry(1., 14401., 40000.),
                   entry(1., 7200., 38999.), entry(1., 7200., 42001.)]
        self.assertIsNone(select_target(outside, config))
        for setup in (3600., 7200., 10800., 14400.):
            for labour in (39000., 42000.):
                with self.subTest(setup=setup, labour=labour):
                    target = entry(setup=setup, labour=labour)
                    self.assertIs(select_target(outside + [target], config), target)

    def test_lowest_tardiness_then_labour_then_setup(self):
        candidates = [entry(110., 3600., 39000.), entry(100., 14400., 40000.),
                      entry(100., 7200., 41000.), entry(100., 10800., 40000.)]
        self.assertIs(select_target(candidates, MilpConfig()), candidates[3])

    def test_invalid_selection_bounds(self):
        for values in ({'setup_min_seconds': -1.}, {'setup_max_seconds': 3599.},
                       {'labour_min': -1.}, {'labour_min': 42001.},
                       {'setup_min_seconds': float('nan')},
                       {'setup_max_seconds': float('inf')},
                       {'labour_min': float('nan')}, {'labour_cap': float('inf')}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                MilpConfig(**values)

    def test_cli_defaults_and_custom_bounds(self):
        parser = build_parser()
        args = parser.parse_args([])
        self.assertEqual((args.milp_setup_min_seconds, args.milp_setup_max_seconds,
                          args.milp_labour_min, args.milp_labour_cap),
                         (3600., 14400., 39000., 42000.))
        args = parser.parse_args(['--milp-setup-min-seconds', '7200',
                                  '--milp-setup-max-seconds', '10800',
                                  '--milp-labour-min', '40000', '--milp-labour-cap', '41000'])
        config = MilpConfig(setup_min_seconds=args.milp_setup_min_seconds,
                            setup_max_seconds=args.milp_setup_max_seconds,
                            labour_min=args.milp_labour_min, labour_cap=args.milp_labour_cap)
        self.assertIsNone(select_target([entry(setup=3600.)], config))
        self.assertIsNotNone(select_target([entry(setup=7200.)], config))


class CheckpointTests(unittest.TestCase):
    def checkpoint(self, mode, step, target, candidate_objectives):
        """Exercise real selection/acceptance while controlling solver proposals and replay."""
        common = dict(status=0, message='optimal', model_rows=1, model_columns=1,
                      build_seconds=0., solver_seconds=0., relative_gap=0., node_count=0)
        machine = SimpleNamespace(**common, segment=np.array([1, 0, 2, 3]),
                                  binary_variables=1, proxy_objective_hours=0.,
                                  proxy_baseline_hours=1., dual_bound_hours=0.)
        jigs = SimpleNamespace(**common, plan=np.ones((3, 1), dtype=int),
                               proxy_tardiness_delta=-10., proxy_labour_delta=0.,
                               dual_bound=0., sensitivities=())
        data = SimpleNamespace(product_ids=('a', 'b', 'c', 'd'))
        problem = SimpleNamespace(machine=SimpleNamespace(daily_work_seconds=21600.),
                                  jig=SimpleNamespace(daily_work_seconds=28800.))
        replays = [evaluation(target.objectives)] + [evaluation(o) for o in candidate_objectives]
        with ExitStack() as stack:
            replay = stack.enter_context(patch(
                'scheduling_engine.optimisation.milp_integration.evaluate_solution', side_effect=replays))
            stack.enter_context(patch('MILP_post_processing.machine.milp_neighborhood.rank_windows',
                return_value=[SimpleNamespace(start=0, size=4, coil=100, tardiness_seconds=100.)]))
            stack.enter_context(patch('MILP_post_processing.machine.milp_neighborhood.solve_window',
                                      return_value=machine))
            stack.enter_context(patch('MILP_post_processing.jigs.milp_neighborhood.rank_windows',
                return_value=[SimpleNamespace(days=(0, 1, 2), score=1.)]))
            jig_solve = stack.enter_context(patch(
                'MILP_post_processing.jigs.milp_neighborhood.solve_window', return_value=jigs))
            result = run_checkpoint(step, [target], data, problem, MilpConfig(mode=mode))
        self.assertEqual(replay.call_count, len(replays))
        return result, jig_solve

    def test_preserves_selected_setup_in_all_modes_and_jig_stages(self):
        for setup in (3600., 7200., 10800., 14400.):
            for mode, step in (('machine', 25), ('jigs', 25), ('jigs', 60),
                               ('both', 25), ('both', 60)):
                with self.subTest(setup=setup, mode=mode, step=step):
                    target = entry(setup=setup)
                    objectives = []
                    if mode in ('machine', 'both'):
                        objectives.append([90., setup, 41000.])
                    if mode in ('jigs', 'both'):
                        objectives.append([90., setup, 40000.] if step == 60
                                          else [80., setup, 41000.])
                    (archive, incumbent, final, attempts), jig_solve = self.checkpoint(
                        mode, step, target, objectives)
                    self.assertIsNotNone(incumbent)
                    np.testing.assert_array_equal(final.objectives, objectives[-1])
                    self.assertTrue(all(a.record['decision'] == 'accepted' for a in attempts))
                    self.assertTrue(all(a.record['setup_target_seconds'] == setup for a in attempts))
                    self.assertEqual(archive[-1].objectives[1], setup)
                    json.dumps([a.record for a in attempts])
                    if mode == 'both':
                        np.testing.assert_array_equal(jig_solve.call_args.args[0].machine_sequence,
                                                      [1, 0, 2, 3])
                        self.assertEqual(attempts[1].record['parent_origin'],
                                         'accepted_machine_proposal_at_same_checkpoint')

    def test_rejects_changed_setup_even_when_still_within_selection_range(self):
        for mode, step in (('machine', 25), ('jigs', 25), ('jigs', 60)):
            with self.subTest(mode=mode, step=step):
                target = entry(setup=10800.)
                (archive, incumbent, _, attempts), _ = self.checkpoint(
                    mode, step, target, [[90., 7200., 40000.]])
                self.assertIsNone(incumbent)
                self.assertIs(archive[0], target)
                self.assertEqual(attempts[0].record['reason'], 'selected schedule setup changed')

    def test_retains_non_increasing_labour_acceptance_rule(self):
        for mode in ('machine', 'jigs'):
            with self.subTest(mode=mode):
                (_, incumbent, _, attempts), _ = self.checkpoint(
                    mode, 25, entry(), [[90., 10800., 41001.]])
                self.assertIsNone(incumbent)
                self.assertEqual(attempts[0].record['reason'], 'labour budget increased')

    def test_skip_records_bounds_without_a_fixed_setup_target(self):
        target = entry(setup=18000.)
        archive, incumbent, _, attempts = run_checkpoint(
            25, [target], None, None, MilpConfig(mode='both'))
        self.assertIs(archive[0], target)
        self.assertIsNone(incumbent)
        record = attempts[0].record
        self.assertEqual(record['decision'], 'skipped')
        self.assertEqual((record['setup_min_seconds'], record['setup_max_seconds'],
                          record['labour_min'], record['labour_cap']),
                         (3600., 14400., 39000., 42000.))
        self.assertNotIn('setup_target_seconds', record)



if __name__ == "__main__":
    unittest.main()
