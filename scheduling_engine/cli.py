"""Command-line settings for reproducible SA experiments."""
from __future__ import annotations
import argparse
from pathlib import Path
from scheduling_engine.config import AnnealingConfig, JigConfig, MachineConfig, ProblemConfig, TrolleyConfig
from scheduling_engine.paths import INPUT_CSV, OUTPUT_DIRECTORY


def build_parser():
    p = argparse.ArgumentParser(description="Three-objective SA with detailed Pareto schedule exports")
    p.add_argument("--data", type=Path, default=INPUT_CSV,
                   metavar="INPUT_CSV", help="your product CSV path; replace the configured placeholder")
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIRECTORY,
                   metavar="OUTPUT_DIRECTORY", help="your results root; a unique run subfolder is created")
    p.add_argument("--run-id", help="optional run label; an existing folder is never overwritten")
    p.add_argument("--move-config", type=int, choices=(1, 2, 3), default=3,
                   help="1: R100/O0/C0; 2: R70/O30/C0; 3: R70/O15/C15. Machine/jig always 50/50")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--calibration-seed", type=int, default=20261003,
                   help="common calibration seed, distinct from evaluation seeds")
    p.add_argument("--start-date", help="production start date (YYYY-MM-DD); otherwise inferred from deadlines")
    p.add_argument("--start-offset-days", type=int, default=7)
    p.add_argument("--machine-work-time-unit", choices=("seconds", "minutes", "hours"), default="seconds")
    p.add_argument("--machine-work-time-column", help="exact input column to use instead of the nominal machine-time column")
    p.add_argument("--calibration-file", type=Path,
                   help="reuse this calibration's temperature and objective scales explicitly across scenarios")
    p.add_argument("--experiment-metadata", type=Path,
                   help="JSON scenario manifest copied into the run and checked against the input SHA-256")
    machine = p.add_argument_group("machine and trolleys")
    machine.add_argument("--machine-daily-hours", type=float, default=6)
    machine.add_argument("--machine-setup-hours", type=float, default=1)
    machine.add_argument("--machine-operators", type=int, default=1)
    machine.add_argument("--machine-operator-hourly-cost", type=float, default=20)
    machine.add_argument("--trolley-capacity", type=int, default=6)
    machine.add_argument("--trolley-transfer-seconds", type=float, default=120)
    machine.add_argument("--trolley-count", type=int, default=30)
    machine.add_argument("--loaded-arrival-seconds", type=float, default=90)
    machine.add_argument("--empty-return-seconds", type=float, default=84)
    machine.add_argument("--empty-arrival-seconds", type=float, default=54)
    jig = p.add_argument_group("jigs; initial plan is fixed at two jigs with two workers each")
    jig.add_argument("--jig-daily-hours", type=float, default=8)
    jig.add_argument("--number-of-jigs", type=int, default=4)
    jig.add_argument("--max-workers-per-jig", type=int, default=4)
    jig.add_argument("--worker-efficiency-exponent", type=float, default=0.9)
    jig.add_argument("--worker-hourly-cost", type=float, default=20)
    jig.add_argument("--assembly-seconds-per-screw", type=float, default=60)
    jig.add_argument("--handling-seconds-per-stud", type=float, default=10)
    jig.add_argument("--horizon-buffer-working-days", type=int, default=20)
    search = p.add_argument_group("SA and early stopping")
    search.add_argument("--temperature-steps", type=int, default=200)
    search.add_argument("--neighbours-per-step", type=int, default=17960)
    search.add_argument("--cooling-rate", type=float, default=0.988)
    search.add_argument("--initial-temperature", type=float)
    search.add_argument("--acceptance-probability", type=float, default=0.8)
    search.add_argument("--temperature-samples", type=int, default=100)
    search.add_argument("--workers", type=int, default=1, help="parallel processes; 0 uses all logical CPUs")
    search.add_argument("--batch-multiplier", type=int, default=8,
                        help="batch size = effective workers times this value; keep fixed in comparisons")
    search.add_argument("--archive-limit", type=int, default=500)
    search.add_argument("--early-stopping-min-steps", type=int, default=60)
    search.add_argument("--early-stopping-patience", type=int, default=10, help="0 disables early stopping")
    search.add_argument("--early-stopping-hypervolume-relative-gain", type=float, default=0.015)
    search.add_argument("--early-stopping-hypervolume-reference-margin", type=float, default=1)
    search.add_argument("--early-stopping-tardiness-seconds", type=float, default=600)
    search.add_argument("--early-stopping-setup-seconds", type=float, default=3600)
    search.add_argument("--early-stopping-labour-cost", type=float, default=120)
    hybrid = p.add_argument_group("optional in-loop MILP")
    hybrid.add_argument("--milp-mode", choices=("none", "machine", "jigs", "both"), default="none")
    hybrid.add_argument("--milp-steps", type=int, nargs="+", default=(25, 45, 60))
    hybrid.add_argument("--milp-setup-min-seconds", type=float, default=3600,
                        help="inclusive minimum setup duration for archive target selection")
    hybrid.add_argument("--milp-setup-max-seconds", type=float, default=14400,
                        help="inclusive maximum setup duration; each MILP preserves its target's setup")
    hybrid.add_argument("--milp-labour-min", type=float, default=39000,
                        help="inclusive minimum labour cost for archive target selection")
    hybrid.add_argument("--milp-labour-cap", type=float, default=42000,
                        help="inclusive maximum target labour cost and candidate cost cap")
    hybrid.add_argument("--milp-machine-call-seconds", type=float, default=30)
    hybrid.add_argument("--milp-jig-call-seconds", type=float, default=8)
    hybrid.add_argument("--milp-machine-window-size", type=int, default=24)
    hybrid.add_argument("--milp-machine-stride", type=int, default=6)
    hybrid.add_argument("--milp-max-displacement", type=int, default=12)
    hybrid.add_argument("--milp-jig-window-weekdays", type=int, default=3)
    hybrid.add_argument("--milp-max-changed-cells", type=int, default=3)
    hybrid.add_argument("--max-search-seconds", type=float,
                        help="search walltime cap including MILP; initialization and exports are separate")
    return p


def problem_from_args(a):
    if a.number_of_jigs < 2 or a.max_workers_per_jig < 2:
        raise ValueError("The fixed initial plan requires at least two jigs and two workers per jig")
    return ProblemConfig(
        machine=MachineConfig(a.machine_daily_hours * 3600, a.machine_setup_hours * 3600,
                              a.machine_operators, a.machine_operator_hourly_cost),
        jig=JigConfig(a.jig_daily_hours * 3600, a.number_of_jigs, a.max_workers_per_jig,
                      a.worker_efficiency_exponent, a.worker_hourly_cost,
                      a.assembly_seconds_per_screw, a.handling_seconds_per_stud),
        trolley=TrolleyConfig(a.trolley_capacity, a.trolley_transfer_seconds,
                              a.trolley_count, a.loaded_arrival_seconds,
                              a.empty_return_seconds, a.empty_arrival_seconds),
    )


def annealing_from_args(a):
    return AnnealingConfig(**{
        field: getattr(a, field) for field in AnnealingConfig.__dataclass_fields__
        if field != "initial_acceptance_probability"
    }, initial_acceptance_probability=a.acceptance_probability)
