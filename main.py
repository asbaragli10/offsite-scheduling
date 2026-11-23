"""Run one experiment with fixed initialization and complete schedule exports."""
from __future__ import annotations

from datetime import datetime
import json
import importlib.util
import importlib
from importlib.metadata import version as package_version
import os
import sys
from time import perf_counter
import numpy as np
import pandas as pd

from jssp_merit.cli import build_parser, problem_from_args, annealing_from_args
from jssp_merit.data import load_products
from jssp_merit.experiments import MODEL_VERSION, MOVE_PRESETS, create_run_directory, neighbourhood_for_preset, shared_calibration
from jssp_merit.experiments import fixed_calibration, sha256_file
from jssp_merit.model import build_initial_solution, evaluate_solution
from jssp_merit.model.calendar import SECONDS_PER_DAY
from jssp_merit.optimisation import run_parallel_pareto_annealing
from jssp_merit.optimisation.milp_integration import MilpConfig
from jssp_merit.reporting import write_outputs
from jssp_merit.reporting.writers import write_milp_attempts
from jssp_merit.reporting.provenance import snapshot_source, write_manifest
from jssp_merit.paths import configure_io_paths


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    started = perf_counter()
    destination = None
    try:
        configure_io_paths(args)
        problem = problem_from_args(args)
        annealing = annealing_from_args(args)
        milp = MilpConfig(
            mode=args.milp_mode, steps=tuple(args.milp_steps),
            setup_min_seconds=args.milp_setup_min_seconds,
            setup_max_seconds=args.milp_setup_max_seconds,
            labour_min=args.milp_labour_min, labour_cap=args.milp_labour_cap,
            machine_call_seconds=args.milp_machine_call_seconds,
            jig_call_seconds=args.milp_jig_call_seconds,
            machine_window_size=args.milp_machine_window_size,
            machine_stride=args.milp_machine_stride,
            max_displacement=args.milp_max_displacement,
            jig_window_weekdays=args.milp_jig_window_weekdays,
            max_changed_cells=args.milp_max_changed_cells)
        if milp.mode != "none" and importlib.util.find_spec("scipy") is None:
            raise ValueError("Hybrid MILP requires SciPy; install requirements-hybrid.txt")
        if milp.mode != "none":
            try:
                if milp.mode in ("machine", "both"):
                    importlib.import_module("MILP_post_processing.machine.milp_neighborhood")
                if milp.mode in ("jigs", "both"):
                    importlib.import_module("MILP_post_processing.jigs.milp_neighborhood")
            except ImportError as exc:
                raise ValueError(f"Cannot load hybrid MILP dependencies: {exc}") from exc
        if args.seed == args.calibration_seed:
            raise ValueError("Use a calibration seed distinct from the evaluation seed")
        neighbourhood = neighbourhood_for_preset(args.move_config)
        data = load_products(args.data, start_date=args.start_date, start_offset_days=args.start_offset_days,
                             machine_work_time_unit=args.machine_work_time_unit,
                             machine_work_time_column=args.machine_work_time_column,
                             trolley_capacity=problem.trolley.capacity)
        if args.calibration_file and args.initial_temperature is not None:
            raise ValueError("--calibration-file and --initial-temperature cannot be combined")
        experiment = None
        if args.experiment_metadata:
            experiment = json.loads(args.experiment_metadata.read_text(encoding="utf-8"))
            if experiment.get("input_sha256") != sha256_file(args.data):
                raise ValueError("Experiment metadata does not match the selected input CSV")
            for key, value in experiment.get("search_settings", {}).items():
                if getattr(args, key, object()) != value:
                    raise ValueError(f"Experiment metadata disagrees with --{key.replace('_', '-')}")
            if experiment.get("calibration_sha256") and (
                    not args.calibration_file or sha256_file(args.calibration_file) != experiment["calibration_sha256"]):
                raise ValueError("Experiment metadata disagrees with calibration file")
        destination = create_run_directory(args.output_dir, args.move_config, args.seed, args.run_id)
        if experiment is not None:
            (destination / "experiment.json").write_text(json.dumps(experiment, indent=2), encoding="utf-8")
        run_metadata = {
            "model_version": MODEL_VERSION, "run_folder": destination.name,
            "move_config": args.move_config, "jig_percentages_R_O_C": MOVE_PRESETS[args.move_config],
            "seed": args.seed, "effective_workers": args.workers or (os.cpu_count() or 1),
            "milp_mode": args.milp_mode, "max_search_seconds": args.max_search_seconds,
            "milp_solver": "scipy.optimize.milp (HiGHS)" if milp.mode != "none" else None,
            "scipy_version": package_version("scipy") if milp.mode != "none" else None,
            "arguments": {k: str(v) if hasattr(v, "__fspath__") else v for k, v in vars(args).items()},
            "command_arguments": list(sys.argv[1:] if argv is None else argv),
            "experiment": experiment,
            "started_at": datetime.now().astimezone().isoformat(),
        }
        (destination / "run_settings.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
        snapshot_source(destination)
        status_path = destination / "run_status.json"
        status_path.write_text(json.dumps({"status": "running", **run_metadata}, indent=2), encoding="utf-8")
        log_path = destination / "optimisation_progress.txt"

        def log(event, **values):
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"timestamp": datetime.now().astimezone().isoformat(),
                                        "event": event, **values},
                                        default=lambda value: value.tolist()
                                        if isinstance(value, (np.ndarray, np.generic))
                                        else str(value)) + "\n")

        def progress(snapshot):
            log("temperature_step", **snapshot)
            print(f"step {snapshot['temperature_step']}: elapsed={snapshot['elapsed_seconds']:.1f}s "
                  f"T={snapshot['temperature']:.5g} HV_live={snapshot['normalised_hypervolume']:.7g} "
                  f"stopHV_gain={snapshot['hypervolume_relative_gain']:.4g} "
                  f"archive={snapshot['archive_size']} stagnation={snapshot['no_meaningful_progress_steps']}", flush=True)

        def checkpoint(item):
            with (destination / "archive_checkpoints.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(item) + "\n")

        def milp_progress(item):
            log("milp_checkpoint", **item)
            print(f"MILP step {item['temperature_step']} {item.get('side', 'selection')}: "
                  f"{item['decision']} ({item['reason']})", flush=True)

        log("run_start", **run_metadata)
        print(f"Run folder: {destination.resolve()}", flush=True)
        initial = build_initial_solution(data, problem, active_jigs=2, workers_per_active_jig=2,
                                         horizon_buffer_working_days=args.horizon_buffer_working_days)
        initial_evaluation = evaluate_solution(initial, data, problem, collect_segments=True)
        last_day = int(max(initial_evaluation.jig_completion_seconds.max(),
                           initial_evaluation.trolley_worker_return_seconds.max()) // SECONDS_PER_DAY)
        initial.staffing_plan[last_day + 1:] = 0
        initial_evaluation = evaluate_solution(initial, data, problem, collect_segments=True)
        initialization_seconds = perf_counter() - started
        calibration_started = perf_counter()
        print("Loading or computing common scenario calibration...", flush=True)
        calibration = fixed_calibration(args.calibration_file, data, problem, input_path=args.data) if args.calibration_file else shared_calibration(
            args.output_dir, initial, initial_evaluation, data, problem,
            input_path=args.data, samples=args.temperature_samples, calibration_seed=args.calibration_seed,
            acceptance_probability=args.acceptance_probability, initial_temperature=args.initial_temperature)
        calibration_seconds = perf_counter() - calibration_started
        (destination / "calibration.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")
        log("calibration_complete", temperature=calibration["temperature"], elapsed_seconds=calibration_seconds)
        result = run_parallel_pareto_annealing(
            initial, data, problem, neighbourhood, annealing,
            initial_temperature=calibration["temperature"], objective_scales=np.asarray(calibration["objective_scales"]),
            initial_evaluation=initial_evaluation, progress=progress, checkpoint=checkpoint,
            milp_progress=milp_progress,
            milp_config=milp, max_search_seconds=args.max_search_seconds)
        export_started = perf_counter()
        print(f"Search stopped: {result.stop_reason}. Exporting {len(result.archive)} Pareto schedules...", flush=True)
        write_outputs(destination, initial_solution=initial, initial_evaluation=initial_evaluation,
                      result=result, data=data, problem=problem, neighbourhood=neighbourhood,
                      annealing=annealing, input_file=args.data, machine_work_time_unit=args.machine_work_time_unit,
                      machine_work_time_column=args.machine_work_time_column)
        write_milp_attempts(destination, result.milp_attempts, data, problem)
        summary_path = destination / "run_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.update(model_version=MODEL_VERSION, scenario_id=calibration["scenario_id"],
                       run_metadata=run_metadata, milp_mode=args.milp_mode,
                       milp_checkpoint_steps=list(milp.steps),
                       milp_attempts=len(result.milp_attempts),
                       milp_accepted=sum(item.record["decision"] == "accepted"
                                         for item in result.milp_attempts),
                       max_search_seconds=args.max_search_seconds)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        attempted = sum(item["attempts"] for item in result.history)
        feasible = sum(item["feasible_candidates"] for item in result.history)
        noops = sum(item["noop_moves"] for item in result.history)
        hv_area = sum(float(a["hypervolume"]) * (float(b["elapsed_seconds"]) - float(a["elapsed_seconds"]))
                      for a, b in zip(result.checkpoints, result.checkpoints[1:]))
        metrics = {
            "run_folder": destination.name, "move_config": args.move_config, "seed": args.seed,
            "milp_mode": args.milp_mode, "milp_attempts": len(result.milp_attempts),
            "milp_accepted": sum(item.record["decision"] == "accepted"
                                 for item in result.milp_attempts),
            "milp_machine_calls": sum(item.record.get("side") == "machine" and
                                      item.record["decision"] != "skipped"
                                      for item in result.milp_attempts),
            "milp_jig_calls": sum(item.record.get("side") == "jigs" and
                                  item.record["decision"] != "skipped"
                                  for item in result.milp_attempts),
            "milp_machine_accepted": sum(item.record.get("side") == "machine" and
                                          item.record["decision"] == "accepted"
                                          for item in result.milp_attempts),
            "milp_jig_accepted": sum(item.record.get("side") == "jigs" and
                                      item.record["decision"] == "accepted"
                                      for item in result.milp_attempts),
            "milp_call_elapsed_seconds": sum(item.record.get("call_elapsed_seconds", 0.0)
                                             for item in result.milp_attempts),
            "max_search_seconds": args.max_search_seconds,
            "live_reference_hypervolume": result.checkpoints[-1]["hypervolume"],
            "time_averaged_hypervolume_own_duration": hv_area / result.elapsed_seconds,
            "search_elapsed_seconds": result.elapsed_seconds,
            "initialization_seconds": initialization_seconds, "calibration_seconds": calibration_seconds,
            "export_seconds": perf_counter() - export_started,
            "completed_temperature_steps": len(result.history), "stopped_early": result.stopped_early,
            "stop_reason": result.stop_reason, "final_temperature": result.history[-1]["temperature"],
            "pareto_archive_size": len(result.archive), "candidate_attempts": attempted,
            "feasible_candidates": feasible, "infeasible_rate": (attempted - feasible) / attempted,
            "noop_rate": noops / attempted, "feasible_candidates_per_second": feasible / result.elapsed_seconds,
            "accepted_proposals": sum(item["accepted_batches"] for item in result.history),
            "total_elapsed_seconds": perf_counter() - started,
        }
        (destination / "run_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        pd.DataFrame([metrics]).to_csv(destination / "run_metrics.csv", index=False)
        log("run_complete", **metrics)
        status_path.write_text(json.dumps({"status": "complete", **metrics}, indent=2), encoding="utf-8")
        write_manifest(destination)
    except (OSError, ValueError) as exc:
        if destination:
            (destination / "run_status.json").write_text(json.dumps({"status": "failed", "error": str(exc)}, indent=2), encoding="utf-8")
        parser.error(str(exc))
    print(f"Saved complete run: {destination.resolve()}")
    print(f"Search elapsed: {result.elapsed_seconds:.2f}s; "
          f"live-reference HV (diagnostic): {metrics['live_reference_hypervolume']:.7g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
