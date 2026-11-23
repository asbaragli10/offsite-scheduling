"""Experiment presets, unique run folders and shared scenario calibration."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
import re
import tempfile
import numpy as np

from scheduling_engine.config import NeighbourhoodConfig
from scheduling_engine.model import OBJECTIVE_NAMES
from scheduling_engine.optimisation import estimate_initial_temperature


MODEL_VERSION = "sa-3obj-finite-trolleys-v3"
# Percentages conditional on selecting the jig family (always probability 0.5).
MOVE_PRESETS = {1: (100, 0, 0), 2: (70, 30, 0), 3: (70, 15, 15)}


def neighbourhood_for_preset(preset: int) -> NeighbourhoodConfig:
    r, o, c = MOVE_PRESETS[preset]
    return NeighbourhoodConfig(
        machine_probability=0.5, workforce_probability=0.5,
        workforce_redistribute_weight=r / 100,
        workforce_open_weight=o / 100, workforce_close_weight=c / 100,
    )


def create_run_directory(root: Path, preset: int, seed: int, run_id: str | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    r, o, c = MOVE_PRESETS[preset]
    if run_id is not None:
        if not re.fullmatch(r"[A-Za-z0-9-]+", str(run_id)):
            raise ValueError("run-id must contain letters, numbers or hyphens")
        path = root / f"run{run_id}_R{r}_O{o}_C{c}_seed{seed}"
        path.mkdir(exist_ok=False)
        return path
    numbers = [int(match.group(1)) for path in root.iterdir()
               if (match := re.match(r"run(\d+)_", path.name))]
    number = max(numbers, default=0) + 1
    while True:
        path = root / f"run{number}_R{r}_O{o}_C{c}_seed{seed}"
        try:
            path.mkdir(exist_ok=False)
            return path
        except FileExistsError:
            number += 1


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixed_calibration(path, data, problem, *, input_path):
    """Explicitly transfer nominal calibration, retaining donor and recipient identities."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    temperature = float(payload["temperature"])
    scales = np.asarray(payload["objective_scales"], dtype=float)
    if not np.isfinite(temperature) or temperature <= 0 or scales.shape != (3,) or not np.all(np.isfinite(scales) & (scales > 0)):
        raise ValueError("Fixed calibration requires positive finite temperature and three objective scales")
    if payload.get("signature", {}).get("objective_order") != list(OBJECTIVE_NAMES):
        raise ValueError("Fixed calibration has incompatible objective order")
    signature = {
        "policy": "fixed_nominal", "donor_sha256": sha256_file(Path(path)),
        "input_sha256": sha256_file(Path(input_path)), "problem": asdict(problem),
        "production_start": data.start_date.isoformat(),
        "machine_work_seconds_sha256": hashlib.sha256(data.machine_work_seconds.tobytes()).hexdigest(),
        "objective_order": list(OBJECTIVE_NAMES),
    }
    return {"scenario_id": hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest(),
            "signature": signature, "policy": "fixed_nominal", "donor_calibration": payload,
            "temperature": temperature, "objective_scales": scales.tolist()}


def shared_calibration(
    root, solution, evaluation, data, problem, *, input_path, samples,
    calibration_seed, acceptance_probability, initial_temperature=None,
):
    """Reuse one deterministic calibration across ALL evaluation seeds/presets.

    The calibration neighbourhood is always preset 3. The requested run's
    seed and move mix deliberately do not enter the fingerprint.
    """
    signature = {
        "model_version": MODEL_VERSION, "input_sha256": sha256_file(Path(input_path)),
        "production_start": data.start_date.isoformat(), "problem": asdict(problem),
        "machine_work_seconds_sha256": hashlib.sha256(data.machine_work_seconds.tobytes()).hexdigest(),
        "initial_sequence": solution.machine_sequence.tolist(),
        "initial_staffing": solution.staffing_plan.tolist(),
        "samples": samples, "calibration_seed": calibration_seed,
        "acceptance_probability": acceptance_probability, "initial_temperature_override": initial_temperature,
        "objective_order": list(OBJECTIVE_NAMES),
    }
    package = Path(__file__).resolve().parent
    core_files = [package / "config.py", package / "optimisation" / "temperature.py"]
    core_files += sorted((package / "model").glob("*.py")) + sorted((package / "neighbourhoods").glob("*.py"))
    signature["calibration_code_sha256"] = hashlib.sha256(
        b"".join(path.read_bytes() for path in core_files)).hexdigest()
    encoded = json.dumps(signature, sort_keys=True).encode()
    scenario_id = hashlib.sha256(encoded).hexdigest()
    folder = Path(root) / "calibration"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{scenario_id}.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["signature"] != signature:
            raise ValueError("Calibration signature does not match this scenario")
        return payload
    result = estimate_initial_temperature(
        solution, data, problem, neighbourhood_for_preset(3), samples=samples,
        seed=calibration_seed, acceptance_probability=acceptance_probability,
        initial_evaluation=evaluation,
    )
    payload = {
        "scenario_id": scenario_id, "signature": signature,
        "temperature": result.temperature if initial_temperature is None else initial_temperature,
        "objective_scales": result.objective_scales.tolist(),
        "feasible_samples": result.feasible_samples,
        "median_positive_deterioration": result.median_positive_deterioration,
    }
    # Unique temporary file also makes simultaneous runs safe.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=folder, delete=False) as stream:
        json.dump(payload, stream, indent=2)
        temp = stream.name
    os.replace(temp, path)
    return payload
