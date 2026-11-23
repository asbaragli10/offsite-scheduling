"""Archive progress monitoring independent of the SA transition rule."""
from __future__ import annotations
import numpy as np
from scheduling_engine.config import AnnealingConfig
from .hypervolume import dominated_hypervolume


class ProgressMonitor:
    def __init__(self, initial_values: np.ndarray, objective_scales: np.ndarray, config: AnnealingConfig):
        self.config = config
        self.scales = np.asarray(objective_scales, dtype=float)
        self.hv_milestone_archive = np.asarray(initial_values, dtype=float).copy()
        self.best = initial_values.min(axis=0).copy()
        self.stagnation = 0

    def update(self, step: int, values: np.ndarray) -> dict[str, float | int]:
        """Compare milestone and current archives under the same inclusive reference."""
        values = np.asarray(values, dtype=float)
        new_best = values.min(axis=0)
        improved = new_best <= self.best - np.asarray(self.config.early_stopping_tolerances) + 1e-9
        self.best = np.where(improved, new_best, self.best)

        current_scaled = values / self.scales
        milestone_scaled = self.hv_milestone_archive / self.scales
        reference = np.maximum(current_scaled.max(axis=0), milestone_scaled.max(axis=0))
        reference += self.config.early_stopping_hypervolume_reference_margin
        old_hv = dominated_hypervolume(milestone_scaled, reference)
        new_hv = dominated_hypervolume(current_scaled, reference)
        gain = max(0.0, (new_hv - old_hv) / max(abs(old_hv), 1e-12))
        hv_progress = gain + 1e-12 >= self.config.early_stopping_hypervolume_relative_gain
        progress = bool(np.any(improved) or hv_progress)
        active = step > self.config.early_stopping_min_steps
        if step == self.config.early_stopping_min_steps:
            self.best = new_best.copy()
            self.hv_milestone_archive = values.copy()
            self.stagnation = 0
        elif not active or progress:
            self.stagnation = 0
            if progress:
                self.hv_milestone_archive = values.copy()
        else:
            self.stagnation += 1
        stop = active and self.config.early_stopping_patience > 0 and self.stagnation >= self.config.early_stopping_patience
        return {
            "early_stopping_monitoring_active": int(active), "hypervolume_relative_gain": gain,
            "hypervolume_improved": int(hv_progress), "objective_progress": int(np.any(improved)),
            "meaningful_progress": int(progress), "no_meaningful_progress_steps": self.stagnation,
            "early_stop": int(stop),
        }
