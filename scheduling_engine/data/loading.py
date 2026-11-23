"""Portable CSV interface for the integrated production model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


CANONICAL_COLUMNS = (
    "Unique Identifier",
    "Coil Size",
    "Machine Work Time",
    "Screws",
    "Studs",
    "Production Deadline",
)

_ALIASES = {
    "uniqueidentifier": "Unique Identifier",
    "productid": "Unique Identifier",
    "assemblyref": "Unique Identifier",
    "coilsize": "Coil Size",
    "coil": "Coil Size",
    "machineworktime": "Machine Work Time",
    "howickworktime": "Machine Work Time",
    "worktime": "Machine Work Time",
    "screws": "Screws",
    "dimplecount": "Screws",
    "studs": "Studs",
    "productiondeadline": "Production Deadline",
    "deadline": "Production Deadline",
}


@dataclass(frozen=True)
class ProductData:
    """Validated, process-safe arrays used by the simulation."""

    product_ids: tuple[str, ...]
    coil_sizes: np.ndarray
    machine_work_seconds: np.ndarray
    screws: np.ndarray
    studs: np.ndarray
    deadlines: tuple[pd.Timestamp, ...]
    deadline_seconds: np.ndarray
    start_date: pd.Timestamp

    @property
    def number_of_products(self) -> int:
        return len(self.product_ids)


def _normalise_header(value: object) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def _canonicalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renames: dict[object, str] = {}
    seen: set[str] = set()
    for column in frame.columns:
        canonical = _ALIASES.get(_normalise_header(column))
        if canonical is not None:
            if canonical in seen:
                raise ValueError(f"Multiple CSV columns map to {canonical!r}")
            renames[column] = canonical
            seen.add(canonical)
    return frame.rename(columns=renames)


def _parse_deadlines(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.strip()
    parsed = pd.Series(pd.NaT, index=text.index, dtype="datetime64[ns]")
    for date_format in (
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%d-%b-%y",
        "%d/%m/%Y",
        "%d/%m/%Y %H:%M:%S",
    ):
        missing = parsed.isna()
        if not missing.any():
            break
        parsed.loc[missing] = pd.to_datetime(
            text.loc[missing], format=date_format, errors="coerce"
        )
    return parsed


def load_products(
    csv_path: str | Path,
    *,
    start_date: str | pd.Timestamp | None = None,
    start_offset_days: int = 7,
    machine_work_time_unit: str = "seconds",
    machine_work_time_column: str | None = None,
    trolley_capacity: int = 6,
) -> ProductData:
    """Load and validate products.

    Full trolleys contain ``trolley_capacity`` products. A final partial
    trolley is valid and is transferred after the last product is machined.
    """

    path = Path(csv_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Product CSV not found: {path}")
    raw = pd.read_csv(path)
    if machine_work_time_column is not None and machine_work_time_column not in raw.columns:
        raise ValueError(f"Machine time column not found: {machine_work_time_column!r}")
    selected_machine_work = raw[machine_work_time_column].copy() if machine_work_time_column else None
    frame = _canonicalise_columns(raw)
    if selected_machine_work is not None:
        frame["Machine Work Time"] = selected_machine_work
    missing = [column for column in CANONICAL_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError("Input CSV is missing required columns: " + ", ".join(missing))
    if frame.empty:
        raise ValueError("Input CSV contains no products")
    if trolley_capacity <= 0:
        raise ValueError("trolley_capacity must be positive")
    identifiers = frame["Unique Identifier"].astype("string").str.strip()
    coil_sizes = pd.to_numeric(frame["Coil Size"], errors="coerce")
    machine_work = pd.to_numeric(frame["Machine Work Time"], errors="coerce")
    screws = pd.to_numeric(frame["Screws"], errors="coerce")
    studs = pd.to_numeric(frame["Studs"], errors="coerce")
    deadlines = _parse_deadlines(frame["Production Deadline"])

    invalid = (
        identifiers.isna()
        | identifiers.eq("")
        | coil_sizes.isna()
        | machine_work.isna()
        | screws.isna()
        | studs.isna()
        | deadlines.isna()
    )
    if invalid.any():
        rows = (frame.index[invalid] + 2).tolist()
        raise ValueError(f"Invalid or missing values on CSV row(s): {rows[:10]}")
    for label, values in (("Coil Size", coil_sizes), ("Screws", screws), ("Studs", studs)):
        if (values < 0).any() or not np.allclose(values, np.round(values)):
            rows = (frame.index[(values < 0) | (values % 1 != 0)] + 2).tolist()
            raise ValueError(f"{label} must contain non-negative integers; CSV row(s): {rows[:10]}")
    if ((machine_work <= 0) | ~np.isfinite(machine_work)).any():
        rows = (frame.index[(machine_work <= 0) | ~np.isfinite(machine_work)] + 2).tolist()
        raise ValueError(f"Machine Work Time must be positive; CSV row(s): {rows[:10]}")

    unit_factors = {"seconds": 1.0, "minutes": 60.0, "hours": 3600.0}
    if machine_work_time_unit not in unit_factors:
        raise ValueError("machine_work_time_unit must be seconds, minutes, or hours")
    machine_work = machine_work.astype(float) * unit_factors[machine_work_time_unit]

    if start_offset_days < 0:
        raise ValueError("start_offset_days cannot be negative")
    production_start = (
        deadlines.min() - pd.Timedelta(days=start_offset_days)
        if start_date is None
        else pd.Timestamp(start_date)
    )
    production_start = pd.Timestamp(production_start).normalize()
    deadline_seconds = (deadlines - production_start).dt.total_seconds().to_numpy(float)
    if (deadline_seconds < 0).any():
        raise ValueError("Production start date cannot be after a product deadline")

    return ProductData(
        product_ids=tuple(identifiers.astype(str)),
        coil_sizes=coil_sizes.to_numpy(dtype=int),
        machine_work_seconds=machine_work.to_numpy(dtype=float),
        screws=screws.to_numpy(dtype=int),
        studs=studs.to_numpy(dtype=int),
        deadlines=tuple(pd.Timestamp(value) for value in deadlines),
        deadline_seconds=deadline_seconds,
        start_date=production_start,
    )
