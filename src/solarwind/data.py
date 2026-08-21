"""Transparent containers for mission time series."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr


@dataclass(frozen=True)
class SolarWindData:
    """Magnetic-field and proton data on their independent time coordinates."""

    mag: xr.Dataset
    protons: xr.Dataset


def notplot_values(variable: Mapping[str, Any], name: str) -> np.ndarray:
    """Extract a PySPEDAS notplot numerical array with a useful error."""

    if "y" not in variable:
        raise ValueError(f"PySPEDAS variable '{name}' has no numerical 'y' array.")
    return np.asarray(variable["y"])


def notplot_times(variable: Mapping[str, Any], name: str) -> np.ndarray:
    """Normalize current datetime64 or legacy Unix-second PySPEDAS times."""

    if "x" not in variable:
        raise ValueError(f"PySPEDAS variable '{name}' has no time 'x' array.")
    raw_times = np.asarray(variable["x"])
    if np.issubdtype(raw_times.dtype, np.datetime64):
        return raw_times.astype("datetime64[ns]")
    seconds = raw_times.astype(np.float64)
    nanoseconds = np.rint(seconds * 1_000_000_000).astype(np.int64)
    return nanoseconds.astype("datetime64[ns]")


def clean_measurements(values: Any) -> np.ndarray:
    """Represent CDF fill values as NaN without applying science-quality cuts."""

    output = np.asarray(values, dtype=np.float64).copy()
    output[~np.isfinite(output) | (np.abs(output) >= 1.0e30)] = np.nan
    return output


def clean_quality_flags(values: Any, fill_value: int) -> np.ndarray:
    """Keep flags as numbers while representing their explicit CDF fill as NaN."""

    output = np.asarray(values, dtype=np.float64).copy()
    output[output == fill_value] = np.nan
    return output


def validate_vector_components(values: np.ndarray, name: str) -> None:
    """Require a native time-by-three vector array."""

    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(
            f"PySPEDAS variable '{name}' must have shape (time, 3); got {values.shape}."
        )


def required_variable(raw: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    """Return a required loader variable or identify what was actually returned."""

    try:
        return raw[name]
    except KeyError as error:
        available = ", ".join(sorted(raw)) or "none"
        raise RuntimeError(
            f"PySPEDAS did not return required variable '{name}'. Available: {available}."
        ) from error


def clip_dataset(dataset: xr.Dataset, start: str, stop: str) -> xr.Dataset:
    """Enforce a half-open interval even when source CDFs span beyond it."""

    start_time = np.datetime64(start, "ns")
    stop_time = np.datetime64(stop, "ns")
    if stop_time <= start_time:
        raise ValueError("stop must be later than start.")
    clipped = dataset.sel(time=(dataset.time >= start_time) & (dataset.time < stop_time))
    if "quality_time" in clipped.coords:
        clipped = clipped.sel(
            quality_time=(clipped.quality_time >= start_time)
            & (clipped.quality_time < stop_time)
        )
    return clipped
