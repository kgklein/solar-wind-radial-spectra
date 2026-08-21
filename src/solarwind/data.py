"""Transparent containers for mission time series."""

from dataclasses import dataclass

import xarray as xr


@dataclass(frozen=True)
class SolarWindData:
    """Magnetic-field and proton data on their independent time coordinates."""

    mag: xr.Dataset
    protons: xr.Dataset
