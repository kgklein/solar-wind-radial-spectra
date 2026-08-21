"""Load native-cadence solar-wind time series."""

from .data import SolarWindData
from .load import load_timeseries

__all__ = ["SolarWindData", "load_timeseries"]
