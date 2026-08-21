"""Mission-dispatching public data loader."""

from .data import SolarWindData


def load_timeseries(mission: str, start: str, stop: str) -> SolarWindData:
    """Load native-cadence time series for a supported mission."""

    normalized_mission = mission.strip().lower()
    if normalized_mission != "psp":
        raise NotImplementedError(f"Mission '{normalized_mission}' is not implemented yet.")

    # Keep the optional, slow PySPEDAS import out of lightweight dispatch tests.
    from .psp import load_psp_timeseries

    return load_psp_timeseries(start=start, stop=stop)
