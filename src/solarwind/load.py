"""Mission-dispatching public data loader."""

from .data import SolarWindData


def load_timeseries(mission: str, start: str, stop: str) -> SolarWindData:
    """Load native-cadence time series for a supported mission."""

    normalized_mission = " ".join(mission.strip().lower().replace("-", " ").split())
    mission_aliases = {
        "psp": "psp",
        "solo": "solo",
        "solar orbiter": "solo",
    }
    canonical_mission = mission_aliases.get(normalized_mission)

    # Keep the optional, slow PySPEDAS import out of lightweight dispatch tests.
    if canonical_mission == "psp":
        from .psp import load_psp_timeseries

        return load_psp_timeseries(start=start, stop=stop)
    if canonical_mission == "solo":
        from .solo import load_solo_timeseries

        return load_solo_timeseries(start=start, stop=stop)
    raise NotImplementedError(f"Mission '{normalized_mission}' is not implemented yet.")
