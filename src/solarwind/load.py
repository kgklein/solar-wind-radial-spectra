"""Mission-dispatching public data loader."""

from .cache import configure_pyspedas_cache
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
    if canonical_mission is None:
        raise NotImplementedError(f"Mission '{normalized_mission}' is not implemented yet.")

    configure_pyspedas_cache()

    # Keep the optional, slow PySPEDAS import out of lightweight dispatch tests.
    if canonical_mission == "psp":
        from .psp import load_psp_timeseries

        return load_psp_timeseries(start=start, stop=stop)
    if canonical_mission == "solo":
        from .solo import load_solo_timeseries

        return load_solo_timeseries(start=start, stop=stop)

    raise AssertionError(f"Unhandled canonical mission '{canonical_mission}'.")
