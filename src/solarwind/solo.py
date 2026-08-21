"""Thin adapter from Solar Orbiter PySPEDAS products to standardized xarray data."""

from typing import Any

import numpy as np
import xarray as xr

from .data import (
    SolarWindData,
    clean_measurements,
    clean_quality_flags,
    clip_dataset,
    notplot_times,
    notplot_values,
    required_variable,
    validate_vector_components,
)

MAG_VARIABLES = {
    "field": "B_RTN",
    "samples_per_second": "VECTOR_TIME_RESOLUTION",
    "vector_range": "VECTOR_RANGE",
    "quality_bitmask": "QUALITY_BITMASK",
    "quality_flag": "QUALITY_FLAG",
}

PAS_VARIABLES = {
    "density": "N",
    "velocity": "V_RTN",
    "temperature": "T",
    "info": "Info",
    "quality_factor": "quality_factor",
}

MAG_QUALITY_FLAG_NOTES = (
    "Official MAG high-level quality flag: 0 bad; 1 known problems/use at your "
    "own risk; 2 survey data/possibly not publication quality; 3 good for "
    "publication subject to PI approval; 4 excellent/specially treated. Values "
    "are retained without filtering."
)
MAG_QUALITY_BITMASK_NOTES = (
    "Official MAG detailed quality bit mask. Values: 1 L2 data from the inboard "
    "sensor; 2 instrument time not synchronized with spacecraft time; 4 MAG "
    "heater operating; 8 interference tones removed; 16 thruster influence "
    "removed; 32 spacecraft interference detected but not removed; 64 solar "
    "array movement; 128 interference from another instrument detected and "
    "removed. Values are retained without filtering."
)
PAS_INFO_NOTES = (
    "Official PAS acquisition information: 0 ground; 1 normal; 2 snapshot; "
    "3 burst; 4 engineering; 5 calibration. Values are retained without mode "
    "selection."
)


def _check_length(values: np.ndarray, sample_count: int, name: str) -> None:
    if values.shape[0] != sample_count:
        raise ValueError(f"Solar Orbiter timestamps and '{name}' samples differ in length.")


def build_magnetic_dataset(
    time: Any,
    components: Any,
    *,
    quality_flag: Any | None = None,
    quality_bitmask: Any | None = None,
    vector_range: Any | None = None,
    samples_per_second: Any | None = None,
) -> xr.Dataset:
    """Build the standardized SolO/MAG dataset from native RTN arrays."""

    field = clean_measurements(components)
    validate_vector_components(field, MAG_VARIABLES["field"])
    time_array = np.asarray(time, dtype="datetime64[ns]")
    sample_count = time_array.size
    _check_length(field, sample_count, MAG_VARIABLES["field"])

    dataset = xr.Dataset(
        data_vars={
            "B_R": ("time", field[:, 0]),
            "B_T": ("time", field[:, 1]),
            "B_N": ("time", field[:, 2]),
            "B_mag": ("time", np.sqrt(np.sum(field**2, axis=1))),
        },
        coords={"time": time_array},
        attrs={
            "mission": "Solar Orbiter",
            "instrument": "MAG",
            "data_level": "L2",
            "coordinate_system": "RTN",
            "datatype": "rtn-normal",
            "product": "solo_L2_mag-rtn-normal",
            "source_variable": MAG_VARIABLES["field"],
        },
    )
    for name in ("B_R", "B_T", "B_N", "B_mag"):
        dataset[name].attrs["units"] = "nT"

    if quality_flag is not None:
        values = clean_quality_flags(quality_flag, fill_value=np.iinfo(np.uint8).max - 1)
        _check_length(values, sample_count, MAG_VARIABLES["quality_flag"])
        dataset["quality_flag"] = ("time", values)
        dataset["quality_flag"].attrs.update(
            units="1",
            source_variable=MAG_VARIABLES["quality_flag"],
            description=MAG_QUALITY_FLAG_NOTES,
        )

    if quality_bitmask is not None:
        values = clean_quality_flags(
            quality_bitmask, fill_value=np.iinfo(np.uint16).max - 1
        )
        _check_length(values, sample_count, MAG_VARIABLES["quality_bitmask"])
        dataset["quality_bitmask"] = ("time", values)
        dataset["quality_bitmask"].attrs.update(
            units="1",
            source_variable=MAG_VARIABLES["quality_bitmask"],
            description=MAG_QUALITY_BITMASK_NOTES,
        )

    if vector_range is not None:
        values = clean_quality_flags(vector_range, fill_value=np.iinfo(np.uint8).max - 1)
        _check_length(values, sample_count, MAG_VARIABLES["vector_range"])
        dataset["vector_range"] = ("time", values)
        dataset["vector_range"].attrs.update(
            units="1",
            source_variable=MAG_VARIABLES["vector_range"],
            description="Official MAG component range setting; retained without filtering.",
        )

    if samples_per_second is not None:
        values = clean_measurements(samples_per_second)
        _check_length(values, sample_count, MAG_VARIABLES["samples_per_second"])
        dataset["samples_per_second"] = ("time", values)
        dataset["samples_per_second"].attrs.update(
            units="s^-1",
            source_units="number of magnetic vectors per second",
            source_variable=MAG_VARIABLES["samples_per_second"],
        )

    return dataset


def build_proton_dataset(
    time: Any,
    velocity: Any,
    density: Any,
    temperature: Any,
    *,
    info: Any | None = None,
    quality_factor: Any | None = None,
) -> xr.Dataset:
    """Build the standardized SolO/SWA-PAS dataset from native moment arrays."""

    velocity_array = clean_measurements(velocity)
    validate_vector_components(velocity_array, PAS_VARIABLES["velocity"])
    density_array = clean_measurements(density)
    temperature_array = clean_measurements(temperature)
    time_array = np.asarray(time, dtype="datetime64[ns]")
    sample_count = time_array.size
    for name, values in (
        (PAS_VARIABLES["velocity"], velocity_array),
        (PAS_VARIABLES["density"], density_array),
        (PAS_VARIABLES["temperature"], temperature_array),
    ):
        _check_length(values, sample_count, name)

    dataset = xr.Dataset(
        data_vars={
            "V_R": ("time", velocity_array[:, 0]),
            "V_T": ("time", velocity_array[:, 1]),
            "V_N": ("time", velocity_array[:, 2]),
            "V_mag": ("time", np.sqrt(np.sum(velocity_array**2, axis=1))),
            "n_p": ("time", density_array),
            "T_p": ("time", temperature_array),
        },
        coords={"time": time_array},
        attrs={
            "mission": "Solar Orbiter",
            "instrument": "SWA/PAS",
            "data_level": "L2",
            "coordinate_system": "RTN",
            "datatype": "pas-grnd-mom",
            "product": "solo_L2_swa-pas-grnd-mom",
            "source_velocity_variable": PAS_VARIABLES["velocity"],
        },
    )
    for name in ("V_R", "V_T", "V_N", "V_mag"):
        dataset[name].attrs["units"] = "km/s"
    dataset["n_p"].attrs.update(units="cm^-3", source_units="particles cm^-3")
    dataset["T_p"].attrs["units"] = "eV"

    if info is not None:
        values = clean_quality_flags(info, fill_value=np.iinfo(np.uint8).max)
        _check_length(values, sample_count, PAS_VARIABLES["info"])
        dataset["info"] = ("time", values)
        dataset["info"].attrs.update(
            units="1",
            source_variable=PAS_VARIABLES["info"],
            description=PAS_INFO_NOTES,
        )

    if quality_factor is not None:
        values = clean_measurements(quality_factor)
        _check_length(values, sample_count, PAS_VARIABLES["quality_factor"])
        dataset["quality_factor"] = ("time", values)
        dataset["quality_factor"].attrs.update(
            units="1",
            source_variable=PAS_VARIABLES["quality_factor"],
            description="Official PAS quality factor; retained without quality cuts.",
        )

    return dataset


def _optional_values(raw: dict[str, Any], name: str) -> np.ndarray | None:
    variable = raw.get(name)
    return notplot_values(variable, name) if variable is not None else None


def load_solo_timeseries(start: str, stop: str) -> SolarWindData:
    """Download SolO MAG and SWA/PAS data and return native-cadence datasets."""

    import pyspedas
    from pyspedas.tplot_tools import cdf_to_tplot

    # The MAG quality variables are CDF metadata, which solo.mag(notplot=True)
    # omits. Download once through the official loader, then use PySPEDAS's CDF
    # extractor with metadata enabled to retain those official quality fields.
    mag_files = pyspedas.projects.solo.mag(
        trange=[start, stop],
        datatype="rtn-normal",
        level="l2",
        downloadonly=True,
    )
    if not mag_files:
        raise RuntimeError(f"No Solar Orbiter MAG rtn-normal data for {start} to {stop}.")
    raw_mag = cdf_to_tplot(
        list(mag_files),
        varnames=list(MAG_VARIABLES.values()),
        get_metadata=True,
        notplot=True,
    )

    raw_protons = pyspedas.projects.solo.swa(
        trange=[start, stop],
        datatype="pas-grnd-mom",
        level="l2",
        varnames=list(PAS_VARIABLES.values()),
        get_support_data=True,
        notplot=True,
        time_clip=True,
    )
    if not raw_mag:
        raise RuntimeError(f"No Solar Orbiter MAG rtn-normal data for {start} to {stop}.")
    if not raw_protons:
        raise RuntimeError(
            f"No Solar Orbiter SWA/PAS ground moments for {start} to {stop}."
        )

    field = required_variable(raw_mag, MAG_VARIABLES["field"])
    mag = build_magnetic_dataset(
        notplot_times(field, MAG_VARIABLES["field"]),
        notplot_values(field, MAG_VARIABLES["field"]),
        quality_flag=_optional_values(raw_mag, MAG_VARIABLES["quality_flag"]),
        quality_bitmask=_optional_values(raw_mag, MAG_VARIABLES["quality_bitmask"]),
        vector_range=_optional_values(raw_mag, MAG_VARIABLES["vector_range"]),
        samples_per_second=_optional_values(
            raw_mag, MAG_VARIABLES["samples_per_second"]
        ),
    )

    velocity = required_variable(raw_protons, PAS_VARIABLES["velocity"])
    protons = build_proton_dataset(
        notplot_times(velocity, PAS_VARIABLES["velocity"]),
        notplot_values(velocity, PAS_VARIABLES["velocity"]),
        notplot_values(
            required_variable(raw_protons, PAS_VARIABLES["density"]),
            PAS_VARIABLES["density"],
        ),
        notplot_values(
            required_variable(raw_protons, PAS_VARIABLES["temperature"]),
            PAS_VARIABLES["temperature"],
        ),
        info=_optional_values(raw_protons, PAS_VARIABLES["info"]),
        quality_factor=_optional_values(raw_protons, PAS_VARIABLES["quality_factor"]),
    )

    return SolarWindData(
        mag=clip_dataset(mag, start=start, stop=stop),
        protons=clip_dataset(protons, start=start, stop=stop),
    )
