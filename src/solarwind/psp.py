"""Thin adapter from PSP PySPEDAS products to standardized xarray data."""

from collections.abc import Mapping
from typing import Any

import numpy as np
import xarray as xr

from .data import SolarWindData

MAG_VARIABLE = "psp_fld_l2_mag_RTN"
FIELDS_QUALITY_VARIABLE = "psp_fld_l2_quality_flags"

PROTON_VARIABLES = {
    "density": "psp_spi_DENS",
    "velocity": "psp_spi_VEL_RTN_SUN",
    "temperature": "psp_spi_TEMP",
    "sun_distance": "psp_spi_SUN_DIST",
    "quality": "psp_spi_QUALITY_FLAG",
}

# The SPAN-I CDF reports SUN_DIST in km. This is the IAU definition of one au.
AU_IN_KM = 149_597_870.7

FIELDS_QUALITY_NOTES = (
    "Official FIELDS bit mask on its native one-minute quality_time coordinate. "
    "Zero means no flags; values are not used to remove magnetic measurements."
)
SPAN_QUALITY_NOTES = (
    "Official SPAN-I two-byte bit mask. Values are retained for inspection and "
    "are not used to remove proton measurements."
)


def _values(variable: Mapping[str, Any], name: str) -> np.ndarray:
    """Extract a PySPEDAS notplot numerical array with a useful error."""

    if "y" not in variable:
        raise ValueError(f"PySPEDAS variable '{name}' has no numerical 'y' array.")
    return np.asarray(variable["y"])


def _times(variable: Mapping[str, Any], name: str) -> np.ndarray:
    """Normalize current datetime64 or legacy Unix-second PySPEDAS times."""

    if "x" not in variable:
        raise ValueError(f"PySPEDAS variable '{name}' has no time 'x' array.")
    raw_times = np.asarray(variable["x"])
    if np.issubdtype(raw_times.dtype, np.datetime64):
        return raw_times.astype("datetime64[ns]")
    seconds = raw_times.astype(np.float64)
    nanoseconds = np.rint(seconds * 1_000_000_000).astype(np.int64)
    return nanoseconds.astype("datetime64[ns]")


def _measurements(values: Any) -> np.ndarray:
    """Represent CDF fill values as NaN without applying science-quality cuts."""

    output = np.asarray(values, dtype=np.float64).copy()
    output[~np.isfinite(output) | (np.abs(output) >= 1.0e30)] = np.nan
    return output


def _quality_flags(values: Any, fill_value: int) -> np.ndarray:
    """Keep bit masks as numbers while representing their CDF fill as NaN."""

    output = np.asarray(values, dtype=np.float64).copy()
    output[output == fill_value] = np.nan
    return output


def _validate_components(values: np.ndarray, name: str) -> None:
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(
            f"PySPEDAS variable '{name}' must have shape (time, 3); got {values.shape}."
        )


def build_magnetic_dataset(
    time: Any,
    components: Any,
    *,
    quality_time: Any | None = None,
    quality_flags: Any | None = None,
) -> xr.Dataset:
    """Build the standardized magnetic dataset from native RTN arrays."""

    field = _measurements(components)
    _validate_components(field, MAG_VARIABLE)
    time_array = np.asarray(time, dtype="datetime64[ns]")
    if field.shape[0] != time_array.size:
        raise ValueError("Magnetic timestamps and samples have different lengths.")

    dataset = xr.Dataset(
        data_vars={
            "B_R": ("time", field[:, 0]),
            "B_T": ("time", field[:, 1]),
            "B_N": ("time", field[:, 2]),
            "B_mag": ("time", np.sqrt(np.sum(field**2, axis=1))),
        },
        coords={"time": time_array},
        attrs={
            "mission": "Parker Solar Probe",
            "instrument": "FIELDS",
            "data_level": "L2",
            "coordinate_system": "RTN",
            "datatype": "mag_rtn",
            "source_variable": MAG_VARIABLE,
        },
    )
    for name in ("B_R", "B_T", "B_N", "B_mag"):
        dataset[name].attrs["units"] = "nT"

    if quality_time is not None and quality_flags is not None:
        quality_time_array = np.asarray(quality_time, dtype="datetime64[ns]")
        flags = _quality_flags(quality_flags, fill_value=np.iinfo(np.uint32).max)
        if flags.size != quality_time_array.size:
            raise ValueError("FIELDS quality timestamps and flags have different lengths.")
        dataset = dataset.assign_coords(quality_time=quality_time_array)
        dataset["quality_flag"] = ("quality_time", flags)
        dataset["quality_flag"].attrs.update(
            units="1",
            source_variable=FIELDS_QUALITY_VARIABLE,
            description=FIELDS_QUALITY_NOTES,
        )

    return dataset


def build_proton_dataset(
    time: Any,
    velocity: Any,
    density: Any,
    temperature: Any,
    *,
    sun_distance_km: Any | None = None,
    quality_flags: Any | None = None,
) -> xr.Dataset:
    """Build the standardized SPAN-I proton dataset from native arrays."""

    velocity_array = _measurements(velocity)
    _validate_components(velocity_array, PROTON_VARIABLES["velocity"])
    time_array = np.asarray(time, dtype="datetime64[ns]")
    density_array = _measurements(density)
    temperature_array = _measurements(temperature)
    sample_count = time_array.size
    if any(
        values.shape[0] != sample_count
        for values in (velocity_array, density_array, temperature_array)
    ):
        raise ValueError("SPAN-I timestamps and moment arrays have different lengths.")

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
            "mission": "Parker Solar Probe",
            "instrument": "SWEAP/SPAN-I",
            "data_level": "L3",
            "coordinate_system": "RTN, Sun frame",
            "datatype": "sf00_l3_mom",
            "source_velocity_variable": PROTON_VARIABLES["velocity"],
        },
    )
    for name in ("V_R", "V_T", "V_N", "V_mag"):
        dataset[name].attrs["units"] = "km/s"
    dataset["n_p"].attrs["units"] = "cm^-3"
    dataset["T_p"].attrs["units"] = "eV"

    if sun_distance_km is not None:
        distance = _measurements(sun_distance_km)
        if distance.shape[0] != sample_count:
            raise ValueError("SPAN-I timestamps and Sun-distance samples differ in length.")
        dataset["r_au"] = ("time", distance / AU_IN_KM)
        dataset["r_au"].attrs.update(
            units="au",
            source_units="km",
            source_variable=PROTON_VARIABLES["sun_distance"],
            conversion="1 au = 149597870.7 km (IAU definition)",
        )

    if quality_flags is not None:
        flags = _quality_flags(quality_flags, fill_value=np.iinfo(np.uint16).max)
        if flags.shape[0] != sample_count:
            raise ValueError("SPAN-I timestamps and quality flags differ in length.")
        dataset["quality_flag"] = ("time", flags)
        dataset["quality_flag"].attrs.update(
            units="1",
            source_variable=PROTON_VARIABLES["quality"],
            description=SPAN_QUALITY_NOTES,
        )

    return dataset


def _required(raw: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    try:
        return raw[name]
    except KeyError as error:
        available = ", ".join(sorted(raw)) or "none"
        raise RuntimeError(
            f"PySPEDAS did not return required variable '{name}'. Available: {available}."
        ) from error


def _clip(dataset: xr.Dataset, start: str, stop: str) -> xr.Dataset:
    """Enforce a half-open requested interval even if a CDF spans beyond it."""

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


def load_psp_timeseries(start: str, stop: str) -> SolarWindData:
    """Download PSP FIELDS and SPAN-I data and return native-cadence datasets."""

    import pyspedas

    # Restrict CDF variables to the measurements used here. This avoids loading
    # large ancillary SPAN spectrograms and avoids duplicate downloads.
    raw_mag = pyspedas.projects.psp.fields(
        trange=[start, stop],
        datatype="mag_rtn",
        level="l2",
        varnames=[MAG_VARIABLE, FIELDS_QUALITY_VARIABLE],
        get_support_data=True,
        notplot=True,
        time_clip=True,
    )
    raw_protons = pyspedas.projects.psp.spi(
        trange=[start, stop],
        datatype="sf00_l3_mom",
        level="l3",
        varnames=[
            "DENS",
            "VEL_RTN_SUN",
            "TEMP",
            "SUN_DIST",
            "QUALITY_FLAG",
        ],
        get_support_data=True,
        notplot=True,
        time_clip=True,
    )
    if not raw_mag:
        raise RuntimeError(f"No PSP/FIELDS mag_rtn data returned for {start} to {stop}.")
    if not raw_protons:
        raise RuntimeError(f"No PSP/SPAN-I proton moments returned for {start} to {stop}.")

    mag_variable = _required(raw_mag, MAG_VARIABLE)
    mag_quality = raw_mag.get(FIELDS_QUALITY_VARIABLE)
    mag = build_magnetic_dataset(
        _times(mag_variable, MAG_VARIABLE),
        _values(mag_variable, MAG_VARIABLE),
        quality_time=(
            _times(mag_quality, FIELDS_QUALITY_VARIABLE)
            if mag_quality is not None
            else None
        ),
        quality_flags=(
            _values(mag_quality, FIELDS_QUALITY_VARIABLE)
            if mag_quality is not None
            else None
        ),
    )

    velocity = _required(raw_protons, PROTON_VARIABLES["velocity"])
    density = _required(raw_protons, PROTON_VARIABLES["density"])
    temperature = _required(raw_protons, PROTON_VARIABLES["temperature"])
    distance = raw_protons.get(PROTON_VARIABLES["sun_distance"])
    proton_quality = raw_protons.get(PROTON_VARIABLES["quality"])
    protons = build_proton_dataset(
        _times(velocity, PROTON_VARIABLES["velocity"]),
        _values(velocity, PROTON_VARIABLES["velocity"]),
        _values(density, PROTON_VARIABLES["density"]),
        _values(temperature, PROTON_VARIABLES["temperature"]),
        sun_distance_km=(
            _values(distance, PROTON_VARIABLES["sun_distance"])
            if distance is not None
            else None
        ),
        quality_flags=(
            _values(proton_quality, PROTON_VARIABLES["quality"])
            if proton_quality is not None
            else None
        ),
    )
    return SolarWindData(
        mag=_clip(mag, start=start, stop=stop),
        protons=_clip(protons, start=start, stop=stop),
    )
