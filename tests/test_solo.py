import numpy as np
import pytest

from solarwind.load import load_timeseries
from solarwind.solo import build_magnetic_dataset, build_proton_dataset


@pytest.mark.parametrize("mission", ["solo", "SolO", "SOLO", "solar orbiter"])
def test_solo_mission_normalization_and_dispatch(monkeypatch, mission):
    sentinel = object()
    monkeypatch.setattr("solarwind.solo.load_solo_timeseries", lambda start, stop: sentinel)

    assert load_timeseries(mission, "2022-01-01", "2022-01-02") is sentinel


def test_solo_mag_dataset_components_magnitude_quality_and_fill():
    time = np.array(
        ["2022-01-01T00:00:00", "2022-01-01T00:00:00.125"],
        dtype="datetime64[ns]",
    )
    dataset = build_magnetic_dataset(
        time,
        components=np.array([[3.0, 4.0, 12.0], [-1.0e31, 1.0, 2.0]]),
        quality_flag=np.array([3, 254], dtype=np.uint8),
        quality_bitmask=np.array([128, 65534], dtype=np.uint16),
        vector_range=np.array([3, 254], dtype=np.uint8),
        samples_per_second=np.array([8.0, -1.0e31]),
    )

    expected = {
        "B_R",
        "B_T",
        "B_N",
        "B_mag",
        "quality_flag",
        "quality_bitmask",
        "vector_range",
        "samples_per_second",
    }
    assert set(dataset.data_vars) == expected
    assert dataset["B_mag"].values[0] == 13.0
    assert np.isnan(dataset["B_R"].values[1])
    assert np.isnan(dataset["B_mag"].values[1])
    assert np.isnan(dataset["quality_flag"].values[1])
    assert np.isnan(dataset["quality_bitmask"].values[1])
    assert np.isnan(dataset["vector_range"].values[1])
    assert np.isnan(dataset["samples_per_second"].values[1])
    assert dataset["B_R"].attrs["units"] == "nT"
    assert dataset.attrs["mission"] == "Solar Orbiter"
    assert dataset.attrs["instrument"] == "MAG"
    assert dataset.attrs["datatype"] == "rtn-normal"


def test_solo_pas_dataset_components_magnitude_metadata_and_fill():
    time = np.array(
        ["2022-01-01T00:00:00", "2022-01-01T00:00:04"],
        dtype="datetime64[ns]",
    )
    dataset = build_proton_dataset(
        time,
        velocity=np.array([[3.0, 4.0, 0.0], [-1.0e31, 1.0, 2.0]]),
        density=np.array([10.0, -1.0e31]),
        temperature=np.array([20.0, -1.0e31]),
        info=np.array([0, 255], dtype=np.uint8),
        quality_factor=np.array([1.0, -1.0e31]),
    )

    expected = {
        "V_R",
        "V_T",
        "V_N",
        "V_mag",
        "n_p",
        "T_p",
        "info",
        "quality_factor",
    }
    assert set(dataset.data_vars) == expected
    assert dataset["V_mag"].values[0] == 5.0
    for name in ("V_R", "V_mag", "n_p", "T_p", "info", "quality_factor"):
        assert np.isnan(dataset[name].values[1])
    assert dataset["V_R"].attrs["units"] == "km/s"
    assert dataset["n_p"].attrs["units"] == "cm^-3"
    assert dataset["T_p"].attrs["units"] == "eV"
    assert dataset.attrs["mission"] == "Solar Orbiter"
    assert dataset.attrs["instrument"] == "SWA/PAS"
    assert dataset.attrs["datatype"] == "pas-grnd-mom"
    assert "without mode selection" in dataset["info"].attrs["description"]
