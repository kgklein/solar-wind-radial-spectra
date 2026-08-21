import numpy as np
import pytest

from solarwind.load import load_timeseries
from solarwind.plot import _continuous_slices
from solarwind.psp import AU_IN_KM, _times, build_magnetic_dataset, build_proton_dataset


def test_mission_name_normalization(monkeypatch):
    sentinel = object()

    monkeypatch.setattr("solarwind.psp.load_psp_timeseries", lambda start, stop: sentinel)

    assert load_timeseries(" PSP ", "2022-01-01", "2022-01-02") is sentinel


def test_unsupported_mission_is_clear():
    with pytest.raises(NotImplementedError, match="Mission 'solo' is not implemented yet"):
        load_timeseries("SolO", "2022-01-01", "2022-01-02")


def test_magnetic_dataset_components_magnitude_metadata_and_fill():
    time = np.array(["2022-01-01T00:00:00", "2022-01-01T00:00:01"], dtype="datetime64[ns]")
    components = np.array([[3.0, 4.0, 12.0], [-1.0e31, 1.0, 2.0]])
    quality_time = time[:1]
    dataset = build_magnetic_dataset(
        time,
        components,
        quality_time=quality_time,
        quality_flags=np.array([2], dtype=np.uint32),
    )

    assert set(dataset.data_vars) == {"B_R", "B_T", "B_N", "B_mag", "quality_flag"}
    np.testing.assert_allclose(dataset["B_mag"].values[0], 13.0)
    assert np.isnan(dataset["B_R"].values[1])
    assert np.isnan(dataset["B_mag"].values[1])
    assert dataset["B_R"].attrs["units"] == "nT"
    assert dataset.attrs["coordinate_system"] == "RTN"
    assert dataset["quality_flag"].dims == ("quality_time",)
    assert "thruster firing" in dataset["quality_flag"].attrs["description"]


def test_proton_dataset_variables_magnitude_units_and_distance_conversion():
    time = np.array(["2022-01-01T00:00:00", "2022-01-01T00:00:02"], dtype="datetime64[ns]")
    velocity = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 10.0]])
    dataset = build_proton_dataset(
        time,
        velocity,
        density=np.array([10.0, 11.0]),
        temperature=np.array([20.0, 21.0]),
        sun_distance_km=np.array([AU_IN_KM, AU_IN_KM / 2]),
        quality_flags=np.array([0, 4], dtype=np.uint16),
    )

    expected = {"V_R", "V_T", "V_N", "V_mag", "n_p", "T_p", "r_au", "quality_flag"}
    assert set(dataset.data_vars) == expected
    np.testing.assert_allclose(dataset["V_mag"], [5.0, 10.0])
    np.testing.assert_allclose(dataset["r_au"], [1.0, 0.5])
    assert dataset["V_R"].attrs["units"] == "km/s"
    assert dataset["n_p"].attrs["units"] == "cm^-3"
    assert dataset["T_p"].attrs["units"] == "eV"
    assert dataset.attrs["datatype"] == "sf00_l3_mom"
    assert "counter overflow" in dataset["quality_flag"].attrs["description"]


def test_component_shape_validation():
    with pytest.raises(ValueError, match=r"shape \(time, 3\)"):
        build_magnetic_dataset(
            np.array(["2022-01-01"], dtype="datetime64[ns]"),
            np.array([[1.0, 2.0]]),
        )


@pytest.mark.parametrize(
    "raw",
    [
        np.array(["2022-01-01T00:00:00"], dtype="datetime64[ns]"),
        np.array([1640995200.0]),
    ],
)
def test_pyspedas_time_forms_are_normalized(raw):
    result = _times({"x": raw}, "synthetic")

    assert result.dtype == np.dtype("datetime64[ns]")
    assert result[0] == np.datetime64("2022-01-01T00:00:00", "ns")


def test_plot_slices_break_at_timestamp_gaps():
    time = np.array(
        [
            "2022-01-01T00:00:00",
            "2022-01-01T00:00:01",
            "2022-01-01T00:00:02",
            "2022-01-01T00:01:00",
            "2022-01-01T00:01:01",
        ],
        dtype="datetime64[ns]",
    )

    segments = list(_continuous_slices(time))

    assert segments == [slice(0, 3), slice(3, 5)]
