"""Command-line diagnostic plotting for native-cadence solar-wind data."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from .data import SolarWindData
from .load import load_timeseries


def _approximate_cadence(dataset: xr.Dataset) -> str:
    if dataset.sizes.get("time", 0) < 2:
        return "unavailable"
    differences_ns = np.diff(dataset.time.values).astype("timedelta64[ns]").astype(np.int64)
    positive_differences = differences_ns[differences_ns > 0]
    if positive_differences.size == 0:
        return "unavailable"
    seconds = float(np.median(positive_differences)) / 1.0e9
    if seconds < 1.0:
        return f"{seconds * 1000:.3g} ms (~{1 / seconds:.3g} Hz)"
    return f"{seconds:.3g} s"


def _dataset_summary(name: str, dataset: xr.Dataset) -> str:
    count = dataset.sizes.get("time", 0)
    if count == 0:
        return f"{name}: no samples"
    start = np.datetime_as_string(dataset.time.values[0], unit="s")
    stop = np.datetime_as_string(dataset.time.values[-1], unit="s")
    return (
        f"{name}: {count:,} samples, {start} to {stop}, "
        f"approximate cadence {_approximate_cadence(dataset)}"
    )


def print_summary(data: SolarWindData) -> None:
    """Print sample counts, ranges, and native cadence estimates."""

    print(_dataset_summary("FIELDS magnetic field", data.mag))
    print(_dataset_summary("SPAN-I proton moments", data.protons))
    if "r_au" in data.protons and data.protons["r_au"].count() > 0:
        minimum = float(data.protons["r_au"].min(skipna=True))
        maximum = float(data.protons["r_au"].max(skipna=True))
        print(f"Heliocentric distance: {minimum:.4f} to {maximum:.4f} au")


def diagnostic_plot(data: SolarWindData) -> plt.Figure:
    """Create an unsmoothed diagnostic plot on each product's native timestamps."""

    figure, axes = plt.subplots(6, 1, figsize=(12, 12), sharex=True, constrained_layout=True)
    mag_time = data.mag.time.values
    proton_time = data.protons.time.values

    for component in ("B_R", "B_T", "B_N"):
        axes[0].plot(mag_time, data.mag[component], linewidth=0.45, label=component)
    axes[0].set_ylabel("B [nT]")
    axes[0].legend(loc="upper right", ncols=3)
    axes[0].set_title("Parker Solar Probe native-cadence time series")

    axes[1].plot(mag_time, data.mag["B_mag"], color="black", linewidth=0.45)
    axes[1].set_ylabel("|B| [nT]")

    for component in ("V_R", "V_T", "V_N"):
        axes[2].plot(proton_time, data.protons[component], linewidth=0.7, label=component)
    axes[2].set_ylabel("V [km/s]")
    axes[2].legend(loc="upper right", ncols=3)

    axes[3].plot(proton_time, data.protons["V_mag"], color="black", linewidth=0.7)
    axes[3].set_ylabel("|V| [km/s]")

    axes[4].plot(proton_time, data.protons["n_p"], color="tab:purple", linewidth=0.7)
    axes[4].set_ylabel("$n_p$ [cm$^{-3}$]")

    axes[5].plot(proton_time, data.protons["T_p"], color="tab:red", linewidth=0.7)
    axes[5].set_ylabel("$T_p$ [eV]")
    axes[5].set_xlabel("UTC")

    for axis in axes:
        axis.grid(alpha=0.25)
    return figure


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mission", required=True, help="Mission name (currently PSP only)")
    parser.add_argument("--start", required=True, help="Interval start in ISO-8601 format")
    parser.add_argument("--stop", required=True, help="Interval stop in ISO-8601 format")
    parser.add_argument("--output", type=Path, help="Save the plot instead of opening a window")
    return parser


def main() -> None:
    args = _parser().parse_args()
    data = load_timeseries(args.mission, args.start, args.stop)
    print_summary(data)
    figure = diagnostic_plot(data)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.output, dpi=150)
        print(f"Saved diagnostic plot to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
