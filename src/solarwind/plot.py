"""Command-line diagnostic plotting for native-cadence solar-wind data."""

import argparse
from collections.abc import Iterator, Sequence
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


def _continuous_slices(time: np.ndarray) -> Iterator[slice]:
    """Split plotting lines at timestamp gaps without changing the data cadence."""

    if time.size == 0:
        return
    differences_ns = np.diff(time).astype("timedelta64[ns]").astype(np.int64)
    positive_differences = differences_ns[differences_ns > 0]
    if positive_differences.size == 0:
        yield slice(0, time.size)
        return

    typical_ns = float(np.median(positive_differences))
    gap_starts = np.flatnonzero(differences_ns > 10 * typical_ns) + 1
    boundaries = np.concatenate(([0], gap_starts, [time.size]))
    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        yield slice(int(start), int(stop))


def _plot_native(
    axis,
    time: np.ndarray,
    values: np.ndarray,
    *,
    segments: Sequence[slice] | None = None,
    **kwargs,
) -> None:
    """Plot every native sample while preventing lines across time gaps."""

    label = kwargs.pop("label", None)
    plot_segments = tuple(_continuous_slices(time)) if segments is None else segments
    series_color = kwargs.pop("color", None)
    for segment_number, segment in enumerate(plot_segments):
        segment_label = label if segment_number == 0 else "_nolegend_"
        lines = axis.plot(
            time[segment],
            values[segment],
            label=segment_label,
            color=series_color,
            **kwargs,
        )
        if series_color is None:
            series_color = lines[0].get_color()


def print_summary(data: SolarWindData) -> None:
    """Print sample counts, ranges, and native cadence estimates."""

    print(f"Mission: {data.mag.attrs.get('mission', 'unknown')}")
    mag_label = f"{data.mag.attrs.get('instrument', 'mag')} {data.mag.attrs.get('datatype', '')}"
    proton_label = (
        f"{data.protons.attrs.get('instrument', 'protons')} "
        f"{data.protons.attrs.get('datatype', '')}"
    )
    print(_dataset_summary(mag_label.strip(), data.mag))
    print(_dataset_summary(proton_label.strip(), data.protons))
    if "r_au" in data.protons and data.protons["r_au"].count() > 0:
        minimum = float(data.protons["r_au"].min(skipna=True))
        maximum = float(data.protons["r_au"].max(skipna=True))
        print(f"Heliocentric distance: {minimum:.4f} to {maximum:.4f} au")


def diagnostic_plot(data: SolarWindData) -> plt.Figure:
    """Create an unsmoothed diagnostic plot on each product's native timestamps."""

    figure, axes = plt.subplots(6, 1, figsize=(12, 12), sharex=True, constrained_layout=True)
    mag_time = data.mag.time.values
    proton_time = data.protons.time.values
    mag_segments = tuple(_continuous_slices(mag_time))
    proton_segments = tuple(_continuous_slices(proton_time))

    for component in ("B_R", "B_T", "B_N"):
        _plot_native(
            axes[0],
            mag_time,
            data.mag[component].values,
            segments=mag_segments,
            linewidth=0.45,
            label=component,
        )
    axes[0].set_ylabel("B [nT]")
    axes[0].legend(loc="upper right", ncols=3)
    mission = data.mag.attrs.get("mission", "Solar wind")
    axes[0].set_title(f"{mission} native-cadence time series")

    _plot_native(
        axes[1],
        mag_time,
        data.mag["B_mag"].values,
        segments=mag_segments,
        color="black",
        linewidth=0.45,
    )
    axes[1].set_ylabel("|B| [nT]")

    for component in ("V_R", "V_T", "V_N"):
        _plot_native(
            axes[2],
            proton_time,
            data.protons[component].values,
            segments=proton_segments,
            linewidth=0.7,
            label=component,
        )
    axes[2].set_ylabel("V [km/s]")
    axes[2].legend(loc="upper right", ncols=3)

    _plot_native(
        axes[3],
        proton_time,
        data.protons["V_mag"].values,
        segments=proton_segments,
        color="black",
        linewidth=0.7,
    )
    axes[3].set_ylabel("|V| [km/s]")

    _plot_native(
        axes[4],
        proton_time,
        data.protons["n_p"].values,
        segments=proton_segments,
        color="tab:purple",
        linewidth=0.7,
    )
    axes[4].set_ylabel("$n_p$ [cm$^{-3}$]")

    _plot_native(
        axes[5],
        proton_time,
        data.protons["T_p"].values,
        segments=proton_segments,
        color="tab:red",
        linewidth=0.7,
    )
    axes[5].set_ylabel("$T_p$ [eV]")
    axes[5].set_xlabel("UTC")

    for axis in axes:
        axis.grid(alpha=0.25)
    return figure


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mission", required=True, help="Mission name (PSP or SolO)")
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
