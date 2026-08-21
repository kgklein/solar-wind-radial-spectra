# Solar Wind Radial Spectra

This small research repository handles the data-access and diagnostic-plotting
part of an undergraduate project comparing solar-wind turbulence at different
heliocentric distances. The current version supports Parker Solar Probe (PSP)
and Solar Orbiter (SolO).

Spectral analysis is intentionally not implemented. PSD methods, interval
selection, filtering, fitting, and related scientific choices are part of the
student's research project rather than this starter code.

## Installation

Python 3.11 or newer is required. A virtual environment is recommended:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
```

PySPEDAS files are cached under `data/psp` and `data/solar-orbiter`, so files
already downloaded by either mission adapter are reused. Set
`SPEDAS_DATA_DIR` before running if you intentionally want a different cache
root; PySPEDAS's mission-specific `PSP_DATA_DIR` and `SOLO_DATA_DIR` overrides
remain supported.

## First plot

Load the example day and open a six-panel diagnostic plot:

```bash
./example_run.sh
```

The helper uses `.venv/bin/python` directly, so activating the environment is
not required. The equivalent explicit PSP command is:

```bash
python scripts/plot_interval.py \
    --mission PSP \
    --start 2022-02-25T00:00:00 \
    --stop 2022-02-26T00:00:00
```

For Solar Orbiter, change only the mission:

```bash
python scripts/plot_interval.py \
    --mission SolO \
    --start 2022-02-25T00:00:00 \
    --stop 2022-02-26T00:00:00
```

The helper accepts the same override as `./example_run.sh --mission SolO`.
Add `--output figures/interval.png` to save the figure instead. The command
prints each product's time range, sample count, and approximate native cadence.
The plot does not smooth, filter, interpolate, resample, or otherwise modify
the measurements. Data gaps and missing values remain visible.

Be aware that full-resolution FIELDS data are large: the example day contains
about 25 million magnetic samples and downloads four six-hour CDF files.

## Products and returned data

The public loader is:

```python
from solarwind import load_timeseries

data = load_timeseries(
    mission="psp",
    start="2022-02-25T00:00:00",
    stop="2022-02-26T00:00:00",
)

data.mag       # xarray.Dataset on the native FIELDS timestamps
data.protons   # xarray.Dataset on the independent SPAN-I timestamps
```

Mission names are case-insensitive. Missions other than PSP or SolO fail
explicitly. `solo`, `SolO`, `SOLO`, and `solar orbiter` are equivalent.

### Parker Solar Probe

The magnetic dataset comes from PSP/FIELDS L2 `mag_rtn`, the full-resolution
RTN magnetic-field product. It exposes `B_R`, `B_T`, `B_N`, and `B_mag` in nT.
The adapter intentionally does not substitute `4_per_cycle` or one-minute
magnetic data.

The proton dataset comes from PSP SWEAP/SPAN-I L3 `sf00_l3_mom`. It exposes
`V_R`, `V_T`, `V_N`, and `V_mag` in km/s, `n_p` in cm^-3, `T_p` in eV, and
`r_au`. The source `SUN_DIST` values are in km and are converted using the IAU
definition `1 au = 149597870.7 km`.

The datasets retain official quality bit masks without applying quality cuts.
SPAN-I `quality_flag` uses the proton time coordinate. The FIELDS quality flag
has its own one-minute `quality_time` coordinate because it is not sampled at
the magnetic waveform cadence. Each flag variable's `description` attribute
records the official CDF bit meanings. CDF fill values are represented as
`NaN`.

### Solar Orbiter

The magnetic dataset comes from Solar Orbiter MAG L2 `rtn-normal`, the
normal-mode RTN product. It exposes `B_R`, `B_T`, `B_N`, and `B_mag` in nT at
the native cadence. It does not substitute the one-minute or burst product.
The official high-level `quality_flag`, detailed `quality_bitmask`,
`vector_range`, and `samples_per_second` metadata are retained without cuts.

The proton dataset comes from the official SWA/PAS L2
`solo_L2_swa-pas-grnd-mom` ground-calculated proton moments product, exposed by
PySPEDAS as datatype `pas-grnd-mom`. It exposes `V_R`, `V_T`, `V_N`, and
`V_mag` in km/s, `n_p` in cm^-3, and `T_p` in the source unit eV. The PAS
`info` operating-mode indicator and `quality_factor` are retained without mode
selection or quality cuts. This product does not contain a
heliocentric-distance variable, so the SolO dataset does not invent or derive
`r_au`.

PySPEDAS's high-level SolO MAG loader omits variables marked as CDF metadata,
even when support data are requested. The adapter therefore downloads the
official CDF through `solo.mag(downloadonly=True)` and reads the selected MAG
science and quality variables using PySPEDAS's own CDF importer with metadata
enabled. PAS works directly with `notplot=True`.

Magnetic and proton timestamps are never synchronized, interpolated, or
resampled.
