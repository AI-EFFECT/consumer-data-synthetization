import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

CLI_DIR = Path(__file__).resolve().parent
SYNTHETIC_DATA_DIR = CLI_DIR.parent / "synthetic_data_generation"
sys.path.insert(0, str(SYNTHETIC_DATA_DIR))

# ---------------------------------------------------------------------------
# Edit these to control which meters get pretrained and what data is pulled.
# One model is trained per device_id/parameter pair.
# ---------------------------------------------------------------------------
# Full known device_id list, for reference / to copy IDs from - not used
# directly. Paste the ones you actually want to run into DEVICE_IDS below.
Smart_Meters_IDs = [
    "100",
    "101",
    "102",
    "103",
    "104",
    "105",
    "106",
    "107",
    "108",
    "26",
    "27",
    "28",
    "29",
    "30",
    "31",
    "32",
    "33",
    "34",
    "35",
    "36",
    "37",
    "38",
    "39",
    "40",
    "41",
    "42",
    "43",
    "44",
    "45",
    "46",
    "47",
    "48",
    "49",
    "50",
    "51",
    "52",
    "53",
    "54",
    "55",
    "56",
    "57",
    "58",
    "59",
    "60",
    "61",
    "62",
    "68",
    "70",
    "71",
    "72",
    "73",
    "74",
    "75",
    "76",
    "77",
    "78",
    "79",
    "84",
    "85",
    "86",
    "87",
    "88",
    "89",
    "90",
    "92",
    "93",
    "94",
    "95",
    "96",
    "97",
    "98",
    "99",
    "CLCECE000515",
    "CLCECE000921",
    "CLCECE000987",
    "CLCECE001121",
    "CLCECE001335",
    "CLCECE001338",
    "CLCECE001452",
    "CLCECE001610",
    "CLCECE001616",
    "CLCECE001840",
    "CLCECE001843",
    "CLCECE002226",
    "CLCECE002901",
    "CLCECE003420",
    "CLCECE003551",
    "CLCECE005204",
    "CLCECE005277",
    "CLCECE005480",
    "CLCECE005844",
    "CLCECE006039",
    "CLCECE007075",
    "CLCECE008082",
    "CLCECE008350",
    "CLCECE008774",
    "CLCECE008876",
    "CLCECE009476",
    "CLCECE009598",
    "CLCECE009681",
    "CLCECE009768",
    "CLCECE009783",
    "CLITCE000134",
    "CLITCE000268",
    "CLITCE000607",
    "CLITCE000626",
    "CLITCE001176",
    "CLITCE001397",
    "CLITCE001519",
    "CLITCE001747",
    "CLITCE002013",
    "CLITCE002176",
    "CLITCE002213",
    "CLITCE002311",
    "CLITCE002327",
    "CLITCE002376",
    "CLITCE002674",
    "CLITCE002796",
    "CLITCE002797",
    "CLITCE002926",
    "CLITCE003052",
    "CLITCE003346",
    "CLITCE003581",
    "CLITCE003596",
    "CLITCE003646",
    "CLITCE004109",
    "CLITCE004193",
    "CLITCE004255",
    "CLITCE004504",
    "CLITCE004748",
    "CLITCE004951",
    "CLITCE005621",
    "CLITCE005684",
    "CLITCE005815",
    "CLITCE005970",
    "CLITCE006102",
    "CLITCE006419",
    "CLITCE006555",
    "CLITCE006756",
    "CLITCE006775",
    "CLITCE006786",
    "CLITCE006850",
    "CLITCE007111",
    "CLITCE007250",
    "CLITCE007282",
    "CLITCE007354",
    "CLITCE007669",
    "CLITCE008048",
    "CLITCE008193",
    "CLITCE009309",
    "CLITCE009696",
    "CLITCE009972",
    "mater_004",
    "mater_005",
    "meter_001",
    "meter_002",
    "meter_869310062981256"
  ]
# Paste the device_id(s) you actually want to run right now - this is what
# `python train.py` uses by default when --device-ids isn't passed. Keep it
# short; pick IDs from Smart_Meters_IDs above (or from a diagnose.py run).
DEVICE_IDS = [
        100]
PARAMETERS = ["energyActiveImportValue-Wh"]
# Where each device's pulled data lives, inside its own model folder. Written
# by fetch_raw.py - run that first; this script never touches the network.
RAW_FILENAME = "raw.csv"

# Physical ceiling on a single meter's average power over one interval. Used to
# reject counter readings that cannot belong to this meter.
#
# The readings are a cumulative Wh register, and a handful of rows per device
# carry a value belonging to some other meter entirely - same IMEI and place in
# the response, but a counter tens of MWh away from its neighbours (they also
# tend to carry a different deviceIDValue, though that is a symptom rather than
# a reliable marker). Differencing across such a row produces a pair of equal
# and opposite spikes of 1e7-1e9 W, which then set the min/max that DGAN's
# ZERO_ONE normalisation scales everything by - so a couple of bad rows in a
# few thousand flatten the entire real signal to ~1e-7 of the model's range.
#
# Measured across the fleet, legitimate peak power (per-model p99.9) has a
# median of ~3.1 kW and only one model exceeds 10 kW, while contaminated rows
# start at ~1e7 W. 25 kW sits an order of magnitude above any real consumer
# here and four orders below the corruption, so the separation is not delicate.
MAX_POWER_W = 25_000
TARGET_RESOLUTION_MINUTES = 15
# ---------------------------------------------------------------------------


def _model_name_from_device_id(device_id: str, parameter: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", f"{device_id}_{parameter}").strip(".-")
    if not name:
        raise ValueError(
            f"Could not derive a model name from device_id={device_id!r}, "
            f"parameter={parameter!r}"
        )
    return name


def _model_path(model_root: Path, model_name: str, namespace: str | None) -> Path:
    if namespace:
        return model_root / namespace / model_name
    return model_root / model_name


def _write_progress(model_path: Path, progress_info) -> None:
    info = {
        "epoch": progress_info.epoch + 1,
        "total_epochs": progress_info.total_epochs,
        "batch": progress_info.batch + 1,
        "total_batches": progress_info.total_batches,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }
    model_path.mkdir(parents=True, exist_ok=True)
    with (model_path / "progress.json").open("w", encoding="utf-8") as f:
        json.dump(info, f)
    print(json.dumps(info), flush=True)


# Columns the API returns alongside the requested parameter. They are constant
# or non-numeric for a single-device query, so they carry no training signal
# and would break DGAN's feature-type detection.
_RAW_METADATA_COLUMNS = (
    "deviceSystemIDValue",
    "date_bucket",
    "deviceIDValue",
    "deviceTypeValue",
    "devicePlaceValue",
    "deviceLabelValue",
)


def load_raw_readings(model_path: Path, parameter: str) -> pd.DataFrame:
    """Read one device's pulled readings from its model folder's raw.csv.

    Returns just the timestamp (renamed to "datetime", which is what
    to_interval_power and diagnose.py expect) and the requested parameter,
    sorted ascending in time. raw.csv preserves the API's own row ordering,
    which zigzags - date_bucket ascends while timestamps descend *within* each
    bucket - so sorting here is not optional.
    """
    raw_csv = model_path / RAW_FILENAME
    if not raw_csv.exists():
        raise FileNotFoundError(
            f"{raw_csv} not found. Run fetch_raw.py first to pull this device's data."
        )
    df = pd.read_csv(raw_csv)
    if parameter not in df.columns:
        raise ValueError(
            f"{raw_csv} has no column {parameter!r} (columns: {list(df.columns)}). "
            "Refetch with fetch_raw.py --parameters."
        )
    if "measurementDatetimeValue" not in df.columns:
        raise ValueError(f"{raw_csv} has no measurementDatetimeValue column.")

    df = df.drop(columns=[c for c in _RAW_METADATA_COLUMNS if c in df.columns])
    df = df.rename(columns={"measurementDatetimeValue": "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    return df.sort_values("datetime").reset_index(drop=True)


def _detect_resolution_minutes(
    df: pd.DataFrame, value_col: str | None = None, time_col: str = "datetime"
) -> float:
    """Median gap between consecutive real readings, in minutes.

    Devices report at different native resolutions - confirmed by probing all
    of Smart_Meters_IDs against the live endpoint: 74 devices at 5min, 72 at
    15min, 4 at 10min. The split does NOT follow the id naming families (the
    CLCECE block is genuinely mixed), so this has to be measured per device.

    Rows with a null value are excluded before measuring. The endpoint returns
    a row per expected slot even when it has no reading, and 150 of 157 devices
    have at least some nulls - on the heavily-null ones (80%+) the gap median
    over *all* rows reports the polling cadence rather than the cadence of
    actual readings, which is what matters here.
    """
    times = pd.to_datetime(df[time_col], utc=True)
    if value_col is not None and value_col in df.columns:
        times = times[pd.to_numeric(df[value_col], errors="coerce").notna()]
    diffs_minutes = times.sort_values().diff().dropna().dt.total_seconds() / 60
    if diffs_minutes.empty:
        raise ValueError("Not enough distinct timestamps to detect a reporting resolution.")
    return float(diffs_minutes.median())


def _pick_sample_len(sequence_len: int, target_ratio: int = 24) -> int:
    """Largest divisor of sequence_len closest to sequence_len/target_ratio.

    dgan_wrapper.train() requires sample_len to divide sequence_len exactly.
    target_ratio=24 mirrors the sequence_len/sample_len ratio used for the
    earlier CEVE-source models (1440/60), applied to whatever sequence_len
    this device's detected resolution produces for a 24h window.
    """
    target = max(1, round(sequence_len / target_ratio))
    divisors = [d for d in range(1, sequence_len + 1) if sequence_len % d == 0]
    return min(divisors, key=lambda d: abs(d - target))


def _drop_implausible_readings(
    series: pd.Series, max_power_w: float, window: int = 9
) -> tuple[pd.Series, int]:
    """Drop counter readings that cannot belong to this meter.

    Compares each reading against a centred rolling median of its neighbours
    rather than against a running total, so an isolated bad row is judged by
    the readings around it and cannot anchor the scan or cascade. The
    tolerance is how far the counter could plausibly drift across the window
    at `max_power_w` - real readings stay far inside it, mis-attributed ones
    miss by orders of magnitude.

    Filtering here, in the counter domain, matters: dropping the row removes
    both the spike into it and the spike back out. Clipping the power values
    afterwards would leave the second half of the pair behind.
    """
    if len(series) < 3:
        return series, 0
    local_median = series.rolling(window, center=True, min_periods=3).median()
    step_seconds = series.index.to_series().diff().dt.total_seconds().median()
    if not np.isfinite(step_seconds) or step_seconds <= 0:
        step_seconds = 900.0
    window_hours = max(window * step_seconds / 3600.0, 1 / 60)
    tolerance_wh = max_power_w * window_hours
    keep = (series - local_median).abs() <= tolerance_wh
    keep |= local_median.isna()
    return series[keep], int((~keep).sum())


def build_attribute_df(processed_df: pd.DataFrame, time_col: str = "datetime") -> pd.DataFrame:
    """Per-sequence conditioning attributes, given per row.

    Only `is_weekend` for now. Day-of-week was the obvious richer choice, but
    the fetched history gives a device 30-41 complete days, which is 4-6
    examples per weekday - too few for a GAN to learn seven distinct shapes
    rather than memorise the examples. A binary weekday/weekend split keeps
    roughly 22/9 days per class and is also where the real load-shape
    difference lives. Month/season is not available at all: the data spans
    late April to late August.

    Values must be constant within a sequence to be meaningful, which holds
    only because to_interval_power day-aligns.
    """
    weekday = processed_df[time_col].dt.dayofweek
    return pd.DataFrame({"is_weekend": (weekday >= 5).astype(int)})


def _keep_complete_days(
    out: pd.DataFrame, time_col: str, target_minutes: int
) -> tuple[pd.DataFrame, int, int]:
    """Keep only whole calendar days, so one sequence is one day.

    dgan_wrapper reshapes into sequences by row count, so without this a
    "24h sequence" starts at whatever time the device's first reading landed
    on (16:00 for device 100) and silently jumps across any blanked outage -
    measured on the previous run, 14 of device 101's 50 sequences spanned a
    time discontinuity. DGAN learns structure indexed by timestep position, so
    an arbitrary phase offset smears the daily load profile it should be
    learning, and a per-sequence calendar attribute (day-of-week) has no
    well-defined value at all.

    A day is kept only if every one of its slots survived the regrid, so the
    sequences handed to the model are contiguous, midnight-aligned and free of
    interpolated-over gaps.
    """
    slots_per_day = (24 * 60) // target_minutes
    if out.empty:
        return out, 0, 0
    dates = out[time_col].dt.date
    counts = dates.map(dates.value_counts())
    kept = out[counts == slots_per_day].reset_index(drop=True)
    return kept, len(kept) // slots_per_day, len(out) // slots_per_day


def to_interval_power(
    df: pd.DataFrame,
    value_col: str,
    time_col: str = "datetime",
    target_minutes: int = TARGET_RESOLUTION_MINUTES,
    max_gap_minutes: int = 60,
    max_power_w: float = MAX_POWER_W,
    day_align: bool = True,
) -> pd.DataFrame:
    """Convert a cumulative energy counter (e.g. energyActiveImportValue-Wh)
    into average power on a uniform `target_minutes` grid, whatever the
    device's own reporting resolution is.

    Every model trains at the same resolution, so devices reporting faster
    than the target (about half the fleet reports every 5min) are downsampled
    here. Because the source is a *cumulative* counter, that is exact rather
    than approximate: the energy consumed over an interval is the counter
    delta across its endpoints, E[t] - E[t-target], no matter how many
    readings fall in between. So we put the counter on the target grid first
    and difference that, instead of computing power at the native resolution
    and averaging it - averaging would have to guess how to weight intervals
    with missing readings, and gets the answer wrong whenever a bin is only
    partly covered.

    Readings never land exactly on the grid (seconds of jitter, and a 10min
    device shares no grid points with a 15min grid at all), so the counter is
    interpolated onto the grid in the time domain. A grid point is only
    trusted when the real readings bracketing it are no more than
    `max_gap_minutes` apart; inside a longer outage the interpolated value is
    a straight line through unknown territory, so it is blanked and the power
    values spanning it are dropped rather than reporting a smooth invented
    load.
    """
    series = (
        df.assign(**{time_col: pd.to_datetime(df[time_col], utc=True)})
        .set_index(time_col)[value_col]
        .pipe(lambda s: pd.to_numeric(s, errors="coerce"))
        .dropna()
        .sort_index()
    )
    series = series[~series.index.duplicated(keep="first")]
    series, n_dropped = _drop_implausible_readings(series, max_power_w)
    if n_dropped:
        print(f"  dropped {n_dropped} implausible counter reading(s)", flush=True)
    if len(series) < 2:
        return pd.DataFrame({time_col: [], value_col: []})

    freq = f"{target_minutes}min"
    grid = pd.date_range(
        series.index.min().ceil(freq), series.index.max().floor(freq), freq=freq
    )
    if len(grid) < 2:
        return pd.DataFrame({time_col: [], value_col: []})

    # Time-weighted interpolation of the counter onto the grid: evaluate the
    # counter at each grid instant using the real readings on either side.
    on_grid = (
        series.reindex(series.index.union(grid))
        .interpolate(method="time", limit_area="inside")
        .reindex(grid)
    )

    # Blank grid points that sit inside a real outage. `prev`/`next` are the
    # timestamps of the nearest actual readings bracketing each grid point;
    # when those are far apart the interpolated value is pure invention.
    stamps = pd.Series(series.index, index=series.index)
    prev_t = stamps.reindex(grid, method="ffill")
    next_t = stamps.reindex(grid, method="bfill")
    bracket_minutes = (next_t - prev_t).dt.total_seconds() / 60
    on_grid[(bracket_minutes > max_gap_minutes) | bracket_minutes.isna()] = np.nan

    interval_hours = target_minutes / 60
    power = (on_grid.diff() / interval_hours).dropna()
    # Anything left outside the physical ceiling straddles a filtered reading or
    # a grid edge; an import register cannot run backwards either.
    power = power[(power >= 0) & (power <= max_power_w)]
    out = power.reset_index()
    out.columns = [time_col, value_col]
    if day_align:
        out, kept_days, raw_days = _keep_complete_days(out, time_col, target_minutes)
        if raw_days and kept_days < raw_days:
            print(
                f"  day-aligned: kept {kept_days} complete day(s), dropped "
                f"{raw_days - kept_days} partial/gappy",
                flush=True,
            )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretrain shared local DoppelGANger models from the raw readings "
        "fetch_raw.py pulled into each model folder, one model per "
        "device_id/parameter pair. Runs entirely offline."
    )
    parser.add_argument(
        "--parameters",
        default=None,
        help="Comma-separated list of energy parameters to query (e.g. "
        "energyActiveImportValue-Wh). Defaults to the PARAMETERS list at the top "
        "of this file. One model is trained per device_id/parameter pair.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Train every device that has a raw.csv under --model-root, instead of "
        "the DEVICE_IDS list. Use this after a full fetch_raw.py run.",
    )
    parser.add_argument(
        "--device-ids",
        default=None,
        help="Comma-separated device_id (deviceSystemIDValue) list. Defaults to the "
        "DEVICE_IDS list at the top of this file.",
    )
    parser.add_argument(
        "--model-root",
        default=str(CLI_DIR / "models"),
        help="Root folder where trained models are stored.",
    )
    parser.add_argument(
        "--namespace",
        default=None,
        help="Optional namespace. Omit this to publish root-level models for all users.",
    )
    parser.add_argument(
        "--index-col",
        default="datetime",
        help="Name of the timestamp/index column in the fetched data. Use an empty "
        "string for none.",
    )
    parser.add_argument(
        "--target-resolution-minutes",
        type=int,
        default=TARGET_RESOLUTION_MINUTES,
        help="Time step every model is trained at. Devices reporting faster than "
        "this are downsampled onto it; devices reporting slower are skipped. "
        f"Default {TARGET_RESOLUTION_MINUTES}.",
    )
    # sequence_len/sample_len follow from the target resolution (one sequence
    # = 24h), so they are the same for every device now. Override to pin a
    # different sequence length.
    parser.add_argument(
        "--sequence-len",
        type=int,
        default=None,
        help="Fix sequence_len instead of deriving 24h worth of samples from "
        "--target-resolution-minutes.",
    )
    parser.add_argument(
        "--sample-len",
        type=int,
        default=None,
        help="Fix sample_len instead of auto-deriving it from sequence_len "
        "(must divide sequence_len exactly).",
    )
    parser.add_argument(
        "--conditional",
        action="store_true",
        help="Condition the model on per-sequence calendar attributes (is_weekend) "
        "so generation can be steered, instead of training an unconditional model.",
    )
    parser.add_argument(
        "--max-power-w",
        type=float,
        default=MAX_POWER_W,
        help="Physical ceiling on one interval's average power. Counter readings "
        "implying more than this are treated as belonging to another meter and "
        f"dropped. Default {MAX_POWER_W}.",
    )
    parser.add_argument(
        "--max-gap-minutes",
        type=int,
        default=60,
        help="A point on the target grid is only trusted when the device's real "
        "readings bracketing it are at most this far apart. Inside a longer "
        "outage the interpolated counter is invention, so those points are "
        "blanked and the power values spanning them dropped rather than "
        "reporting a smooth made-up load.",
    )
    # IMPORTANT: dgan_wrapper.train() clamps batch_size to min(batch_size,
    # n_sequences) and drops the last partial batch (drop_last=True), so a
    # batch_size close to or above n_sequences silently collapses most/all
    # epochs into 0-1 gradient updates. With ~19 sequences, keep this small
    # so there are still multiple mini-batches per epoch.
    parser.add_argument("--batch-size", type=int, default=4)
    # Bumped from 10: fewer sequences per epoch (4 batches/epoch at these
    # settings) means more epochs are needed for a comparable total number
    # of gradient steps.
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Retrain models even when model.pt already exists.",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Report each device's native cadence, usable rows and available "
        "sequences from its raw.csv instead of training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_root = Path(args.model_root).resolve()
    index_col = args.index_col or None
    if args.all:
        # Every model folder that fetch_raw.py has already populated. The folder
        # name is "<device_id>_<parameter>", so strip the parameter suffix back
        # off to recover the device id.
        device_ids = sorted(
            {
                raw.parent.name[: -(len(p) + 1)]
                for p in PARAMETERS
                for raw in model_root.glob(f"*_{p}/raw.csv")
            }
        )
    elif args.device_ids:
        device_ids = [s.strip() for s in args.device_ids.split(",") if s.strip()]
    else:
        device_ids = DEVICE_IDS
    parameters = (
        [p.strip() for p in args.parameters.split(",") if p.strip()]
        if args.parameters
        else PARAMETERS
    )

    if not device_ids:
        raise SystemExit("No device_ids configured. Edit DEVICE_IDS or pass --device-ids.")
    if not parameters:
        raise SystemExit("No parameters configured. Edit PARAMETERS or pass --parameters.")

    if not args.inspect:
        model_root.mkdir(parents=True, exist_ok=True)
        from dgan_wrapper import save_train_result, train

    for device_id in device_ids:
        for parameter in parameters:
            model_name = _model_name_from_device_id(device_id, parameter)
            target_path = _model_path(model_root, model_name, args.namespace)
            if not args.inspect and (target_path / "model.pt").exists() and not args.overwrite:
                print(
                    f"Skipping existing model {target_path}. Pass --overwrite to retrain.",
                    flush=True,
                )
                continue

            try:
                train_df = load_raw_readings(target_path, parameter)
            except (FileNotFoundError, ValueError) as e:
                print(f"device_id={device_id!r}: {e} Skipping.", flush=True)
                continue

            if train_df.empty:
                print(f"No rows in {target_path / RAW_FILENAME}, skipping.", flush=True)
                continue

            # Carry the fetch's own record forward so a trained model still says
            # which pull it came from once raw.csv is overwritten by a later one.
            raw_provenance = {}
            raw_meta_path = target_path / "raw_meta.json"
            if raw_meta_path.exists():
                with raw_meta_path.open(encoding="utf-8") as f:
                    raw_meta = json.load(f)
                raw_provenance = {
                    k: raw_meta.get(k)
                    for k in ("fetched_at", "source", "base_url", "span_start", "span_end", "n_rows")
                }

            target_minutes = args.target_resolution_minutes
            try:
                native_minutes = round(_detect_resolution_minutes(train_df, parameter))
            except ValueError as e:
                print(f"device_id={device_id!r}: {e}, skipping.", flush=True)
                continue

            # A device that reports more slowly than the target can't be put on
            # the target grid without inventing readings it never took, so skip
            # it rather than training on upsampled filler.
            if native_minutes > target_minutes:
                print(
                    f"device_id={device_id!r}: reports every ~{native_minutes}min, coarser "
                    f"than the {target_minutes}min target - skipping (upsampling would "
                    "invent readings).",
                    flush=True,
                )
                continue

            processed_df = to_interval_power(
                train_df,
                parameter,
                target_minutes=target_minutes,
                max_gap_minutes=args.max_gap_minutes,
                max_power_w=args.max_power_w,
            )

            sequence_len = args.sequence_len or round(24 * 60 / target_minutes)
            sample_len = args.sample_len or _pick_sample_len(sequence_len)

            if args.inspect:
                print(
                    f"device_id={device_id!r} parameter={parameter!r}: "
                    f"{len(train_df)} raw rows, median cadence ~{native_minutes}min "
                    f"-> {target_minutes}min target, "
                    f"{len(processed_df)} rows after regrid+power-conversion, "
                    f"auto sequence_len={sequence_len} sample_len={sample_len} "
                    f"({len(processed_df) // sequence_len} sequence(s) available)",
                    flush=True,
                )
                print(processed_df.head(3).to_string(), flush=True)
                continue

            if len(processed_df) < 2 * sequence_len:
                print(
                    f"device_id={device_id!r}: only {len(processed_df)} usable rows after "
                    f"processing, need >= {2 * sequence_len} for sequence_len={sequence_len} "
                    "(at least 2 sequences) - skipping.",
                    flush=True,
                )
                continue

            print(
                f"Training shared synthetic model '{model_name}' from device_id={device_id!r} "
                f"(native~{native_minutes}min -> {target_minutes}min, "
                f"sequence_len={sequence_len}, sample_len={sample_len})",
                flush=True,
            )
            # Isolate each device's training so this behaves like running the
            # CLI separately per device_id: one device erroring out (bad
            # data, a CUDA hiccup, ...) shouldn't lose the rest of the batch.
            try:
                attribute_df = (
                    build_attribute_df(processed_df, index_col or "datetime")
                    if args.conditional
                    else None
                )
                train_result = train(
                    processed_df,
                    index_col=index_col,
                    sequence_len=sequence_len,
                    sample_len=sample_len,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    progress_callback=lambda info, path=target_path: _write_progress(path, info),
                    attribute_df=attribute_df,
                )
                save_train_result(train_result, str(target_path))
                processed_df.to_csv(target_path / "train_data.csv", index=False)

                metadata = {
                    "model_name": model_name,
                    "namespace": args.namespace,
                    "shared": args.namespace is None,
                    "source": str(target_path / RAW_FILENAME),
                    "raw_fetch": raw_provenance,
                    "device_id": device_id,
                    "parameter": parameter,
                    "index_col": index_col,
                    "native_resolution_minutes": native_minutes,
                    "resolution_minutes": target_minutes,
                    "max_gap_minutes": args.max_gap_minutes,
                    "max_power_w": args.max_power_w,
                    # Physical bounds of the quantity, so generation can clip to
                    # them: an import register cannot run backwards, and
                    # to_interval_power already enforces the same range here.
                    "value_floor": 0.0,
                    "value_ceiling": args.max_power_w,
                    "conditional_attributes": (
                        list(attribute_df.columns) if attribute_df is not None else None
                    ),
                    "sequence_len": sequence_len,
                    "sample_len": sample_len,
                    "batch_size": args.batch_size,
                    "epochs": args.epochs,
                    "trained_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                }
                with (target_path / "metadata.json").open("w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2)
                print(f"Saved shared synthetic model to {target_path}", flush=True)
            except Exception as e:
                print(
                    f"Training failed for device_id={device_id!r}, parameter={parameter!r}: "
                    f"{e}. Skipping to the next device.",
                    flush=True,
                )
                continue


if __name__ == "__main__":
    main()
