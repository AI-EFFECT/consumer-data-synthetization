# Synthetic Model Pretrainer CLI

Local-only CLI for pretraining shared synthetic data models from smart-meter
data. It fetches readings from the `data_provision` service (one `device_id`
per model) and trains a DoppelGANger model per device. It is not a Docker
service and is not started with the platform stack.

## Setup

From `services/synthetic_model_pretrainer`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`data_provision` must be reachable (defaults to `http://localhost:8002`). Its
`/sentinel/dataspace/smart_meter/reading-by-device-day` endpoint is a `GET`,
so no auth token is required.

## Configure which meters to pretrain

Device ids are real smart-meter identifiers, so they are **configuration, not
source**: they live in this folder's `.env`, which git ignores. Start from the
template:

```bash
cp .env.example .env
```

```bash
# every known id — fetch_raw.py's default fleet
PRETRAIN_DEVICE_IDS=device-a,device-b,device-c
# what `python train.py` trains when --device-ids is not passed
PRETRAIN_RUN_DEVICE_IDS=device-a
# upstream host fetch_raw.py reads from
VCPES_HOST=
VCPES_PORT=8000
```

Nothing else the scripts need is secret, so the remaining knobs stay as
constants at the top of `train.py`:

```python
PARAMETERS = ["energyActiveImportValue-Wh"]
START_DATE = "2025-10-01T00:00:00Z"
END_DATE = "2026-07-01T00:00:00Z"
DATA_PROVISION_URL = "http://localhost:8002"
```

One model is trained per `device_id`/`parameter` pair. `PRETRAIN_DEVICE_IDS` is
the full known fleet; `PRETRAIN_RUN_DEVICE_IDS` is the short list you actually
want to run right now. Both, and every constant above, can be overridden
per-run with the matching CLI flag (`--device-ids`, `--parameters`,
`--start-date`, ...). Without either variable set, the scripts stop with a
message naming the file to fill in rather than silently doing nothing.

> [!IMPORTANT]
> Never paste a real device id back into a tracked file. The point of the
> pseudonyms the generation service serves (see its README) is undone if the
> identifiers are sitting in the repository.

## What the data looks like, and what the CLI does to it

The endpoint returns a **cumulative** energy counter
(`energyActiveImportValue-Wh` is a meter register that only counts up), not
per-interval consumption. Two things follow from that:

- **Rows arrive in a zigzag order.** `date_bucket` ascends but timestamps
  *within* each bucket descend, so reading the raw response top-to-bottom the
  counter appears to fall all day and jump at each day boundary. Everything in
  the CLI sorts by time first; this only misleads when inspecting the raw JSON
  by hand.
- **Power is a difference of the counter.** `to_interval_power()` converts the
  register into average power over each interval,
  `(E[t] - E[t-step]) / step_hours`, in **watts**.

Devices do not all report at the same cadence — probing the live endpoint
shows roughly half the fleet at 5 minutes, half at 15, and a few at 10. The
naming families are not a reliable guide (one id prefix covers both), so the
native resolution is measured per device from the gaps between its non-null
readings.

**Every model is trained at a single resolution**, `TARGET_RESOLUTION_MINUTES`
(15 min by default). Faster devices are downsampled onto that grid. Because
the source is cumulative, this is exact rather than approximate: the energy in
a 15-minute interval is the counter delta across its endpoints no matter how
many readings fall in between. The counter is put on the target grid *first*
and differenced there, rather than computing power natively and averaging it.
A device reporting more slowly than the target is skipped rather than
upsampled into readings it never took.

Nulls are common (150 of 157 devices have some; a few are 80%+ null) and are
dropped before anything else. A point on the target grid is only kept when the
device's real readings bracketing it are at most `--max-gap-minutes` apart —
inside a longer outage the interpolated counter is invention, so those points
are blanked and the power values spanning them dropped.

`sequence_len` is one day at the target resolution (96 samples at 15 min) and
`sample_len` is derived from it.

## Pull the original data first (`fetch_raw.py`)

`fetch_raw.py` archives the unprocessed API response for every device under
`models/<device_id>/raw.csv`, so analysis starts from exactly what the API
returned — original row order, no cleaning or unit conversion. It fetches
`energyActiveImportValue-Wh` only; `--parameters` takes a comma-separated list
if you ever need others.

```powershell
python fetch_raw.py
```

It calls the VCPES/Sentinel API **directly** by default, which needs a token:

```
SENTINEL_V08_TOKEN=<token>
```

in `services/.env` (the same file the platform services read). Without it the
script refuses to start rather than failing later with a 401. Pass
`--via data-provision` to go through the relay instead — useful from a host
that can only reach the platform services.

Throttled by default (2s between devices, 0.5s between pagination pages, 3
retries with backoff). A full run is ~157 devices × ~4 pages, roughly 20–25
minutes. It is resumable: devices that already have a `raw.csv` are skipped,
so it is safe to interrupt and rerun. `--overwrite` forces a refetch,
`--limit N` does a trial run.

Outputs:

```
models/<device_id>/raw.csv         original API response, unmodified
models/<device_id>/raw_meta.json   window, pages, span, per-parameter non-null counts
models/raw_fetch_index.csv         one row per device, written incrementally
```

> [!NOTE]
> `raw.csv` keeps the API's own row ordering, which zigzags (`date_bucket`
> ascends, timestamps descend *within* each bucket). Sort by
> `measurementDatetimeValue` before reading it.

## Inspect the response first

```powershell
python train.py --inspect
```

fetches one window per configured `device_id` and prints its native
resolution, how many rows survive the regrid, and how many sequences that
leaves, instead of training.

## Train shared models

```powershell
python train.py
```

runs with the constants at the top of the file. Each configured `device_id`
creates one model, e.g. `100` with parameter `energyActiveImportValue-Wh`
creates `models/100_energyActiveImportValue-Wh`. By default, models are
written to:

```powershell
services/synthetic_model_pretrainer/models
```

Point `services/synthetic_data_generation` at that folder with
`SYNTHETIC_MODEL_ROOT` so every user of the generation API can list and
generate from these root-level shared models.

## Options

```powershell
python train.py `
  --data-provision-url http://localhost:8002 `
  --parameters energyActiveImportValue-Wh `
  --start-date 2025-10-01T00:00:00Z `
  --end-date 2026-07-01T00:00:00Z `
  --device-ids device-a,device-b `
  --model-root .\models `
  --index-col datetime `
  --target-resolution-minutes 15 `
  --max-gap-minutes 60 `
  --batch-size 4 `
  --epochs 300
```

`--sequence-len` and `--sample-len` override the values derived from
`--target-resolution-minutes`; `sample_len` must divide `sequence_len`
exactly. Use `--index-col ""` if the fetched data has no timestamp column, and
`--overwrite` to retrain a model that already has `model.pt`.

## Evaluate what each model can actually generate (`capabilities.py`)

A trained model's generator is recurrent, so it rolls out to any length, but its
quality is not length-independent: the recurrent state drifts outside the regime
it was trained on. `capabilities.py` measures, per model, which horizons hold up.

```powershell
python capabilities.py --seed 0 --runs 5 --num-examples 40
```

Takes about three minutes for the whole fleet. For each model it generates
several independent batches at 24/48/72/96/168 h, averages the metrics over the
runs, and compares them against that model's own `train_data.csv` on:

- **scale** (`median_ratio`) and **spread** (`iqr_ratio`) against the real series;
- **physical validity**: share of generated values that are negative, which an
  import register cannot produce, measured *before* clipping;
- **mode collapse**: share of near-constant examples;
- **daily shape**, relative to what is achievable for that device.

That last one matters. Rather than an absolute correlation bar, each device's own
split-half reliability is measured first (halve its real days at random, average
each half into a daily profile, correlate the two). That is the ceiling: no model
can match the real profile more closely than the real data matches itself. Across
the fleet the median is 0.70, but about a quarter of meters score below 0.5,
meaning they have no repeatable daily pattern at all. Those are flagged
(`daily_profile_learnable: false`) and skip the shape check rather than being
penalised for a property of the data.

Models trained on fewer than `MIN_REAL_DAYS` (14) complete days are not certified
at any horizon: the statistics they would be compared against come from the same
few days, so both sides are noise.

`offered_hours` stops at the first horizon that fails, since a horizon passing
while a shorter one fails is sampling noise rather than a capability.

Results are written into each model's `metadata.json` under `capabilities`, and
collected into `models/capabilities_index.json`. The platform lists only models
certified for at least 24 hours, and offers each one only at its verified
horizons.

> [!NOTE]
> Decisions near a threshold are not fully stable: roughly 10% of models change
> their offered horizons between seeds even with `--runs 5`, because much of the
> fleet sits close to the limits rather than because of sampling noise.

`diagnose.py` is a separate, older tool with its own `OK`/`WARN`/`DISCARD`
verdicts. It judges only the trained length and **does not** gate what the
platform serves; `capabilities.py` does.

## Generate from an already-trained model (no interface needed)

`generate.py` loads a model already trained by `train.py` and writes the
synthetic output as a CSV directly into that model's own folder — useful for
a quick check without going through the `synthetic_data_generation`
API/interface:

```powershell
python generate.py 0cb815fd7f50 --num-examples 3
```

This writes `models/0cb815fd7f50/generated_<timestamp>.csv`, in the same
shape (`example_id` column included) as the generation API's CSV export. Use
`--namespace` if the model was trained under one, `--model-root` to point at
a different models folder, and `--output` to pick a specific filename
instead of the timestamped default.

## Diagnose model quality

`diagnose.py` generates sample output from every trained model under
`models/` and checks it for signs of a bad fit, so you don't have to eyeball
each one:

```powershell
python diagnose.py
```

For each model it reports:

- **Mode collapse** — share of generated examples that are near-constant.
- **Range violation** — share of generated values that fall outside the real
  training data's observed min/max (with a margin). Requires `train_data.csv`,
  which only the CLI's `main()` (not the training API) saves alongside a model.
- **Autocorrelation match** — mean absolute difference between the real and
  synthetic autocorrelation at several lags, i.e. whether the generated series
  keeps the same temporal structure as the real one instead of just matching
  its overall value distribution.
- **Cross-feature correlation drift** — Frobenius distance between the real
  and synthetic feature-correlation matrices. Only meaningful for
  multi-feature models; today's models are single-feature (`value`), so this
  reports `None`.

Each model gets a verdict (`OK` / `WARN` / `DISCARD` / `BAD_DATA`) from
heuristic thresholds at the top of the script — tune them once a few real
runs show what's normal for this platform's meters. `BAD_DATA` means the
*real* training data itself has ~0 variance (e.g. a meter that reported flat
or zero import for the whole fetch window) — a near-constant synthetic
output is then the correct fit, not mode collapse, so it's reported
separately from `DISCARD` rather than penalizing the model. Retraining won't
fix a `BAD_DATA` verdict; re-fetch a window with more variance for that
consumer instead. A full JSON report is written to
`models/diagnostics_report.json` (`--output` to change the path). Use
`--model <name>` to check a single model, and `--num-examples` to control how
many samples are generated per model (default 20 — more gives a steadier
mode-collapse/range estimate at the cost of runtime).

Generation is stochastic and unseeded by default, so low-probability metrics
like `range_violation_ratio` will jitter somewhat between runs of the same
model - that's sampling noise, not the model changing. Pass `--seed <int>` to
fix the RNG so repeated runs (e.g. comparing a model before/after a
hyperparameter change) produce the same synthetic draws and are directly
comparable, or pass `--runs 10` to diagnose each model 10 times (each an
independent sample) and report the mean ± std of every metric instead of a
single noisy draw - the printed std also tells you how much a one-off number
can be trusted. `range_violation_ratio` tends to have real spread across runs
(it hinges on a few extreme values); `acf_mae` is usually far steadier, since
it reflects the average shape rather than the tails.
