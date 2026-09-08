"""Pull the original, unprocessed smart-meter readings for each device and
store them under models/<device_id>/raw.csv.

This is deliberately a dumb archiver: it applies no cleaning, no unit
conversion and no reordering, so whatever analysis comes next starts from
exactly what the API returned.

By default it talks to the VCPES/Sentinel API directly, skipping the
data_provision relay - the relay adds nothing here beyond a hop and an auth
header we can set ourselves. Pass `--via data-provision` to go back through
it (useful from a host that can only reach the platform services).

Three things it has to get right, because each one silently truncates or
breaks the fetch:

1. The direct API needs `Authorization: Token <SENTINEL_V08_TOKEN>` and lives
   on port 8000. Port 80 on the same host serves an HTML UI, not the API, so
   pointing at it yields a web page rather than JSON.
2. `parameters` is comma-joined for the direct API but must be repeated query
   params for data_provision, whose `List[SmartMeterParameter]` validation
   rejects the joined form with a 422.
3. The upstream pages by *time*, in yearly chunks, and an intermediate chunk
   can legitimately be empty while later ones hold data. Stopping at the first
   empty page throws away everything after it - for device 100 that is the
   difference between 1978 rows and 3476. `next_url`'s own host is also
   unusable (it advertises port 80), so only its query is reused.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd
import requests

CLI_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CLI_DIR))

# This folder's own .env (device ids, upstream host) plus the platform's, so
# SENTINEL_V08_TOKEN only has to be configured in one place. train.py loads the
# same files on import; loading them here too keeps the module runnable on its
# own.
try:
    from dotenv import load_dotenv

    load_dotenv(CLI_DIR / ".env")
    load_dotenv(CLI_DIR.parent / ".env")
    load_dotenv(CLI_DIR.parent.parent / ".env")
except ImportError:  # dotenv is optional; a plain env var works too
    pass

from train import Smart_Meters_IDs, _model_name_from_device_id  # noqa: E402

# data_provision's route for the same upstream endpoint, used by
# --via data-provision.
DATA_PROVISION_URL = "http://localhost:8002"
SMART_METER_ENDPOINT = "/sentinel/dataspace/smart_meter/reading-by-device-day"

# Mirrors services/data_provision/config.py so both point at the same host.
# Configured in .env rather than defaulted here: the hostname identifies the
# operator whose meters this reads.
VCPES_HOST = os.getenv("VCPES_HOST") or ""
VCPES_PORT = os.getenv("VCPES_PORT") or "8000"
VCPES_BASE_URL = f"http://{VCPES_HOST}:{VCPES_PORT}"
SENTINEL_ENDPOINT = "/dataspace/smart_meter/reading-by-device-day"

DEFAULT_PARAMETERS = ["energyActiveImportValue-Wh"]

# Wide enough to cover everything the upstream holds; empty chunks at either
# end are walked over rather than treated as the end of the data.
START_DATE = "2024-01-01T00:00:00Z"
END_DATE = "2027-01-01T00:00:00Z"
MAX_PAGES = 20


def _encode(base: dict, parameters: list[str], direct: bool) -> list[tuple[str, str]]:
    """Query params, with `parameters` encoded the way this target expects."""
    items = list(base.items())
    if direct:
        items.append(("parameters", ",".join(parameters)))
    else:
        items.extend(("parameters", p) for p in parameters)
    return items


def _next_page_params(next_url: str, direct: bool) -> list[tuple[str, str]]:
    """Reuse only `next_url`'s query - its host advertises the wrong port."""
    query = parse_qs(urlparse(next_url).query)
    parameters: list[str] = []
    for value in query.pop("parameters", []):
        parameters.extend(p for p in value.split(",") if p)
    base = {k: v[0] for k, v in query.items()}
    return _encode(base, parameters, direct)


def fetch_raw(
    device_id: str,
    parameters: list[str],
    start_date: str,
    end_date: str,
    direct: bool = True,
    base_url: str | None = None,
    token: str | None = None,
    retries: int = 3,
    page_delay: float = 0.5,
) -> tuple[list[dict], int]:
    """Every row the API will give for one device, following pagination."""
    if direct:
        url = f"{(base_url or VCPES_BASE_URL).rstrip('/')}{SENTINEL_ENDPOINT}"
        headers = {"Authorization": f"Token {token}"} if token else {}
    else:
        url = f"{(base_url or DATA_PROVISION_URL).rstrip('/')}{SMART_METER_ENDPOINT}"
        headers = {}

    params = _encode(
        {
            "start_date": start_date,
            "end_date": end_date,
            "format": "json",
            "deviceSystemIDValue": device_id,
        },
        parameters,
        direct,
    )

    rows: list[dict] = []
    pages = 0
    while True:
        for attempt in range(retries):
            try:
                response = requests.get(url, params=params, headers=headers, timeout=600)
                if response.status_code == 401:
                    raise SystemExit(
                        "401 from the Sentinel API: SENTINEL_V08_TOKEN is missing or "
                        f"invalid. Set it in {CLI_DIR.parent / '.env'} (or the "
                        "environment), or use --via data-provision."
                    )
                response.raise_for_status()
                break
            except requests.RequestException:
                if attempt == retries - 1:
                    raise
                time.sleep(2 * (attempt + 1))
        payload = response.json()
        page = payload.get("data")
        rows.extend(page if isinstance(page, list) else [])
        pages += 1
        next_url = payload.get("next_url")
        # NB: do not stop on an empty page - see the module docstring.
        if not next_url or pages >= MAX_PAGES:
            break
        params = _next_page_params(next_url, direct)
        time.sleep(page_delay)
    return rows, pages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--via",
        choices=("sentinel", "data-provision"),
        default="sentinel",
        help="Call the VCPES/Sentinel API directly (default) or go through the "
        "data_provision relay.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help=f"Override the target base URL (default {VCPES_BASE_URL} for "
        f"sentinel, {DATA_PROVISION_URL} for data-provision).",
    )
    parser.add_argument("--start-date", default=START_DATE)
    parser.add_argument("--end-date", default=END_DATE)
    parser.add_argument(
        "--device-ids",
        default=None,
        help="Comma-separated subset. Defaults to every id in PRETRAIN_DEVICE_IDS.",
    )
    parser.add_argument(
        "--parameters",
        default=None,
        help="Comma-separated parameters to fetch. Defaults to "
        f"{','.join(DEFAULT_PARAMETERS)}. See the SmartMeterParameter literal in "
        "services/data_provision/routers/sentinel.py for the full set.",
    )
    parser.add_argument("--out-root", default=str(CLI_DIR / "models"))
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Refetch devices that already have a raw.csv.",
    )
    # The API has been intermittently returning 502s under load, so the
    # defaults here are deliberately unhurried. Nothing about this is urgent.
    parser.add_argument(
        "--delay", type=float, default=2.0, help="Seconds between devices (default 2)."
    )
    parser.add_argument(
        "--page-delay",
        type=float,
        default=0.5,
        help="Seconds between pagination requests (default 0.5).",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Stop after this many devices."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    direct = args.via == "sentinel"
    token = os.getenv("SENTINEL_V08_TOKEN")
    if direct and not token:
        raise SystemExit(
            "SENTINEL_V08_TOKEN is not set - the Sentinel API rejects unauthenticated "
            f"requests with a 401. Add it to {CLI_DIR.parent / '.env'} (or export it), "
            "or pass --via data-provision to use the relay instead."
        )
    if direct and not args.base_url and not VCPES_HOST:
        raise SystemExit(
            f"VCPES_HOST is not set. Add it to {CLI_DIR / '.env'} (see .env.example), "
            "pass --base-url, or use --via data-provision."
        )

    target_url = args.base_url or (VCPES_BASE_URL if direct else DATA_PROVISION_URL)
    print(f"Fetching via {args.via}: {target_url}", flush=True)

    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    device_ids = (
        [d.strip() for d in args.device_ids.split(",") if d.strip()]
        if args.device_ids
        else Smart_Meters_IDs
    )
    if not device_ids:
        raise SystemExit(
            f"No device_ids configured. Set PRETRAIN_DEVICE_IDS in {CLI_DIR / '.env'} "
            "(see .env.example) or pass --device-ids."
        )
    parameters = (
        [p.strip() for p in args.parameters.split(",") if p.strip()]
        if args.parameters
        else DEFAULT_PARAMETERS
    )
    if args.limit:
        device_ids = device_ids[: args.limit]

    index = []
    for i, device_id in enumerate(device_ids, 1):
        # One folder per device/parameter pair, named exactly as train.py names a
        # trained model, so a device's raw data and its model live together
        # rather than in two folders that look unrelated.
        targets = {
            parameter: out_root / _model_name_from_device_id(device_id, parameter)
            for parameter in parameters
        }
        if all((t / "raw.csv").exists() for t in targets.values()) and not args.overwrite:
            print(f"[{i}/{len(device_ids)}] {device_id}: raw.csv exists, skipping.", flush=True)
            continue

        started = time.time()
        try:
            rows, pages = fetch_raw(
                device_id,
                parameters,
                args.start_date,
                args.end_date,
                direct=direct,
                base_url=args.base_url,
                token=token,
                page_delay=args.page_delay,
            )
        except SystemExit:
            raise
        except Exception as e:
            print(f"[{i}/{len(device_ids)}] {device_id}: FAILED {type(e).__name__}: {e}", flush=True)
            index.append({"device_id": device_id, "n_rows": 0, "error": f"{type(e).__name__}: {e}"})
            continue

        elapsed = round(time.time() - started, 1)
        df = pd.DataFrame(rows)
        # Columns the API returns regardless of which parameters were asked for.
        meta_cols = [c for c in df.columns if c not in parameters]

        for parameter, target in targets.items():
            target.mkdir(parents=True, exist_ok=True)
            cols = meta_cols + [c for c in (parameter,) if c in df.columns]
            # Row order and row content are untouched; with the default single
            # parameter this is the whole response verbatim.
            df[cols].to_csv(target / "raw.csv", index=False)

            entry = {
                "device_id": device_id,
                "parameter": parameter,
                "model_name": target.name,
                "n_rows": len(df),
                "n_pages": pages,
                "elapsed_s": elapsed,
                "error": None,
            }
            if len(df) and "measurementDatetimeValue" in df.columns:
                t = pd.to_datetime(df["measurementDatetimeValue"], utc=True)
                entry["span_start"] = str(t.min())
                entry["span_end"] = str(t.max())
                entry["span_days"] = round((t.max() - t.min()).total_seconds() / 86400, 2)
                if parameter in df.columns:
                    entry["n_nonnull"] = int(
                        pd.to_numeric(df[parameter], errors="coerce").notna().sum()
                    )

            meta = dict(entry)
            meta.update({
                "source": f"{args.via}:{SENTINEL_ENDPOINT if direct else SMART_METER_ENDPOINT}",
                "base_url": target_url,
                "start_date": args.start_date,
                "end_date": args.end_date,
                "columns": cols,
                "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "note": "Original API response, unmodified: original row order, no cleaning or unit conversion.",
            })
            with (target / "raw_meta.json").open("w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            index.append(entry)

        print(
            f"[{i}/{len(device_ids)}] {device_id}: {len(df)} rows, {pages} page(s), "
            f"{index[-1].get('span_days', '-')} days, {elapsed}s -> "
            f"{', '.join(t.name for t in targets.values())}",
            flush=True,
        )
        pd.DataFrame(index).to_csv(out_root / "raw_fetch_index.csv", index=False)
        time.sleep(args.delay)

    if index:
        pd.DataFrame(index).to_csv(out_root / "raw_fetch_index.csv", index=False)
        print(f"\nIndex written to {out_root / 'raw_fetch_index.csv'}")


if __name__ == "__main__":
    main()
