"""Measure, per model, which generation horizons actually hold up.

A trained model's generator is an LSTM, so it rolls out to any length - but its
quality is not length-independent: the hidden state drifts outside the regime it
was trained on. Rather than assume 24h everywhere or advertise a week nobody
should use, this measures each horizon against the model's own real training
data and records the verdict alongside the model.

The result is written into each model's metadata.json under "capabilities", and
collected into models/capabilities_index.json for the platform to serve.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

CLI_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CLI_DIR))
sys.path.insert(0, str(CLI_DIR.parent / "synthetic_data_generation"))

from dgan_wrapper import generate, load_train_result  # noqa: E402
from diagnose import (  # noqa: E402
    ACF_LAGS,
    RANGE_MARGIN_RATIO,
    _acf_per_sequence,
    _reshape_sequences,
)

# Horizons offered to users, in hours. 24h is the trained length for every
# current model; the rest are rollouts beyond it.
DEFAULT_HORIZONS_H = (24, 48, 72, 96, 168)

# A horizon is "supported" only if all of these hold. Calibrated against what
# the models actually produce at their *trained* length, so the bar is "no worse
# than the model already is at 24h", not an absolute quality standard.
MEDIAN_RATIO_OK = (0.5, 2.0)  # synthetic median / real median
IQR_RATIO_OK = (0.4, 4.0)  # synthetic IQR / real IQR
FLAT_RATIO_MAX = 0.3  # share of near-constant examples

# An import register cannot run backwards, so any negative value the generator
# emits is physically impossible. Generation clips them away, but clipping only
# hides the symptom: the negatives are the left tail of an over-dispersed
# distribution, so clipping piles that mass on zero instead. A model producing
# many of them is not fit to serve regardless of what the other metrics say.
# Measured across the fleet the median is 14%, and horizons that passed every
# other check reached 42% - hence a cap rather than trusting the clip.
NEGATIVE_RATIO_MAX = 0.20

# A model trained on a handful of days cannot be judged: the very statistics we
# compare against are estimated from those same few days, so both sides are
# noise and a horizon can pass by accident. An earlier run offered a 168h
# horizon off 3 days of real data, which is not a capability.
MIN_REAL_DAYS = 14

# Daily shape is judged *relative to what is learnable for that device*, not
# against an absolute correlation. Real load profiles differ hugely in how
# repeatable they are: splitting each device's real days in half and correlating
# the two mean profiles gives a median of 0.70, but 27% of devices score below
# 0.5 - those meters have no stable daily pattern in the first place, so holding
# their model to an absolute bar would penalise it for a property of the data.
PROFILE_CORR_RELATIVE_MIN = 0.6  # fraction of the device's own reliability ceiling
PROFILE_RELIABILITY_MIN = 0.3  # below this, the device has no learnable daily profile


def _profile_reliability(real: np.ndarray, steps_per_day: int, seed: int = 0) -> float | None:
    """Split-half reliability of the device's own mean daily profile.

    Randomly halves the real days, averages each half into a profile, and
    correlates them. This is the ceiling: no model can match the real profile
    more closely than the real data matches itself.
    """
    n_days = len(real) // steps_per_day
    if n_days < 6:
        return None
    days = real[: n_days * steps_per_day].reshape(n_days, steps_per_day)
    order = np.random.default_rng(seed).permutation(n_days)
    half = n_days // 2
    a = days[order[:half]].mean(axis=0)
    b = days[order[half : 2 * half]].mean(axis=0)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _metrics(real: np.ndarray, synth: np.ndarray, steps_per_day: int) -> dict:
    """Compare a rollout against the model's real training data."""
    r_med, s_med = float(np.median(real)), float(np.median(synth))
    r_iqr = float(np.subtract(*np.percentile(real, [75, 25])))
    s_iqr = float(np.subtract(*np.percentile(synth, [75, 25])))

    # Daily shape: average each over whole days and correlate the profiles.
    n_days = synth.shape[1] // steps_per_day
    r_prof = real[: (len(real) // steps_per_day) * steps_per_day].reshape(
        -1, steps_per_day
    ).mean(axis=0)
    profile_corr = None
    if n_days >= 1:
        s_prof = synth[:, : n_days * steps_per_day].reshape(-1, steps_per_day).mean(axis=0)
        if r_prof.std() > 1e-9 and s_prof.std() > 1e-9:
            profile_corr = float(np.corrcoef(r_prof, s_prof)[0, 1])

    # Robust range, same bounds diagnose.py uses.
    lo, hi = np.percentile(real, [0.5, 99.5])
    margin = (hi - lo) * RANGE_MARGIN_RATIO
    range_violation = float(((synth < lo - margin) | (synth > hi + margin)).mean())

    per_example_std = synth.std(axis=1)
    flat_ratio = float((per_example_std < 1e-6).mean())
    # An import register cannot run backwards, so real interval power is never
    # negative. Anything the generator emits below zero is physically impossible
    # and is a symptom of over-dispersion pushing the left tail past zero.
    negative_ratio = float((synth < 0).mean())

    lags = [l for l in ACF_LAGS if l < synth.shape[1]]
    acf_mae = None
    if lags:
        r_seq = _reshape_sequences(real.reshape(-1, 1), steps_per_day)
        s_acf = _acf_per_sequence(synth[:, :, None], lags)
        r_acf = _acf_per_sequence(r_seq, lags)
        if not np.all(np.isnan(r_acf)) and not np.all(np.isnan(s_acf)):
            acf_mae = float(np.nanmean(np.abs(r_acf - s_acf)))

    return {
        "median_ratio": (s_med / r_med) if r_med else None,
        "iqr_ratio": (s_iqr / r_iqr) if r_iqr > 1e-9 else None,
        "profile_corr": profile_corr,
        "range_violation": range_violation,
        "flat_ratio": flat_ratio,
        "negative_ratio": negative_ratio,
        "acf_mae": acf_mae,
        "synth_median_w": s_med,
        "real_median_w": r_med,
    }


_AVERAGED = (
    "median_ratio", "iqr_ratio", "profile_corr", "range_violation",
    "flat_ratio", "negative_ratio", "acf_mae", "synth_median_w",
)


def _average_runs(runs: list[dict]) -> dict:
    """Mean of each metric across independent draws, plus its spread.

    Every metric here is estimated from a finite sample of generated sequences,
    so a single draw carries real sampling noise - measured across two seeds,
    14% of models changed their offered horizons and 9 entered or left the list
    entirely. Deciding on the mean of several draws removes most of that; the
    recorded _std says how much a given decision can be trusted.

    The mean is used rather than requiring every draw to pass: a hard "all runs
    must pass" rule would cut borderline models instead of averaging them, and
    the fleet has too few passing models to spend that way.
    """
    out = dict(runs[-1])
    for key in _AVERAGED:
        values = [
            r[key] for r in runs
            if r.get(key) is not None and not (isinstance(r[key], float) and np.isnan(r[key]))
        ]
        if values:
            out[key] = float(np.mean(values))
            out[f"{key}_std"] = float(np.std(values))
        else:
            out[key] = None
            out[f"{key}_std"] = None
    return out


def _supported(m: dict, reliability: float | None) -> tuple[bool, list[str]]:
    reasons = []
    mr, ir, pc = m["median_ratio"], m["iqr_ratio"], m["profile_corr"]
    if mr is None or not (MEDIAN_RATIO_OK[0] <= mr <= MEDIAN_RATIO_OK[1]):
        reasons.append(f"median {mr:.2f}x real" if mr else "median undefined")
    if ir is None or not (IQR_RATIO_OK[0] <= ir <= IQR_RATIO_OK[1]):
        reasons.append(f"spread {ir:.2f}x real" if ir else "spread undefined")
    if m["flat_ratio"] > FLAT_RATIO_MAX:
        reasons.append(f"{m['flat_ratio']:.0%} flat examples")
    if m["negative_ratio"] > NEGATIVE_RATIO_MAX:
        reasons.append(
            f"{m['negative_ratio']:.0%} of generated values are negative, which an "
            "import register cannot produce"
        )

    # Shape is only judged where there is a stable daily profile to reproduce.
    if reliability is not None and reliability >= PROFILE_RELIABILITY_MIN:
        if pc is None:
            reasons.append("shape undefined")
        else:
            relative = pc / reliability
            if relative < PROFILE_CORR_RELATIVE_MIN:
                reasons.append(
                    f"daily shape {relative:.0%} of what this device allows "
                    f"(corr {pc:.2f} vs ceiling {reliability:.2f})"
                )
    return (not reasons), reasons


def evaluate_model(
    model_dir: Path, horizons_h, num_examples: int, resolution_min: int, runs: int = 1
) -> dict:
    train_csv = model_dir / "train_data.csv"
    result = {"model": model_dir.name}
    try:
        tr = load_train_result(str(model_dir))
    except Exception as e:
        return {**result, "error": f"could not load model: {e}"}
    if not train_csv.exists():
        return {**result, "error": "no train_data.csv to compare against"}

    feature = list(tr.features)[0]
    real = pd.read_csv(train_csv)[feature].to_numpy(dtype=float)
    steps_per_day = (24 * 60) // resolution_min
    trained_steps = int(tr.model.max_sequence_len)
    real_days = int(len(real) // steps_per_day)
    reliability = _profile_reliability(real, steps_per_day)
    learnable = reliability is not None and reliability >= PROFILE_RELIABILITY_MIN

    horizons = []
    for hours in horizons_h:
        steps = hours * 60 // resolution_min
        if steps % tr.model.sample_len:
            continue
        try:
            run_metrics = []
            for _ in range(max(1, runs)):
                dfs = generate(tr, num_examples=num_examples, horizon_steps=steps)
                synth = np.stack([d[feature].to_numpy(dtype=float) for d in dfs])
                run_metrics.append(_metrics(real, synth, steps_per_day))
        except Exception as e:
            horizons.append({"hours": hours, "steps": steps, "supported": False,
                             "reasons": [f"generation failed: {e}"]})
            continue
        m = _average_runs(run_metrics)
        ok, reasons = _supported(m, reliability)
        if real_days < MIN_REAL_DAYS:
            ok = False
            reasons = [
                f"only {real_days} real day(s) of training data "
                f"(minimum {MIN_REAL_DAYS} to certify a horizon)"
            ] + reasons
        rel_corr = (
            m["profile_corr"] / reliability
            if (learnable and m["profile_corr"] is not None)
            else None
        )
        horizons.append({
            "hours": hours, "steps": steps, "is_trained_length": steps == trained_steps,
            "runs": max(1, runs), "supported": ok, "reasons": reasons,
            "profile_corr_relative": round(rel_corr, 4) if rel_corr is not None else None,
            **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()},
        })

    supported = [h["hours"] for h in horizons if h["supported"]]
    # Only contiguous support from the trained length outward is offered: a
    # horizon that passes while a shorter one fails is far more likely to be
    # noise than a real capability.
    offered = []
    for h in horizons:
        if h["supported"]:
            offered.append(h["hours"])
        else:
            break
    result.update({
        "resolution_minutes": resolution_min,
        "trained_sequence_hours": trained_steps * resolution_min // 60,
        "real_days_trained_on": real_days,
        "enough_real_data": real_days >= MIN_REAL_DAYS,
        "daily_profile_reliability": round(reliability, 4) if reliability is not None else None,
        "daily_profile_learnable": learnable,
        "horizons": horizons,
        "supported_hours": supported,
        "offered_hours": offered,
        "max_offered_hours": max(offered) if offered else None,
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_examples_per_horizon": num_examples,
        "runs_per_horizon": max(1, runs),
    })
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-root", default=str(CLI_DIR / "models"))
    p.add_argument("--model", default=None, help="Evaluate a single model folder name.")
    p.add_argument("--horizons", default=",".join(str(h) for h in DEFAULT_HORIZONS_H),
                   help="Comma-separated horizons in hours.")
    p.add_argument("--num-examples", type=int, default=40,
                   help="Samples generated per horizon, per run (default 40).")
    p.add_argument("--runs", type=int, default=5,
                   help="Independent draws per horizon; metrics are averaged over "
                        "them before the thresholds are applied (default 5). A "
                        "single draw let 14%% of models change horizons between seeds.")
    p.add_argument("--resolution-minutes", type=int, default=15)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--max-negative-ratio",
        type=float,
        default=NEGATIVE_RATIO_MAX,
        help=f"Reject a horizon whose generated output is more than this share "
        f"negative (default {NEGATIVE_RATIO_MAX}).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
    global NEGATIVE_RATIO_MAX
    NEGATIVE_RATIO_MAX = args.max_negative_ratio
    root = Path(args.model_root).resolve()
    horizons_h = [int(x) for x in args.horizons.split(",") if x.strip()]
    models = sorted({p.parent for p in root.rglob("model.pt")})
    if args.model:
        models = [m for m in models if m.name == args.model]
        if not models:
            raise SystemExit(f"No model named {args.model!r} under {root}")

    index = []
    for i, model_dir in enumerate(models, 1):
        res = evaluate_model(
            model_dir, horizons_h, args.num_examples, args.resolution_minutes, args.runs
        )
        if "error" in res:
            print(f"[{i}/{len(models)}] {model_dir.name}: {res['error']}", flush=True)
            index.append(res)
            continue

        meta_path = model_dir / "metadata.json"
        meta = {}
        if meta_path.exists():
            with meta_path.open(encoding="utf-8") as f:
                meta = json.load(f)
        meta["capabilities"] = {k: v for k, v in res.items() if k != "model"}
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        index.append({
            "model": res["model"],
            "device_id": meta.get("device_id"),
            "parameter": meta.get("parameter"),
            "resolution_minutes": res["resolution_minutes"],
            "real_days_trained_on": res["real_days_trained_on"],
            "offered_hours": res["offered_hours"],
            "max_offered_hours": res["max_offered_hours"],
        })
        print(f"[{i}/{len(models)}] {res['model']}: offers {res['offered_hours'] or 'nothing'}", flush=True)

    out = root / "capabilities_index.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump({"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "models": index}, f, indent=2)
    print(f"\nIndex written to {out}")


if __name__ == "__main__":
    main()
