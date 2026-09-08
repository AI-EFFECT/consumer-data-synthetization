import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

CLI_DIR = Path(__file__).resolve().parent
SYNTHETIC_DATA_DIR = CLI_DIR.parent / "synthetic_data_generation"
sys.path.insert(0, str(SYNTHETIC_DATA_DIR))

from dgan_wrapper import generate, load_train_result  # noqa: E402

# Lags (in samples) at which autocorrelation is compared. Filtered down to
# those shorter than a model's own sequence_len before use, since a lag past
# the end of every sequence is meaningless.
ACF_LAGS = [1, 5, 10, 30, 60]

# Heuristic thresholds turning raw metrics into a pass/warn/discard verdict.
# Not statistically derived - tune once a few real diagnostic runs show what
# "normal" looks like for this platform's meters.
FLAT_EXAMPLE_RATIO_WARN = 0.3  # share of generated examples that are near-constant
FLAT_EXAMPLE_RATIO_DISCARD = 0.6
RANGE_VIOLATION_RATIO_WARN = 0.05  # share of generated values outside real min/max (+margin)
RANGE_VIOLATION_RATIO_DISCARD = 0.2
RANGE_MARGIN_RATIO = 0.1  # how far past the real min/max a value may go before counting as a violation
ACF_MAE_WARN = 0.25  # mean |real_acf - synth_acf| across the compared lags
ACF_MAE_DISCARD = 0.5
REAL_DATA_FLAT_STD = 1e-6  # below this, the real training data itself has no variance to learn from

# Distribution checks, stated in robust units (median / IQR) rather than
# mean / std. This is deliberate: when a model's real data carries outliers,
# its std and its min/max are inflated by those same outliers, so every check
# expressed in those terms silently stops firing on exactly the models that
# most need catching. A previous run graded a model OK whose real mean was
# 164 W and whose synthetic mean was -2.3 MW, because the real range spanned
# +/-1e9 and nothing could fall outside it. Median and IQR do not move.
MEDIAN_SHIFT_WARN = 1.0  # |synth_median - real_median|, in real IQRs
MEDIAN_SHIFT_DISCARD = 3.0
IQR_RATIO_WARN = (0.5, 2.0)  # synth IQR / real IQR must stay inside this
IQR_RATIO_DISCARD = (0.25, 4.0)
REAL_DATA_FLAT_IQR = 1e-9  # below this the real data has no spread to compare against


def _find_models(model_root: Path) -> list[Path]:
    return sorted({p.parent for p in model_root.rglob("model.pt")})


def _sequence_len(model_dir: Path, fallback: int) -> int:
    metadata_path = model_dir / "metadata.json"
    if metadata_path.exists():
        with metadata_path.open(encoding="utf-8") as f:
            sequence_len = json.load(f).get("sequence_len")
        if sequence_len:
            return int(sequence_len)
    # metadata.json is missing sequence_len (or the whole file): the indices.csv
    # saved by dgan_wrapper.train() is always exactly sequence_len rows long
    # (train_result.indices = train_df[index_col].head(sequence_len)).
    indices_path = model_dir / "indices.csv"
    if indices_path.exists():
        return len(pd.read_csv(indices_path))
    return fallback


def _reshape_sequences(values: np.ndarray, sequence_len: int) -> np.ndarray:
    """(n_rows, n_features) -> (n_sequences, sequence_len, n_features), dropping any remainder."""
    n_sequences = values.shape[0] // sequence_len
    if n_sequences < 1:
        return values[:0].reshape(0, sequence_len, values.shape[1])
    return values[: n_sequences * sequence_len].reshape(n_sequences, sequence_len, values.shape[1])


def _acf_per_sequence(sequences: np.ndarray, lags: list[int]) -> np.ndarray:
    """Mean autocorrelation at `lags`, averaged over sequences and features.

    Zero-variance sequences (a mode-collapsed synthetic example, or a flat
    real window) are excluded rather than folded in as an artificial ACF of
    0 - flat examples are already caught separately by the flat-ratio check.
    """
    if sequences.shape[0] == 0 or not lags:
        return np.full(len(lags), np.nan)
    n_features = sequences.shape[-1]
    per_seq_acfs = []
    for seq in sequences:
        for feat in range(n_features):
            series = seq[:, feat]
            if series.std() < 1e-9:
                continue
            centered = series - series.mean()
            denom = np.dot(centered, centered)
            per_seq_acfs.append([np.dot(centered[:-lag], centered[lag:]) / denom for lag in lags])
    if not per_seq_acfs:
        return np.full(len(lags), np.nan)
    return np.mean(np.array(per_seq_acfs), axis=0)


def _correlation_delta(real_seqs: np.ndarray, synth_seqs: np.ndarray) -> float | None:
    """Frobenius distance between real and synthetic feature-correlation matrices.

    None for single-feature models: with one column there is no cross-feature
    correlation to preserve, so the metric is undefined rather than 0.
    """
    n_features = real_seqs.shape[-1]
    if n_features < 2 or real_seqs.shape[0] == 0:
        return None
    real_corr = np.corrcoef(real_seqs.reshape(-1, n_features), rowvar=False)
    synth_corr = np.corrcoef(synth_seqs.reshape(-1, n_features), rowvar=False)
    return float(np.linalg.norm(real_corr - synth_corr))


def _verdict(
    flat_ratio,
    range_violation_ratio,
    acf_mae,
    real_data_flat: bool = False,
    median_shift=None,
    iqr_ratio=None,
) -> tuple[str, list[str]]:
    reasons = []
    severity = 0  # 0=OK, 1=WARN, 2=DISCARD

    if real_data_flat:
        # The real training data itself has ~0 variance (e.g. a meter that reported
        # constant/zero import for the whole fetch window), so a near-constant
        # synthetic output is the correct fit, not mode collapse. Skip flat_ratio
        # entirely here rather than penalizing the model for a data problem.
        reasons.append(
            "training data has no variance (real std < "
            f"{REAL_DATA_FLAT_STD:g}) - a constant synthetic output is the correct fit here, "
            "not mode collapse; retraining won't help, this consumer's fetched window needs "
            "more variance"
        )
    elif flat_ratio is not None:
        if flat_ratio >= FLAT_EXAMPLE_RATIO_DISCARD:
            reasons.append(f"{flat_ratio:.0%} of generated examples are near-constant (mode collapse)")
            severity = max(severity, 2)
        elif flat_ratio >= FLAT_EXAMPLE_RATIO_WARN:
            reasons.append(f"{flat_ratio:.0%} of generated examples are near-constant (mode collapse)")
            severity = max(severity, 1)

    if range_violation_ratio is not None:
        if range_violation_ratio >= RANGE_VIOLATION_RATIO_DISCARD:
            reasons.append(f"{range_violation_ratio:.0%} of generated values fall well outside the real data's range")
            severity = max(severity, 2)
        elif range_violation_ratio >= RANGE_VIOLATION_RATIO_WARN:
            reasons.append(f"{range_violation_ratio:.0%} of generated values fall outside the real data's range")
            severity = max(severity, 1)

    if median_shift is not None and not np.isnan(median_shift):
        if median_shift >= MEDIAN_SHIFT_DISCARD:
            reasons.append(
                f"synthetic values sit {median_shift:.1f} real IQRs away from the real "
                "median (wrong scale entirely)"
            )
            severity = max(severity, 2)
        elif median_shift >= MEDIAN_SHIFT_WARN:
            reasons.append(
                f"synthetic median is {median_shift:.1f} real IQRs off the real median"
            )
            severity = max(severity, 1)

    if iqr_ratio is not None and not np.isnan(iqr_ratio):
        low, high = IQR_RATIO_DISCARD
        warn_low, warn_high = IQR_RATIO_WARN
        if iqr_ratio <= low or iqr_ratio >= high:
            reasons.append(
                f"synthetic spread is {iqr_ratio:.2f}x the real IQR "
                f"({'over' if iqr_ratio >= high else 'under'}-dispersed)"
            )
            severity = max(severity, 2)
        elif iqr_ratio <= warn_low or iqr_ratio >= warn_high:
            reasons.append(f"synthetic spread is {iqr_ratio:.2f}x the real IQR")
            severity = max(severity, 1)

    if acf_mae is not None and not np.isnan(acf_mae):
        if acf_mae >= ACF_MAE_DISCARD:
            reasons.append(f"autocorrelation shape doesn't match real data (mean abs diff {acf_mae:.2f})")
            severity = max(severity, 2)
        elif acf_mae >= ACF_MAE_WARN:
            reasons.append(f"autocorrelation shape only loosely matches real data (mean abs diff {acf_mae:.2f})")
            severity = max(severity, 1)

    if real_data_flat and severity == 0:
        return "BAD_DATA", reasons
    return {0: "OK", 1: "WARN", 2: "DISCARD"}[severity], reasons


def diagnose_model(model_dir: Path, num_examples: int) -> dict:
    report: dict = {"model": model_dir.name, "path": str(model_dir)}

    try:
        train_result = load_train_result(str(model_dir))
    except Exception as e:
        report["verdict"] = "ERROR"
        report["reasons"] = [f"could not load model: {e}"]
        return report

    feature_cols = list(train_result.features)
    sequence_len = _sequence_len(
        model_dir, fallback=getattr(train_result.model, "max_sequence_len", 100)
    )
    lags = [lag for lag in ACF_LAGS if lag < sequence_len]

    synthetic_dfs = generate(train_result, num_examples=num_examples)
    synth_values = np.stack([df[feature_cols].to_numpy(dtype=float) for df in synthetic_dfs])
    report["num_examples"] = len(synthetic_dfs)

    # Worst-case (max) per-example, per-feature std, so an example only counts
    # as "flat" once every one of its features has collapsed - a genuinely
    # constant feature among several varying ones isn't mode collapse.
    per_example_std = synth_values.std(axis=1).max(axis=1)
    flat_examples = int((per_example_std < 1e-6).sum())
    flat_ratio = flat_examples / len(synthetic_dfs)
    report["flat_examples"] = flat_examples
    report["flat_ratio"] = flat_ratio

    synth_acf = _acf_per_sequence(synth_values, lags)

    range_violation_ratio = None
    acf_mae = None
    correlation_delta = None
    median_shift = None
    iqr_ratio = None
    real_data_flat = False

    real_csv = model_dir / "train_data.csv"
    if not real_csv.exists():
        report["real_data_note"] = (
            "no train_data.csv saved for this model (only the pretrainer CLI saves it); "
            "range and autocorrelation comparisons against real data are skipped."
        )
    else:
        real_df = pd.read_csv(real_csv)
        available_cols = [c for c in feature_cols if c in real_df.columns]
        if not available_cols:
            report["real_data_note"] = "train_data.csv has none of this model's feature columns; skipped."
        else:
            real_values = real_df[available_cols].to_numpy(dtype=float)
            real_seqs = _reshape_sequences(real_values, sequence_len)
            col_idx = [feature_cols.index(c) for c in available_cols]
            comparable_synth = synth_values[..., col_idx]

            real_data_flat = bool(np.all(real_values.std(axis=0) < REAL_DATA_FLAT_STD))
            report["real_data_flat"] = real_data_flat

            # Robust bounds: the 0.5/99.5 percentiles rather than min/max, so a
            # couple of extreme real values cannot widen the acceptable band to
            # the point where nothing can ever violate it.
            real_min = np.percentile(real_values, 0.5, axis=0)
            real_max = np.percentile(real_values, 99.5, axis=0)
            margin = (real_max - real_min) * RANGE_MARGIN_RATIO
            out_of_range = (comparable_synth < real_min - margin) | (comparable_synth > real_max + margin)
            range_violation_ratio = float(out_of_range.mean())

            real_acf = _acf_per_sequence(real_seqs, lags)
            if lags and not np.all(np.isnan(real_acf)):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    acf_mae = float(np.nanmean(np.abs(real_acf - synth_acf)))

            correlation_delta = _correlation_delta(real_seqs, comparable_synth)

            flat_synth = comparable_synth.reshape(-1, len(available_cols))

            # Robust location/spread comparison, averaged over features.
            real_median = np.median(real_values, axis=0)
            synth_median = np.median(flat_synth, axis=0)
            real_iqr = np.subtract(*np.percentile(real_values, [75, 25], axis=0))
            synth_iqr = np.subtract(*np.percentile(flat_synth, [75, 25], axis=0))
            with np.errstate(divide="ignore", invalid="ignore"):
                shifts = np.abs(synth_median - real_median) / real_iqr
                ratios = synth_iqr / real_iqr
            usable = real_iqr > REAL_DATA_FLAT_IQR
            if usable.any():
                median_shift = float(np.nanmean(shifts[usable]))
                iqr_ratio = float(np.nanmean(ratios[usable]))
            report["real_median"] = dict(zip(available_cols, real_median.tolist()))
            report["synth_median"] = dict(zip(available_cols, synth_median.tolist()))
            report["real_iqr"] = dict(zip(available_cols, real_iqr.tolist()))
            report["synth_iqr"] = dict(zip(available_cols, synth_iqr.tolist()))
            report["real_mean"] = dict(zip(available_cols, real_values.mean(axis=0).tolist()))
            report["synth_mean"] = dict(zip(available_cols, flat_synth.mean(axis=0).tolist()))
            report["real_std"] = dict(zip(available_cols, real_values.std(axis=0).tolist()))
            report["synth_std"] = dict(zip(available_cols, flat_synth.std(axis=0).tolist()))
            report["lags"] = lags
            report["real_acf"] = real_acf.tolist()
            report["synth_acf"] = synth_acf.tolist()

    report["range_violation_ratio"] = range_violation_ratio
    report["acf_mae"] = acf_mae
    report["feature_correlation_delta"] = correlation_delta
    report["median_shift"] = median_shift
    report["iqr_ratio"] = iqr_ratio

    verdict, reasons = _verdict(
        flat_ratio, range_violation_ratio, acf_mae, real_data_flat, median_shift, iqr_ratio
    )
    report["verdict"] = verdict
    report["reasons"] = reasons
    return report


_AVERAGED_METRICS = (
    "flat_ratio",
    "range_violation_ratio",
    "acf_mae",
    "feature_correlation_delta",
    "median_shift",
    "iqr_ratio",
)


def diagnose_model_averaged(model_dir: Path, num_examples: int, runs: int) -> dict:
    """Run diagnose_model `runs` times and average the noisy scalar metrics.

    Each run draws a fresh, independent sample of `num_examples` synthetic
    sequences, so run-to-run values (especially range_violation_ratio, which
    can hinge on how spread-out a handful of examples happen to be) carry
    real sampling noise on top of any actual model-quality signal. Averaging
    over several runs is equivalent to one big run of num_examples*runs
    examples for these particular metrics (they're all plain means), but
    doing it as repeated runs also surfaces the run-to-run std, which is
    useful on its own for judging how much a single number can be trusted.
    """
    run_reports = [diagnose_model(model_dir, num_examples) for _ in range(runs)]

    if run_reports[0].get("verdict") == "ERROR":
        return run_reports[0]

    report = dict(run_reports[-1])  # descriptive fields (real_mean, acf vectors, ...) from the last run
    report["runs"] = runs
    for metric in _AVERAGED_METRICS:
        values = [r[metric] for r in run_reports if r.get(metric) is not None and not (
            isinstance(r[metric], float) and np.isnan(r[metric])
        )]
        if values:
            report[metric] = float(np.mean(values))
            report[f"{metric}_std"] = float(np.std(values))
        else:
            report[metric] = None
            report[f"{metric}_std"] = None

    verdict, reasons = _verdict(
        report["flat_ratio"], report["range_violation_ratio"], report["acf_mae"],
        report.get("real_data_flat", False),
        report.get("median_shift"), report.get("iqr_ratio"),
    )
    report["verdict"] = verdict
    report["reasons"] = reasons
    return report


def _print_summary(reports: list[dict]) -> None:
    def _fmt(r: dict, metric: str) -> str | None:
        value = r.get(metric)
        if value is None:
            return None
        std = r.get(f"{metric}_std")
        if std is None or r.get("runs", 1) <= 1:
            return f"{value:.3f}"
        return f"{value:.3f}±{std:.3f}"

    summary_df = pd.DataFrame(
        [
            {
                "model": r["model"],
                "verdict": r["verdict"],
                "flat_ratio": _fmt(r, "flat_ratio"),
                "range_violation": _fmt(r, "range_violation_ratio"),
                "acf_mae": _fmt(r, "acf_mae"),
                "median_shift": _fmt(r, "median_shift"),
                "iqr_ratio": _fmt(r, "iqr_ratio"),
                "feature_corr_delta": _fmt(r, "feature_correlation_delta"),
            }
            for r in reports
        ]
    )
    pd.set_option("display.width", 120)
    print(summary_df.to_string(index=False))

    for r in reports:
        if r.get("reasons"):
            print(f"\n{r['model']} [{r['verdict']}]:")
            for reason in r["reasons"]:
                print(f"  - {reason}")
        if r.get("real_data_note"):
            print(f"\n{r['model']}: {r['real_data_note']}")

    discard = [r["model"] for r in reports if r["verdict"] == "DISCARD"]
    if discard:
        print(f"\nRecommended to discard/retrain: {', '.join(discard)}")

    bad_data = [r["model"] for r in reports if r["verdict"] == "BAD_DATA"]
    if bad_data:
        print(
            f"\nTraining data itself has no variance, retraining won't help - "
            f"re-fetch a richer window instead: {', '.join(bad_data)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose pretrained synthetic models: generate sample output from "
        "each and compare it against its real training data (where available) on "
        "value range, autocorrelation, and cross-feature correlation, flagging models "
        "worth discarding or retraining."
    )
    parser.add_argument(
        "--model-root",
        default=str(CLI_DIR / "models"),
        help="Root folder to scan recursively for trained models (model.pt files).",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=20,
        help="Number of synthetic examples to generate per model for the diagnostic. "
        "Generation is stochastic and unseeded by default, so metrics like "
        "range_violation_ratio have run-to-run noise at low values - raise this "
        "for a steadier estimate, or set --seed for reproducible comparisons "
        "across runs (e.g. before/after a hyperparameter change).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Fix the PyTorch RNG seed before generating, so repeated runs (or runs "
        "compared before/after a config change) produce the same synthetic samples "
        "instead of a fresh random draw each time.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Diagnose each model this many times (independent samples of --num-examples "
        "each) and report the mean ± std of the noisy metrics. Use this instead of just "
        "raising --num-examples when you also want to see how much a single run's "
        "numbers can be trusted, e.g. --runs 10.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Diagnose a single model folder name instead of every model under --model-root.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write the full JSON report. Defaults to <model-root>/diagnostics_report.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
    model_root = Path(args.model_root).resolve()
    models = _find_models(model_root)
    if args.model:
        models = [m for m in models if m.name == args.model]
        if not models:
            raise SystemExit(f"No model named {args.model!r} found under {model_root}")
    if not models:
        raise SystemExit(f"No trained models (model.pt) found under {model_root}")

    reports = []
    for model_dir in models:
        print(f"Diagnosing {model_dir.name} ({args.runs} run(s)) ...", flush=True)
        reports.append(diagnose_model_averaged(model_dir, args.num_examples, args.runs))

    print()
    _print_summary(reports)

    output_path = Path(args.output) if args.output else model_root / "diagnostics_report.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "models": reports}, f, indent=2
        )
    print(f"\nFull report saved to {output_path}")


if __name__ == "__main__":
    main()
