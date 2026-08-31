import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

CLI_DIR = Path(__file__).resolve().parent
SYNTHETIC_DATA_DIR = CLI_DIR.parent / "synthetic_data_generation"
sys.path.insert(0, str(SYNTHETIC_DATA_DIR))

from dgan_wrapper import generate, load_train_result  # noqa: E402


def _model_path(model_root: Path, model_name: str, namespace: str | None) -> Path:
    if namespace:
        return model_root / namespace / model_name
    return model_root / model_name


# Categorical slots 1 (blue) and 8 (orange) from the platform's validated palette.
_REAL_COLOR = "#2a78d6"
_SYNTHETIC_COLOR = "#eb6834"
_GRID_COLOR = "#e1e0d9"
_TEXT_COLOR = "#0b0b0b"
_MUTED_COLOR = "#898781"


def _plot_comparison(
    model_name: str,
    real_csv: Path,
    combined_df: pd.DataFrame,
    example_id: int,
    output_path: Path,
) -> None:
    """Save a PNG comparing the real training series to one generated example.

    The generated data's timestamps are copied from the first `sequence_len`
    rows of the real training data (see dgan_wrapper.train), so slicing the
    real series to the same length gives a like-for-like, same-time-axis
    comparison rather than two unrelated date ranges.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    real_df = pd.read_csv(real_csv, parse_dates=["datetime"])
    synth_df = combined_df[combined_df["example_id"] == example_id].copy()
    synth_df["datetime"] = pd.to_datetime(synth_df["datetime"])
    real_window = real_df.head(len(synth_df))

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)
    ax.plot(
        real_window["datetime"],
        real_window["value"],
        color=_REAL_COLOR,
        linewidth=2,
        solid_capstyle="round",
        solid_joinstyle="round",
        label="Real",
    )
    ax.plot(
        synth_df["datetime"],
        synth_df["value"],
        color=_SYNTHETIC_COLOR,
        linewidth=2,
        solid_capstyle="round",
        solid_joinstyle="round",
        label="Synthetic",
    )

    ax.set_title(f"{model_name}: real vs. synthetic", color=_TEXT_COLOR, fontsize=12)
    ax.set_ylabel("value", color=_TEXT_COLOR)
    ax.tick_params(colors=_MUTED_COLOR)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_GRID_COLOR)
    ax.grid(True, color=_GRID_COLOR, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    legend = ax.legend(frameon=False, loc="upper right")
    for text in legend.get_texts():
        text.set_color(_TEXT_COLOR)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, facecolor="#fcfcfb")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic data from an already-trained model and "
        "save the result as a CSV inside that model's own folder, without "
        "going through the synthetic_data_generation API/interface."
    )
    parser.add_argument(
        "model_name", help="Name of the pretrained model (its folder name, e.g. a shelly_id)."
    )
    parser.add_argument(
        "--namespace",
        default=None,
        help="Optional namespace the model was trained under (models/<namespace>/<model_name>).",
    )
    parser.add_argument(
        "--model-root",
        default=str(CLI_DIR / "models"),
        help="Root folder where trained models are stored.",
    )
    parser.add_argument(
        "--num-examples", type=int, default=1, help="Number of examples to generate."
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV filename. Defaults to generated_<timestamp>.csv inside the model's folder.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Also save a PNG comparing the real training data to one generated "
        "example, for use in documentation.",
    )
    parser.add_argument(
        "--plot-example",
        type=int,
        default=1,
        help="Which example_id to plot against the real data (default: 1).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_root = Path(args.model_root).resolve()
    model_path = _model_path(model_root, args.model_name, args.namespace)

    if not (model_path / "model.pt").exists():
        raise SystemExit(f"No trained model found at {model_path} (missing model.pt).")

    print(f"Loading model from {model_path} ...", flush=True)
    train_result = load_train_result(str(model_path))

    # Physical bounds recorded at training time. Without them the generator,
    # whose output layer is unbounded, emits negative average power for a
    # counter that cannot run backwards.
    floor = ceiling = None
    metadata_path = model_path / "metadata.json"
    if metadata_path.exists():
        with metadata_path.open(encoding="utf-8") as f:
            metadata = json.load(f)
        floor, ceiling = metadata.get("value_floor"), metadata.get("value_ceiling")

    print(f"Generating {args.num_examples} example(s) ...", flush=True)
    synthetic_dfs = generate(
        train_result,
        num_examples=args.num_examples,
        clip_min=floor,
        clip_max=ceiling,
    )

    # Match the shape of the synthetic_data_generation API's CSV export.
    for i, df in enumerate(synthetic_dfs):
        df["example_id"] = i + 1
    combined_df = pd.concat(synthetic_dfs, ignore_index=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_name = args.output or f"generated_{timestamp}.csv"
    output_path = model_path / output_name
    combined_df.to_csv(output_path, index=False)
    print(f"Saved {len(combined_df)} rows to {output_path}", flush=True)

    if args.plot:
        real_csv = model_path / "train_data.csv"
        if not real_csv.exists():
            print(
                f"Warning: {real_csv} not found, skipping comparison plot. "
                "(Only models trained after the train_data.csv change have it.)",
                flush=True,
            )
        elif args.plot_example not in combined_df["example_id"].unique():
            print(
                f"Warning: example_id={args.plot_example} was not generated "
                f"(--num-examples={args.num_examples}), skipping comparison plot.",
                flush=True,
            )
        else:
            plot_path = model_path / f"comparison_{timestamp}.png"
            _plot_comparison(
                args.model_name, real_csv, combined_df, args.plot_example, plot_path
            )
            print(f"Saved comparison plot to {plot_path}", flush=True)


if __name__ == "__main__":
    main()
