import os
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel

from doppelganger import DGTorch, OutputType, prepare_data

# Common names used for the time index column across CSV sources.
_INDEX_COL_ALIASES = ("datetime", "timestamp")


class TrainResult(BaseModel):
    model: DGTorch
    features: pd.Series
    indices: Optional[pd.Series] = None
    # Names of the per-sequence attribute columns the model was conditioned on,
    # in the order prepare_data saw them. Empty for models trained on the dummy
    # zero attribute, which carries no information.
    attribute_names: Optional[List[str]] = None

    class Config:
        arbitrary_types_allowed = True


def train(
    train_df: pd.DataFrame,
    sequence_len: int = 10,
    sample_len: int = 1,
    index_col: Optional[str] = "timestamp",
    batch_size: int = 1000,
    epochs: int = 15000,
    progress_callback: Optional[callable] = None,
    attribute_df: Optional[pd.DataFrame] = None,
    **model_kwargs,
) -> TrainResult:
    """
    Trains the DoppelGANger model using the PyTorch implementation.

    Extra keyword arguments are forwarded straight to DGTorch, so training
    hyperparameters that aren't part of this wrapper's own signature can be
    tuned per-run without editing this file. The ones that matter for an
    unstable/over-dispersed generator (synthetic std well above the real
    data's) are `gradient_penalty_coef` (default 10.0 - raise to damp a
    generator that produces out-of-range spikes), `generator_learning_rate`
    and `discriminator_learning_rate` (both default 1e-3), and the network
    size knobs `feature_num_units` / `feature_num_layers`.
    """
    if sequence_len % sample_len != 0:
        raise ValueError(
            f"sample_len {sample_len} is not a divisor of sequence_len {sequence_len}"
        )

    if index_col is not None and index_col not in train_df.columns:
        fallback = next(
            (
                alias
                for alias in _INDEX_COL_ALIASES
                if alias != index_col and alias in train_df.columns
            ),
            None,
        )
        if fallback is None:
            raise ValueError(f"index_col {index_col} not found in train_df")
        index_col = fallback

    feature_df = train_df.drop(columns=[index_col]) if index_col else train_df
    features_np = feature_df.to_numpy()

    # Reshape data into sequences
    n_sequences = features_np.shape[0] // sequence_len

    if n_sequences < 2:
        raise ValueError(
            f"Not enough data! The uploaded file allows for only {n_sequences} sequence(s) "
            f"of length {sequence_len}. The model requires at least 2 sequences to train. "
            f"Please upload a longer CSV (min {sequence_len * 2} rows) or reduce the sequence_len."
        )

    features_reshaped = features_np[: (n_sequences * sequence_len)].reshape(
        n_sequences, sequence_len, features_np.shape[1]
    )

    # Determine feature types
    feature_types = []
    for col in feature_df.columns:
        if pd.api.types.is_numeric_dtype(feature_df[col]):
            feature_types.append(OutputType.CONTINUOUS)
        else:
            feature_types.append(OutputType.DISCRETE)

    # Per-sequence attributes. DoppelGANger conditions the feature generator on
    # these (Generator.forward concatenates them onto the LSTM input at every
    # timestep), so real values here are what makes conditional generation
    # possible. `attribute_df` is given per row, aligned with train_df; the
    # first row of each sequence supplies that sequence's value, which is only
    # meaningful when sequences are day-aligned.
    #
    # With no attributes supplied DoppelGANger still requires at least one, so
    # fall back to a constant zero column - it carries no information and the
    # model is then unconditional.
    attribute_names = None
    if attribute_df is not None and not attribute_df.empty:
        attrs_np = attribute_df.to_numpy()[: (n_sequences * sequence_len)]
        attrs_np = attrs_np.reshape(n_sequences, sequence_len, -1)[:, 0, :]
        attribute_types = [
            OutputType.CONTINUOUS
            if pd.api.types.is_float_dtype(attribute_df[col])
            else OutputType.DISCRETE
            for col in attribute_df.columns
        ]
        attribute_names = list(attribute_df.columns)
    else:
        attrs_np = np.zeros((features_reshaped.shape[0], 1))
        attribute_types = [OutputType.CONTINUOUS]

    dg_data = prepare_data(
        original_attributes=attrs_np,
        attribute_types=attribute_types,
        original_features=features_reshaped,
        feature_types=feature_types,
    )

    # Instantiate and train the model
    model = DGTorch(
        attribute_outputs=dg_data.attribute_outputs,
        additional_attribute_outputs=dg_data.additional_attribute_outputs,
        feature_outputs=dg_data.feature_outputs,
        max_sequence_len=sequence_len,
        sample_len=sample_len,
        cuda=torch.cuda.is_available(),  # Use GPU if available
        **model_kwargs,
    )

    dataset_tensors = [torch.from_numpy(dg_data.attributes).float()]
    if dg_data.additional_attributes is not None:
        dataset_tensors.append(torch.from_numpy(dg_data.additional_attributes).float())
    dataset_tensors.append(torch.from_numpy(dg_data.features).float())

    dataset = torch.utils.data.TensorDataset(*dataset_tensors)

    model.train(
        dataset,
        # Ensure batch_size is at least 2, but not larger than total sequences
        batch_size=max(2, min(batch_size, features_reshaped.shape[0])),
        num_epochs=epochs,
        progress_callback=progress_callback,
    )

    indices = train_df[index_col].head(sequence_len) if index_col else None

    return TrainResult(
        model=model,
        features=feature_df.columns.to_series(),
        indices=indices,
        attribute_names=attribute_names,
    )


def _extend_index(indices: pd.Series, n_steps: int) -> pd.Series:
    """A time axis of `n_steps`, continuing indices' own start and spacing.

    indices holds the first max_sequence_len timestamps of the training data,
    so it is exactly one trained sequence long. A longer rollout needs more
    labels than that; rather than repeat or truncate, continue the same
    regular grid.
    """
    stamps = pd.to_datetime(indices, utc=True)
    if len(stamps) < 2:
        return indices
    step = stamps.iloc[1] - stamps.iloc[0]
    extended = pd.date_range(stamps.iloc[0], periods=n_steps, freq=step)
    return pd.Series(extended.astype(str), name=indices.name)


def generate(
    train_result: TrainResult,
    num_examples: int = 1,
    horizon_steps: Optional[int] = None,
    clip_min: Optional[float] = None,
    clip_max: Optional[float] = None,
) -> List[pd.DataFrame]:
    """Generate synthetic sequences.

    `horizon_steps` asks for a rollout longer (or shorter) than the length the
    model was trained on. The generator is an LSTM and rolls out to any
    multiple of sample_len, but quality degrades with distance from the trained
    length - see each model's metadata "capabilities" block for which horizons
    were measured to hold up.

    `clip_min`/`clip_max` enforce a physical bound the generator does not know
    about. Its output layer is unbounded, so for a quantity with a hard floor -
    average power from a cumulative *import* register, which cannot run
    backwards - it happily emits negative values (measured at ~14% of samples,
    up to 28% on some models). Clipping removes the impossible values but does
    not fix the cause: the negatives are the left tail of an over-dispersed
    distribution, so a model producing many of them is over-dispersed whether
    or not it is clipped. capabilities.py records the pre-clip negative_ratio
    for that reason.
    """
    # Request at least 2 examples to avoid BatchNorm issues with batch size of 1
    num_to_generate = max(num_examples, 2)

    trained_len = train_result.model.max_sequence_len
    # We ignore the first return value (generated attributes) as it corresponds to our dummy attribute
    if horizon_steps is None or horizon_steps == trained_len:
        _, synthetic_features = train_result.model.generate(num_to_generate)
    else:
        _, synthetic_features = train_result.model.generate_horizon(
            num_to_generate, horizon_steps
        )

    # Trim to the requested number of examples
    synthetic_features = synthetic_features[:num_examples]

    if clip_min is not None or clip_max is not None:
        synthetic_features = np.clip(synthetic_features, clip_min, clip_max)

    synthetic_dfs = []
    for single_example_features in synthetic_features:
        synthetic_df = pd.DataFrame(
            single_example_features,
            columns=train_result.features,
        )
        if train_result.indices is not None:
            index_name = train_result.indices.name
            n_steps = len(single_example_features)
            index_values = (
                train_result.indices
                if n_steps == len(train_result.indices)
                else _extend_index(train_result.indices, n_steps)
            )
            synthetic_df[index_name] = index_values.to_numpy()
            synthetic_df = synthetic_df[[index_name] + list(train_result.features)]
        synthetic_dfs.append(synthetic_df)

    return synthetic_dfs


def save_train_result(train_result: TrainResult, folder_path: str):
    os.makedirs(folder_path, exist_ok=True)

    # Save model using torch.save
    model_path = os.path.join(folder_path, "model.pt")
    torch.save(train_result.model, model_path)

    # Save feature column names
    features_path = os.path.join(folder_path, "features.csv")
    train_result.features.to_csv(features_path, index=False, header=False)

    if train_result.attribute_names:
        with open(os.path.join(folder_path, "attributes.csv"), "w", encoding="utf-8") as f:
            f.write("\n".join(train_result.attribute_names) + "\n")

    # Save indices (e.g., timestamps)
    if train_result.indices is not None:
        indices_path = os.path.join(folder_path, "indices.csv")
        train_result.indices.to_csv(indices_path, index=False, header=True)


def load_train_result(path: str) -> TrainResult:
    model_path = os.path.join(path, "model.pt")
    features_path = os.path.join(path, "features.csv")
    indices_path = os.path.join(path, "indices.csv")

    map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # weights_only defaults to True as of PyTorch 2.6, which can't deserialize our
    # custom DGTorch/Generator classes. These are our own trained model files, not
    # third-party pickles, so full unpickling is safe here.
    model = torch.load(model_path, map_location=map_location, weights_only=False)
    # model.device is a plain string baked in at training time (e.g. "cuda" on a
    # GPU box); map_location only remaps tensor storages, so it's left stale.
    # Realign it and the model's own tensors with where they actually landed.
    model.device = str(map_location)
    model.to(model.device)
    features = pd.read_csv(features_path, header=None, index_col=None).squeeze(
        "columns"
    )

    indices = None
    if os.path.exists(indices_path):
        indices = pd.read_csv(indices_path, index_col=None).squeeze("columns")

    attribute_names = None
    attributes_path = os.path.join(path, "attributes.csv")
    if os.path.exists(attributes_path):
        with open(attributes_path, encoding="utf-8") as f:
            attribute_names = [line.strip() for line in f if line.strip()]

    return TrainResult(
        model=model, features=features, indices=indices, attribute_names=attribute_names
    )
