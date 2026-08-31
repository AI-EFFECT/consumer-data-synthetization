"""
Doppelganger implementation in PyTorch.
Based on https://github.com/gretelai/doppelganger-torch
"""

from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import List, Union

import numpy as np
import torch
from torch.utils import tensorboard


class ProgressInfo:
    def __init__(self, epoch, total_epochs, batch, total_batches):
        self.epoch = epoch
        self.total_epochs = total_epochs
        self.batch = batch
        self.total_batches = total_batches


def autocovariance(a):
    if len(a.shape) == 3:
        if a.shape[2] != 1:
            raise RuntimeError(
                f"Unexpected shape={a.shape} for autocovariance calculation."
            )
        a = a.reshape(a.shape[0], a.shape[1])
    s = np.zeros(2 * a.shape[1] - 1)
    for i in range(a.shape[0]):
        ts = (
            a[i, :]
            - a[
                i,
                :,
            ].mean()
        )
        ss = np.correlate(ts, ts, mode="full")
        s += ss
    s = s[s.size // 2 :]
    counts = np.arange(len(s), 0, -1) * a.shape[0]
    s /= counts
    s /= a.var()
    return s


class OutputType(str, Enum):
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"


class Normalization(str, Enum):
    ZERO_ONE = "zero_one"
    MINUSONE_ONE = "minusone_one"


@dataclass(frozen=True)
class Output:
    name: str

    def get_dim(self):
        return 1


@dataclass(frozen=True)
class DiscreteOutput(Output):
    dim: int

    def get_dim(self):
        return self.dim


@dataclass(frozen=True)
class ContinuousOutput(Output):
    normalization: Normalization
    global_min: float
    global_max: float
    is_feature_normalized: bool
    is_example_normalized: bool


def prepare_data(
    original_attributes: np.ndarray,
    attribute_types: List[OutputType],
    original_features: np.ndarray,
    feature_types: List[OutputType],
    normalization: Normalization = Normalization.ZERO_ONE,
    is_feature_normalized: bool = True,
    is_example_normalized: bool = True,
):
    def make_output(
        index: int,
        t: OutputType,
        data: np.ndarray,
        is_feature_normalized: bool,
        is_example_normalized: bool,
    ) -> Output:
        if t == OutputType.CONTINUOUS:
            output = ContinuousOutput(
                name="a" + str(index),
                normalization=normalization,
                global_min=np.min(data),
                global_max=np.max(data),
                is_feature_normalized=is_feature_normalized,
                is_example_normalized=is_example_normalized,
            )
        elif t == OutputType.DISCRETE:
            output = DiscreteOutput(
                name="a" + str(index), dim=1 + np.int32(np.max(data))
            )
        else:
            raise RuntimeError(f"Unknown output type={t}")
        return output

    attribute_outputs = [
        make_output(
            index,
            t,
            original_attributes[:, index],
            is_feature_normalized=is_feature_normalized,
            is_example_normalized=False,
        )
        for index, t in enumerate(attribute_types)
    ]
    feature_outputs = [
        make_output(
            index,
            t,
            original_features[:, :, index],
            is_feature_normalized=is_feature_normalized,
            is_example_normalized=is_example_normalized,
        )
        for index, t in enumerate(feature_types)
    ]
    attributes, _, _ = transform(
        original_attributes, attribute_outputs, variable_dim_index=1
    )
    features, additional_attribute_outputs, additional_attributes = transform(
        original_features, feature_outputs, variable_dim_index=2
    )
    return DGData(
        attributes=attributes,
        additional_attributes=additional_attributes,
        features=features,
        attribute_outputs=attribute_outputs,
        additional_attribute_outputs=additional_attribute_outputs,
        feature_outputs=feature_outputs,
    )


def rescale(
    original: np.ndarray,
    normalization: Normalization,
    global_min: Union[float, np.ndarray],
    global_max: Union[float, np.ndarray],
) -> np.ndarray:
    range_val = global_max - global_min
    if isinstance(range_val, np.ndarray):
        range_val[range_val == 0] = 1e-6
    elif range_val == 0:
        range_val = 1e-6
    if normalization == Normalization.ZERO_ONE:
        return (original - global_min) / range_val
    elif normalization == Normalization.MINUSONE_ONE:
        return (2.0 * (original - global_min) / range_val) - 1.0


def rescale_inverse(
    transformed: np.ndarray,
    normalization: Normalization,
    global_min: Union[float, np.ndarray],
    global_max: Union[float, np.ndarray],
) -> np.ndarray:
    if normalization == Normalization.ZERO_ONE:
        return transformed * (global_max - global_min) + global_min
    elif normalization == Normalization.MINUSONE_ONE:
        return ((transformed + 1) / 2) * (global_max - global_min) + global_min


def transform(
    original_data: np.ndarray, outputs: List[Output], variable_dim_index: int
):
    additional_attribute_outputs = []
    additional_attribute_parts = []
    parts = []
    for index, output in enumerate(outputs):
        if isinstance(output, DiscreteOutput):
            if variable_dim_index == 1:
                indices = original_data[:, index].astype(int)
            elif variable_dim_index == 2:
                indices = original_data[:, :, index].astype(int)
            else:
                raise RuntimeError(
                    f"Unsupported variable_dim_index={variable_dim_index}"
                )
            if variable_dim_index == 1:
                b = np.zeros((len(indices), output.dim))
                b[np.arange(len(indices)), indices] = 1
            elif variable_dim_index == 2:
                b = np.zeros((indices.shape[0], indices.shape[1], output.dim))

                def all_idx(idx, axis):
                    grid = np.ogrid[tuple(map(slice, idx.shape))]
                    grid.insert(axis, idx)
                    return tuple(grid)

                b[all_idx(indices, axis=2)] = 1
            parts.append(b)
        elif isinstance(output, ContinuousOutput):
            if variable_dim_index == 1:
                raw = original_data[:, index]
            elif variable_dim_index == 2:
                raw = original_data[:, :, index]
            else:
                raise RuntimeError(
                    f"Unsupported variable_dim_index={variable_dim_index}"
                )
            if output.is_feature_normalized:
                feature_scaled = rescale(
                    raw, output.normalization, output.global_min, output.global_max
                )
            else:
                feature_scaled = raw
            if output.is_example_normalized:
                if variable_dim_index != 2:
                    raise RuntimeError(
                        "is_example_normalized only applies to features where the data has 3 dimensions"
                    )
                mins = np.min(feature_scaled, axis=1)
                maxes = np.max(feature_scaled, axis=1)
                additional_attribute_outputs.append(
                    ContinuousOutput(
                        name=output.name + "_midpoint",
                        normalization=output.normalization,
                        global_min=0.0
                        if output.normalization == Normalization.ZERO_ONE
                        else -1.0,
                        global_max=1.0,
                        is_feature_normalized=False,
                        is_example_normalized=False,
                    )
                )
                additional_attribute_outputs.append(
                    ContinuousOutput(
                        name=output.name + "_half_range",
                        normalization=Normalization.ZERO_ONE,
                        global_min=0.0,
                        global_max=1.0,
                        is_feature_normalized=False,
                        is_example_normalized=False,
                    )
                )
                additional_attribute_parts.append(
                    ((mins + maxes) / 2).reshape(mins.shape[0], 1)
                )
                additional_attribute_parts.append(
                    ((maxes - mins) / 2).reshape(mins.shape[0], 1)
                )
                mins = np.broadcast_to(
                    mins.reshape(mins.shape[0], 1),
                    (mins.shape[0], feature_scaled.shape[1]),
                )
                maxes = np.broadcast_to(
                    maxes.reshape(maxes.shape[0], 1),
                    (mins.shape[0], feature_scaled.shape[1]),
                )
                scaled = rescale(feature_scaled, output.normalization, mins, maxes)
            else:
                scaled = feature_scaled
            if variable_dim_index == 1:
                scaled = scaled.reshape(original_data.shape[0], 1)
            elif variable_dim_index == 2:
                scaled = scaled.reshape(
                    original_data.shape[0], original_data.shape[1], 1
                )
            parts.append(scaled)
        else:
            raise RuntimeError(f"Unsupported output type, class={type(output)}'")
    additional_attributes = None
    if additional_attribute_parts:
        additional_attributes = np.concatenate(additional_attribute_parts, axis=1)
    return (
        np.concatenate(parts, axis=variable_dim_index),
        additional_attribute_outputs,
        additional_attributes,
    )


def inverse_transform(
    transformed_data: np.ndarray,
    outputs: List[Output],
    variable_dim_index: int,
    additional_attributes: np.ndarray = None,
    additional_attribute_outputs: List[Output] = None,
):
    parts = []
    transformed_index = 0
    additional_attribute_index = 0
    for index, output in enumerate(outputs):
        if isinstance(output, DiscreteOutput):
            if variable_dim_index == 1:
                onehot = transformed_data[
                    :, transformed_index : (transformed_index + output.dim)
                ]
            elif variable_dim_index == 2:
                onehot = transformed_data[
                    :, :, transformed_index : (transformed_index + output.dim)
                ]
            else:
                raise RuntimeError(
                    f"Unsupported variable_dim_index={variable_dim_index}"
                )
            indices = np.argmax(onehot, axis=variable_dim_index)
            target_shape = list(transformed_data.shape)
            target_shape[-1] = 1
            indices = indices.reshape(target_shape)
            parts.append(indices)
            transformed_index += output.dim
        elif isinstance(output, ContinuousOutput):
            if variable_dim_index == 1:
                transformed = transformed_data[:, transformed_index]
            elif variable_dim_index == 2:
                transformed = transformed_data[:, :, transformed_index]
            else:
                raise RuntimeError(
                    f"Unsupported variable_dim_index={variable_dim_index}"
                )
            if output.is_example_normalized:
                if variable_dim_index != 2:
                    raise RuntimeError(
                        "is_example_normalized only applies to features where the data has 3 dimensions"
                    )
                if (
                    additional_attributes is None
                    or additional_attribute_outputs is None
                ):
                    raise RuntimeError(
                        "Must provide additional_attributes and additional_attribute_outputs if is_example_normalized=True"
                    )
                midpoint = additional_attributes[:, additional_attribute_index]
                half_range = additional_attributes[:, additional_attribute_index + 1]
                additional_attribute_index += 2
                mins = midpoint - half_range
                maxes = midpoint + half_range
                mins = np.expand_dims(mins, 1)
                maxes = np.expand_dims(maxes, 1)
                example_scaled = rescale_inverse(
                    transformed,
                    normalization=output.normalization,
                    global_min=mins,
                    global_max=maxes,
                )
            else:
                example_scaled = transformed
            if output.is_feature_normalized:
                original = rescale_inverse(
                    example_scaled,
                    output.normalization,
                    output.global_min,
                    output.global_max,
                )
            else:
                original = example_scaled
            target_shape = list(transformed_data.shape)
            target_shape[-1] = 1
            original = original.reshape(target_shape)
            parts.append(original)
            transformed_index += 1
        else:
            raise RuntimeError(f"Unsupported output type, class={type(output)}'")
    return np.concatenate(parts, axis=variable_dim_index)


@dataclass(frozen=True)
class DGData:
    attributes: np.ndarray
    additional_attributes: Union[np.ndarray, None]
    features: np.ndarray
    attribute_outputs: List[Output]
    additional_attribute_outputs: List[Output]
    feature_outputs: List[Output]


class Merger(torch.nn.Module):
    def __init__(self, modules, dim_index: int):
        super(Merger, self).__init__()
        if isinstance(modules, torch.nn.ModuleList):
            self.layers = modules
        else:
            self.layers = torch.nn.ModuleList(modules)
        self.dim_index = dim_index

    def forward(self, input):
        return torch.cat([m(input) for m in self.layers], dim=self.dim_index)


class OutputDecoder(torch.nn.Module):
    def __init__(self, input_dim: int, outputs: List[Output], dim_index: int):
        super(OutputDecoder, self).__init__()
        if outputs is None or len(outputs) == 0:
            raise RuntimeError("OutputDecoder received no outputs")
        self.dim_index = dim_index
        self.generators = torch.nn.ModuleList()
        for output in outputs:
            if isinstance(output, DiscreteOutput):
                self.generators.append(
                    torch.nn.Sequential(
                        OrderedDict(
                            [
                                (
                                    "linear",
                                    torch.nn.Linear(input_dim, output.get_dim()),
                                ),
                                ("softmax", torch.nn.Softmax(dim=dim_index)),
                            ]
                        )
                    )
                )
            elif isinstance(output, ContinuousOutput):
                if output.normalization == Normalization.ZERO_ONE:
                    normalizer = torch.nn.Sigmoid()
                elif output.normalization == Normalization.MINUSONE_ONE:
                    normalizer = torch.nn.Tanh()
                else:
                    raise RuntimeError(
                        f"Unsupported normalization='{output.normalization}'"
                    )
                self.generators.append(
                    torch.nn.Sequential(
                        OrderedDict(
                            [
                                (
                                    "linear",
                                    torch.nn.Linear(input_dim, output.get_dim()),
                                ),
                                ("normalization", normalizer),
                            ]
                        )
                    )
                )
            else:
                raise RuntimeError(f"Unsupported output type, class={type(output)}'")

    def forward(self, input):
        outputs = [generator(input) for generator in self.generators]
        merged = torch.cat(outputs, dim=self.dim_index)
        return merged


class SelectLastCell(torch.nn.Module):
    def forward(self, x):
        out, _ = x
        return out


class Generator(torch.nn.Module):
    def __init__(
        self,
        attribute_outputs: List[Output],
        additional_attribute_outputs: Union[List[Output], None],
        feature_outputs: List[Output],
        max_sequence_len: int,
        sample_len: int,
        attribute_noise_dim: int,
        feature_noise_dim: int,
        attribute_num_units: int,
        attribute_num_layers: int,
        feature_num_units: int,
        feature_num_layers: int,
    ):
        super(Generator, self).__init__()
        assert max_sequence_len % sample_len == 0
        self.sample_len = sample_len
        self.max_sequence_len = max_sequence_len
        self.attribute_gen = self._make_attribute_generator(
            attribute_outputs,
            attribute_noise_dim,
            attribute_num_units,
            attribute_num_layers,
        )
        attribute_dim = sum(output.get_dim() for output in attribute_outputs)
        if additional_attribute_outputs:
            self.additional_attribute_gen = self._make_attribute_generator(
                additional_attribute_outputs,
                attribute_noise_dim + attribute_dim,
                attribute_num_units,
                attribute_num_layers,
            )
            additional_attribute_dim = sum(
                output.get_dim() for output in additional_attribute_outputs
            )
        else:
            self.additional_attribute_gen = None
            additional_attribute_dim = 0
        self.feature_gen = torch.nn.Sequential(
            OrderedDict(
                [
                    (
                        "lstm",
                        torch.nn.LSTM(
                            attribute_dim
                            + additional_attribute_dim
                            + feature_noise_dim,
                            feature_num_units,
                            feature_num_layers,
                            batch_first=True,
                        ),
                    ),
                    ("selector", SelectLastCell()),
                    (
                        "merger",
                        Merger(
                            [
                                OutputDecoder(
                                    feature_num_units, feature_outputs, dim_index=2
                                )
                                for _ in range(self.sample_len)
                            ],
                            dim_index=2,
                        ),
                    ),
                ]
            )
        )

    def _make_attribute_generator(
        self, outputs: List[Output], input_dim: int, num_units: int, num_layers: int
    ):
        seq = []
        last_dim = input_dim
        for _ in range(num_layers):
            seq.append(torch.nn.Linear(last_dim, num_units))
            seq.append(torch.nn.ReLU())
            seq.append(torch.nn.BatchNorm1d(num_units))
            last_dim = num_units
        seq.append(OutputDecoder(last_dim, outputs, dim_index=1))
        return torch.nn.Sequential(*seq)

    def forward(self, attribute_noise: torch.Tensor, feature_noise: torch.Tensor):
        attributes = self.attribute_gen(attribute_noise)
        if self.additional_attribute_gen:
            attributes_no_gradient = attributes.detach()
            additional_attribute_gen_input = torch.cat(
                (attributes_no_gradient, attribute_noise), dim=1
            )
            additional_attributes = self.additional_attribute_gen(
                additional_attribute_gen_input
            )
        else:
            additional_attributes = None
        if self.additional_attribute_gen:
            combined_attributes = torch.cat((attributes, additional_attributes), dim=1)
        else:
            combined_attributes = attributes
        combined_attributes_no_gradient = combined_attributes.detach()
        reshaped_attributes = torch.reshape(
            combined_attributes_no_gradient, (combined_attributes.shape[0], 1, -1)
        )
        reshaped_attributes = reshaped_attributes.expand(-1, feature_noise.shape[1], -1)
        feature_gen_input = torch.cat((reshaped_attributes, feature_noise), 2)
        features = self.feature_gen(feature_gen_input)
        features = torch.reshape(
            features, (features.shape[0], self.max_sequence_len, -1)
        )
        if self.additional_attribute_gen:
            return attributes, additional_attributes, features
        else:
            return attributes, features


class Discriminator(torch.nn.Module):
    def __init__(self, input_dim: int, num_layers: int = 5, num_units: int = 200):
        super(Discriminator, self).__init__()
        seq = []
        last_dim = input_dim
        for _ in range(num_layers):
            seq.append(torch.nn.Linear(last_dim, num_units))
            seq.append(torch.nn.ReLU())
            last_dim = num_units
        seq.append(torch.nn.Linear(last_dim, 1))
        self.seq = torch.nn.Sequential(*seq)

    def forward(self, input: torch.Tensor):
        return self.seq(input)


def interpolate(x1, x2, alpha):
    diff = x2 - x1
    expanded_dims = [1 for _ in diff.shape]
    expanded_dims[0] = -1
    reshaped_alpha = alpha.reshape(expanded_dims).expand(diff.shape)
    return x1 + reshaped_alpha * diff


def apply_named(m: torch.nn.Module, prefix: str, func):
    func(m, prefix)
    for name, child in m.named_children():
        apply_named(child, prefix + "." + name, func)


def log_weights(m: torch.nn.Module, tb_writer: tensorboard.SummaryWriter, global_step):
    def f(a, prefix):
        for name, param in a.named_parameters(recurse=False):
            tb_writer.add_histogram(
                "weights/" + prefix + "." + name, param, global_step
            )

    apply_named(m, "", f)


class DGTorch(torch.nn.Module):  # Make it a module to save/load easily
    def __init__(
        self,
        attribute_outputs: List[Output],
        additional_attribute_outputs: Union[List[Output], None],
        feature_outputs: List[Output],
        max_sequence_len: int,
        sample_len: int,
        attribute_noise_dim: int = 10,
        feature_noise_dim: int = 10,
        attribute_num_layers: int = 3,
        attribute_num_units: int = 100,
        feature_num_layers: int = 1,
        feature_num_units: int = 100,
        gradient_penalty_coef: float = 10.0,
        generator_learning_rate: float = 0.001,
        generator_beta1: float = 0.5,
        discriminator_learning_rate: float = 0.001,
        discriminator_beta1: float = 0.5,
        use_attribute_discriminator: bool = False,
        attribute_gradient_penalty_coef: float = 10.0,
        attribute_loss_coef: float = 1.0,
        attribute_discriminator_learning_rate: float = 0.001,
        attribute_discriminator_beta1: float = 0.5,
        forget_bias: bool = False,
        cuda: bool = True,
    ):
        super(DGTorch, self).__init__()  # Add this for torch.nn.Module
        if max_sequence_len % sample_len != 0:
            raise RuntimeError(
                f"max_sequence_len={max_sequence_len} must be divisible by sample_len={sample_len}"
            )
        self.EPS = 1e-8
        self.attribute_outputs = attribute_outputs
        self.additional_attribute_outputs = additional_attribute_outputs
        self.feature_outputs = feature_outputs
        self.gradient_penalty_coef = gradient_penalty_coef
        self.generator_learning_rate = generator_learning_rate
        self.generator_beta1 = generator_beta1
        self.discriminator_learning_rate = discriminator_learning_rate
        self.discriminator_beta1 = discriminator_beta1
        self.attribute_gradient_penalty_coef = attribute_gradient_penalty_coef
        self.attribute_loss_coef = attribute_loss_coef
        self.attribute_discriminator_learning_rate = (
            attribute_discriminator_learning_rate
        )
        self.attribute_discriminator_beta1 = attribute_discriminator_beta1

        # Store dims as instance attributes for pickle compatibility
        self.attribute_noise_dim = attribute_noise_dim
        self.feature_noise_dim = feature_noise_dim
        self.max_sequence_len = max_sequence_len
        self.sample_len = sample_len

        if cuda and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.generator = Generator(
            attribute_outputs,
            additional_attribute_outputs,
            feature_outputs,
            max_sequence_len,
            sample_len,
            attribute_noise_dim,
            feature_noise_dim,
            attribute_num_units,
            attribute_num_layers,
            feature_num_units,
            feature_num_layers,
        )
        self.generator.to(self.device)
        attribute_dim = sum(output.get_dim() for output in attribute_outputs)
        additional_attribute_dim = 0
        if self.additional_attribute_outputs:
            additional_attribute_dim = sum(
                output.get_dim() for output in self.additional_attribute_outputs
            )
        feature_dim = sum(output.get_dim() for output in feature_outputs)
        self.feature_discriminator = Discriminator(
            attribute_dim + additional_attribute_dim + max_sequence_len * feature_dim,
            num_layers=5,
            num_units=200,
        )
        self.feature_discriminator.to(self.device)
        self.attribute_discriminator = None
        if use_attribute_discriminator:
            self.attribute_discriminator = Discriminator(
                attribute_dim + additional_attribute_dim, num_layers=5, num_units=200
            )
            self.attribute_discriminator.to(self.device)


        if forget_bias:

            def init_weights(m):
                if "LSTM" in str(m.__class__):
                    for name, param in m.named_parameters(recurse=False):
                        if "bias_hh" in name:
                            with torch.no_grad():
                                hidden_size = m.hidden_size
                                a = -np.sqrt(1.0 / hidden_size)
                                b = np.sqrt(1.0 / hidden_size)
                                bias_ii = torch.Tensor(hidden_size)
                                bias_ig_io = torch.Tensor(hidden_size * 2)
                                bias_if = torch.Tensor(hidden_size)
                                torch.nn.init.uniform_(bias_ii, a, b)
                                torch.nn.init.uniform_(bias_ig_io, a, b)
                                torch.nn.init.ones_(bias_if)
                                new_param = torch.cat(
                                    [bias_ii, bias_if, bias_ig_io], dim=0
                                )
                                param.copy_(new_param)

            self.generator.apply(init_weights)

    def attribute_noise_func(self, batch_size):
        return torch.randn(batch_size, self.attribute_noise_dim, device=self.device)

    def feature_noise_func(self, batch_size):
        return torch.randn(
            batch_size,
            self.max_sequence_len // self.sample_len,
            self.feature_noise_dim,
            device=self.device,
        )

    def generate(
        self, batch_size: int = None, attribute_noise=None, feature_noise=None
    ):
        if batch_size is not None:
            attribute_noise = self.attribute_noise_func(batch_size)
            feature_noise = self.feature_noise_func(batch_size)
        else:
            if attribute_noise is None or feature_noise is None:
                raise RuntimeError(
                    "generate() must receive either batch_size or both attribute_noise and feature_noise"
                )
            attribute_noise = attribute_noise.to(self.device)
            feature_noise = feature_noise.to(self.device)
        # generate() is inference-only, but self.generator's BatchNorm1d layers
        # default to training mode (nothing ever calls .eval() - DGTorch.train()
        # above is a training-loop method, not the nn.Module mode toggle, so the
        # standard train()/eval() machinery is unusable on this class). Left in
        # training mode, BatchNorm normalizes against the live statistics of
        # whichever random batch this call happens to draw, so the output for a
        # given latent code silently depends on what else is in the same batch -
        # eval() (called on the submodule directly, bypassing the override) makes
        # generation deterministic per latent code, using the fixed running
        # statistics learned during training instead.
        was_training = self.generator.training
        self.generator.eval()
        try:
            batch = self.generator(attribute_noise, feature_noise)
        finally:
            self.generator.train(was_training)
        batch = [x.cpu().detach().numpy() for x in batch]
        if self.additional_attribute_outputs:
            transformed_attributes, additional_attributes, transformed_features = batch
            attributes = inverse_transform(
                transformed_attributes, self.attribute_outputs, variable_dim_index=1
            )
            features = inverse_transform(
                transformed_features,
                self.feature_outputs,
                variable_dim_index=2,
                additional_attributes=additional_attributes,
                additional_attribute_outputs=self.additional_attribute_outputs,
            )
        else:
            transformed_attributes, transformed_features = batch
            attributes = inverse_transform(
                transformed_attributes, self.attribute_outputs, variable_dim_index=1
            )
            features = inverse_transform(
                transformed_features, self.feature_outputs, variable_dim_index=2
            )
        return attributes, features

    def generate_horizon(self, batch_size: int, sequence_len: int):
        """Generate sequences of `sequence_len` steps, not just max_sequence_len.

        The generator's feature path is an LSTM that emits `sample_len` steps
        per noise block, so it rolls out to any multiple of sample_len - its
        weights do not depend on sequence length. Only the *discriminator* is
        fixed-width (its input Linear is sized max_sequence_len * feature_dim),
        and that is used solely during training.

        Quality is not length-independent even though the architecture is: the
        LSTM state drifts outside the regime it was trained on, so output well
        past max_sequence_len degrades. Callers should check per-horizon
        quality rather than assume a longer rollout is usable.
        """
        if sequence_len % self.sample_len != 0:
            raise ValueError(
                f"sequence_len={sequence_len} must be a multiple of sample_len={self.sample_len}"
            )
        blocks = sequence_len // self.sample_len
        g = self.generator
        attribute_noise = self.attribute_noise_func(batch_size)
        feature_noise = torch.randn(
            batch_size, blocks, self.feature_noise_dim, device=self.device
        )
        was_training = g.training
        g.eval()
        try:
            attributes = g.attribute_gen(attribute_noise)
            if g.additional_attribute_gen:
                additional = g.additional_attribute_gen(
                    torch.cat((attributes.detach(), attribute_noise), dim=1)
                )
                combined = torch.cat((attributes, additional), dim=1)
            else:
                additional = None
                combined = attributes
            reshaped = combined.detach().reshape(combined.shape[0], 1, -1)
            reshaped = reshaped.expand(-1, blocks, -1)
            features = g.feature_gen(torch.cat((reshaped, feature_noise), 2))
            features = torch.reshape(features, (features.shape[0], sequence_len, -1))
        finally:
            g.train(was_training)

        features_np = features.detach().cpu().numpy()
        attributes_np = attributes.detach().cpu().numpy()
        if additional is not None:
            additional_np = additional.detach().cpu().numpy()
            features_out = inverse_transform(
                features_np,
                self.feature_outputs,
                variable_dim_index=2,
                additional_attributes=additional_np,
                additional_attribute_outputs=self.additional_attribute_outputs,
            )
        else:
            features_out = inverse_transform(
                features_np, self.feature_outputs, variable_dim_index=2
            )
        attributes_out = inverse_transform(
            attributes_np, self.attribute_outputs, variable_dim_index=1
        )
        return attributes_out, features_out

    def _discriminate(self, batch):
        inputs = list(batch)
        inputs[-1] = torch.reshape(inputs[-1], (inputs[-1].shape[0], -1))
        input = torch.cat(inputs, dim=1)
        output = self.feature_discriminator(input)
        return output

    def _discriminate_attributes(self, batch):
        if not self.attribute_discriminator:
            raise RuntimeError(
                "discriminate_attributes called with no attribute_discriminator"
            )
        input = torch.cat(batch, dim=1)
        output = self.attribute_discriminator(input)
        return output

    def _get_gradient_penalty(self, generated_batch, real_batch, discriminator_func):
        alpha = torch.rand(generated_batch[0].shape[0], device=self.device)
        interpolated_batch = [
            interpolate(g, r, alpha).requires_grad_(True)
            for g, r in zip(generated_batch, real_batch)
        ]
        interpolated_output = discriminator_func(interpolated_batch)
        gradients = torch.autograd.grad(
            interpolated_output,
            interpolated_batch,
            grad_outputs=torch.ones(interpolated_output.shape, device=self.device),
            retain_graph=True,
            create_graph=True,
        )
        squared_sums = [
            torch.sum(torch.square(g.view(g.size(0), -1))) for g in gradients
        ]
        norm = torch.sqrt(sum(squared_sums) + self.EPS)
        return ((norm - 1.0) ** 2).mean()

    def add_batch_summary(
        self, tb_writer: tensorboard.SummaryWriter, batch, prefix: str, global_step: int
    ):
        attributes = batch[0]
        index = 0
        for output in self.attribute_outputs:
            if isinstance(output, DiscreteOutput):
                probs = attributes[:, index : (index + output.dim)]
                indices = torch.argmax(probs, dim=1)
                tb_writer.add_histogram(
                    prefix + "/attributes/" + output.name, indices, global_step
                )
                tb_writer.add_histogram(
                    prefix + "/attributes/" + output.name + "_probs", probs, global_step
                )
            elif isinstance(output, ContinuousOutput):
                tb_writer.add_histogram(
                    prefix + "/attributes/" + output.name,
                    attributes[:, index],
                    global_step,
                )
            index += output.get_dim()
        if self.additional_attribute_outputs:
            additional_attributes = batch[1]
            index = 0
            for output in self.additional_attribute_outputs:
                if isinstance(output, DiscreteOutput):
                    probs = additional_attributes[:, index : (index + output.dim)]
                    indices = torch.argmax(probs, dim=1)
                    tb_writer.add_histogram(
                        prefix + "/additional_attributes/" + output.name,
                        indices,
                        global_step,
                    )
                    tb_writer.add_histogram(
                        prefix + "/additional_attributes/" + output.name + "_probs",
                        probs,
                        global_step,
                    )
                elif isinstance(output, ContinuousOutput):
                    tb_writer.add_histogram(
                        prefix + "/additional_attributes/" + output.name,
                        additional_attributes[:, index],
                        global_step,
                    )
                index += output.get_dim()

    def train(
        self,
        dataset,
        batch_size: int,
        num_epochs: int,
        discriminator_rounds: int = 1,
        generator_rounds: int = 1,
        tb_writer: tensorboard.SummaryWriter = None,
        log_activations: bool = False,
        progress_callback: callable = None,
    ):
        loader = torch.utils.data.DataLoader(
            dataset, batch_size, shuffle=True, drop_last=True
        )
        opt_discriminator = torch.optim.Adam(
            self.feature_discriminator.parameters(),
            lr=self.discriminator_learning_rate,
            betas=(self.discriminator_beta1, 0.999),
        )
        opt_attribute_discriminator = None
        if self.attribute_discriminator is not None:
            opt_attribute_discriminator = torch.optim.Adam(
                self.attribute_discriminator.parameters(),
                lr=self.attribute_discriminator_learning_rate,
                betas=(self.attribute_discriminator_beta1, 0.999),
            )
        opt_generator = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.generator_learning_rate,
            betas=(self.generator_beta1, 0.999),
        )
        global_step = 0
        for epoch in range(num_epochs):
            for batch_number, real_batch in enumerate(loader):
                if progress_callback:
                    progress_callback(
                        ProgressInfo(
                            epoch=epoch,
                            total_epochs=num_epochs,
                            batch=batch_number,
                            total_batches=len(loader),
                        )
                    )
                global_step += 1
                attribute_noise = self.attribute_noise_func(batch_size)
                feature_noise = self.feature_noise_func(batch_size)
                generated_batch = self.generator(attribute_noise, feature_noise)
                real_batch = [x.to(self.device) for x in real_batch]
                for _ in range(discriminator_rounds):
                    opt_discriminator.zero_grad()
                    generated_output = self._discriminate(generated_batch)
                    real_output = self._discriminate(real_batch)
                    loss_generated = torch.mean(generated_output)
                    loss_real = -torch.mean(real_output)
                    loss_gradient_penalty = self._get_gradient_penalty(
                        generated_batch, real_batch, self._discriminate
                    )
                    loss = (
                        loss_generated
                        + loss_real
                        + self.gradient_penalty_coef * loss_gradient_penalty
                    )
                    loss.backward(retain_graph=True)
                    opt_discriminator.step()
                    if opt_attribute_discriminator is not None:
                        opt_attribute_discriminator.zero_grad()
                        generated_output = self._discriminate_attributes(
                            generated_batch[:-1]
                        )
                        real_output = self._discriminate_attributes(real_batch[:-1])
                        loss_generated = torch.mean(generated_output)
                        loss_real = -torch.mean(real_output)
                        loss_gradient_penalty = self._get_gradient_penalty(
                            generated_batch[:-1],
                            real_batch[:-1],
                            self._discriminate_attributes,
                        )
                        attribute_loss = (
                            loss_generated
                            + loss_real
                            + self.attribute_gradient_penalty_coef
                            * loss_gradient_penalty
                        )
                        attribute_loss.backward(retain_graph=True)
                        opt_attribute_discriminator.step()
                for _ in range(generator_rounds):
                    opt_generator.zero_grad()
                    generated_output = self._discriminate(generated_batch)
                    if self.attribute_discriminator:
                        attribute_generated_output = self._discriminate_attributes(
                            generated_batch[:-1]
                        )
                        loss = -torch.mean(
                            generated_output
                        ) + self.attribute_loss_coef * -torch.mean(
                            attribute_generated_output
                        )
                    else:
                        loss = -torch.mean(generated_output)
                    loss.backward()
                    opt_generator.step()
