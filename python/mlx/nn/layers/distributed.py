# Copyright © 2024 Apple Inc.

import math
from functools import lru_cache
from typing import Callable, Optional, Union

import mlx.core as mx
from mlx.nn.layers.base import Module
from mlx.nn.layers.linear import Linear
from mlx.nn.layers.quantized import QuantizedLinear
from mlx.utils import tree_map_with_path


@lru_cache
def sum_gradients(group):
    if group.size() == 1:
        return lambda x: x

    @mx.custom_function
    def f(x):
        return x

    @f.vjp
    def f(x, dx, _):
        return mx.distributed.all_sum(dx, group=group)

    return f


def _split(weight, segments, axis):
    """Equivalent to mx.split but allows for fractional segments."""
    if isinstance(segments, int) or isinstance(segments[0], int):
        return mx.split(weight, segments, axis=axis)

    N = weight.shape[axis]
    indices = [int(s * N) for s in segments]
    return mx.split(weight, indices, axis=axis)


def compute_shard_sizes(
    dim: int,
    N: int,
    unit: int = 1,
    weights: Optional[list[float]] = None,
) -> list[int]:
    """Distribute *dim* elements across *N* shards.

    When *unit* > 1 every shard size is a multiple of *unit*.

    When *weights* is ``None`` the split is as even as possible (the first
    ``(dim // unit) % N`` shards receive one extra *unit*).

    When *weights* is given it must have length *N* and contain relative
    weights (e.g. ``[2, 1, 1]`` gives rank 0 twice as many units).  The
    weights are normalised internally and each shard size is rounded to the
    nearest multiple of *unit* with a greedy correction pass so the sizes
    sum to exactly *dim*.
    """
    assert unit >= 1, f"unit must be >= 1, got {unit}"
    if unit > 1:
        assert dim % unit == 0, f"dim ({dim}) must be divisible by unit ({unit})"
    assert dim // unit >= N, (
        f"not enough units to distribute: dim={dim}, unit={unit}, "
        f"n_units={dim // unit}, N={N}"
    )
    n_units = dim // unit

    if weights is None:
        base, rem = divmod(n_units, N)
        return [(base + (1 if i < rem else 0)) * unit for i in range(N)]

    assert len(weights) == N, f"len(weights) ({len(weights)}) must equal N ({N})"
    assert all(w > 0 for w in weights), "all weights must be positive"
    total_w = sum(weights)

    # Ideal (fractional) unit counts per shard.
    ideal = [w / total_w * n_units for w in weights]
    # Round each to nearest integer.
    counts = [max(1, round(v)) for v in ideal]

    # Greedy correction: adjust counts so they sum to n_units.
    diff = sum(counts) - n_units
    if diff > 0:
        # Over-allocated — take from shards that were rounded up the most.
        order = sorted(range(N), key=lambda i: counts[i] - ideal[i], reverse=True)
        for i in order:
            if diff == 0:
                break
            if counts[i] > 1:
                counts[i] -= 1
                diff -= 1
    elif diff < 0:
        # Under-allocated — give to shards that were rounded down the most.
        order = sorted(range(N), key=lambda i: counts[i] - ideal[i])
        for i in order:
            if diff == 0:
                break
            counts[i] += 1
            diff += 1

    return [c * unit for c in counts]


def _sizes_to_boundary_indices(sizes: list[int]) -> list[int]:
    """Convert a list of shard sizes to cumulative split boundary indices.

    ``[4, 3, 3]`` → ``[4, 7]``
    """
    indices: list[int] = []
    cumsum = 0
    for s in sizes[:-1]:
        cumsum += s
        indices.append(cumsum)
    return indices


def _uneven_split(
    weight: mx.array,
    N: int,
    axis: int,
    unit: int = 1,
    weights: Optional[list[float]] = None,
) -> list[mx.array]:
    """Like ``mx.split(weight, N, axis)`` but allows ``weight.shape[axis] % N != 0``.

    When *unit* > 1, shard sizes are multiples of *unit*.
    When *weights* is given, distribute proportionally (see :func:`compute_shard_sizes`).
    """
    dim = weight.shape[axis]
    if weights is None and unit == 1 and dim % N == 0:
        return mx.split(weight, N, axis=axis)
    return mx.split(
        weight,
        _sizes_to_boundary_indices(compute_shard_sizes(dim, N, unit, weights)),
        axis=axis,
    )


def _shard(
    parameters: dict,
    sharding_predicate: Callable,
    group: Optional[mx.distributed.Group] = None,
):
    """Returns a new parameter tree with the weights sharded according to the
    sharding_predicate.

    The sharding predicate should return the sharding axis and optionally also
    the segments that comprise the weight.
    """
    group = group or mx.distributed.init()
    N = group.size()
    r = group.rank()

    def _shard_fn(path, weight):
        if not isinstance(weight, mx.array):
            return weight

        s = sharding_predicate(path, weight)
        if s is None:
            return weight

        axis = None
        segments = 1
        unit = 1
        shard_weights = None
        if isinstance(s, int):
            axis = s
        elif isinstance(s, tuple):
            if len(s) == 2:
                axis, segments = s
            elif len(s) == 3:
                axis, segments, unit = s
            elif len(s) == 4:
                axis, segments, unit, shard_weights = s
            else:
                raise ValueError(
                    "The sharding function should return int or "
                    "tuple of (axis, segments[, unit[, weights]])"
                )
        else:
            raise ValueError(
                "The sharding function should return int or "
                "tuple of (axis, segments[, unit[, weights]])"
            )

        return mx.contiguous(
            mx.concatenate(
                [
                    _uneven_split(part, N, axis, unit, shard_weights)[r]
                    for part in _split(weight, segments, axis)
                ],
                axis=axis,
            )
        )

    return tree_map_with_path(_shard_fn, parameters)


def _all_to_sharded(segments, unit=1, weights=None):
    """Simple predicate to shard fully connected layers such that a common
    representation becomes a sharded representation."""

    def _shard_fn(path, weight):
        if path.endswith("bias"):
            return -1, segments, unit, weights
        return max(weight.ndim - 2, 0), segments, unit, weights

    return _shard_fn


def _sharded_to_all(segments, unit=1, weights=None):
    """Simple predicate to shard fully connected layers such that a sharded
    representation becomes a common representation."""

    def _shard_fn(path, weight):
        if path.endswith("bias"):
            return None
        return -1, segments, unit, weights

    return _shard_fn


def _shard_quantized_a2s(
    parameters: dict,
    group: mx.distributed.Group,
    group_size: int,
    segments: Union[int, list] = 1,
    unit: int = 1,
    weights: Optional[list[float]] = None,
):
    """Shard quantized parameters for the all-to-sharded case.

    When *unit* >= *group_size*, pads the output dimension (axis 0) to the
    next multiple of *group_size* before splitting.  This ensures shard sizes
    match the paired sharded-to-all layer's group-aligned input dimension.
    """
    if unit >= group_size:
        output_dim = parameters["weight"].shape[0]
        padded = ((output_dim + group_size - 1) // group_size) * group_size
        pad_size = padded - output_dim
        if pad_size > 0:
            padded_params: dict = {}
            for k, v in parameters.items():
                if isinstance(v, mx.array):
                    pad_shape = list(v.shape)
                    pad_shape[0] = pad_size
                    padded_params[k] = mx.concatenate(
                        [v, mx.zeros(pad_shape, dtype=v.dtype)], axis=0
                    )
                else:
                    padded_params[k] = v
            parameters = padded_params

    return _shard(parameters, _all_to_sharded(segments, unit, weights), group)


def _check_sharding(sharding):
    if sharding not in ("all-to-sharded", "sharded-to-all"):
        raise ValueError(
            (
                f"Sharding type {sharding=} not supported, "
                "choose one of 'all-to-sharded' or 'sharded-to-all'"
            )
        )


def _shard_quantized_s2a(
    parameters: dict,
    group: mx.distributed.Group,
    group_size: int,
    bits: int,
    segments: Union[int, list] = 1,
    unit: int = 1,
    weights: Optional[list[float]] = None,
):
    """Shard quantized parameters for the sharded-to-all case.

    Standard ``_shard`` with ``_uneven_split`` fails here because weight,
    scales, and biases have different physical sizes on axis -1 (the input
    dimension).  Independent uneven splits would give inconsistent logical
    boundaries.

    Instead we split in *logical* space with ``group_size`` alignment, then
    derive per-parameter physical split indices.
    """
    N = group.size()
    r = group.rank()

    scales = parameters.get("scales")
    if scales is None:
        # Not actually quantized — fall back to generic path.
        return _shard(parameters, _sharded_to_all(segments, unit, weights), group)

    # Ensure unit is at least group_size and a multiple of it, so we
    # never split inside a quantization group.
    unit = ((unit + group_size - 1) // group_size) * group_size
    unit_qg = unit // group_size

    # Number of quantization groups along the input axis.
    num_quant_groups = scales.shape[-1]
    group_counts = compute_shard_sizes(num_quant_groups, N, unit_qg, weights)

    weight_ppg = group_size * bits // 32  # packed elements per quant-group
    scale_ppg = 1  # one scale / one bias per quant-group

    result: dict = {}
    for key, param in parameters.items():
        if not isinstance(param, mx.array):
            result[key] = param
            continue

        if key == "bias":
            # Linear bias — not split in sharded-to-all.
            result[key] = param
        elif key == "weight":
            sizes = [gc * weight_ppg for gc in group_counts]
            indices = _sizes_to_boundary_indices(sizes)
            if segments != 1:
                seg_parts = _split(param, segments, -1)
                result[key] = mx.contiguous(
                    mx.concatenate(
                        [mx.split(sp, indices, axis=-1)[r] for sp in seg_parts],
                        axis=-1,
                    )
                )
            else:
                result[key] = mx.contiguous(mx.split(param, indices, axis=-1)[r])
        elif key in ("scales", "biases"):
            sizes = [gc * scale_ppg for gc in group_counts]
            indices = _sizes_to_boundary_indices(sizes)
            if segments != 1:
                seg_parts = _split(param, segments, -1)
                result[key] = mx.contiguous(
                    mx.concatenate(
                        [mx.split(sp, indices, axis=-1)[r] for sp in seg_parts],
                        axis=-1,
                    )
                )
            else:
                result[key] = mx.contiguous(mx.split(param, indices, axis=-1)[r])
        else:
            # Unknown parameter — pass through unchanged.
            result[key] = param

    return result


def shard_inplace(
    module: Module,
    sharding: Union[str, Callable],
    *,
    segments: Union[int, list] = 1,
    unit: int = 1,
    weights: Optional[list[float]] = None,
    group: Optional[mx.distributed.Group] = None,
):
    """Shard a module in-place by updating its parameter dictionary with the
    sharded parameter dictionary.

    The ``sharding`` argument can be any callable that given the path and the
    weight returns the sharding axis and optionally also the segments that
    comprise the unsharded weight. For instance if the weight is a fused QKV
    matrix the segments should be 3.

    .. note::
        The module doesn't change so in order for distributed communication to
        happen the module needs to natively support it and for it to be enabled.

    Args:
        module (mlx.nn.Module): The parameters of this module will be sharded
            in-place.
        sharding (str or callable): One of "all-to-sharded" and
            "sharded-to-all" or a callable that returns the sharding axis and
            segments.
        segments (int or list): The segments to use if ``sharding`` is a
            string. Default: ``1``.
        unit (int): Split granularity — shard sizes will be multiples of
            *unit*. Pass ``head_dim`` for attention projections. Default: ``1``.
        weights (list[float], optional): Relative weights for proportional
            distribution (e.g. ``[2, 1, 1]``). Length must equal the group
            size. Default: ``None`` (even split).
        group (mlx.core.distributed.Group): The distributed group to shard
            across. If not set, the global group will be used. Default: ``None``.
    """
    group = group or mx.distributed.init()

    if isinstance(sharding, str):
        _check_sharding(sharding)
        is_quantized = hasattr(module, "group_size") and hasattr(module, "bits")
        if sharding == "sharded-to-all" and is_quantized:
            module.update(
                _shard_quantized_s2a(
                    module.parameters(),
                    group,
                    module.group_size,
                    module.bits,
                    segments,
                    unit,
                    weights,
                )
            )
        elif sharding == "all-to-sharded" and is_quantized:
            module.update(
                _shard_quantized_a2s(
                    module.parameters(),
                    group,
                    module.group_size,
                    segments,
                    unit,
                    weights,
                )
            )
        else:
            predicate = (
                _all_to_sharded(segments, unit, weights)
                if sharding == "all-to-sharded"
                else _sharded_to_all(segments, unit, weights)
            )
            module.update(_shard(module.parameters(), predicate, group))
    else:
        # Custom callable predicate.
        module.update(_shard(module.parameters(), sharding, group))


def shard_linear(
    module: Module,
    sharding: str,
    *,
    segments: Union[int, list] = 1,
    unit: int = 1,
    weights: Optional[list[float]] = None,
    group: Optional[mx.distributed.Group] = None,
):
    """Create a new linear layer that has its parameters sharded and also
    performs distributed communication either in the forward or backward
    pass.

    .. note::
        Contrary to ``shard_inplace``, the original layer is not changed but a
        new layer is returned.

    Args:
        module (mlx.nn.Module): The linear layer to be sharded.
        sharding (str): One of "all-to-sharded" and
            "sharded-to-all" that defines the type of sharding to perform.
        segments (int or list): The segments to use. Default: ``1``.
        unit (int): Split granularity — shard sizes will be multiples of
            *unit*. Pass ``head_dim`` for attention projections. Default: ``1``.
        weights (list[float], optional): Relative weights for proportional
            distribution (e.g. ``[2, 1, 1]``). Length must equal the group
            size. Default: ``None`` (even split).
        group (mlx.core.distributed.Group): The distributed group to shard
            across. If not set, the global group will be used. Default: ``None``.
    """
    _check_sharding(sharding)
    group = group or mx.distributed.init()

    is_linear = isinstance(module, Linear)

    cls_map = {
        ("all-to-sharded", True): AllToShardedLinear,
        ("all-to-sharded", False): QuantizedAllToShardedLinear,
        ("sharded-to-all", True): ShardedToAllLinear,
        ("sharded-to-all", False): QuantizedShardedToAllLinear,
    }
    cls = cls_map[sharding, is_linear]

    # Bypass __init__ — avoids dim % N validation and throwaway weight alloc.
    sl = cls.__new__(cls)
    Module.__init__(sl)
    sl.group = group
    if not is_linear:
        sl.group_size = module.group_size
        sl.bits = module.bits
        sl.mode = getattr(module, "mode", "affine")

    # Shard parameters — use setattr since the bare module has no keys yet.
    params = module.parameters()

    if cls is QuantizedShardedToAllLinear:
        sharded = _shard_quantized_s2a(
            params,
            group,
            module.group_size,
            module.bits,
            segments,
            unit,
            weights,
        )
    elif cls is QuantizedAllToShardedLinear:
        sharded = _shard_quantized_a2s(
            params,
            group,
            module.group_size,
            segments,
            unit,
            weights,
        )
    else:
        predicate = (
            _all_to_sharded(segments, unit, weights)
            if sharding == "all-to-sharded"
            else _sharded_to_all(segments, unit, weights)
        )
        sharded = _shard(params, predicate, group)

    for k, v in sharded.items():
        setattr(sl, k, v)

    if not is_linear:
        sl.freeze()

    return sl


class AllToShardedLinear(Module):
    """Each member of the group applies part of the affine transformation such
    that the result is sharded across the group.

    The gradients are automatically aggregated from each member of the group.

    Args:
        input_dims (int): The dimensionality of the input features
        output_dims (int): The dimensionality of the output features
        bias (bool, optional): If set to ``False`` the the layer will not use a
            bias. Default is ``True``.
        group (mx.distributed.Group, optional): The sharding will happen across
            this group. If not set then the global group is used. Default is
            ``None``.
    """

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        bias: bool = True,
        group: Optional[mx.distributed.Group] = None,
    ):
        super().__init__()

        # Initialize the parameters
        scale = math.sqrt(1.0 / input_dims)
        self.group = group or mx.distributed.init()
        N = self.group.size()

        if (output_dims % N) != 0:
            raise ValueError(
                f"Cannot shard the output of size {output_dims} across {N} devices."
            )

        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(output_dims // N, input_dims),
        )
        if bias:
            self.bias = mx.random.uniform(
                low=-scale,
                high=scale,
                shape=(output_dims // N,),
            )

    def _extra_repr(self) -> str:
        out_dims, in_dims = self.weight.shape
        N = self.group.size()
        out_dims *= N
        return f"input_dims={in_dims}, output_dims={out_dims}, bias={'bias' in self}"

    def __call__(self, x: mx.array) -> mx.array:
        # Aggregate the gradients coming from each shard
        x = sum_gradients(self.group)(x)

        # Compute the affine projection
        if "bias" in self:
            x = mx.addmm(self["bias"], x, self["weight"].T)
        else:
            x = x @ self["weight"].T
        return x

    @classmethod
    def from_linear(
        cls,
        linear_layer: Module,
        *,
        segments: Union[int, list] = 1,
        group: Optional[mx.distributed.Group] = None,
    ):
        group = group or mx.distributed.init()
        output_dims, input_dims = linear_layer.weight.shape

        sl = cls(input_dims, output_dims, hasattr(linear_layer, "bias"), group)
        sl.update(_shard(linear_layer.parameters(), _all_to_sharded(segments), group))

        return sl


class ShardedToAllLinear(Module):
    """Each member of the group applies part of the affine transformation and
    then aggregates the results.

    All nodes will have the same exact result after this layer.

    :class:`ShardedToAllLinear` provides a classmethod :meth:`from_linear` to
    convert linear layers to sharded :obj:`ShardedToAllLinear` layers.

    Args:
        input_dims (int): The dimensionality of the input features
        output_dims (int): The dimensionality of the output features
        bias (bool, optional): If set to ``False`` the the layer will not use a
            bias. Default is ``True``.
        group (mx.distributed.Group, optional): The sharding will happen across
            this group. If not set then the global group is used. Default is
            ``None``.
    """

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        bias: bool = True,
        group: Optional[mx.distributed.Group] = None,
    ):
        super().__init__()

        # Initialize the parameters
        scale = math.sqrt(1.0 / input_dims)
        self.group = group or mx.distributed.init()
        N = self.group.size()

        if (input_dims % N) != 0:
            raise ValueError(
                f"The input of size {input_dims} cannot be sharded across {N} devices."
            )

        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(output_dims, input_dims // N),
        )
        if bias:
            self.bias = mx.random.uniform(
                low=-scale,
                high=scale,
                shape=(output_dims,),
            )

    def _extra_repr(self) -> str:
        N = self.group.size()
        out_dims, in_dims = self.weight.shape
        in_dims *= N
        return f"input_dims={in_dims}, output_dims={out_dims}, bias={'bias' in self}"

    def __call__(self, x: mx.array) -> mx.array:
        x = x @ self["weight"].T

        x = mx.distributed.all_sum(x, group=self.group)

        if "bias" in self:
            x = x + self["bias"]

        return x

    @classmethod
    def from_linear(
        cls,
        linear_layer: Module,
        *,
        segments: Union[int, list] = 1,
        group: Optional[mx.distributed.Group] = None,
    ):
        group = group or mx.distributed.init()
        output_dims, input_dims = linear_layer.weight.shape

        sl = cls(input_dims, output_dims, hasattr(linear_layer, "bias"), group)
        sl.update(_shard(linear_layer.parameters(), _sharded_to_all(segments), group))

        return sl


class QuantizedAllToShardedLinear(Module):
    """Each member of the group applies part of the affine transformation with
    a quantized matrix such that the result is sharded across the group.

    It is the quantized equivalent of :class:`mlx.nn.AllToShardedLinear`.
    Similar to :class:`mlx.nn.QuantizedLinear` its parameters are frozen and
    will not be included in any gradient computation.

    Args:
        input_dims (int): The dimensionality of the input features.
        output_dims (int): The dimensionality of the output features.
        bias (bool, optional): If set to ``False`` then the layer will not use
            a bias. Default: ``True``.
        group_size (int, optional): The group size to use for the quantized
            weight. See :func:`~mlx.core.quantize`. Default: ``64``.
        bits (int, optional): The bit width to use for the quantized weight.
            See :func:`~mlx.core.quantize`. Default: ``4``.
        mode (str, optional): The quantization method to use (see
            :func:`~mlx.core.quantize`). Default: ``"affine"``.
        group (mx.distributed.Group, optional): The sharding will happen across
            this group. If not set then the global group is used. Default is
            ``None``.
    """

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        bias: bool = True,
        group_size: int = 64,
        bits: int = 4,
        mode: str = "affine",
        group: Optional[mx.distributed.Group] = None,
    ):
        super().__init__()

        # Quantization config
        self.group_size = group_size
        self.bits = bits
        self.mode = mode

        # Initialize the quantized weight
        scale = math.sqrt(1.0 / input_dims)
        self.group = group or mx.distributed.init()
        N = self.group.size()

        if (output_dims % N) != 0:
            raise ValueError(
                f"Cannot shard the output of size {output_dims} across {N} devices."
            )

        weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(output_dims // N, input_dims),
        )
        self.weight, self.scales, *biases = mx.quantize(
            weight, group_size, bits, mode=mode
        )
        self.biases = biases[0] if biases else None

        # And bias if needed
        if bias:
            self.bias = mx.zeros((output_dims // N,))

        # Freeze this model's parameters
        self.freeze()

    def unfreeze(self, *args, **kwargs):
        """Wrap unfreeze so that we unfreeze any layers we might contain but
        our parameters will remain frozen."""
        super().unfreeze(*args, **kwargs)
        self.freeze(recurse=False)

    def _extra_repr(self) -> str:
        out_dims, in_dims = self.weight.shape
        in_dims = (in_dims * 32) // self.bits
        out_dims *= self.group.size()
        return (
            f"input_dims={in_dims}, output_dims={out_dims}, bias={'bias' in self}, "
            f"group_size={self.group_size}, bits={self.bits}, mode={self.mode}"
        )

    def __call__(self, x: mx.array) -> mx.array:
        # Aggregate the gradients coming from each shard
        x = sum_gradients(self.group)(x)

        x = mx.quantized_matmul(
            x,
            self["weight"],
            scales=self["scales"],
            biases=self.get("biases"),
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
        )
        if "bias" in self:
            x = x + self["bias"]
        return x

    @classmethod
    def from_quantized_linear(
        cls,
        quantized_linear_layer: Module,
        *,
        segments: Union[int, list] = 1,
        group: Optional[mx.distributed.Group] = None,
    ):
        group = group or mx.distributed.init()
        output_dims, input_dims = quantized_linear_layer.weight.shape
        input_dims = (input_dims * 32) // quantized_linear_layer.bits

        sl = cls(
            input_dims,
            output_dims,
            hasattr(quantized_linear_layer, "bias"),
            group_size=quantized_linear_layer.group_size,
            bits=quantized_linear_layer.bits,
            mode=getattr(quantized_linear_layer, "mode", "affine"),
            group=group,
        )
        sl.update(
            _shard(
                quantized_linear_layer.parameters(),
                _all_to_sharded(segments),
                group,
            )
        )

        return sl


class QuantizedShardedToAllLinear(Module):
    """Each member of the group applies part of the affine transformation using
    the quantized matrix and then aggregates the results.

    All nodes will have the same exact result after this layer.

    It is the quantized equivalent of :class:`mlx.nn.ShardedToAllLinear`.
    Similar to :class:`mlx.nn.QuantizedLinear` its parameters are frozen and
    will not be included in any gradient computation.

    Args:
        input_dims (int): The dimensionality of the input features.
        output_dims (int): The dimensionality of the output features.
        bias (bool, optional): If set to ``False`` then the layer will not use
            a bias. Default: ``True``.
        group_size (int, optional): The group size to use for the quantized
            weight. See :func:`~mlx.core.quantize`. Default: ``64``.
        bits (int, optional): The bit width to use for the quantized weight.
            See :func:`~mlx.core.quantize`. Default: ``4``.
        mode (str, optional): The quantization method to use (see
            :func:`~mlx.core.quantize`). Default: ``"affine"``.
        group (mx.distributed.Group, optional): The sharding will happen across
            this group. If not set then the global group is used. Default is
            ``None``.
    """

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        bias: bool = True,
        group_size: int = 64,
        bits: int = 4,
        mode: str = "affine",
        group: Optional[mx.distributed.Group] = None,
    ):
        super().__init__()

        # Quantization config
        self.group_size = group_size
        self.bits = bits
        self.mode = mode

        # Initialize the quantized weight
        scale = math.sqrt(1.0 / input_dims)
        self.group = group or mx.distributed.init()
        N = self.group.size()

        if (input_dims % N) != 0:
            raise ValueError(
                f"The input of size {input_dims} cannot be sharded across {N} devices."
            )

        weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(output_dims, input_dims // N),
        )
        self.weight, self.scales, *biases = mx.quantize(
            weight, group_size, bits, mode=mode
        )
        self.biases = biases[0] if biases else None

        # And bias if needed
        if bias:
            self.bias = mx.zeros((output_dims,))

        # Freeze this model's parameters
        self.freeze()

    def unfreeze(self, *args, **kwargs):
        """Wrap unfreeze so that we unfreeze any layers we might contain but
        our parameters will remain frozen."""
        super().unfreeze(*args, **kwargs)
        self.freeze(recurse=False)

    def _extra_repr(self) -> str:
        out_dims, in_dims = self.weight.shape
        in_dims = (in_dims * 32) // self.bits * self.group.size()
        return (
            f"input_dims={in_dims}, output_dims={out_dims}, bias={'bias' in self}, "
            f"group_size={self.group_size}, bits={self.bits}, mode={self.mode}"
        )

    def __call__(self, x: mx.array) -> mx.array:
        # Pad input if shorter than the expanded weight (happens when the
        # original dim is not divisible by group_size — the last rank's
        # shard has a partial quantization group with padding in the weight).
        expected = (self["weight"].shape[-1] * 32) // self.bits
        if x.shape[-1] < expected:
            pad = expected - x.shape[-1]
            x = mx.concatenate(
                [x, mx.zeros((*x.shape[:-1], pad), dtype=x.dtype)], axis=-1
            )
        x = mx.quantized_matmul(
            x,
            self["weight"],
            scales=self["scales"],
            biases=self.get("biases"),
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
        )
        x = mx.distributed.all_sum(x, group=self.group)
        if "bias" in self:
            x = x + self["bias"]
        return x

    @classmethod
    def from_quantized_linear(
        cls,
        quantized_linear_layer: Module,
        *,
        segments: Union[int, list] = 1,
        group: Optional[mx.distributed.Group] = None,
    ):
        group = group or mx.distributed.init()
        output_dims, input_dims = quantized_linear_layer.weight.shape
        input_dims = (input_dims * 32) // quantized_linear_layer.bits

        sl = cls(
            input_dims,
            output_dims,
            hasattr(quantized_linear_layer, "bias"),
            group_size=quantized_linear_layer.group_size,
            bits=quantized_linear_layer.bits,
            mode=getattr(quantized_linear_layer, "mode", "affine"),
            group=group,
        )
        sl.update(
            _shard(
                quantized_linear_layer.parameters(),
                _sharded_to_all(segments),
                group,
            )
        )

        return sl
