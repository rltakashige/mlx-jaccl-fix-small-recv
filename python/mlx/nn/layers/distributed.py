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


def compute_shard_sizes(dim: int, N: int, unit: int = 1) -> list[int]:
    """Distribute *dim* elements across *N* shards as evenly as possible.

    When *unit* > 1 every shard size is a multiple of *unit*.
    The first ``(dim // unit) % N`` shards receive one extra *unit*.
    """
    if unit > 1 and dim % unit != 0:
        raise ValueError(f"dim ({dim}) must be divisible by unit ({unit})")
    n_units = dim // unit
    base, rem = divmod(n_units, N)
    return [(base + (1 if i < rem else 0)) * unit for i in range(N)]


def _sizes_to_indices(sizes: list[int]) -> list[int]:
    """Convert a list of shard sizes to cumulative split indices.

    ``[4, 3, 3]`` → ``[4, 7]``
    """
    indices: list[int] = []
    cumsum = 0
    for s in sizes[:-1]:
        cumsum += s
        indices.append(cumsum)
    return indices


def _uneven_split(weight: mx.array, N: int, axis: int, unit: int = 1) -> list[mx.array]:
    """Like ``mx.split(weight, N, axis)`` but allows ``weight.shape[axis] % N != 0``.

    When *unit* > 1, shard sizes are multiples of *unit*.
    """
    dim = weight.shape[axis]
    if unit == 1 and dim % N == 0:
        return mx.split(weight, N, axis=axis)
    return mx.split(
        weight, _sizes_to_indices(compute_shard_sizes(dim, N, unit)), axis=axis
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
        if isinstance(s, int):
            axis = s
        elif isinstance(s, tuple):
            if len(s) == 2:
                axis, segments = s
            elif len(s) == 3:
                axis, segments, unit = s
            else:
                raise ValueError(
                    "The sharding function should return int, "
                    "tuple[int, segments], or tuple[int, segments, unit]"
                )
        else:
            raise ValueError(
                "The sharding function should return int, "
                "tuple[int, segments], or tuple[int, segments, unit]"
            )

        return mx.contiguous(
            mx.concatenate(
                [
                    _uneven_split(part, N, axis, unit)[r]
                    for part in _split(weight, segments, axis)
                ],
                axis=axis,
            )
        )

    return tree_map_with_path(_shard_fn, parameters)


def _all_to_sharded(segments, unit=1):
    """Simple predicate to shard fully connected layers such that a common
    representation becomes a sharded representation."""

    def _shard_fn(path, weight):
        if path.endswith("bias"):
            return -1, segments, unit
        return max(weight.ndim - 2, 0), segments, unit

    return _shard_fn


def _sharded_to_all(segments, unit=1):
    """Simple predicate to shard fully connected layers such that a sharded
    representation becomes a common representation."""

    def _shard_fn(path, weight):
        if path.endswith("bias"):
            return None
        return -1, segments, unit

    return _shard_fn


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
        return _shard(parameters, _sharded_to_all(segments, unit), group)

    # Convert element-space unit to quantization-group-space unit.
    if unit > 1:
        if unit % group_size != 0:
            raise ValueError(
                f"unit ({unit}) must be divisible by group_size ({group_size}) "
                "for quantized sharded-to-all sharding"
            )
        unit_qg = unit // group_size
    else:
        unit_qg = 1

    # Number of quantization groups along the input axis.
    num_quant_groups = scales.shape[-1]
    group_counts = compute_shard_sizes(num_quant_groups, N, unit_qg)

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
            indices = _sizes_to_indices(sizes)
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
            indices = _sizes_to_indices(sizes)
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
                )
            )
        else:
            predicate = (
                _all_to_sharded(segments, unit)
                if sharding == "all-to-sharded"
                else _sharded_to_all(segments, unit)
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
    if sharding == "sharded-to-all" and not is_linear:
        sharded = _shard_quantized_s2a(
            module.parameters(), group, module.group_size, module.bits, segments, unit
        )
    else:
        predicate = (
            _all_to_sharded(segments, unit)
            if sharding == "all-to-sharded"
            else _sharded_to_all(segments, unit)
        )
        sharded = _shard(module.parameters(), predicate, group)

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
