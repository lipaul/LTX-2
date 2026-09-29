import atexit
import json
import os
import threading
import weakref
from collections import OrderedDict
from pathlib import Path

import safetensors
import torch

from ltx_core.loader.fuse_loras import FuseRule, bf16_fuse_rule
from ltx_core.loader.kernels import TRITON_AVAILABLE
from ltx_core.loader.module_ops import ModuleOps
from ltx_core.loader.primitives import StateDict
from ltx_core.loader.sd_ops import KeyValueOperationResult, SDOps
from ltx_core.model.transformer.model import LTXModel
from ltx_core.quantization.policy import QuantizationPolicy

BLOCK_SIZE = 1024


def fused_add_round_launch(target_weight: torch.Tensor, original_weight: torch.Tensor, seed: int) -> torch.Tensor:
    if not TRITON_AVAILABLE:
        raise RuntimeError(
            "fused_add_round_launch requires Triton, which is not available on this platform. "
            "Callers should gate on ltx_core.loader.kernels.TRITON_AVAILABLE and use a "
            "deterministic-rounding fallback instead."
        )
    import triton  # noqa: PLC0415

    from ltx_core.loader.kernels import fused_add_round_kernel  # noqa: PLC0415

    if original_weight.dtype == torch.float8_e4m3fn:
        exponent_bits, mantissa_bits, exponent_bias = 4, 3, 7
    elif original_weight.dtype == torch.float8_e5m2:
        exponent_bits, mantissa_bits, exponent_bias = 5, 2, 15  # noqa: F841
    else:
        raise ValueError("Unsupported dtype")

    if target_weight.dtype != torch.bfloat16:
        raise ValueError("target_weight dtype must be bfloat16")

    # Calculate grid and block sizes
    n_elements = original_weight.numel()
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # Launch kernel
    fused_add_round_kernel[grid](
        original_weight,
        target_weight,
        seed,
        n_elements,
        exponent_bias,
        mantissa_bits,
        BLOCK_SIZE,
    )
    return target_weight


def _naive_weight_or_bias_downcast(key: str, value: torch.Tensor) -> list[KeyValueOperationResult]:
    """
    Downcast the weight or bias to the float8_e4m3fn dtype.
    """
    return [KeyValueOperationResult(key, value.to(dtype=torch.float8_e4m3fn))]


# Triton's fp8e4nv (torch.float8_e4m3fn) is only a legal kernel dtype on
# sm_89+ (Ada). Below that, tl.load of an e4m3fn tensor fails at compile with
# "type fp8e4nv not supported in this architecture". PyTorch eager conversion
# still works -- that is the software path used when Triton is missing.
_TRITON_FP8E4NV_MIN_CAPABILITY = (8, 9)


def _triton_supports_fp8e4nv(device: torch.device) -> bool:
    """Return whether Triton's fused fp8 kernel can compile for *device*.
    ``device`` is the tensor's device, not the process's current CUDA device,
    so mixed-GPU hosts pick the capability that will actually run the kernel.
    """
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability(device) >= _TRITON_FP8E4NV_MIN_CAPABILITY


def _use_triton_fp8_kernel(device: torch.device) -> bool:
    """True when ``fused_add_round_launch`` can compile on *device*."""
    return TRITON_AVAILABLE and _triton_supports_fp8e4nv(device)


def _upcast_and_round(
    weight: torch.Tensor, dtype: torch.dtype, with_stochastic_rounding: bool = False, seed: int = 0
) -> torch.Tensor:
    """
    Upcast the weight to the given dtype and optionally apply stochastic rounding.
    Input weight needs to have float8_e4m3fn or float8_e5m2 dtype.
    Stochastic rounding is implemented via a Triton kernel. When Triton is not
    available (e.g., on Windows) or cannot compile ``fp8e4nv`` (pre-Ada GPUs),
    this falls back to deterministic (nearest) rounding via ``weight.to(dtype)``.
    """
    # R9A (additive): vectorized fp8(e4m3fn)->bf16 widening kernel.
    # Off by default; LTX_R9A_FP8K=1 enables it. Falls back to weight.to(dtype)
    # on any unsupported case / exception.
    if (
        weight.dtype == torch.float8_e4m3fn
        and dtype == torch.bfloat16
        and not with_stochastic_rounding
    ):
        try:
            from ltx_core.model.transformer import r8_sycl as _R8

            _g = _R8.r9a_fp8_widen(weight)
            if _g is not None:
                return _g
        except Exception:  # noqa: BLE001
            pass
    if not with_stochastic_rounding or not _use_triton_fp8_kernel(weight.device):
        return weight.to(dtype)
    return fused_add_round_launch(torch.zeros_like(weight, dtype=dtype), weight, seed)


_R18_FP8_BF16_CACHE_LOCK = threading.RLock()
_R18_FP8_BF16_CACHE_USED_BYTES = 0
_R18_FP8_BF16_CACHE_STATS = {
    "calls": 0,
    "hits": 0,
    "misses": 0,
    "stored_bytes": 0,
    "peak_bytes": 0,
    "budget_bytes": 0,
}


def _r18_fp8_bf16_cache_enabled() -> bool:
    if os.environ.get("LTX_ALL_ORIG", "0") == "1":
        return False
    value = os.environ.get("LTX_R18_FP8_BF16_CACHE", "0")
    if value not in {"0", "1"}:
        raise ValueError("LTX_R18_FP8_BF16_CACHE must be 0 or 1")
    return value == "1"


def _r18_fp8_bf16_cache_budget_bytes() -> int:
    value = os.environ.get("LTX_R18_FP8_BF16_CACHE_GIB", "6")
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError("LTX_R18_FP8_BF16_CACHE_GIB must be an integer") from exc
    if not 1 <= parsed <= 12:
        raise ValueError("LTX_R18_FP8_BF16_CACHE_GIB must be in [1,12]")
    return parsed * 1024**3


def _r18_cached_bf16_upcast(
    layer: torch.nn.Linear,
    slot: str,
    source: torch.Tensor,
    target_dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    global _R18_FP8_BF16_CACHE_USED_BYTES
    key = _r4_cache_key(source, target_dtype)
    entry = getattr(layer, slot, None)
    if entry is not None and entry[0] == key:
        _R18_FP8_BF16_CACHE_STATS["hits"] += 1
        return entry[1]
    _R18_FP8_BF16_CACHE_STATS["misses"] += 1
    fresh = _upcast_and_round(source, target_dtype, False, seed)
    fresh_bytes = fresh.numel() * fresh.element_size()
    budget = _r18_fp8_bf16_cache_budget_bytes()
    with _R18_FP8_BF16_CACHE_LOCK:
        old = getattr(layer, slot, None)
        if old is not None:
            _R18_FP8_BF16_CACHE_USED_BYTES -= old[1].numel() * old[1].element_size()
        if _R18_FP8_BF16_CACHE_USED_BYTES + fresh_bytes <= budget:
            _R18_FP8_BF16_CACHE_USED_BYTES += fresh_bytes
            setattr(layer, slot, (key, fresh))
            _R18_FP8_BF16_CACHE_STATS["stored_bytes"] = _R18_FP8_BF16_CACHE_USED_BYTES
        else:
            setattr(layer, slot, None)
        _R18_FP8_BF16_CACHE_STATS["used_bytes"] = _R18_FP8_BF16_CACHE_USED_BYTES
        _R18_FP8_BF16_CACHE_STATS["peak_bytes"] = max(
            _R18_FP8_BF16_CACHE_STATS["peak_bytes"],
            _R18_FP8_BF16_CACHE_USED_BYTES,
        )
        _R18_FP8_BF16_CACHE_STATS["budget_bytes"] = budget
    return fresh


def _r18_print_fp8_bf16_cache_stats() -> None:
    if _R18_FP8_BF16_CACHE_STATS["calls"] > 0:
        print(
            "R18_FP8_BF16_CACHE_STATS "
            + json.dumps(_R18_FP8_BF16_CACHE_STATS, sort_keys=True),
            flush=True,
        )


atexit.register(_r18_print_fp8_bf16_cache_stats)


# R20: bounded keyed oneDNN FP8 prepack path for repeated FP8 linear weights.
_R20_FP8_CACHE_MAX_ENTRIES = 2048
_R20_FP8_CACHE_BUDGET_BYTES = 16 * 1024**2
_R20_FP8_CACHE_LOCK = threading.RLock()
_R20_FP8_CACHE_LRU: OrderedDict[int, tuple[object, object]] = OrderedDict()
_R20_FP8_CACHE_USED_BYTES = 0
_R20_FP8_CACHE_STATS = {
    "forward_calls": 0,
    "direct_calls": 0,
    "weight_hits": 0,
    "weight_misses": 0,
    "bias_widenings": 0,
    "admitted": 0,
    "rejected": 0,
    "evicted": 0,
    "used_bytes": 0,
    "peak_bytes": 0,
}
_R20_FP8_QPARAM_CACHE: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

try:
    _R20_QPREPACK = torch.ops.onednn.qlinear_prepack
    _R20_QPOINTWISE = torch.ops.onednn.qlinear_pointwise
except AttributeError:
    _R20_QPREPACK = None
    _R20_QPOINTWISE = None


class _R20Fp8CacheEntry:
    __slots__ = ("key", "prepacked_weight", "bias", "nbytes")

    def __init__(self, key: tuple, prepacked_weight: torch.Tensor, bias: torch.Tensor | None, nbytes: int) -> None:
        self.key = key
        self.prepacked_weight = prepacked_weight
        self.bias = bias
        self.nbytes = nbytes


def _r20_fp8_cache_enabled() -> bool:
    if os.environ.get("LTX_ALL_ORIG", "0") == "1":
        return False
    value = os.environ.get("LTX_R20_FP8_CACHE", "0")
    if value not in {"0", "1"}:
        raise ValueError("LTX_R20_FP8_CACHE must be 0 or 1")
    return value == "1"


def _r20_tensor_cache_key(source: torch.Tensor, target_dtype: torch.dtype) -> tuple:
    version = None if torch.is_inference(source) else int(source._version)
    return (
        source.data_ptr(),
        version,
        tuple(source.shape),
        tuple(source.stride()),
        source.dtype,
        str(source.device),
        target_dtype,
    )


def _r20_linear_cache_key(
    layer: torch.nn.Linear,
    target_dtype: torch.dtype,
    shape_key: tuple[int, int],
) -> tuple:
    bias_key = _r20_tensor_cache_key(layer.bias, target_dtype) if layer.bias is not None else None
    return _r20_tensor_cache_key(layer.weight, target_dtype), bias_key, shape_key


def _r20_direct_admissible(layer: torch.nn.Linear, input: torch.Tensor) -> bool:
    weight = layer.weight
    bias = layer.bias
    return (
        _R20_QPREPACK is not None
        and _R20_QPOINTWISE is not None
        and input.is_xpu
        and input.dtype == torch.bfloat16
        and input.numel() > 0
        and weight.is_xpu
        and weight.dtype == torch.float8_e4m3fn
        and weight.dim() == 2
        and weight.is_contiguous()
        and input.shape[-1] == weight.shape[1]
        and (bias is None or (bias.dtype in (torch.float8_e4m3fn, torch.bfloat16) and bias.is_contiguous()))
    )


def _r20_qparams(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    key = str(device)
    with _R20_FP8_CACHE_LOCK:
        params = _R20_FP8_QPARAM_CACHE.get(key)
        if params is None:
            params = (
                torch.ones((), dtype=torch.float32, device=device),
                torch.zeros((), dtype=torch.float8_e4m3fn, device=device),
            )
            _R20_FP8_QPARAM_CACHE[key] = params
        return params


def _r20_cache_ref(layer: torch.nn.Linear, identifier: int) -> object:
    def remove(ref: object) -> None:
        global _R20_FP8_CACHE_USED_BYTES
        with _R20_FP8_CACHE_LOCK:
            current = _R20_FP8_CACHE_LRU.get(identifier)
            if current is None or current[0] is not ref:
                return
            _R20_FP8_CACHE_LRU.pop(identifier, None)
            _R20_FP8_CACHE_USED_BYTES -= current[1].nbytes
            _R20_FP8_CACHE_STATS["evicted"] += 1
            _R20_FP8_CACHE_STATS["used_bytes"] = _R20_FP8_CACHE_USED_BYTES

    return weakref.ref(layer, remove)


def _r20_remove_entry_locked(identifier: int, layer: torch.nn.Linear) -> None:
    global _R20_FP8_CACHE_USED_BYTES
    old = _R20_FP8_CACHE_LRU.pop(identifier, None)
    if old is not None:
        entry = old[1]
        _R20_FP8_CACHE_USED_BYTES -= entry.nbytes
        _R20_FP8_CACHE_STATS["used_bytes"] = _R20_FP8_CACHE_USED_BYTES
        if getattr(layer, "_r20_fp8_cache_entry", None) is entry:
            delattr(layer, "_r20_fp8_cache_entry")


def _r20_admit_entry_locked(
    layer: torch.nn.Linear,
    identifier: int,
    entry: _R20Fp8CacheEntry,
) -> bool:
    global _R20_FP8_CACHE_USED_BYTES
    _r20_remove_entry_locked(identifier, layer)
    while _R20_FP8_CACHE_LRU and (
        len(_R20_FP8_CACHE_LRU) >= _R20_FP8_CACHE_MAX_ENTRIES
        or entry.nbytes > _R20_FP8_CACHE_BUDGET_BYTES - _R20_FP8_CACHE_USED_BYTES
    ):
        old_identifier, old_ref_entry = next(iter(_R20_FP8_CACHE_LRU.items()))
        old_layer = old_ref_entry[0]()
        if old_layer is None:
            _R20_FP8_CACHE_LRU.pop(old_identifier, None)
            _R20_FP8_CACHE_USED_BYTES -= old_ref_entry[1].nbytes
            _R20_FP8_CACHE_STATS["used_bytes"] = _R20_FP8_CACHE_USED_BYTES
            continue
        _r20_remove_entry_locked(old_identifier, old_layer)
        _R20_FP8_CACHE_STATS["evicted"] += 1
    if entry.nbytes > _R20_FP8_CACHE_BUDGET_BYTES - _R20_FP8_CACHE_USED_BYTES:
        _R20_FP8_CACHE_STATS["rejected"] += 1
        return False
    _R20_FP8_CACHE_LRU[identifier] = (_r20_cache_ref(layer, identifier), entry)
    _R20_FP8_CACHE_USED_BYTES += entry.nbytes
    _R20_FP8_CACHE_STATS["admitted"] += 1
    _R20_FP8_CACHE_STATS["used_bytes"] = _R20_FP8_CACHE_USED_BYTES
    _R20_FP8_CACHE_STATS["peak_bytes"] = max(
        _R20_FP8_CACHE_STATS["peak_bytes"], _R20_FP8_CACHE_USED_BYTES
    )
    setattr(layer, "_r20_fp8_cache_entry", entry)
    return True


def _r20_cache_stats_snapshot() -> dict[str, int | float | None]:
    with _R20_FP8_CACHE_LOCK:
        total = _R20_FP8_CACHE_STATS["weight_hits"] + _R20_FP8_CACHE_STATS["weight_misses"]
        result: dict[str, int | float | None] = dict(_R20_FP8_CACHE_STATS)
        result["hit_rate"] = (_R20_FP8_CACHE_STATS["weight_hits"] / total) if total else None
        result["max_entries"] = _R20_FP8_CACHE_MAX_ENTRIES
        result["budget_bytes"] = _R20_FP8_CACHE_BUDGET_BYTES
        return result


def r20_fp8_cache_stats() -> dict[str, int | float | None]:
    return _r20_cache_stats_snapshot()


def _r20_print_cache_stats() -> None:
    if _R20_FP8_CACHE_STATS["direct_calls"] == 0:
        return
    print("R20_FP8_CACHE_STATS " + json.dumps(_r20_cache_stats_snapshot(), sort_keys=True), flush=True)


atexit.register(_r20_print_cache_stats)


def _r20_direct_fp8_linear(layer: torch.nn.Linear, input: torch.Tensor) -> torch.Tensor | None:
    if not _r20_direct_admissible(layer, input):
        return None
    global _R20_FP8_CACHE_USED_BYTES
    identifier = id(layer)
    shape_key = (input.numel() // input.shape[-1], input.shape[-1])
    key = _r20_linear_cache_key(layer, torch.bfloat16, shape_key)
    entry = getattr(layer, "_r20_fp8_cache_entry", None)
    with _R20_FP8_CACHE_LOCK:
        _R20_FP8_CACHE_STATS["forward_calls"] += 1
        if entry is not None and entry.key == key:
            current = _R20_FP8_CACHE_LRU.get(identifier)
            if current is None or current[1] is not entry:
                _R20_FP8_CACHE_LRU[identifier] = (_r20_cache_ref(layer, identifier), entry)
            else:
                _R20_FP8_CACHE_LRU.move_to_end(identifier)
            _R20_FP8_CACHE_STATS["direct_calls"] += 1
            _R20_FP8_CACHE_STATS["weight_hits"] += 1
            prepacked_weight = entry.prepacked_weight
            bias = entry.bias
        else:
            _R20_FP8_CACHE_STATS["weight_misses"] += 1
            prepacked_weight = _R20_QPREPACK(layer.weight, [shape_key[0], shape_key[1]])
            bias = layer.bias
            if bias is not None and bias.dtype == torch.float8_e4m3fn:
                with _R20_FP8_CACHE_LOCK:
                    _R20_FP8_CACHE_STATS["bias_widenings"] += 1
                bias = _upcast_and_round(bias, torch.bfloat16, False, layer._seed)
            nbytes = 0 if bias is None else bias.numel() * bias.element_size()
            new_entry = _R20Fp8CacheEntry(key, prepacked_weight, bias, nbytes)
            with _R20_FP8_CACHE_LOCK:
                _r20_admit_entry_locked(layer, identifier, new_entry)
            _R20_FP8_CACHE_STATS["direct_calls"] += 1

    scale, zero_point = _r20_qparams(input.device)
    matrix = input if input.dim() == 2 else input.reshape(-1, input.shape[-1])
    if not matrix.is_contiguous():
        matrix = matrix.contiguous()
    output = _R20_QPOINTWISE(
        matrix,
        1.0,
        0,
        prepacked_weight,
        scale,
        zero_point,
        bias,
        1.0,
        0,
        torch.bfloat16,
        "none",
        [],
        "none",
    )
    return output.reshape(*input.shape[:-1], output.shape[-1])


# --- R4 (additive): bounded per-layer bf16 cache for fp8 -> bf16 upcast (Scheme C).
#
# Off by default. Set ``LTX_R4_FP8CACHE=1`` to enable; unset or ``=0`` keeps the
# original path bit-for-bit. ``LTX_ALL_ORIG=1`` also forces the original path
# (round-0 baseline, same convention as the R1/R2/R3/R5/R6 switches).
_R4_FP8CACHE_BUDGET_BYTES = 5 * 1024**3  # 5 GiB total across all layers
_R4_FP8CACHE_MIN_NUMEL = 4096  # smaller tensors skip the cache (recomputed directly)

_r4_fp8cache_used_bytes = 0
_r4_fp8cache_lock = threading.Lock()


def _r4_fp8cache_enabled() -> bool:
    """True only when the R4 cache is explicitly switched on."""
    if os.environ.get("LTX_ALL_ORIG", "0") == "1":
        return False
    return os.environ.get("LTX_R4_FP8CACHE", "0") == "1"


def _r4_cache_key(source: torch.Tensor, target_dtype: torch.dtype) -> tuple:
    """Identity of an upcast result: source storage + version + shape + dtypes.

    Inference tensors (``torch.is_inference``) carry no version counter and
    are immutable, so the key omits ``_version`` for them and relies on
    data_ptr / shape / dtype / device.
    """
    if torch.is_inference(source):
        return (
            source.data_ptr(),
            tuple(source.shape),
            source.dtype,
            str(source.device),
            target_dtype,
        )
    return (
        source.data_ptr(),
        source._version,
        tuple(source.shape),
        source.dtype,
        str(source.device),
        target_dtype,
    )


def _r4_cached_upcast(
    layer: torch.nn.Linear,
    slot: str,
    source: torch.Tensor,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    """Return ``source.to(target_dtype)`` via the per-layer bounded bf16 cache.

    * Hit (same key): return the stored bf16 tensor (bitwise identical).
    * Small tensors (``numel < _R4_FP8CACHE_MIN_NUMEL``): computed directly,
      never stored.
    * Miss: compute directly, then admit only if the 5 GiB budget still holds;
      over budget the fresh tensor is returned without storing (fallback).
    * Errors are never swallowed: only the budget gate triggers the fallback.
    """
    global _r4_fp8cache_used_bytes
    if source.numel() < _R4_FP8CACHE_MIN_NUMEL:
        return source.to(target_dtype)
    key = _r4_cache_key(source, target_dtype)
    entry = getattr(layer, slot, None)
    if entry is not None and entry[0] == key:
        cached = entry[1]
        if cached.dtype == target_dtype and tuple(cached.shape) == tuple(source.shape):
            return cached
    fresh = source.to(target_dtype)
    fresh_bytes = fresh.numel() * fresh.element_size()
    with _r4_fp8cache_lock:
        old = getattr(layer, slot, None)
        if old is not None:
            _r4_fp8cache_used_bytes -= old[1].numel() * old[1].element_size()
        if _r4_fp8cache_used_bytes + fresh_bytes <= _R4_FP8CACHE_BUDGET_BYTES:
            _r4_fp8cache_used_bytes += fresh_bytes
            setattr(layer, slot, (key, fresh))
        else:
            # Over budget: fall back to the direct result; store nothing.
            setattr(layer, slot, None)
    return fresh


class Fp8CastLinear(torch.nn.Linear):
    """nn.Linear storing weights in fp8, upcasting to input dtype during forward.
    Used via __class__ reassignment (not subclassing) so existing weight tensors
    are preserved in-place. Class-level forward is required for torch.compile
    compatibility — instance-level closure monkey-patches cause graph breaks.
    """

    _with_stochastic_rounding: bool
    _seed: int

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002, type: ignore[override]
        if (
            _r18_fp8_bf16_cache_enabled()
            and not self._with_stochastic_rounding
            and input.dtype == torch.bfloat16
            and self.weight.dtype == torch.float8_e4m3fn
            and (self.bias is None or self.bias.dtype == torch.float8_e4m3fn)
            and not torch.is_grad_enabled()
        ):
            _R18_FP8_BF16_CACHE_STATS["calls"] += 1
            weight_up = _r18_cached_bf16_upcast(
                self, "_r18_w_up", self.weight, input.dtype, self._seed
            )
            bias_up = (
                _r18_cached_bf16_upcast(
                    self, "_r18_b_up", self.bias, input.dtype, self._seed
                )
                if self.bias is not None
                else None
            )
            return torch.nn.functional.linear(input, weight_up, bias_up)
        if (
            _r20_fp8_cache_enabled()
            and not self._with_stochastic_rounding
            and os.environ.get("LTX_R18_SYCLEFF", "0") != "1"
            and not torch.is_grad_enabled()
        ):
            direct_output = _r20_direct_fp8_linear(self, input)
            if direct_output is not None:
                return direct_output
        if _r4_fp8cache_enabled() and not self._with_stochastic_rounding:
            # R4 (Scheme C): per-layer bounded bf16 cache for the fp8 -> bf16
            # upcast. Stochastic rounding bypasses the cache (each call draws
            # a new sample); small and over-budget tensors fall back to the
            # direct conversion inside _r4_cached_upcast.
            w_up = _r4_cached_upcast(self, "_r4_w_up", self.weight, input.dtype)
            b_up = (
                _r4_cached_upcast(self, "_r4_b_up", self.bias, input.dtype)
                if self.bias is not None
                else None
            )
            return torch.nn.functional.linear(input, w_up, b_up)
        if (
            not self._with_stochastic_rounding
            and input.dtype == torch.bfloat16
            and self.weight.dtype == torch.float8_e4m3fn
            and (self.bias is None or self.bias.dtype == torch.float8_e4m3fn)
        ):
            from ltx_core.model.transformer.r8_sycl import (
                r18_sycleff_on,
                r9a_fp8_widen_batch,
                r9a_fp8_widen_pair,
            )

            if r18_sycleff_on():
                descriptors = [self.weight]
                if self.bias is not None:
                    descriptors.append(self.bias)
                batch = r9a_fp8_widen_batch(descriptors)
                if batch is not None:
                    return torch.nn.functional.linear(
                        input, batch[0], batch[1] if self.bias is not None else None
                    )
            # R41: one batch launch for the weight and its bias, independent of
            # LTX_R18_SYCLEFF. The two tensors are independent upcasts, so a single
            # R8Fp8WidenBatchKernel launch replaces two, removing one launch per
            # bias-bearing fp8 linear.
            pair = r9a_fp8_widen_pair(self.weight, self.bias)
            if pair is not None:
                return torch.nn.functional.linear(input, pair[0], pair[1])
        w_up = _upcast_and_round(self.weight, input.dtype, self._with_stochastic_rounding, self._seed)
        b_up = (
            _upcast_and_round(self.bias, input.dtype, self._with_stochastic_rounding, self._seed)
            if self.bias is not None
            else None
        )
        return torch.nn.functional.linear(input, w_up, b_up)

def _replace_fwd_with_upcast(layer: torch.nn.Linear, with_stochastic_rounding: bool = False, seed: int = 0) -> None:
    """
    Intended to be applied via __class__ reassignment to existing nn.Linear
    instances. Forward remains defined at the class level, which is required for
    torch.compile compatibility — instance-level closure monkey-patches cause
    graph breaks.
    Also retypes ``weight`` and ``bias`` to fp8 so the meta param dtype matches
    the post-load tensor dtype (sd_ops downcasts checkpoint bf16 -> fp8 at load).
    Block streaming relies on this to derive pool buffer layout from the meta
    model without an eager checkpoint read.
    """
    layer.__class__ = Fp8CastLinear
    layer._with_stochastic_rounding = with_stochastic_rounding
    layer._seed = seed
    layer.weight = torch.nn.Parameter(
        torch.empty(layer.weight.shape, dtype=torch.float8_e4m3fn, device=layer.weight.device),
        requires_grad=layer.weight.requires_grad,
    )
    if layer.bias is not None:
        layer.bias = torch.nn.Parameter(
            torch.empty(layer.bias.shape, dtype=torch.float8_e4m3fn, device=layer.bias.device),
            requires_grad=layer.bias.requires_grad,
        )


# Module-name suffixes for the Linears that participate in fp8 cast. Used by
# both the upcast matcher and the sd_ops downcast map so the two cannot drift.
# - ``.to_q`` / ``.to_k`` / ``.to_v`` / ``.to_out.0`` have a leading dot so they
#   only match the attention Linears at ``...attnN.to_q`` etc.
# - ``ff.net.0.proj`` / ``ff.net.2`` are intentionally **dotless** so they match
#   both video FF (``...ff.net.0.proj``) and audio FF (``...audio_ff.net.0.proj``).
_FP8_CAST_KEY_PREFIX = "transformer_blocks."
_FP8_CAST_LINEAR_SUFFIXES: tuple[str, ...] = (
    ".to_q",
    ".to_k",
    ".to_v",
    ".to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
)


def _is_fp8_cast_linear(module_name: str) -> bool:
    """Return True if *module_name* names a Linear that should be fp8-cast."""
    if _FP8_CAST_KEY_PREFIX not in module_name:
        return False
    return any(module_name.endswith(suffix) for suffix in _FP8_CAST_LINEAR_SUFFIXES)


def _amend_forward_with_upcast(
    model: torch.nn.Module, with_stochastic_rounding: bool = False, seed: int = 0
) -> torch.nn.Module:
    """
    Replace the forward method of the fp8-cast Linear layers (per
    :data:`_FP8_CAST_LINEAR_SUFFIXES`) to forward with upcast and optional
    stochastic rounding.
    Only the Linears whose weights are downcast by :data:`TRANSFORMER_LINEAR_DOWNCAST_MAP`
    are retyped. Linears outside that subset (e.g. ``to_gate_logits``) are left as
    plain ``nn.Linear`` so the meta-model param dtype matches the loaded checkpoint
    dtype.
    """
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) and _is_fp8_cast_linear(name):
            _replace_fwd_with_upcast(m, with_stochastic_rounding, seed)
    return model


def _build_transformer_linear_downcast_map() -> SDOps:
    """Build the sd_ops downcast map from the same suffix registry as the matcher."""
    ops = SDOps("TRANSFORMER_LINEAR_DOWNCAST_MAP")
    for suffix in _FP8_CAST_LINEAR_SUFFIXES:
        ops = ops.with_kv_operation(
            key_prefix=_FP8_CAST_KEY_PREFIX,
            key_suffix=suffix + ".weight",
            operation=_naive_weight_or_bias_downcast,
        ).with_kv_operation(
            key_prefix=_FP8_CAST_KEY_PREFIX,
            key_suffix=suffix + ".bias",
            operation=_naive_weight_or_bias_downcast,
        )
    return ops


TRANSFORMER_LINEAR_DOWNCAST_MAP = _build_transformer_linear_downcast_map()

UPCAST_DURING_INFERENCE = ModuleOps(
    name="upcast_fp8_during_linear_forward",
    matcher=lambda model: isinstance(model, LTXModel),
    mutator=lambda model: _amend_forward_with_upcast(model, False),
)


class UpcastWithStochasticRounding(ModuleOps):
    """
    ModuleOps for upcasting the model's float8_e4m3fn weights and biases to the bfloat16 dtype
    and applying stochastic rounding during linear forward.
    """

    def __new__(cls, seed: int = 0):
        return super().__new__(
            cls,
            name="upcast_fp8_during_linear_forward_with_stochastic_rounding",
            matcher=lambda model: isinstance(model, LTXModel),
            mutator=lambda model: _amend_forward_with_upcast(model, True, seed),
        )


def fuse_cast_fp8_weight(
    delta_bf16: torch.Tensor,
    weight_fp8: torch.Tensor,
) -> torch.Tensor:
    """Return ``(delta_bf16 + dequantize(weight_fp8)).to(weight_fp8.dtype)``.
    CUDA with Triton on sm_89+ uses stochastic rounding via the fused kernel;
    otherwise falls back to a deterministic bf16 add (no Triton, or a GPU
    whose Triton cannot compile ``fp8e4nv``). ``delta_bf16`` is the bf16
    accumulator and is mutated in place.
    """
    if delta_bf16.dtype != torch.bfloat16:
        raise ValueError(f"delta_bf16 must be bfloat16, got {delta_bf16.dtype}")
    if _use_triton_fp8_kernel(weight_fp8.device):
        fused_add_round_launch(delta_bf16, weight_fp8, seed=0)
    else:
        delta_bf16.add_(weight_fp8.to(dtype=torch.bfloat16))
    return delta_bf16.to(dtype=weight_fp8.dtype)


def _fp8_cast_fuse(
    key: str,
    weight: torch.Tensor,
    deltas: torch.Tensor,
    model_sd: StateDict,
) -> dict[str, torch.Tensor]:
    """Cast the dequantized FP8 weight + BF16 deltas back to ``weight.dtype``
    (FP8) via the fused-add-round kernel on CUDA when Triton can compile
    ``fp8e4nv``, otherwise via a deterministic software add.
    Only a subset of linears are FP8-downcast (see ``TRANSFORMER_LINEAR_DOWNCAST_MAP``);
    LoRAs may also target layers left in BF16 (e.g. audio ``add_q/k/v_proj``, cross-modal
    projections). For those, fall back to a plain BF16 fuse.
    """
    if weight.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        return bf16_fuse_rule(key, weight, deltas, model_sd)
    return {key: fuse_cast_fp8_weight(deltas, weight)}


fp8_cast_fuse_rule = FuseRule(aggregation_dtype=torch.bfloat16, fuse_fn=_fp8_cast_fuse)


# Raw safetensors storage prefix shared by every diffusion-transformer
# parameter (and every prequant `*_scale` sibling). Verified against
# ltx-2.3-22b-{dev,distilled}-fp8.safetensors: 2924/2924 and 2992/2992 of
# the scale keys start with this exact prefix.
_RAW_DIFFUSION_MODEL_PREFIX = "model.diffusion_model."


def _read_scales(checkpoint_path: str | Path) -> dict[str, torch.Tensor]:
    """Return ``{post_rename_param_key: scale_tensor}`` for every prequant
    ``*_scale`` sibling in *checkpoint_path*.
    Keys are returned in the post-rename form the loader will pass to the
    sd-op (e.g. ``transformer_blocks.0.attn1.to_q.weight``) -- the raw
    ``model.diffusion_model.`` prefix and the ``_scale`` suffix are both
    stripped. Catches both ``.weight_scale`` and ``.bias_scale``; the
    latter is absent in the current LTX-2.3 prequant checkpoints but
    accepted for forward compatibility.
    """
    out: dict[str, torch.Tensor] = {}
    with safetensors.safe_open(str(checkpoint_path), framework="pt", device="cpu") as h:
        raw_keys = h.keys()
        for k in raw_keys:
            if not k.endswith("_scale"):
                continue
            if not k.startswith(_RAW_DIFFUSION_MODEL_PREFIX):
                raise ValueError(
                    f"Scale key {k!r} does not start with the expected raw prefix {_RAW_DIFFUSION_MODEL_PREFIX!r}"
                )
            param_key = k.removeprefix(_RAW_DIFFUSION_MODEL_PREFIX).removesuffix("_scale")
            out[param_key] = h.get_tensor(k)
    return out


def _build_prequant_fold_sd_ops(scales: dict[str, torch.Tensor]) -> SDOps:
    """Build sd-ops that fold prequant ``*_scale`` siblings into their parent
    tensor at load time.
    *scales* is keyed by the **post-rename** param key (e.g.
    ``transformer_blocks.0.attn1.to_q.weight``); see :func:`_read_scales`.
    Four ``with_kv_operation`` entries (symmetric for ``.weight`` and ``.bias``):
    * ``.weight`` / ``.bias`` -> if a sibling scale exists in *scales*, fold;
      then delegate to ``TRANSFORMER_LINEAR_DOWNCAST_MAP`` (downcast covered
      Linears, pass everything else through). Without a scale, delegate
      directly.
    * ``.weight_scale`` / ``.bias_scale`` -> drop (the scale is consumed by
      the fold). Raises if the scale key doesn't correspond to a known
      entry in *scales* -- that means the file shipped a scale we didn't
      pre-register, which would silently desync the fold.
    """

    def _on_param(param_key: str, value: torch.Tensor) -> list[KeyValueOperationResult]:
        scale = scales.get(param_key)
        if scale is None:
            return TRANSFORMER_LINEAR_DOWNCAST_MAP.apply_to_key_value(param_key, value)
        scale = scale.to(device=value.device)
        if scale.ndim != 0:
            raise ValueError(f"Unsupported scale shape {tuple(scale.shape)} for {param_key}")
        bf16 = (value.to(torch.float32) * scale).to(torch.bfloat16)
        # Delegate the final fp8-vs-bf16 decision to the downcast map: Linears
        # outside the fp8 subset (e.g. to_gate_logits) stay bf16 to match the
        # plain nn.Linear that the upcast matcher leaves untouched.
        return TRANSFORMER_LINEAR_DOWNCAST_MAP.apply_to_key_value(param_key, bf16)

    def _drop_scale(scale_key: str, _value: torch.Tensor) -> list[KeyValueOperationResult]:
        param_key = scale_key.removesuffix("_scale")
        if param_key not in scales:
            raise ValueError(
                f"Scale key {scale_key!r} has no matching entry in the prequant scales dict; "
                f"_read_scales and the loader's rename map have drifted"
            )
        return []

    # Register the drop ops first so the dict-membership sanity check is the
    # earliest sd-op that can fire on a scale key -- we crash on a stray scale
    # before any silently mismatched fold has a chance to land in the state
    # dict. Registration order is irrelevant for correctness (no overlap
    # between matchers) but communicates intent.
    return (
        SDOps("FP8_CAST_PREQUANT_AWARE")
        .with_kv_operation(key_suffix=".weight_scale", operation=_drop_scale)
        .with_kv_operation(key_suffix=".bias_scale", operation=_drop_scale)
        .with_kv_operation(key_suffix=".weight", operation=_on_param)
        .with_kv_operation(key_suffix=".bias", operation=_on_param)
    )


def build_policy(checkpoint_path: str | Path) -> QuantizationPolicy:
    """FP8 casting with upcasting during inference.
    *checkpoint_path* is required (mirroring ``fp8_scaled_mm.build_policy``).
    For prequantized fp8 checkpoints, sibling ``*_scale`` tensors (weight or
    bias) are folded into the parent at load time.
    """
    scales = _read_scales(checkpoint_path)
    return QuantizationPolicy(
        sd_ops=_build_prequant_fold_sd_ops(scales),
        module_ops=(UPCAST_DURING_INFERENCE,),
        fuse_rule=fp8_cast_fuse_rule,
    )
