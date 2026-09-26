"""R26 XPU kernels for Diffusion VAE normalization and absolute RoPE chains."""

from __future__ import annotations

import os
from pathlib import Path

import torch


_LIB: object | None = None
_LIB_READY = False
_ROPE_TABLE_CACHE: dict[
    tuple[object, ...],
    tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]],
] = {}
_AXIS_TABLE_CACHE: dict[
    tuple[object, ...],
    tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]],
] = {}


def r26_on() -> bool:
    """Return whether the R26 VAE chain is enabled."""
    if os.environ.get("LTX_ALL_ORIG", "0") == "1":
        return False
    value = os.environ.get("LTX_R26_VAE_CHAIN", "0")
    if value not in {"0", "1"}:
        raise ValueError("LTX_R26_VAE_CHAIN must be 0 or 1")
    return value == "1"


def _library() -> object | None:
    global _LIB, _LIB_READY
    if not r26_on():
        return None
    if _LIB_READY:
        return _LIB
    _LIB_READY = True
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        return None
    here = Path(__file__).resolve()
    candidates = [
        Path("/home/simon/ltx2.5-demo-main/LTX-2/packages/ltx-kernels/csrc/sycl_r26/libr26_vae.so"),
        here.parents[6] / "ltx-kernels" / "csrc" / "sycl_r26" / "libr26_vae.so",
    ]
    library_path = next((path for path in candidates if path.is_file()), None)
    if library_path is None:
        return None
    torch.ops.load_library(str(library_path))
    _LIB = torch.ops.libr26_vae
    return _LIB


def r26_norm_affine(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    out: torch.Tensor,
    eps: float,
) -> bool:
    """Write RMSNorm followed by channel scale and shift into ``out``."""
    lib = _library()
    if lib is None or x.ndim < 1:
        return False
    c = int(x.shape[-1])
    rows = int(x.numel()) // c if c > 0 else 0
    tensors = (x, norm_weight, scale, shift, out)
    if not (
        64 <= c <= 256
        and c % 8 == 0
        and rows > 0
        and out.shape == x.shape
        and all(t.is_xpu and t.dtype == torch.bfloat16 for t in tensors)
        and all(t.is_contiguous() and t.stride(-1) == 1 for t in tensors)
        and all(t.device == x.device for t in tensors)
        and all(t.data_ptr() % 16 == 0 for t in tensors)
        and norm_weight.numel() == c
        and scale.numel() == c
        and shift.numel() == c
    ):
        return False
    lib.r26_norm_affine(x, norm_weight, scale, shift, out, float(eps))
    return True


def r26_norm_affine_precomputed(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    affine_scale: torch.Tensor,
    shift: torch.Tensor,
    out: torch.Tensor,
    eps: float,
) -> bool:
    """Write RMSNorm with a precomputed bf16 channel affine scale into ``out``."""
    lib = _library()
    if lib is None or x.ndim < 1:
        return False
    c = int(x.shape[-1])
    rows = int(x.numel()) // c if c > 0 else 0
    tensors = (x, norm_weight, affine_scale, shift, out)
    if not (
        64 <= c <= 256
        and c % 8 == 0
        and rows > 0
        and out.shape == x.shape
        and all(t.is_xpu and t.dtype == torch.bfloat16 for t in tensors)
        and all(t.is_contiguous() and t.stride(-1) == 1 for t in tensors)
        and all(t.device == x.device for t in tensors)
        and all(t.data_ptr() % 16 == 0 for t in tensors)
        and norm_weight.numel() == c
        and affine_scale.numel() == c
        and shift.numel() == c
    ):
        return False
    lib.r26_norm_affine_precomputed(
        x, norm_weight, affine_scale, shift, out, float(eps)
    )
    return True


def _version_counter(tensor: torch.Tensor) -> int:
    """Return the tensor version counter; inference-mode tensors return -1.

    Inference-mode tensors carry no version counter and are reported as -1, a
    value outside the range of any real version counter.
    """
    if torch.is_inference(tensor):
        return -1
    return tensor._version


def _axis_tables(pos: torch.Tensor, inv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    angle = pos[:, None] * inv[None, :]
    return angle.cos().reshape(-1), angle.sin().reshape(-1)


def _cached_axis_tables(
    device: torch.device,
    length: int,
    inv: torch.Tensor,
    positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if positions is None:
        key = (str(device), length, id(inv), _version_counter(inv))
    else:
        key = (
            str(device),
            length,
            id(positions),
            _version_counter(positions),
            id(inv),
            _version_counter(inv),
        )
    cached = _AXIS_TABLE_CACHE.get(key)
    if cached is not None:
        return cached[0], cached[1]
    if positions is None:
        positions = torch.arange(length, device=device, dtype=torch.float32)
    cos, sin = _axis_tables(positions, inv)
    _AXIS_TABLE_CACHE[key] = (cos, sin, (inv, positions))
    return cos, sin


def _rope_tables(
    query_device: torch.device,
    t: int,
    h: int,
    w: int,
    d_t: int,
    d_h: int,
    d_w: int,
    inv_t: torch.Tensor,
    inv_h: torch.Tensor,
    inv_w: torch.Tensor,
    w_pos: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    key = (
        str(query_device),
        t,
        h,
        w,
        d_t,
        d_h,
        d_w,
        id(inv_t),
        id(inv_h),
        id(inv_w),
        id(w_pos),
        _version_counter(inv_t),
        _version_counter(inv_h),
        _version_counter(inv_w),
        _version_counter(w_pos),
    )
    cached = _ROPE_TABLE_CACHE.get(key)
    if cached is not None:
        return cached[0], cached[1]
    cos_t, sin_t = _cached_axis_tables(query_device, t, inv_t)
    cos_h, sin_h = _cached_axis_tables(query_device, h, inv_h)
    cos_w, sin_w = _cached_axis_tables(query_device, w, inv_w, w_pos)
    cos = torch.cat((cos_t, cos_h, cos_w))
    sin = torch.cat((sin_t, sin_h, sin_w))
    _ROPE_TABLE_CACHE[key] = (cos, sin, (inv_t, inv_h, inv_w, w_pos))
    return cos, sin


def r26_norm_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    inv_t: torch.Tensor,
    inv_h: torch.Tensor,
    inv_w: torch.Tensor,
    w_pos: torch.Tensor,
    d_t: int,
    d_h: int,
    eps: float,
    query_scale: float = 1.0,
    key_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Fuse VAE q/k head RMSNorm, q scale, and three-axis absolute RoPE."""
    lib = _library()
    if lib is None:
        return None
    if query.ndim != 6 or key.ndim != 6 or query.shape != key.shape:
        return None
    batch, t, h, w, heads, head_dim = (int(v) for v in query.shape)
    d_w = head_dim - int(d_t) - int(d_h)
    if (
        batch <= 0
        or t <= 0
        or h <= 0
        or w <= 0
        or heads <= 0
        or head_dim != 64
        or d_t <= 0
        or d_h <= 0
        or d_t % 2 != 0
        or d_h % 2 != 0
        or d_w <= 0
        or d_w % 2 != 0
    ):
        return None
    tensors = (query, key, query_weight, key_weight, inv_t, inv_h, inv_w, w_pos)
    if not all(tensor.is_xpu for tensor in tensors):
        return None
    if not (
        query.dtype == torch.bfloat16
        and key.dtype == torch.bfloat16
        and query_weight.dtype == torch.bfloat16
        and key_weight.dtype == torch.bfloat16
        and inv_t.dtype == torch.float32
        and inv_h.dtype == torch.float32
        and inv_w.dtype == torch.float32
        and w_pos.dtype == torch.float32
    ):
        return None
    if not (
        query.is_contiguous()
        and key.is_contiguous()
        and query_weight.is_contiguous()
        and key_weight.is_contiguous()
        and inv_t.is_contiguous()
        and inv_h.is_contiguous()
        and inv_w.is_contiguous()
        and w_pos.is_contiguous()
    ):
        return None
    if not (
        query_weight.numel() == head_dim
        and key_weight.numel() == head_dim
        and w_pos.ndim == 1
        and w_pos.numel() == w
        and inv_t.ndim == 1
        and inv_h.ndim == 1
        and inv_w.ndim == 1
        and inv_t.numel() == d_t // 2
        and inv_h.numel() == d_h // 2
        and inv_w.numel() == d_w // 2
        and query.data_ptr() % 16 == 0
        and key.data_ptr() % 16 == 0
        and query_weight.data_ptr() % 16 == 0
        and key_weight.data_ptr() % 16 == 0
    ):
        return None
    if not all(tensor.device == query.device for tensor in tensors):
        return None

    cos, sin = _rope_tables(
        query.device,
        t,
        h,
        w,
        int(d_t),
        int(d_h),
        d_w,
        inv_t,
        inv_h,
        inv_w,
        w_pos,
    )
    result = lib.r26_norm_rope(
        query,
        key,
        query_weight,
        key_weight,
        cos,
        sin,
        batch,
        t,
        h,
        w,
        heads,
        head_dim,
        int(d_t),
        int(d_h),
        float(eps),
        float(query_scale),
        float(key_scale),
    )
    return tuple(result)


def library_path() -> str | None:
    """Return the loaded R26 shared-library path for diagnostics."""
    lib = _library()
    if lib is None:
        return None
    here = Path(__file__).resolve()
    candidates = [
        Path("/home/simon/ltx2.5-demo-main/LTX-2/packages/ltx-kernels/csrc/sycl_r26/libr26_vae.so"),
        here.parents[6] / "ltx-kernels" / "csrc" / "sycl_r26" / "libr26_vae.so",
    ]
    return str(next(path for path in candidates if path.is_file()))
