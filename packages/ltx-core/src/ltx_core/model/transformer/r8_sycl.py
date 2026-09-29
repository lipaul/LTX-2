"""R8 SYCL elementwise kernels (self-written, icpx -fsycl).

Covers the whole elementwise family, including the R7 chains redone in SYCL:
  Batch A  K1 ada_zero 4 -> 1 (rms_norm + fetch + 1p + addcmul)
           cross affine 3 -> 1 (fetch + 1p + addcmul, already normalized)
           K2 post_sa 3 -> 1 (gate fetch + addcmul + rms_norm, dual output)
  Batch B  K3 split-rope negfold 3 -> 1 (mul + 2 x addcmul_)

Numerics: fp32 compute, bf16 IO (same contract as R7, Gate1a/b precedent).
Eager fallbacks are op-for-op identical to the R3/R6 path, so LTX_R8_SYCL=0 /
LTX_ALL_ORIG=1 behave bitwise as before. R7 Triton code is untouched; when both
are on, R8 takes precedence (call sites check R8 first), so R7 stays resident
but unused by default.

Only used for the stock PytorchAdaZeroFunction/PytorchPostSAFunction call sites
(transformer.py gates on isinstance); custom ops configs keep the old path.
"""

from __future__ import annotations

import os

import torch


def _env_on(name: str, default: str = "1") -> bool:
    if os.environ.get("LTX_ALL_ORIG", "0") == "1":
        return False
    return os.environ.get(name, default) != "0"


def r8_on() -> bool:
    """R8 master switch: default on; LTX_R8_SYCL=0 or LTX_ALL_ORIG=1 restores old path."""
    return _env_on("LTX_R8_SYCL")


def r31_compat_on() -> bool:
    """R31 output-audit compat: reproduce round-0 bf16 rounding at the fused
    R3/R6/R8/R10D sites. Default on (METHOD §7); LTX_R31_COMPAT=0 falls back
    to the fused math; LTX_ALL_ORIG=1 forces off."""
    return _env_on("LTX_R31_COMPAT", "1")


def r18_sycleff_on() -> bool:
    """R18 efficiency switch for the three R8 SYCL kernels."""
    if os.environ.get("LTX_ALL_ORIG", "0") == "1":
        return False
    value = os.environ.get("LTX_R18_SYCLEFF", "0")
    if value not in {"0", "1"}:
        raise ValueError("LTX_R18_SYCLEFF must be 0 or 1")
    return value == "1"


def _r18_env_int(name: str, default: int, choices: tuple[int, ...]) -> int:
    value = os.environ.get(name, str(default))
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed not in choices:
        raise ValueError(f"{name} must be one of {choices}")
    return parsed


def _r18_prefetch(name: str, default: int) -> int:
    value = os.environ.get(name, os.environ.get("LTX_R18_PREFETCH", str(default)))
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed not in (0, 1, 2):
        raise ValueError(f"{name} must be 0, 1, or 2")
    return parsed


def _r18_env_bound(name: str, default: int) -> int:
    value = os.environ.get(name, str(default))
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative")
    return parsed


def _r41_ada_local_on() -> bool:
    """R41: route the R31-compat ada to the shared-local-memory reduction.

    ``R8K1AdaCompatLocalKernel`` keeps the R31 round-0 rounding chain and reduces the
    row sums through a ``sycl::local_accessor`` instead of the global scratch pool.
    The reduction sequence is identical, so the output is bitwise equal to
    ``R8K1AdaCompatKernel``. ``LTX_R41_ADALOCAL=0`` (the default) keeps the
    global-scratch kernel.
    """
    if os.environ.get("LTX_ALL_ORIG", "0") == "1":
        return False
    value = os.environ.get("LTX_R41_ADALOCAL", "0")
    if value not in {"0", "1"}:
        raise ValueError("LTX_R41_ADALOCAL must be 0 or 1")
    return value == "1"


def _r41_ada_local_wg() -> int:
    return _r18_env_bound("LTX_R41_ADALOCAL_WG", 256)


def _r41_batch_on() -> bool:
    """R41: widen an fp8 linear's weight and bias in one launch.

    The two tensors are independent, so one ``R8Fp8WidenBatchKernel`` launch covers
    both. ``LTX_R41_BATCH=0`` (the default) keeps the two separate launches.
    """
    if os.environ.get("LTX_ALL_ORIG", "0") == "1":
        return False
    value = os.environ.get("LTX_R41_BATCH", "0")
    if value not in {"0", "1"}:
        raise ValueError("LTX_R41_BATCH must be 0 or 1")
    return value == "1"


def _r41_ada_pipe_on() -> bool:
    """R41: two-row software-pipelined R31-compat ada.

    ``R8K1AdaCompatPipeKernel`` gives one workgroup two consecutive rows and issues the
    second row's x and per-token table loads before the first row's reduction barrier,
    so the memory system is busy while the barrier drains. It also halves the number of
    sequential reduction waves. The arithmetic and the reduction sequence are those of
    ``k1_body<true,true>``, so the output is bitwise equal to
    ``R8K1AdaCompatKernel``. ``LTX_R41_ADAPIPE=0`` (the default) keeps the two-row-free
    global-scratch kernel.
    """
    if os.environ.get("LTX_ALL_ORIG", "0") == "1":
        return False
    value = os.environ.get("LTX_R41_ADAPIPE", "0")
    if value not in {"0", "1"}:
        raise ValueError("LTX_R41_ADAPIPE must be 0 or 1")
    return value == "1"


def _r41_ada_pipe_wg() -> int:
    return _r18_env_bound("LTX_R41_ADAPIPE_WG", 256)


def _r17_mlp_norm_fuse_on() -> bool:
    """R17 switch for the chunked-MLP RMSNorm and affine fusion."""
    if os.environ.get("LTX_ALL_ORIG", "0") == "1":
        return False
    value = os.environ.get("LTX_R17_MLP_NORM_FUSE", "1")
    if value not in {"0", "1"}:
        raise ValueError("LTX_R17_MLP_NORM_FUSE must be 0 or 1")
    return value == "1"


def _r19_norm_rope_on() -> bool:
    """R19 switch for q/k RMSNorm and split rotary embedding fusion."""
    if os.environ.get("LTX_ALL_ORIG", "0") == "1":
        return False
    value = os.environ.get("LTX_R19_NORM_ROPE", "1")
    if value not in {"0", "1"}:
        raise ValueError("LTX_R19_NORM_ROPE must be 0 or 1")
    return value == "1"


def _r21_elemwise_on() -> bool:
    """R21 switch for the DiffVAE SiLU and up-projection product."""
    if os.environ.get("LTX_ALL_ORIG", "0") == "1":
        return False
    value = os.environ.get("LTX_R21_ELEMWISE", "0")
    if value not in {"0", "1"}:
        raise ValueError("LTX_R21_ELEMWISE must be 0 or 1")
    return value == "1"


def r21_swiglu_silu_mul_supported(
    x: torch.Tensor, w_up: torch.Tensor, gate: torch.Tensor
) -> bool:
    """Return whether the R21 XPU SiLU/up-product kernel can write ``gate``."""
    if not _r21_elemwise_on() or _lib() is None:
        return False
    if not (x.ndim == 2 and w_up.ndim == 2 and gate.ndim == 2):
        return False
    rows, dim = int(x.shape[0]), int(x.shape[1])
    hidden = int(w_up.shape[0])
    return (
        hidden > 0
        and tuple(int(v) for v in w_up.shape) == (hidden, dim)
        and tuple(int(v) for v in gate.shape) == (rows, hidden)
        and all(t.is_xpu and t.dtype == torch.bfloat16 for t in (x, w_up, gate))
        and all(t.is_contiguous() and t.stride(-1) == 1 for t in (x, w_up, gate))
        and all(t.device == x.device for t in (w_up, gate))
        and all(t.data_ptr() % 16 == 0 for t in (x, w_up, gate))
        and gate.numel() % 8 == 0
    )


def r21_swiglu_silu_mul(gate: torch.Tensor, up: torch.Tensor) -> bool:
    """Replace an in-place SiLU followed by ``gate.mul_(up)`` with one XPU pass."""
    if not _r21_elemwise_on():
        return False
    lib = _lib()
    if lib is None or not (
        gate.is_xpu
        and up.is_xpu
        and gate.dtype == torch.bfloat16
        and up.dtype == torch.bfloat16
        and gate.shape == up.shape
        and gate.is_contiguous()
        and up.is_contiguous()
        and gate.stride(-1) == 1
        and up.stride(-1) == 1
        and gate.device == up.device
        and gate.data_ptr() % 16 == 0
        and up.data_ptr() % 16 == 0
        and gate.numel() % 8 == 0
    ):
        return False
    lib.r21_swiglu_silu_mul(gate, up)
    return True


def _r21b_elemwise_on() -> bool:
    """R21b switch for the compat SiLU/up-product kernel (default off)."""
    if os.environ.get("LTX_ALL_ORIG", "0") == "1":
        return False
    value = os.environ.get("LTX_R21B_ELEMWISE", "1")
    if value not in {"0", "1"}:
        raise ValueError("LTX_R21B_ELEMWISE must be 0 or 1")
    return value == "1"


def r21b_swiglu_silu_mul_supported(
    x: torch.Tensor, w_up: torch.Tensor, gate: torch.Tensor
) -> bool:
    """Return whether the R21b compat kernel can write ``gate``."""
    if not _r21b_elemwise_on() or _lib() is None:
        return False
    if not (x.ndim == 2 and w_up.ndim == 2 and gate.ndim == 2):
        return False
    rows, dim = int(x.shape[0]), int(x.shape[1])
    hidden = int(w_up.shape[0])
    return (
        hidden > 0
        and tuple(int(v) for v in w_up.shape) == (hidden, dim)
        and tuple(int(v) for v in gate.shape) == (rows, hidden)
        and all(t.is_xpu and t.dtype == torch.bfloat16 for t in (x, w_up, gate))
        and all(t.is_contiguous() and t.stride(-1) == 1 for t in (x, w_up, gate))
        and all(t.device == x.device for t in (w_up, gate))
        and all(t.data_ptr() % 16 == 0 for t in (x, w_up, gate))
        and gate.numel() % 8 == 0
    )


def r21b_swiglu_silu_mul(gate: torch.Tensor, up: torch.Tensor) -> bool:
    """Replace an in-place SiLU followed by ``gate.mul_(up)`` (compat)."""
    if not _r21b_elemwise_on():
        return False
    lib = _lib()
    if lib is None or not (
        gate.is_xpu
        and up.is_xpu
        and gate.dtype == torch.bfloat16
        and up.dtype == torch.bfloat16
        and gate.shape == up.shape
        and gate.is_contiguous()
        and up.is_contiguous()
        and gate.stride(-1) == 1
        and up.stride(-1) == 1
        and gate.device == up.device
        and gate.data_ptr() % 16 == 0
        and up.data_ptr() % 16 == 0
        and gate.numel() % 8 == 0
    ):
        return False
    lib.r21b_swiglu_silu_mul(gate, up)
    return True


_LIB: object | None = None
_LIB_OK: bool | None = None


def _lib() -> object | None:
    """Load libr8_sycl.so once; None when unavailable (CPU / missing .so)."""
    global _LIB, _LIB_OK
    if _LIB_OK is None:
        try:
            if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
                raise RuntimeError("no XPU")
            here = os.path.dirname(os.path.abspath(__file__))
            cands = [os.path.join(here, "libr8_sycl.so")]
            # Built out-of-tree: ltx-kernels csrc dir (keeps binaries out of ltx-core).
            cands.append(
                os.path.normpath(
                    os.path.join(here, "..", "..", "..", "..", "..", "ltx-kernels", "csrc", "sycl_r8", "libr8_sycl.so")
                )
            )
            so = next((c for c in cands if os.path.isfile(c)), cands[0])
            if not os.path.isfile(so):
                raise RuntimeError(f"missing {so}")
            torch.ops.load_library(so)
            torch.ops.libr8.k1_ada  # noqa: B018 - fail fast when symbol missing
            _LIB = torch.ops.libr8
            _LIB_OK = True
        except Exception:  # noqa: BLE001 - any failure means eager fallback
            _LIB = None
            _LIB_OK = False
    return _LIB


def lib_ok() -> bool:
    """True once the SYCL .so is loaded and callable (cached)."""
    return _lib() is not None


def _row_stride(t: torch.Tensor | None) -> int:
    """Row stride for (rows, C) parts; 0 broadcasts (or missing ts)."""
    if t is None or t.dim() < 2 or t.shape[-2] == 1:
        return 0
    return t.stride(-2)


def _parts_ok(n: int, table: torch.Tensor, ts: torch.Tensor | None) -> bool:
    if table.numel() != n or table.dtype != torch.bfloat16 or not table.is_xpu:
        return False
    if ts is None:
        return True
    return (
        ts.is_xpu
        and ts.dtype == torch.bfloat16
        and ts.shape[-1] == n
        and ts.stride(-1) == 1
    )


def _x_ok(x: torch.Tensor) -> bool:
    n = x.shape[-1]
    return (
        x.is_xpu
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and x.stride(-1) == 1
        and 64 <= n <= 8192
    )


def r17_mlp_norm_affine(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    out: torch.Tensor,
    eps: float,
) -> bool:
    """Write one fused RMSNorm and channel affine result into ``out``.

    The return value reports whether the XPU kernel wrote ``out``. Unsupported
    devices, layouts, dtypes, and row counts use the original path in the caller.
    """
    if not _r17_mlp_norm_fuse_on():
        return False
    lib = _lib()
    if lib is None or x.ndim < 1:
        return False
    n = int(x.shape[-1])
    rows = int(x.numel()) // n if n > 0 else 0
    tensors = (x, norm_weight, scale, shift, out)
    if not (
        64 <= n <= 8192
        and n % 8 == 0
        and 0 < rows <= (1 << 30) // 32
        and out.shape == x.shape
        and all(t.is_xpu and t.dtype == torch.bfloat16 for t in tensors)
        and all(t.is_contiguous() and t.stride(-1) == 1 for t in tensors)
        and all(t.device == x.device for t in tensors)
        and all(t.data_ptr() % 16 == 0 for t in tensors)
        and norm_weight.numel() == n
        and scale.numel() == n
        and shift.numel() == n
    ):
        return False
    lib.r17_mlp_norm_affine(x, norm_weight, scale, shift, out, float(eps))
    return True

def _eager_ada(x: torch.Tensor, eps: float, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    from ltx_core.utils import rms_norm as _rms  # noqa: PLC0415

    return torch.addcmul(shift, _rms(x, eps=eps), 1 + scale)


def r8_ada_zero(
    x: torch.Tensor,
    eps: float,
    table_s: torch.Tensor,
    ts_s: torch.Tensor,
    table_h: torch.Tensor,
    ts_h: torch.Tensor,
) -> torch.Tensor:
    """Fused ada_zero (rms_norm + fetch + 1p + addcmul, 4 launches -> 1)."""
    n = x.shape[-1]
    lib = _lib() if r8_on() else None
    if (
        lib is None
        or not _x_ok(x)
        or not _parts_ok(n, table_s, ts_s)
        or not _parts_ok(n, table_h, ts_h)
    ):
        scale = table_s if ts_s is None else table_s + ts_s
        shift = table_h if ts_h is None else table_h + ts_h
        return _eager_ada(x, eps, scale, shift)
    if r18_sycleff_on() and ts_s is not None and ts_h is not None:
        wg = _r18_env_int("LTX_R18_K1_WG", 256, (128, 256, 512))
        vec = _r18_env_int("LTX_R18_K1_VEC", 8, (8, 16))
        prefetch = _r18_prefetch("LTX_R18_K1_PREFETCH", 1)
        mode = os.environ.get("LTX_R18_K1_MODE", "local")
        if mode == "subgroup":
            return lib.k1_ada_subgroup(
                x,
                table_s,
                ts_s,
                _row_stride(ts_s),
                table_h,
                ts_h,
                _row_stride(ts_h),
                float(eps),
                wg,
                vec,
                prefetch,
            )
        if mode == "group":
            return lib.k1_ada_group_reduce(
                x,
                table_s,
                ts_s,
                _row_stride(ts_s),
                table_h,
                ts_h,
                _row_stride(ts_h),
                float(eps),
                wg,
                vec,
                prefetch,
            )
        if mode == "reg" and vec == 8:
            return lib.k1_ada_reg_local(
                x,
                table_s,
                ts_s,
                _row_stride(ts_s),
                table_h,
                ts_h,
                _row_stride(ts_h),
                float(eps),
                wg,
                prefetch,
            )
        if mode not in {"local", "reg", "group"}:
            raise ValueError("LTX_R18_K1_MODE must be local, reg, subgroup, or group")
        return lib.k1_ada_local(
            x,
            table_s,
            ts_s,
            _row_stride(ts_s),
            table_h,
            ts_h,
            _row_stride(ts_h),
            float(eps),
            wg,
            vec,
            prefetch,
        )
    if _r10d_k1hold_on():
        try:
            return lib.k1_ada_hold(x, table_s, ts_s, _row_stride(ts_s), table_h, ts_h, _row_stride(ts_h), float(eps))
        except Exception:  # noqa: BLE001 - .so without the R10D kernel -> two-pass K1
            pass
    if _r41_ada_local_on() and r31_compat_on():
        wg = _r41_ada_local_wg()
        if 32 <= wg <= 1024 and wg % 32 == 0 and wg * 32 <= 1 << 22:
            return lib.k1_ada_compat_local(
                x,
                table_s,
                ts_s,
                _row_stride(ts_s),
                table_h,
                ts_h,
                _row_stride(ts_h),
                float(eps),
                wg,
            )
    if _r41_ada_pipe_on() and r31_compat_on():
        wg = _r41_ada_pipe_wg()
        if n in (wg * 8, 2 * wg * 8):
            return lib.k1_ada_compat_pipe(
                x,
                table_s,
                ts_s,
                _row_stride(ts_s),
                table_h,
                ts_h,
                _row_stride(ts_h),
                float(eps),
                wg,
            )
    return lib.k1_ada(x, table_s, ts_s, _row_stride(ts_s), table_h, ts_h, _row_stride(ts_h), float(eps))


def r8_affine_addcmul(
    x: torch.Tensor,
    table_s: torch.Tensor,
    ts_s: torch.Tensor | None,
    table_h: torch.Tensor,
    ts_h: torch.Tensor | None,
) -> torch.Tensor:
    """Fused fetch + 1p + addcmul for already-normalized inputs (cross path, 3 -> 1)."""
    n = x.shape[-1]
    lib = _lib() if r8_on() else None
    if (
        lib is None
        or not _x_ok(x)
        or not _parts_ok(n, table_s, ts_s)
        or not _parts_ok(n, table_h, ts_h)
    ):
        scale = table_s if ts_s is None else table_s + ts_s
        shift = table_h if ts_h is None else table_h + ts_h
        return torch.addcmul(shift, x, 1 + scale)
    return lib.k1_affine(x, table_s, ts_s, _row_stride(ts_s), table_h, ts_h, _row_stride(ts_h))


def r8_post_sa(
    x: torch.Tensor,
    y: torch.Tensor,
    table_g: torch.Tensor,
    ts_g: torch.Tensor,
    norm_weights: torch.Tensor | None,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused post_sa (gate fetch + addcmul + rms_norm, 3 launches -> 1).

    Returns (fma, normed) like PytorchPostSAFunction.
    """
    from ltx_core.utils import rms_norm as _rms  # noqa: PLC0415

    n = x.shape[-1]
    lib = _lib() if r8_on() else None
    if (
        lib is None
        or norm_weights is not None
        or not _x_ok(x)
        or y.shape != x.shape
        or not y.is_xpu
        or y.dtype != torch.bfloat16
        or not y.is_contiguous()
        or not _parts_ok(n, table_g, ts_g)
    ):
        gate = table_g + ts_g
        fma = torch.addcmul(x, y, gate)
        return fma, _rms(fma, norm_weights, eps=eps)
    return tuple(lib.k2_postsa(x, y, table_g, ts_g, _row_stride(ts_g), float(eps)))


def _r10d_gate_on() -> bool:
    """R10D F1: fuse the text-cross-attention gate mul with the residual add
    (`x = x + attn_out * gate` -> one addcmul). LTX_R10D_GATE=0 restores the
    two-op formula (mul + add)."""
    return _env_on("LTX_R10D_GATE")


def _r10d_gate_on() -> bool:
    """R10D F2: self-written SYCL per-head broadcast gate multiply.
    LTX_R10D_GATE2=0 keeps torch's broadcast mul."""
    return _env_on("LTX_R10D_GATE2")


def r10d_gate_mul(attn_out, gates, heads, dim_head):
    """R10D F2: `attn_out.view(B,T,H,Dh) * gates.unsqueeze(-1)` via SYCL.

    Bitwise identical to the torch formula (bf16, fp32 product, one RNE).
    Returns None when unavailable/unsupported so the caller keeps torch's path.
    """
    lib = _lib() if r8_on() else None
    if lib is None or not _r10d_gate_on():
        return None
    try:
        b, t, _ = attn_out.shape
        if (
            attn_out.is_xpu
            and attn_out.dtype == torch.bfloat16
            and attn_out.is_contiguous()
            and gates.dtype == torch.bfloat16
            and gates.is_contiguous()
            and gates.numel() == b * t * heads
            and dim_head % 8 == 0
            and attn_out.numel() % 8 == 0
            and attn_out.data_ptr() % 16 == 0
            and gates.data_ptr() % 16 == 0
        ):
            return lib.r10d_gate_mul(attn_out, gates.reshape(-1), int(dim_head))
    except Exception:  # noqa: BLE001 - any failure -> torch path
        pass
    return None


def _r10d_gelu_on() -> bool:
    """R10D F3: self-written SYCL tanh-approx GELU.

    STOP-LOSS: bitwise vs torch but only 1.06x (< 1.2x gate) -- torch's VEC8
    GeluTanh is already at ~352 GB/s. Kept as an opt-in experiment
    (LTX_R10D_GELU=1), default OFF.
    """
    return _env_on("LTX_R10D_GELU", "0")


def r10d_gelu(x):
    """R10D F3: tanh-approx GELU via SYCL (bitwise vs torch's GeluTanhFunctor).

    Returns None when unavailable/unsupported so the caller keeps torch's path.
    """
    lib = _lib() if r8_on() else None
    if lib is None or not _r10d_gelu_on():
        return None
    try:
        if (
            x.is_xpu
            and x.dtype == torch.bfloat16
            and x.is_contiguous()
            and x.numel() % 8 == 0
            and x.data_ptr() % 16 == 0
        ):
            return lib.r10d_gelu(x)
    except Exception:  # noqa: BLE001 - any failure -> torch path
        pass
    return None


def r8_cross_attention_adaln(
    x_normed,
    context,
    attn,
    context_mask,
    qsh_tab,
    qsh_ts,
    qsc_tab,
    qsc_ts,
    g_tab,
    g_ts,
    ksh_tab,
    ksh_ts,
    ksc_tab,
    ksc_ts,
    residual=None,
):
    # R8 fused cross-attention ada: 2x (fetch+1p+addcmul 3->1); gate mul + attn unchanged.
    attn_input = r8_affine_addcmul(x_normed, qsc_tab, qsc_ts, qsh_tab, qsh_ts)
    encoder_hidden_states = r8_affine_addcmul(context, ksc_tab, ksc_ts, ksh_tab, ksh_ts)
    gate = g_tab + g_ts
    attn_out = attn(attn_input, context=encoder_hidden_states, mask=context_mask)
    if residual is not None and _r10d_gate_on():
        if r31_compat_on():
            # R31: round-0 two-op formula (mul rounds, then add rounds).
            return residual + attn_out * gate
        # R10D F1: one fused pass instead of mul + add.
        return torch.addcmul(residual, attn_out, gate)
    return attn_out * gate


def _r9d_k3v_on() -> bool:
    """R9D: vectorised K3 (split-rope) sub-switch; LTX_R9D_K3V=0 reverts to scalar K3."""
    return _env_on("LTX_R9D_K3V")


def _r10d_k1hold_on() -> bool:
    """R10D: single-pass K1Ada (x held in registers across the rstd barrier).
    STOP-LOSS: microbench 1.00x (< 1.2x gate) because the second x read is
    already L2/L1-served -- K1Ada sits at the 4-access read ceiling. Kept as an
    opt-in experiment only (LTX_R10D_K1HOLD=1), default OFF."""
    return _env_on("LTX_R10D_K1HOLD", "0")


def _r9d_k3v_ok(t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, dh: int) -> bool:
    """Guards for the u16x8 K3 kernel: unit innermost stride, Dh multiple of 8,
    and 16-byte-aligned row starts (x/cos/sin data_ptr aligned + outer strides x8)."""
    if dh % 8 != 0 or dh == 0:
        return False
    for t_, st in ((t, t.stride()), (cos, cos.stride()), (sin, sin.stride())):
        if st[-1] != 1 or t_.data_ptr() % 16 != 0:
            return False
        if any(s % 8 != 0 for s in st[:3]):
            return False
    return True


def r19_norm_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    cos_freqs: torch.Tensor,
    sin_freqs: torch.Tensor,
    eps: float,
    query_scale: float = 1.0,
    key_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Normalize q/k rows and apply split RoPE in one paired XPU launch."""
    if not (r8_on() and _r19_norm_rope_on()):
        return None
    lib = _lib()
    if lib is None:
        return None
    if (
        query.ndim != 3
        or key.ndim != 3
        or query.shape != key.shape
        or cos_freqs.ndim != 4
        or cos_freqs.shape != sin_freqs.shape
        or cos_freqs.shape[1] <= 0
        or cos_freqs.shape[3] <= 0
    ):
        return None
    batch, tokens, channels = (int(v) for v in query.shape)
    heads = int(cos_freqs.shape[1])
    head_dim = int(cos_freqs.shape[3])
    if (
        channels != heads * head_dim * 2
        or channels % 8 != 0
        or head_dim % 8 != 0
        or cos_freqs.shape[0] not in (1, batch)
        or cos_freqs.shape[2] != tokens
    ):
        return None
    tensors = (query, key, query_weight, key_weight, cos_freqs, sin_freqs)
    if not all(t.is_xpu and t.dtype == torch.bfloat16 for t in tensors):
        return None
    if not (
        query.stride(-1) == 1
        and key.stride(-1) == 1
        and query_weight.is_contiguous()
        and key_weight.is_contiguous()
        and query_weight.numel() == channels
        and key_weight.numel() == channels
        and cos_freqs.stride(-1) == 1
        and sin_freqs.stride(-1) == 1
    ):
        return None
    if not all(t.data_ptr() % 16 == 0 for t in tensors):
        return None
    return tuple(
        lib.r19_norm_rope(
            query,
            key,
            query_weight,
            key_weight,
            cos_freqs,
            sin_freqs,
            batch,
            heads,
            tokens,
            head_dim,
            float(eps),
            float(query_scale),
            float(key_scale),
        )
    )


def r8_split_rope(
    input_tensor: torch.Tensor,
    cos_freqs: torch.Tensor,
    sin_freqs: torch.Tensor,
) -> torch.Tensor:
    """Fused split-rope negfold (mul + 2 x addcmul_ -> 1 launch).

    Covers the R5 negfold contract for 4D (B,H,T,D) tensors AND the 3D
    (B,T,H*D) needs_reshape path used by the DiT attention blocks (views in,
    fused compute, views back out). Falls back to the eager negfold formula
    for anything unexpected.
    """
    from ltx_core.model.transformer.rope import _apply_split_rotary_emb_negfold as _eager  # noqa: PLC0415

    def _fallback():
        return _eager(input_tensor, cos_freqs, sin_freqs)

    lib = _lib() if r8_on() else None
    if lib is None:
        return _fallback()
    if (
        not input_tensor.is_xpu
        or input_tensor.dtype != torch.bfloat16
        or cos_freqs.shape != sin_freqs.shape
        or cos_freqs.ndim != 4
        or input_tensor.ndim not in (3, 4)
        or not (
            cos_freqs.is_xpu
            and sin_freqs.is_xpu
            and cos_freqs.dtype == torch.bfloat16
            and sin_freqs.dtype == torch.bfloat16
        )
    ):
        return _fallback()
    if input_tensor.ndim == 4:
        b, h, t, d = (int(v) for v in input_tensor.shape)
        if d % 2 != 0 or tuple(int(v) for v in cos_freqs.shape[-2:]) != (t, d // 2):
            return _fallback()
        if cos_freqs.shape[0] not in (1, b) or cos_freqs.shape[1] != h:
            return _fallback()
        dh = d // 2
        xs = [int(s) for s in input_tensor.stride()]
        cs = [int(s) for s in cos_freqs.stride()]
        ss = [int(s) for s in sin_freqs.stride()]
        if _r9d_k3v_on() and _r9d_k3v_ok(input_tensor, cos_freqs, sin_freqs, dh):
            try:
                return lib.k3_rope_vec(input_tensor, xs, cos_freqs, cs, sin_freqs, ss, b, h, t, dh)
            except Exception:  # noqa: BLE001 - fall back to scalar K3
                pass
        return lib.k3_rope(input_tensor, xs, cos_freqs, cs, sin_freqs, ss, b, h, t, dh)
    # 3D needs_reshape path: mirror the eager view dance, fuse the middle.
    b_in = int(input_tensor.shape[0])
    b_freq, h, t, dh = (int(v) for v in cos_freqs.shape)
    if b_freq not in (1, b_in):
        return _fallback()
    if int(input_tensor.shape[-1]) != h * dh * 2 or h <= 0 or dh <= 0:
        return _fallback()
    x4 = input_tensor.unflatten(-1, (h, -1)).transpose(1, 2)
    if tuple(int(v) for v in x4.shape) != (b_in, h, t, dh * 2):
        return _fallback()
    xs = [int(s) for s in x4.stride()]
    cs = [int(s) for s in cos_freqs.stride()]
    ss = [int(s) for s in sin_freqs.stride()]
    if _r9d_k3v_on() and _r9d_k3v_ok(x4, cos_freqs, sin_freqs, dh):
        try:
            out4 = lib.k3_rope_vec(x4, xs, cos_freqs, cs, sin_freqs, ss, b_in, h, t, dh)
            return out4.transpose(1, 2).flatten(-2)
        except Exception:  # noqa: BLE001 - fall back to scalar K3
            pass
    out4 = lib.k3_rope(x4, xs, cos_freqs, cs, sin_freqs, ss, b_in, h, t, dh)
    return out4.transpose(1, 2).flatten(-2)


def r18_dual_split_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    cos_freqs: torch.Tensor,
    sin_freqs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Apply one split-rope launch to an eligible q/k pair with shared cos/sin."""
    if not (r8_on() and r18_sycleff_on()):
        return None
    lib = _lib()
    if lib is None:
        return None
    if (
        query.shape != key.shape
        or query.ndim not in (3, 4)
        or cos_freqs.shape != sin_freqs.shape
        or cos_freqs.ndim != 4
        or not (
            query.is_xpu
            and key.is_xpu
            and cos_freqs.is_xpu
            and sin_freqs.is_xpu
            and query.dtype == torch.bfloat16
            and key.dtype == torch.bfloat16
            and cos_freqs.dtype == torch.bfloat16
            and sin_freqs.dtype == torch.bfloat16
        )
    ):
        return None
    if query.ndim == 4:
        b, h, t, d = (int(v) for v in query.shape)
        dh = d // 2
        if (
            d % 2 != 0
            or tuple(int(v) for v in cos_freqs.shape[-2:]) != (t, dh)
            or cos_freqs.shape[0] not in (1, b)
            or cos_freqs.shape[1] != h
        ):
            return None
        max_tasks = _r18_env_bound("LTX_R18_K3_MAX_TASKS", 1 << 20)
        if b * h * t * (dh // 8) > max_tasks:
            return None
        qxs = [int(s) for s in query.stride()]
        kxs = [int(s) for s in key.stride()]
        cs = [int(s) for s in cos_freqs.stride()]
        ss = [int(s) for s in sin_freqs.stride()]
        if not (
            _r9d_k3v_on()
            and _r9d_k3v_ok(query, cos_freqs, sin_freqs, dh)
            and _r9d_k3v_ok(key, cos_freqs, sin_freqs, dh)
        ):
            return None
        return tuple(lib.k3_rope_dual(query, qxs, key, kxs, cos_freqs, cs, sin_freqs, ss, b, h, t, dh))
    b_in = int(query.shape[0])
    b_freq, h, t, dh = (int(v) for v in cos_freqs.shape)
    if b_freq not in (1, b_in) or int(query.shape[-1]) != h * dh * 2:
        return None
    q4 = query.unflatten(-1, (h, -1)).transpose(1, 2)
    k4 = key.unflatten(-1, (h, -1)).transpose(1, 2)
    if tuple(int(v) for v in q4.shape) != (b_in, h, t, dh * 2):
        return None
    max_tasks = _r18_env_bound("LTX_R18_K3_MAX_TASKS", 1 << 20)
    if b_in * h * t * (dh // 8) > max_tasks:
        return None
    if not (
        _r9d_k3v_on()
        and _r9d_k3v_ok(q4, cos_freqs, sin_freqs, dh)
        and _r9d_k3v_ok(k4, cos_freqs, sin_freqs, dh)
    ):
        return None
    qxs = [int(s) for s in q4.stride()]
    kxs = [int(s) for s in k4.stride()]
    cs = [int(s) for s in cos_freqs.stride()]
    ss = [int(s) for s in sin_freqs.stride()]
    outq, outk = lib.k3_rope_dual(q4, qxs, k4, kxs, cos_freqs, cs, sin_freqs, ss, b_in, h, t, dh)
    return outq.transpose(1, 2).flatten(-2), outk.transpose(1, 2).flatten(-2)


def _r9a_fp8k_on() -> bool:
    """R9A: vectorized fp8(e4m3fn)->bf16 widening kernel.

    ADOPTED (2026-09-24): in-situ e2e gain -2.250 s (fp8 dequant 3.520 -> 0.253 s
    + new R8Fp8WidenKernel ~1.0 s), bitwise vs torch. Default ON;
    LTX_R9A_FP8K=0 (or LTX_ALL_ORIG=1) restores the original torch path.
    """
    return _env_on("LTX_R9A_FP8K")


def r9a_fp8_widen(x):
    """R9A: fp8 e4m3fn -> bf16 via SYCL (bitwise vs torch). None => caller uses torch."""
    lib = _lib() if r8_on() else None
    if lib is None or not _r9a_fp8k_on():
        return None
    if not (
        x.is_xpu
        and x.dtype == torch.float8_e4m3fn
        and x.is_contiguous()
        and x.numel() % 8 == 0
        and x.data_ptr() % 16 == 0
    ):
        return None
    if r18_sycleff_on():
        return lib.r9a_fp8_widen_bits(x)
    return lib.r9a_fp8_widen(x)


def r9a_fp8_widen_batch(
    inputs: list[torch.Tensor],
) -> list[torch.Tensor] | None:
    """Widen a small descriptor list in one R18 batch launch.

    The batch kernel converts with ``r18_fp8_bits_to_bf16``; the exhaustive 256-entry
    check and the ``bitwise_vs_baseline`` result of every grouped run in
    ``r41-othnorm-20260928/st1_widen_batch.json`` show it equals the ``R9A_LUT`` path
    the single-tensor op uses, so the op is available independently of
    ``LTX_R18_SYCLEFF``.
    """
    if not (r8_on() and _r9a_fp8k_on()):
        return None
    lib = _lib()
    if lib is None or not inputs or len(inputs) > 4:
        return None
    if not all(
        x.is_xpu
        and x.dtype == torch.float8_e4m3fn
        and x.is_contiguous()
        and x.numel() > 0
        and x.numel() % 8 == 0
        and x.data_ptr() % 16 == 0
        for x in inputs
    ):
        return None
    return list(lib.r9a_fp8_widen_batch(inputs))


def r9a_fp8_widen_pair(
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None] | None:
    """Widen an fp8 weight and its fp8 bias in one batch launch.

    Returns ``None`` when the pair is ineligible so the caller keeps the two separate
    launches.
    """
    if not _r41_batch_on():
        return None
    if bias is None:
        return None
    if not (
        _r9a_fp8k_on()
        and weight.dtype == torch.float8_e4m3fn
        and bias.dtype == torch.float8_e4m3fn
    ):
        return None
    out = r9a_fp8_widen_batch([weight, bias])
    if out is None or len(out) != 2:
        return None
    return out[0], out[1]


def _r9a_v2_on() -> bool:
    """R9A v2 (16-wide) opt-in; LTX_R9A_V2=1."""
    return _env_on("LTX_R9A_V2", "0")


def r9a_fp8_widen16(x):
    """R9A v2: 16-wide fp8->bf16 widening. None => caller falls back."""
    lib = _lib() if r8_on() else None
    if lib is None or not _r9a_v2_on():
        return None
    try:
        if (
            x.is_xpu
            and x.dtype == torch.float8_e4m3fn
            and x.is_contiguous()
            and x.numel() % 16 == 0
            and x.data_ptr() % 16 == 0
        ):
            return lib.r9a_fp8_widen16(x)
    except Exception:  # noqa: BLE001
        pass
    return None
