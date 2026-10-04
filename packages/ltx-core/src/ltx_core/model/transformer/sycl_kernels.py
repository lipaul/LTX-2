"""Hand-written SYCL kernels for the XPU build, launched through torch ops.

The kernels fuse chains of small elementwise operations into single passes:

  ada_zero       rms_norm + table fetch + (1 + scale) + addcmul   4 -> 1
  cross affine   table fetch + (1 + scale) + addcmul              3 -> 1
  post_sa        gate fetch + addcmul + rms_norm, dual output     3 -> 1
  rope negfold   multiply + 2 x addcmul                           3 -> 1
  plus degree-1 helpers: per-head gate broadcast multiply, fp8 -> bf16 widening
  and a tanh-approximation GELU.

Numerics: fp32 accumulate, bf16 in/out.  With LTX_TORCH_EXACT_ROUNDING=1 (the
default) each kernel and its torch fallback round at the points of the plain
two-op torch expression, so both paths produce the same bits; RMSNorm
reductions can differ by one ulp in a few elements because the summation order
differs.  LTX_TORCH_EXACT_ROUNDING=0 selects the cheaper fused rounding.  An
entry point uses the torch expression when its own input guard fails.  A
missing or unloadable libltx_sycl.so raises when an XPU is present.

Only the stock PytorchAdaZeroFunction/PytorchPostSAFunction call sites are
covered; transformer.py gates on ``isinstance`` so custom ops configs keep
their own path.
"""

from __future__ import annotations

import os

import torch


def _env_on(name: str, default: str = "1") -> bool:
    if os.environ.get("LTX_ORIGINAL_PATH", "0") == "1":
        return False
    return os.environ.get(name, default) != "0"


def sycl_enabled() -> bool:
    """Master switch for the SYCL kernel path.

    On by default.  ``LTX_SYCL_KERNELS=0`` (or ``LTX_ORIGINAL_PATH=1``) routes
    every call site back to its plain torch expression."""
    return _env_on("LTX_SYCL_KERNELS")


def torch_exact_rounding_enabled() -> bool:
    """Whether the fused sites reproduce torch's bf16 rounding exactly.

    With this on, each fused site rounds where the equivalent torch expression
    rounds, so its output matches torch bit for bit.  ``LTX_TORCH_EXACT_ROUNDING=0``
    keeps the cheaper fused rounding; ``LTX_ORIGINAL_PATH=1`` forces the
    unfused path."""
    return _env_on("LTX_TORCH_EXACT_ROUNDING", "1")


def _env_bound(name: str, default: int) -> int:
    value = os.environ.get(name, str(default))
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative")
    return parsed


def _mlp_norm_fuse_enabled() -> bool:
    """Switch for the chunked-MLP RMSNorm and affine fusion."""
    if os.environ.get("LTX_ORIGINAL_PATH", "0") == "1":
        return False
    value = os.environ.get("LTX_SYCL_MLP_NORM", "1")
    if value not in {"0", "1"}:
        raise ValueError("LTX_SYCL_MLP_NORM must be 0 or 1")
    return value == "1"


def _norm_rope_enabled() -> bool:
    """Switch for q/k RMSNorm and split rotary embedding fusion."""
    if os.environ.get("LTX_ORIGINAL_PATH", "0") == "1":
        return False
    value = os.environ.get("LTX_SYCL_NORM_ROPE", "1")
    if value not in {"0", "1"}:
        raise ValueError("LTX_SYCL_NORM_ROPE must be 0 or 1")
    return value == "1"


# Cached handle to the loaded torch operator namespace, and whether the load
# attempt has already been made (True / False) or not yet (None).
_LIB: object | None = None
_LIB_OK: bool | None = None


def _lib() -> object | None:
    """Load libltx_sycl.so once.

    Returns None on hosts without an XPU.  With an XPU present, a missing or
    unloadable library raises."""
    global _LIB, _LIB_OK
    if _LIB_OK is None:
        if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            _LIB = None
            _LIB_OK = False
            return None
        here = os.path.dirname(os.path.abspath(__file__))
        cands = [os.path.join(here, "libltx_sycl.so")]
        # Built out-of-tree: ltx-kernels csrc dir (keeps binaries out of ltx-core).
        cands.append(
            os.path.normpath(
                os.path.join(here, "..", "..", "..", "..", "..", "ltx-kernels", "csrc", "sycl_kernels", "libltx_sycl.so")
            )
        )
        so = next((c for c in cands if os.path.isfile(c)), None)
        if so is None:
            raise RuntimeError(f"libltx_sycl.so not found in {cands}")
        torch.ops.load_library(so)
        torch.ops.ltx_sycl.ada_zero  # noqa: B018 - raises when the symbol is missing
        _LIB = torch.ops.ltx_sycl
        _LIB_OK = True
    return _LIB


def lib_ok() -> bool:
    """True once the SYCL .so is loaded and callable (cached)."""
    return _lib() is not None


def _ts_row_stride(x: torch.Tensor, ts: torch.Tensor | None) -> int | None:
    """Row stride of a (B, T, C) time-step part for the kernels' row-linear addressing.

    Row r = b * T + t reads ``ts + r * stride``.  A part with a single row
    (all leading dims 1) broadcasts with stride 0.  Returns None when ``ts``
    cannot be addressed that way for ``x``, e.g. a (1, T, C) part with B > 1."""
    if ts is None or all(s == 1 for s in ts.shape[:-1]):
        return 0
    if x.dim() != 3 or ts.dim() != 3 or tuple(ts.shape[:2]) != tuple(x.shape[:2]):
        return None
    b, t = x.shape[0], x.shape[1]
    rs = ts.stride(1) if t > 1 else ts.stride(0)
    if b > 1 and t > 1 and ts.stride(0) != t * rs:
        return None
    return rs


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
        and n % 8 == 0
    )


def mlp_norm_affine(
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
    if not _mlp_norm_fuse_enabled():
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
    lib.mlp_norm_affine(x, norm_weight, scale, shift, out, float(eps))
    return True

def _eager_ada(x: torch.Tensor, eps: float, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    from ltx_core.utils import rms_norm as _rms  # noqa: PLC0415

    normed = _rms(x, eps=eps)
    if torch_exact_rounding_enabled():
        return normed * (1 + scale) + shift
    return torch.addcmul(shift, normed, 1 + scale)


def _eager_affine(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    if torch_exact_rounding_enabled():
        return x * (1 + scale) + shift
    return torch.addcmul(shift, x, 1 + scale)


def ada_zero(
    x: torch.Tensor,
    eps: float,
    table_s: torch.Tensor,
    ts_s: torch.Tensor,
    table_h: torch.Tensor,
    ts_h: torch.Tensor,
) -> torch.Tensor:
    """Fused ada_zero (rms_norm + fetch + 1p + addcmul, 4 launches -> 1)."""
    n = x.shape[-1]
    lib = _lib() if sycl_enabled() else None
    rs_s = _ts_row_stride(x, ts_s)
    rs_h = _ts_row_stride(x, ts_h)
    if (
        lib is None
        or ts_s is None
        or ts_h is None
        or rs_s is None
        or rs_h is None
        or not _x_ok(x)
        or not _parts_ok(n, table_s, ts_s)
        or not _parts_ok(n, table_h, ts_h)
    ):
        scale = table_s if ts_s is None else table_s + ts_s
        shift = table_h if ts_h is None else table_h + ts_h
        return _eager_ada(x, eps, scale, shift)
    return lib.ada_zero(x, table_s, ts_s, rs_s, table_h, ts_h, rs_h, float(eps))


def affine_addcmul(
    x: torch.Tensor,
    table_s: torch.Tensor,
    ts_s: torch.Tensor | None,
    table_h: torch.Tensor,
    ts_h: torch.Tensor | None,
) -> torch.Tensor:
    """Fused fetch + 1p + addcmul for already-normalized inputs (cross path, 3 -> 1)."""
    n = x.shape[-1]
    lib = _lib() if sycl_enabled() else None
    rs_s = _ts_row_stride(x, ts_s)
    rs_h = _ts_row_stride(x, ts_h)
    if (
        lib is None
        or ts_s is None
        or ts_h is None
        or rs_s is None
        or rs_h is None
        or not _x_ok(x)
        or not _parts_ok(n, table_s, ts_s)
        or not _parts_ok(n, table_h, ts_h)
    ):
        scale = table_s if ts_s is None else table_s + ts_s
        shift = table_h if ts_h is None else table_h + ts_h
        return _eager_affine(x, scale, shift)
    return lib.affine_addcmul(x, table_s, ts_s, rs_s, table_h, ts_h, rs_h)


def post_sa(
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
    lib = _lib() if sycl_enabled() else None
    rs_g = _ts_row_stride(x, ts_g)
    if (
        lib is None
        or norm_weights is not None
        or rs_g is None
        or not _x_ok(x)
        or y.shape != x.shape
        or not y.is_xpu
        or y.dtype != torch.bfloat16
        or not y.is_contiguous()
        or not _parts_ok(n, table_g, ts_g)
    ):
        gate = table_g + ts_g
        fma = x + y * gate if torch_exact_rounding_enabled() else torch.addcmul(x, y, gate)
        return fma, _rms(fma, norm_weights, eps=eps)
    return tuple(lib.post_sa(x, y, table_g, ts_g, rs_g, float(eps)))


def _gate_add_fuse_enabled() -> bool:
    """Whether the text-cross-attention gate multiply is folded into the
    residual add, so ``x = x + attn_out * gate`` becomes one addcmul.
    ``LTX_FUSE_GATE_ADD=0`` keeps the two-op mul-then-add form."""
    return _env_on("LTX_FUSE_GATE_ADD")


def _gate_broadcast_enabled() -> bool:
    """Whether the per-head gate multiply runs in the SYCL broadcast kernel.
    ``LTX_SYCL_GATE_BROADCAST=0`` keeps torch's broadcast multiply."""
    return _env_on("LTX_SYCL_GATE_BROADCAST")


def elementwise_enabled() -> bool:
    """ 向量化 elementwise 開關；``LTX_SYCL_ELEMENTWISE=0`` 或 ``LTX_ORIGINAL_PATH=1`` 走原運算式。"""
    if os.environ.get("LTX_ORIGINAL_PATH", "0") == "1":
        return False
    value = os.environ.get("LTX_SYCL_ELEMENTWISE", "1")
    if value not in {"0", "1"}:
        raise ValueError("LTX_SYCL_ELEMENTWISE must be 0 or 1")
    return value == "1"


def _elementwise_out(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...], int] | None:
    """把被寫入的 (B, T, D) 張量攤成 (T, D)，回傳 (攤平張量, 原形狀, D)。

    要求 16 位元組對齊與內維為 8 的倍數，因為該維以 16 位元組向量寫入。不支援時回 ``None``。
    """
    if not x.is_xpu or x.dtype != torch.bfloat16 or not x.is_contiguous() or x.dim() != 3:
        return None
    d = x.shape[2]
    if d < 8 or d % 8 != 0 or x.data_ptr() % 16 != 0:
        return None
    return x.view(-1, d), tuple(x.shape), d


def _elementwise_operand(b: torch.Tensor, rows: int, d: int) -> torch.Tensor | None:
    """把運算元攤成 2-D：同形 (rows, d)、列廣播 (1, d)、單一值 (1, 1)。不支援時回 ``None``。"""
    if not b.is_xpu or b.dtype != torch.bfloat16 or not b.is_contiguous():
        return None
    if b.numel() == rows * d and b.shape[-1] == d:
        return b.view(-1, d) if b.data_ptr() % 16 == 0 else None
    if b.numel() == d and b.shape[-1] == d and b.data_ptr() % 16 == 0:
        return b.view(1, d)
    if b.numel() == 1:
        return b.view(1, 1)
    return None


def elementwise_fused(x: torch.Tensor, b: torch.Tensor, g: torch.Tensor | None, op: int):
    """ 單核 elementwise：op 0 = x+b、1 = x*b、2 = 相容 addcmul（x + b*g，兩次 bf16 進位）。

    ``b`` 可為同形張量或廣播運算元（單一元素或長度 D）；``g`` 供 op 2／3 使用。
    不支援時回 ``None``。
    """
    lib = _lib() if sycl_enabled() else None
    if lib is None or not elementwise_enabled():
        return None
    x3 = _elementwise_out(x)
    if x3 is None:
        return None
    flat, shape, d = x3
    rows = flat.shape[0]
    b2 = _elementwise_operand(b, rows, d)
    if b2 is None:
        return None
    if op >= 2:
        g2 = None if g is None else _elementwise_operand(g, rows, d)
        if g2 is None or g2.shape[0] != 1:
            return None
        out = lib.elem3(flat, b2, g2, op, 8)
    else:
        out = lib.elem(flat, b2, op, 8)
    return out.view(shape)


def elementwise_addcmul(x: torch.Tensor, b: torch.Tensor, g: torch.Tensor | None):
    """ 單核融合 addcmul：``x + b*g``，單次進位，與 ``torch.addcmul`` 逐位元相同。

    只處理廣播運算元（``g`` 為單一元素或長度 D），因為 torch 在該形狀下走純量路徑。
    不支援時回 ``None``。
    """
    if g is None or g.numel() != 1:
        return None
    return elementwise_fused(x, b, g, 3)


def exact_residual_fuse_enabled() -> bool:
    """torch 精確捨入模式下 FF 殘差 ``x + y * g`` 的單核開關。

    ``LTX_FUSE_EXACT_RESIDUAL``：1 = ElemKernel op 2 單核，0 = torch 乘再加；預設 1。
    ``LTX_ORIGINAL_PATH=1``、``LTX_SYCL_KERNELS=0`` 或 ``LTX_SYCL_ELEMENTWISE=0`` 時為關。
    """
    value = os.environ.get("LTX_FUSE_EXACT_RESIDUAL", "1")
    if value not in {"0", "1"}:
        raise ValueError("LTX_FUSE_EXACT_RESIDUAL must be 0 or 1")
    return value == "1" and sycl_enabled() and elementwise_enabled()


def elementwise_gate_add_exact(x: torch.Tensor, y: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """單核計算 ``x + y * g``，乘積與和各做一次 bf16 捨入（ElemKernel op 2）。

    參數：
        x: (B, T, D) bf16 XPU 連續張量，D 為 8 的倍數，16 位元組對齊。
        y: 與 x 同形的 bf16 XPU 連續張量，16 位元組對齊。
        g: bf16 XPU 連續張量，形狀為 (B, T, D)（每 token gate）或 (1, 1, D)（列廣播 gate），16 位元組對齊。
    回傳：與 x 同形的新張量。
    其他輸入拋出 ``ValueError``；共享庫未載入時拋出 ``RuntimeError``。
    """
    lib = _lib()
    if lib is None:
        raise RuntimeError("elementwise_gate_add_exact: libltx_sycl.so is not loaded")
    x3 = _elementwise_out(x)
    if x3 is None:
        raise ValueError(
            f"elementwise_gate_add_exact: x must be contiguous 16B-aligned bf16 XPU (B,T,D) with D%8==0, "
            f"got shape={tuple(x.shape)} dtype={x.dtype} contiguous={x.is_contiguous()}"
        )
    flat, shape, d = x3
    rows = flat.shape[0]
    if y.shape != x.shape or not y.is_xpu or y.dtype != torch.bfloat16 or not y.is_contiguous() or y.data_ptr() % 16 != 0:
        raise ValueError(
            f"elementwise_gate_add_exact: y must match x {tuple(x.shape)} as contiguous 16B-aligned bf16 XPU, "
            f"got shape={tuple(y.shape)} dtype={y.dtype} contiguous={y.is_contiguous()}"
        )
    per_token = tuple(g.shape) == tuple(shape)
    row_bcast = g.dim() == 3 and g.shape[0] == 1 and g.shape[1] == 1 and g.shape[2] == d
    if (
        not (per_token or row_bcast)
        or not g.is_xpu
        or g.dtype != torch.bfloat16
        or not g.is_contiguous()
        or g.data_ptr() % 16 != 0
    ):
        raise ValueError(
            f"elementwise_gate_add_exact: g must be {tuple(shape)} or (1, 1, {d}) contiguous 16B-aligned bf16 XPU, "
            f"got shape={tuple(g.shape)} stride={tuple(g.stride())} dtype={g.dtype}"
        )
    g2 = g.view(rows, d) if per_token else g.view(1, d)
    return lib.elem3(flat, y.view(rows, d), g2, 2, 8).view(shape)


def gate_inline_enabled() -> bool:
    """torch 精確捨入模式下 FF 殘差 gate 內聯的單核開關。

    ``LTX_FUSE_GATE_INLINE``：1 = ElemKernel op 4 單核（gate 由表列與時間步列在核內相加），
    0 = 先以 torch 算出 gate；預設 1。
    ``LTX_ORIGINAL_PATH=1``、``LTX_SYCL_KERNELS=0`` 或 ``LTX_SYCL_ELEMENTWISE=0`` 時為關。
    """
    value = os.environ.get("LTX_FUSE_GATE_INLINE", "1")
    if value not in {"0", "1"}:
        raise ValueError("LTX_FUSE_GATE_INLINE must be 0 or 1")
    return value == "1" and sycl_enabled() and elementwise_enabled()


def gate_inline_av_enabled() -> bool:
    """torch 精確捨入模式下 a2v / v2a 殘差 gate 與 mask 內聯的單核開關。

    ``LTX_FUSE_GATE_INLINE_AV``：1 = ElemKernel op 5 單核，0 = torch gate 相加、乘 gate、乘 mask、加殘差；
    預設 1。``LTX_ORIGINAL_PATH=1``、``LTX_SYCL_KERNELS=0`` 或 ``LTX_SYCL_ELEMENTWISE=0`` 時為關。
    """
    value = os.environ.get("LTX_FUSE_GATE_INLINE_AV", "1")
    if value not in {"0", "1"}:
        raise ValueError("LTX_FUSE_GATE_INLINE_AV must be 0 or 1")
    return value == "1" and sycl_enabled() and elementwise_enabled()


def _gate_inline_operands(name: str, x: torch.Tensor, y: torch.Tensor, tab: torch.Tensor, ts: torch.Tensor):
    """檢查 gate 內聯運算元並回傳 ``(lib, flat_x, flat_y, tab_row, ts_rs, ts_div, shape, rows)``。

    第 r 列（r = b*T + t）讀取 ts 起點 + (r // ts_div) * ts_rs 的 D 個元素。

    x: (B, T, D) bf16 XPU 連續張量，D 為 8 的倍數，16 位元組對齊。
    y: 與 x 同形的 bf16 XPU 連續張量，16 位元組對齊。
    tab: (1, 1, D) bf16 XPU 連續張量（表列），16 位元組對齊。
    ts: bf16 XPU 張量，內維步幅 1，16 位元組對齊，形狀為下列之一：
        (B, T, D) 每 token 列，列步幅 st 為 8 的倍數且 >= D，B > 1 時批次步幅為 T * st；
        (B, 1, D) 每批次列，批次步幅為 8 的倍數且 >= D；
        (1, 1, D) 共用列。
    其他輸入拋出 ``ValueError``；共享庫未載入時拋出 ``RuntimeError``。
    """
    lib = _lib()
    if lib is None:
        raise RuntimeError(f"{name}: libltx_sycl.so is not loaded")
    x3 = _elementwise_out(x)
    if x3 is None:
        raise ValueError(
            f"{name}: x must be contiguous 16B-aligned bf16 XPU (B,T,D) with D%8==0, "
            f"got shape={tuple(x.shape)} dtype={x.dtype} contiguous={x.is_contiguous()}"
        )
    flat, shape, d = x3
    rows = flat.shape[0]
    b_, t_ = shape[0], shape[1]
    if y.shape != x.shape or not y.is_xpu or y.dtype != torch.bfloat16 or not y.is_contiguous() or y.data_ptr() % 16 != 0:
        raise ValueError(
            f"{name}: y must match x {tuple(shape)} as contiguous 16B-aligned bf16 XPU, "
            f"got shape={tuple(y.shape)} dtype={y.dtype} contiguous={y.is_contiguous()}"
        )
    if (
        tuple(tab.shape) != (1, 1, d)
        or not tab.is_xpu
        or tab.dtype != torch.bfloat16
        or not tab.is_contiguous()
        or tab.data_ptr() % 16 != 0
    ):
        raise ValueError(
            f"{name}: tab must be (1, 1, {d}) contiguous 16B-aligned bf16 XPU, "
            f"got shape={tuple(tab.shape)} stride={tuple(tab.stride())} dtype={tab.dtype}"
        )
    ts_ok = (
        ts.is_xpu and ts.dtype == torch.bfloat16 and ts.dim() == 3 and ts.shape[2] == d
        and ts.stride(2) == 1 and ts.data_ptr() % 16 == 0
    )
    ts_rs, ts_div = -1, 1
    if ts_ok and tuple(ts.shape) == (1, 1, d):
        ts_rs, ts_div = 0, rows
    elif ts_ok and b_ > 1 and tuple(ts.shape) == (b_, 1, d):
        if ts.stride(0) >= d and ts.stride(0) % 8 == 0:
            ts_rs, ts_div = ts.stride(0), t_
    elif ts_ok and tuple(ts.shape) == tuple(shape):
        st = ts.stride(1) if t_ > 1 else ts.stride(0)
        if st >= d and st % 8 == 0 and (b_ == 1 or ts.stride(0) == t_ * st):
            ts_rs, ts_div = st, 1
    if ts_rs < 0:
        raise ValueError(
            f"{name}: ts must be {tuple(shape)}, ({b_}, 1, {d}) or (1, 1, {d}) bf16 XPU with unit inner stride, "
            f"row stride a multiple of 8 and >= {d}, batch stride T*row_stride, 16B-aligned; "
            f"got shape={tuple(ts.shape)} stride={tuple(ts.stride())} dtype={ts.dtype}"
        )
    return lib, flat, y.view(rows, d), tab.view(1, d), ts_rs, ts_div, shape, rows


def elementwise_gate_add_inline(x: torch.Tensor, y: torch.Tensor, tab: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
    """單核計算 ``x + y * (tab + ts)``，gate、乘積、和各做一次 bf16 捨入（ElemKernel op 4）。

    參數：
        x: (B, T, D) bf16 XPU 連續張量，D 為 8 的倍數，16 位元組對齊。
        y: 與 x 同形的 bf16 XPU 連續張量，16 位元組對齊。
        tab: (1, 1, D) bf16 XPU 連續張量（gate 表列），16 位元組對齊。
        ts: (B, T, D) 每 token 時間步列、(B, 1, D) 每批次列或 (1, 1, D) 共用列；
            bf16 XPU，內維步幅 1，16 位元組對齊，步幅條件見 :func:`_gate_inline_operands`。
    回傳：與 x 同形的新張量。
    其他輸入拋出 ``ValueError``；共享庫未載入時拋出 ``RuntimeError``。
    """
    lib, flat, y2, tab2, ts_rs, ts_div, shape, _ = _gate_inline_operands("elementwise_gate_add_inline", x, y, tab, ts)
    return lib.elem_gate(flat, y2, tab2, ts, ts_rs, ts_div, None, 1, 8).view(shape)


def elementwise_gate_mask_add_inline(
    x: torch.Tensor, y: torch.Tensor, tab: torch.Tensor, ts: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """單核計算 ``x + (y * (tab + ts)) * mask``，gate、乘 gate、乘 mask、和各做一次 bf16 捨入（ElemKernel op 5）。

    參數：
        x, y, tab, ts: 同 :func:`elementwise_gate_add_inline`。
        mask: (B, 1, 1) bf16 XPU 連續張量（每批次一個值），B 與 x 相同。
    回傳：與 x 同形的新張量。
    其他輸入拋出 ``ValueError``；共享庫未載入時拋出 ``RuntimeError``。
    """
    name = "elementwise_gate_mask_add_inline"
    lib, flat, y2, tab2, ts_rs, ts_div, shape, rows = _gate_inline_operands(name, x, y, tab, ts)
    b_ = shape[0]
    if (
        mask is None
        or tuple(mask.shape) != (b_, 1, 1)
        or not mask.is_xpu
        or mask.dtype != torch.bfloat16
        or not mask.is_contiguous()
    ):
        raise ValueError(
            f"{name}: mask must be ({b_}, 1, 1) contiguous bf16 XPU, got "
            f"{None if mask is None else (tuple(mask.shape), mask.dtype, mask.is_contiguous())}"
        )
    return lib.elem_gate(flat, y2, tab2, ts, ts_rs, ts_div, mask.view(b_), rows // b_, 8).view(shape)


def gate_broadcast_mul(attn_out, gates, heads, dim_head):
    """`attn_out.view(B,T,H,Dh) * gates.unsqueeze(-1)` via SYCL.

    Bitwise identical to the torch formula (bf16, fp32 product, one RNE).
    Returns None for inputs outside the kernel's contract so the caller keeps
    torch's path."""
    lib = _lib() if sycl_enabled() else None
    if lib is None or not _gate_broadcast_enabled() or attn_out.dim() != 3:
        return None
    b, t, c = attn_out.shape
    if not (
        attn_out.is_xpu
        and attn_out.dtype == torch.bfloat16
        and attn_out.is_contiguous()
        and gates.is_xpu
        and gates.dtype == torch.bfloat16
        and gates.is_contiguous()
        and c == heads * dim_head
        and gates.numel() == b * t * heads
        and dim_head % 8 == 0
        and attn_out.numel() % 8 == 0
        and attn_out.data_ptr() % 16 == 0
        and gates.data_ptr() % 16 == 0
    ):
        return None
    return lib.gate_broadcast_mul(attn_out, gates.reshape(-1), int(dim_head))


def _gelu_tanh_fast_enabled() -> bool:
    """tanh-approx GELU 的 SYCL 開關。

    LTX_TORCH_EXACT_ROUNDING=1 時核內以 libm tanh 與 torch 的 GeluTanhFunctor 同式，輸出逐位元相同；
    LTX_TORCH_EXACT_ROUNDING=0 時核內以 native exp2 計算 x * sigmoid(2u)，進位點不同。
    LTX_SYCL_GELU_FAST=0 走 torch 的 GELU。
    """
    return _env_on("LTX_SYCL_GELU_FAST", "1")


def gelu_tanh_fast(x):
    """SYCL tanh-approx GELU。輸入不在核的契約內時回 None，呼叫端保留原路徑。"""
    lib = _lib() if sycl_enabled() else None
    if lib is None or not _gelu_tanh_fast_enabled():
        return None
    if not (
        x.is_xpu
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and x.numel() % 8 == 0
        and x.data_ptr() % 16 == 0
    ):
        return None
    return lib.gelu_tanh_fast(x)


def xca_gate_inline_enabled() -> bool:
    """torch 精確捨入模式下 text cross-attention gate 內聯的單核開關。

    ``LTX_FUSE_GATE_INLINE_XCA``：1 = ElemKernel op 4 單核（gate 由表列與時間步列在核內相加），
    0 = 先以 torch 算出 gate；預設 1。
    ``LTX_ORIGINAL_PATH=1``、``LTX_SYCL_KERNELS=0`` 或 ``LTX_SYCL_ELEMENTWISE=0`` 時為關。
    """
    value = os.environ.get("LTX_FUSE_GATE_INLINE_XCA", "1")
    if value not in {"0", "1"}:
        raise ValueError("LTX_FUSE_GATE_INLINE_XCA must be 0 or 1")
    return value == "1" and sycl_enabled() and elementwise_enabled()


def cross_attention_adaln(
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
    # SYCL fused cross-attention adaln: two 3->1 chains; residual gate path via
    # ElemKernel op 4 when enabled.
    attn_input = affine_addcmul(x_normed, qsc_tab, qsc_ts, qsh_tab, qsh_ts)
    encoder_hidden_states = affine_addcmul(context, ksc_tab, ksc_ts, ksh_tab, ksh_ts)
    attn_out = attn(attn_input, context=encoder_hidden_states, mask=context_mask)
    if residual is not None and _gate_add_fuse_enabled():
        if torch_exact_rounding_enabled() and xca_gate_inline_enabled():
            if g_ts is None:
                raise ValueError("cross_attention_adaln: g_ts is required for the op 4 path")
            # Single fused pass: gate add, product, and sum in ElemKernel op 4.
            return elementwise_gate_add_inline(residual, attn_out, g_tab, g_ts)
        gate = g_tab + g_ts
        if torch_exact_rounding_enabled():
            # Torch-exact rounding: two ops, each rounding once, matching torch's bf16 result.
            fused = elementwise_fused(residual, attn_out, gate, 2)
            if fused is not None:
                return fused
            return residual + attn_out * gate
        # Single fused pass instead of mul + add.
        return torch.addcmul(residual, attn_out, gate)
    return attn_out * (g_tab + g_ts)


def _rope_negfold_vec_enabled() -> bool:
    """Sub-switch for the vectorised split-rope negfold kernel; LTX_SYCL_ROPE_NEGFOLD_VEC=0 selects the scalar kernel."""
    return _env_on("LTX_SYCL_ROPE_NEGFOLD_VEC")


def _rope_negfold_vec_ok(t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, dh: int) -> bool:
    """Guards for the u16x8 rope_negfold_vec kernel: unit innermost stride, Dh multiple of 8,
    and 16-byte-aligned row starts (x/cos/sin data_ptr aligned + outer strides x8)."""
    if dh % 8 != 0 or dh == 0:
        return False
    for t_, st in ((t, t.stride()), (cos, cos.stride()), (sin, sin.stride())):
        if st[-1] != 1 or t_.data_ptr() % 16 != 0:
            return False
        if any(s % 8 != 0 for s in st[:3]):
            return False
    return True


def norm_rope(
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
    if not (sycl_enabled() and _norm_rope_enabled()):
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
        or cos_freqs.shape[0] != batch
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
        lib.norm_rope(
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


def split_rope(
    input_tensor: torch.Tensor,
    cos_freqs: torch.Tensor,
    sin_freqs: torch.Tensor,
) -> torch.Tensor:
    """Fused split-rope negfold (mul + 2 x addcmul_ -> 1 launch).

    Covers the negfold contract for both 4D (B,H,T,D) tensors and the 3D
    (B,T,H*D) needs_reshape path used by the DiT attention blocks (views in,
    fused compute, views back out). Inputs outside the kernel contract
    (dtype, device, cos/sin batch different from the input batch) use the
    eager negfold formula.
    """
    from ltx_core.model.transformer.rope import _apply_split_rotary_emb_negfold as _eager  # noqa: PLC0415

    def _fallback():
        return _eager(input_tensor, cos_freqs, sin_freqs)

    lib = _lib() if sycl_enabled() else None
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
        if cos_freqs.shape[0] != b or cos_freqs.shape[1] != h:
            return _fallback()
        dh = d // 2
        xs = [int(s) for s in input_tensor.stride()]
        cs = [int(s) for s in cos_freqs.stride()]
        ss = [int(s) for s in sin_freqs.stride()]
        if _rope_negfold_vec_enabled() and _rope_negfold_vec_ok(input_tensor, cos_freqs, sin_freqs, dh):
            return lib.rope_negfold_vec(input_tensor, xs, cos_freqs, cs, sin_freqs, ss, b, h, t, dh)
        return lib.rope_negfold(input_tensor, xs, cos_freqs, cs, sin_freqs, ss, b, h, t, dh)
    # 3D needs_reshape path: mirror the eager view dance, fuse the middle.
    b_in = int(input_tensor.shape[0])
    b_freq, h, t, dh = (int(v) for v in cos_freqs.shape)
    if b_freq != b_in:
        return _fallback()
    if int(input_tensor.shape[-1]) != h * dh * 2 or h <= 0 or dh <= 0:
        return _fallback()
    x4 = input_tensor.unflatten(-1, (h, -1)).transpose(1, 2)
    if tuple(int(v) for v in x4.shape) != (b_in, h, t, dh * 2):
        return _fallback()
    xs = [int(s) for s in x4.stride()]
    cs = [int(s) for s in cos_freqs.stride()]
    ss = [int(s) for s in sin_freqs.stride()]
    if _rope_negfold_vec_enabled() and _rope_negfold_vec_ok(x4, cos_freqs, sin_freqs, dh):
        out4 = lib.rope_negfold_vec(x4, xs, cos_freqs, cs, sin_freqs, ss, b_in, h, t, dh)
        return out4.transpose(1, 2).flatten(-2)
    out4 = lib.rope_negfold(x4, xs, cos_freqs, cs, sin_freqs, ss, b_in, h, t, dh)
    return out4.transpose(1, 2).flatten(-2)


def _fp8_widen_enabled() -> bool:
    """Whether the vectorised fp8(e4m3fn) -> bf16 widening kernel is used.

    Measured end-to-end gain -2.250 s (fp8 dequant 3.520 -> 0.253 s plus
    about 1.0 s in the widening kernel), bitwise against torch.  On by default;
    ``LTX_SYCL_FP8_WIDEN=0`` (or ``LTX_ORIGINAL_PATH=1``) uses torch's own
    cast.
    """
    return _env_on("LTX_SYCL_FP8_WIDEN")


def fp8_widen(x):
    """fp8 e4m3fn -> bf16 via SYCL (bitwise vs torch). None => caller uses torch."""
    lib = _lib() if sycl_enabled() else None
    if lib is None or not _fp8_widen_enabled():
        return None
    if not (
        x.is_xpu
        and x.dtype == torch.float8_e4m3fn
        and x.is_contiguous()
        and x.numel() % 8 == 0
        and x.data_ptr() % 16 == 0
    ):
        return None
    return lib.fp8_widen(x)


