"""R7 elementwise chain fusion (C1): Triton-XPU fused kernels.

Fuses whole chains (not pairwise):
  K1FULL  ada_zero 4 launches -> 1:  out = shift + rms_norm(x) * (1 + scale)
          with scale/shift fetched on the fly (table broadcast + timestep row).
          DO_NORM=0 variant covers cross-attention pre-addcmuls (fetch+1p+addcmul 3->1).
  K2FULL  post_sa 3 launches -> 1:   (fma, normed) = (x + y*gate, rms_norm(fma))
          with gate fetched on the fly; both outputs materialized in one launch.

Numerics: fp32 compute, bf16 IO. Validated closer to fp64 truth than the stepwise
bf16 eager path (R6 precedent). Eager fallbacks below are op-for-op identical to the
current R3/R6 path (same operand order), so LTX_R7_FUSE=0 / LTX_ALL_ORIG=1 /
non-XPU / non-bf16 / non-contiguous all behave bitwise as before.

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


def r7_on() -> bool:
    """R7 master switch: default on; LTX_R7_FUSE=0 or LTX_ALL_ORIG=1 restores old path."""
    return _env_on("LTX_R7_FUSE")


_TRITON_OK: bool | None = None


def triton_ok() -> bool:
    """True once Triton-XPU is importable and an XPU device exists (cached)."""
    global _TRITON_OK
    if _TRITON_OK is None:
        try:
            import triton  # noqa: F401, PLC0415

            _TRITON_OK = hasattr(torch, "xpu") and torch.xpu.is_available()
        except Exception:  # noqa: BLE001
            _TRITON_OK = False
    return bool(_TRITON_OK)


_KERNS: dict = {}


def _kerns() -> dict:
    """Lazily build Triton kernels (no import-time triton dependency)."""
    if "k1" not in _KERNS:
        import triton  # noqa: PLC0415
        import triton.language as tl  # noqa: PLC0415

        @triton.jit
        def _k1_kernel(
            x_ptr,
            tbs_ptr,
            tss_ptr,
            tss_row_stride,
            tbh_ptr,
            tsh_ptr,
            tsh_row_stride,
            o_ptr,
            n_cols,
            eps,
            DO_NORM: tl.constexpr,
            BLOCK: tl.constexpr,
        ):
            pid = tl.program_id(0)
            col = tl.arange(0, BLOCK)
            m = col < n_cols
            x = tl.load(x_ptr + pid * n_cols + col, mask=m, other=0.0).to(tl.float32)
            if DO_NORM:
                rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / n_cols + eps)
                xn = x * rstd
            else:
                xn = x
            tss = tl.load(tss_ptr + pid * tss_row_stride + col, mask=m, other=0.0).to(tl.float32)
            tsh = tl.load(tsh_ptr + pid * tsh_row_stride + col, mask=m, other=0.0).to(tl.float32)
            tbs = tl.load(tbs_ptr + col, mask=m, other=0.0).to(tl.float32)
            tbh = tl.load(tbh_ptr + col, mask=m, other=0.0).to(tl.float32)
            out = ((tbh + tsh) + xn * (1.0 + (tbs + tss))).to(tl.bfloat16)
            tl.store(o_ptr + pid * n_cols + col, out, mask=m)

        @triton.jit
        def _k2_kernel(
            x_ptr,
            y_ptr,
            tbg_ptr,
            tsg_ptr,
            tsg_row_stride,
            fma_ptr,
            out_ptr,
            n_cols,
            eps,
            BLOCK: tl.constexpr,
        ):
            pid = tl.program_id(0)
            col = tl.arange(0, BLOCK)
            m = col < n_cols
            base = pid * n_cols
            x = tl.load(x_ptr + base + col, mask=m, other=0.0).to(tl.float32)
            y = tl.load(y_ptr + base + col, mask=m, other=0.0).to(tl.float32)
            tbg = tl.load(tbg_ptr + col, mask=m, other=0.0).to(tl.float32)
            tsg = tl.load(tsg_ptr + pid * tsg_row_stride + col, mask=m, other=0.0).to(tl.float32)
            fma = x + y * (tbg + tsg)
            rstd = 1.0 / tl.sqrt(tl.sum(fma * fma, axis=0) / n_cols + eps)
            tl.store(fma_ptr + base + col, fma.to(tl.bfloat16), mask=m)
            tl.store(out_ptr + base + col, (fma * rstd).to(tl.bfloat16), mask=m)

        _KERNS["k1"] = _k1_kernel
        _KERNS["k2"] = _k2_kernel
    return _KERNS


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


def _block(n: int) -> int:
    block = 1
    while block < n:
        block *= 2
    return block


def _launch_k1(
    x: torch.Tensor,
    table_s: torch.Tensor,
    ts_s: torch.Tensor | None,
    table_h: torch.Tensor,
    ts_h: torch.Tensor | None,
    eps: float,
    do_norm: bool,
) -> torch.Tensor:
    from ltx_core.utils import rms_norm as _rms  # noqa: PLC0415

    n = x.shape[-1]
    if not (
        r7_on()
        and triton_ok()
        and _x_ok(x)
        and _parts_ok(n, table_s, ts_s)
        and _parts_ok(n, table_h, ts_h)
    ):
        scale = table_s if ts_s is None else table_s + ts_s
        shift = table_h if ts_h is None else table_h + ts_h
        base = _rms(x, eps=eps) if do_norm else x
        return torch.addcmul(shift, base, 1 + scale)
    dev = x.device
    zeros1 = torch.zeros(n, dtype=torch.bfloat16, device=dev)
    out = torch.empty_like(x)
    rows = x.numel() // n
    _kerns()["k1"][(rows,)](
        x,
        table_s,
        ts_s if ts_s is not None else zeros1,
        _row_stride(ts_s),
        table_h,
        ts_h if ts_h is not None else zeros1,
        _row_stride(ts_h),
        out,
        n,
        float(eps),
        DO_NORM=do_norm,
        BLOCK=_block(n),
    )
    return out


def r7_ada_zero(
    x: torch.Tensor,
    eps: float,
    table_s: torch.Tensor,
    ts_s: torch.Tensor,
    table_h: torch.Tensor,
    ts_h: torch.Tensor,
) -> torch.Tensor:
    """Fused ada_zero (rms_norm + fetch + 1p + addcmul, 4 launches -> 1)."""
    return _launch_k1(x, table_s, ts_s, table_h, ts_h, eps, True)


def r7_affine_addcmul(
    x: torch.Tensor,
    table_s: torch.Tensor,
    ts_s: torch.Tensor | None,
    table_h: torch.Tensor,
    ts_h: torch.Tensor | None,
) -> torch.Tensor:
    """Fused fetch + 1p + addcmul for already-normalized inputs (cross path, 3 -> 1)."""
    return _launch_k1(x, table_s, ts_s, table_h, ts_h, 1e-6, False)


def r7_post_sa(
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
    if not (
        r7_on()
        and triton_ok()
        and norm_weights is None
        and _x_ok(x)
        and y.shape == x.shape
        and y.is_xpu
        and y.dtype == torch.bfloat16
        and y.is_contiguous()
        and _parts_ok(n, table_g, ts_g)
    ):
        gate = table_g + ts_g
        fma = torch.addcmul(x, y, gate)
        return fma, _rms(fma, norm_weights, eps=eps)
    fma = torch.empty_like(x)
    out = torch.empty_like(x)
    rows = x.numel() // n
    _kerns()["k2"][(rows,)](
        x, y, table_g, ts_g, _row_stride(ts_g), fma, out, n, float(eps), BLOCK=_block(n)
    )
    return fma, out


def r7_cross_attention_adaln(
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
):
    # R7 fused cross-attention ada: 2x (fetch+1p+addcmul 3->1); gate mul + attn unchanged.
    attn_input = r7_affine_addcmul(x_normed, qsc_tab, qsc_ts, qsh_tab, qsh_ts)
    encoder_hidden_states = r7_affine_addcmul(context, ksc_tab, ksc_ts, ksh_tab, ksh_ts)
    gate = g_tab + g_ts
    return attn(attn_input, context=encoder_hidden_states, mask=context_mask) * gate
