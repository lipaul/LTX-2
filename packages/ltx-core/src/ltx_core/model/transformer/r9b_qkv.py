# R9B fused q/k/v projections (additive; default on).
#
# Self-attention (context_dim is None) fuses to_q/to_k/to_v into one
# Linear(D, 3I) (forward chunk(3)); cross-attention fuses to_k/to_v into
# Linear(Dc, 2I) (chunk(2), to_q stays). Downstream (q_norm/k_norm, RoPE,
# SDPA, to_out) sees identical shapes.
#
# Installed post-load by maybe_fuse_qkv (old weights are catted, then the
# originals are deleted, so VRAM does not double). Attention.forward branches
# on hasattr(self, to_qkv) / hasattr(self, to_kv). LTX_R9B_QKV=0 (or
# LTX_ALL_ORIG=1) keeps the three-linear path bit-for-bit. Numerics are NOT
# bitwise (GEMM N changes tile/summation order); accept per R6/R7 precedent
# (relL2/ULP).
from __future__ import annotations

import logging
import os

import torch

log = logging.getLogger(__name__)


def _env_on(name: str, default: str = "1") -> bool:
    if os.environ.get("LTX_ALL_ORIG", "0") == "1":
        return False
    return os.environ.get(name, default) != "0"


def r9b_on() -> bool:
    return _env_on("LTX_R9B_QKV", "0")  # DEFAULT OFF 2026-09-23: Gate2 +1.450s regression; opt-in only


def _drop_r4_slots(layer: torch.nn.Linear) -> None:
    try:
        from ltx_core.quantization import fp8_cast as _fc
    except Exception:
        return
    for slot in ("_r4_w_up", "_r4_b_up"):
        try:
            entry = getattr(layer, slot, None)
            if entry is not None:
                cached = entry[1]
                _fc._r4_fp8cache_used_bytes -= cached.numel() * cached.element_size()
        except Exception:
            pass


def _retype_fp8(layer: torch.nn.Linear, ref: torch.nn.Linear) -> None:
    from ltx_core.quantization.fp8_cast import Fp8CastLinear

    layer.__class__ = Fp8CastLinear
    layer._with_stochastic_rounding = bool(getattr(ref, "_with_stochastic_rounding", False))
    layer._seed = int(getattr(ref, "_seed", 0))


def fuse_attention(attn: torch.nn.Module) -> str | None:
    if hasattr(attn, "to_qkv") or hasattr(attn, "to_kv"):
        return None
    to_q, to_k, to_v = attn.to_q, attn.to_k, attn.to_v
    inner = to_q.out_features
    if to_k.out_features != inner or to_v.out_features != inner:
        return None
    if to_q.bias is None or to_k.bias is None or to_v.bias is None:
        return None
    is_self = bool(getattr(attn, "_r9b_is_self", to_k.in_features == to_q.in_features))
    is_fp8 = to_q.weight.dtype == torch.float8_e4m3fn
    if is_self:
        # NOTE: no dtype=fp8 at construction (kaiming init lacks fp8);
        # assign the exact-dtype catted parameters directly.
        fused = torch.nn.Linear(to_q.in_features, 3 * inner, bias=True)
        fused.weight = torch.nn.Parameter(
            torch.cat([to_q.weight, to_k.weight, to_v.weight], dim=0).to(to_q.weight.device)
        )
        fused.bias = torch.nn.Parameter(
            torch.cat([to_q.bias, to_k.bias, to_v.bias], dim=0).to(to_q.weight.device)
        )
        if is_fp8:
            _retype_fp8(fused, to_q)
        for layer in (to_q, to_k, to_v):
            _drop_r4_slots(layer)
        del attn.to_q, attn.to_k, attn.to_v
        attn.to_qkv = fused
        return "qkv"
    fused = torch.nn.Linear(to_k.in_features, 2 * inner, bias=True)
    fused.weight = torch.nn.Parameter(
        torch.cat([to_k.weight, to_v.weight], dim=0).to(to_k.weight.device)
    )
    fused.bias = torch.nn.Parameter(
        torch.cat([to_k.bias, to_v.bias], dim=0).to(to_k.weight.device)
    )
    if is_fp8:
        _retype_fp8(fused, to_k)
    for layer in (to_k, to_v):
        _drop_r4_slots(layer)
    del attn.to_k, attn.to_v
    attn.to_kv = fused
    return "kv"


def maybe_fuse_qkv(model: torch.nn.Module) -> torch.nn.Module:
    if not r9b_on():
        return model
    from ltx_core.model.transformer.attention import Attention

    n_qkv = n_kv = 0
    for m in model.modules():
        if isinstance(m, Attention):
            r = fuse_attention(m)
            if r == "qkv":
                n_qkv += 1
            elif r == "kv":
                n_kv += 1
    log.info("R9B fused Attention: %d qkv + %d kv", n_qkv, n_kv)
    return model
