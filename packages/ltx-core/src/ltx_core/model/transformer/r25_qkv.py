from __future__ import annotations

import logging
import os

import torch

log = logging.getLogger(__name__)


def _env_on(name: str, default: str) -> bool:
    if os.environ.get("LTX_ALL_ORIG", "0") == "1":
        return False
    value = os.environ.get(name, default)
    if value not in {"0", "1"}:
        raise ValueError(f"{name} must be 0 or 1")
    return value == "1"


def r25_on() -> bool:
    return _env_on("LTX_R25_QKV_MERGE", "1")


def _is_video_self_attention(name: str, module: torch.nn.Module) -> bool:
    return name.rsplit(".", 1)[-1] == "attn1" and bool(getattr(module, "_r9b_is_self", False))


def _drop_r4_slots(layer: torch.nn.Module) -> None:
    from ltx_core.quantization import fp8_cast

    for slot in ("_r4_w_up", "_r4_b_up"):
        entry = getattr(layer, slot, None)
        if entry is not None:
            cached = entry[1]
            fp8_cast._r4_fp8cache_used_bytes -= cached.numel() * cached.element_size()


def _k_major(weight: torch.Tensor) -> torch.Tensor:
    """Return *weight* as a contiguous (K, N) tensor with its values preserved."""
    if weight.dtype == torch.float8_e4m3fn:
        return weight.float().t().contiguous().to(weight.dtype)
    return weight.t().contiguous()


class KMajorLinear(torch.nn.Module):
    """Linear projection whose weight is stored K-major.

    Storing the weight transposed, ``(in_features, out_features)`` and
    contiguous, makes the widened B operand of the GEMM N-contiguous instead of
    K-contiguous. ``forward`` applies it with a single :func:`torch.addmm` over
    the flattened token axis, so the result shape and dtype match the
    ``torch.nn.Linear`` it replaces.
    """

    def __init__(self, in_features: int, out_features: int, dtype: torch.dtype, stochastic: bool, seed: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self._stochastic = stochastic
        self._seed = seed
        self.weight = torch.nn.Parameter(
            torch.empty((in_features, out_features), dtype=dtype, device="meta"), requires_grad=False
        )
        self.bias = torch.nn.Parameter(
            torch.empty((out_features,), dtype=dtype, device="meta"), requires_grad=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: A002, type: ignore[override]
        from ltx_core.quantization import fp8_cast

        dtype = x.dtype
        if self.weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            weight = fp8_cast._upcast_and_round(self.weight, dtype, self._stochastic, self._seed)
            bias = fp8_cast._upcast_and_round(self.bias, dtype, self._stochastic, self._seed)
        else:
            weight = self.weight
            bias = self.bias
        flat = x.reshape(-1, self.in_features)
        out = torch.addmm(bias, flat, weight)
        return out.view(*x.shape[:-1], self.out_features)


def _regroup_video_self_qkv(attn: torch.nn.Module) -> bool:
    if hasattr(attn, "to_qkv") or hasattr(attn, "to_kv"):
        return False
    to_q, to_k, to_v = attn.to_q, attn.to_k, attn.to_v
    inner = to_q.out_features
    if to_k.out_features != inner or to_v.out_features != inner:
        return False
    if to_q.in_features != to_k.in_features or to_q.in_features != to_v.in_features:
        return False
    if to_q.bias is None or to_k.bias is None or to_v.bias is None:
        return False
    if to_q.weight.dtype != to_k.weight.dtype or to_q.weight.dtype != to_v.weight.dtype:
        return False
    if to_q.weight.device != to_k.weight.device or to_q.weight.device != to_v.weight.device:
        return False

    for name, old in (("to_q", to_q), ("to_k", to_k), ("to_v", to_v)):
        stochastic = bool(getattr(old, "_with_stochastic_rounding", False))
        seed = int(getattr(old, "_seed", 0))
        replacement = KMajorLinear(old.in_features, old.out_features, old.weight.dtype, stochastic, seed)
        replacement.weight = torch.nn.Parameter(
            _k_major(old.weight.data), requires_grad=False
        )
        replacement.bias = torch.nn.Parameter(old.bias.data.clone(), requires_grad=False)
        _drop_r4_slots(old)
        setattr(attn, name, replacement)
    return True


def maybe_fuse_video_self_qkv(model: torch.nn.Module) -> torch.nn.Module:
    """Apply the R25 QKV GEMM route to every video self-attention projection."""
    if not r25_on():
        return model

    from ltx_core.model.transformer.attention import Attention

    candidates = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, Attention)
        and _is_video_self_attention(name, module)
        and not hasattr(module, "to_qkv")
        and not hasattr(module, "to_kv")
    ]
    regrouped = sum(_regroup_video_self_qkv(module) for _, module in candidates)
    log.info("R25 video self QKV K-major projections: %d attentions", regrouped)
    return model
