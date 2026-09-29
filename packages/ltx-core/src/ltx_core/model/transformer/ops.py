import os
from typing import List, Protocol

import torch
from torch import nn

from ltx_core.model.transformer.rope import LTXRopeType, apply_rotary_emb
from ltx_core.utils import rms_norm


def _env_on(name, default="1"):
    # Central env helper: LTX_ALL_ORIG=1 forces round-0 (all fall back to original formulas).
    if os.environ.get("LTX_ALL_ORIG", "0") == "1":
        return False
    return os.environ.get(name, default) != "0"


class PreAttentionCallable(Protocol):
    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        attn_module: nn.Module,
        mask: torch.Tensor | None,
        pe: torch.Tensor | None,
        k_pe: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


class PytorchPreAttention(PreAttentionCallable):
    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        attn_module: nn.Module,
        mask: torch.Tensor | None,  # noqa: ARG002
        pe: torch.Tensor | None,
        k_pe: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            pe is not None
            and _env_on("LTX_R19_NORM_ROPE")
            and attn_module.rope_type == LTXRopeType.SPLIT
            and (k_pe is None or k_pe is pe)
        ):
            from ltx_core.model.transformer.r8_sycl import r19_norm_rope  # noqa: PLC0415

            pair = r19_norm_rope(
                q,
                k,
                attn_module.q_norm.weight,
                attn_module.k_norm.weight,
                pe[0],
                pe[1],
                attn_module.q_norm.eps,
            )
            if pair is not None:
                return pair
        q = attn_module.q_norm(q)
        k = attn_module.k_norm(k)
        if pe is not None:
            if (
                _env_on("LTX_R18_SYCLEFF", "0")
                and attn_module.rope_type == LTXRopeType.SPLIT
                and (k_pe is None or k_pe is pe)
            ):
                from ltx_core.model.transformer.r8_sycl import r18_dual_split_rope  # noqa: PLC0415

                pair = r18_dual_split_rope(q, k, pe[0], pe[1])
                if pair is not None:
                    return pair
            q = apply_rotary_emb(q, pe, attn_module.rope_type)
            k = apply_rotary_emb(k, pe if k_pe is None else k_pe, attn_module.rope_type)
        return q, k


class AdaZeroCallable(Protocol):
    def __call__(
        self,
        x: torch.Tensor,
        eps: float,
        scale: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor: ...


class PytorchAdaZeroFunction(AdaZeroCallable):
    def __call__(
        self,
        x: torch.Tensor,
        eps: float,
        scale: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor:
        # T2 (AdaLN fusion): fold the scale/shift affine into one addcmul kernel
        # (was norm -> (1+scale) add -> mul -> +shift add = 3 elementwise launches;
        # now (1+scale) add -> norm -> addcmul = 2). Numerically ~= 1 ulp (bf16).
        # LTX_R3_FUSE=0 (or LTX_ALL_ORIG=1) restores the original formula.
        if not _env_on("LTX_R3_FUSE"):
            return rms_norm(x, eps=eps) * (1 + scale) + shift
        normed = rms_norm(x, eps=eps)
        return torch.addcmul(shift, normed, 1 + scale)


class PostSACallable(Protocol):
    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        norm_weights: torch.Tensor | None,
        eps: float,
        gate: torch.Tensor,
    ) -> List[torch.Tensor]: ...


class PytorchPostSAFunction(PostSACallable):
    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        norm_weights: torch.Tensor | None,
        eps: float,
        gate: torch.Tensor,
    ) -> List[torch.Tensor]:
        # T2 (AdaLN fusion): mul+add -> single addcmul kernel.
        # LTX_R3_FUSE=0 (or LTX_ALL_ORIG=1) restores the original formula.
        if not _env_on("LTX_R3_FUSE"):
            x_fma = x + y * gate
        else:
            x_fma = torch.addcmul(x, y, gate)
        return x_fma, rms_norm(x_fma, norm_weights, eps=eps)


class GatedAttentionCallable(Protocol):
    def __call__(
        self,
        x: torch.Tensor,
        attn_out: torch.Tensor,
        attn_module: nn.Module,
    ) -> torch.Tensor: ...


class PytorchGatedAttention(GatedAttentionCallable):
    def __call__(
        self,
        x: torch.Tensor,
        attn_out: torch.Tensor,
        attn_module: nn.Module,
    ) -> torch.Tensor:
        gate_logits = attn_module.to_gate_logits(x)  # (B, T, H)
        b, t, _ = attn_out.shape
        gates = 2.0 * torch.sigmoid(gate_logits)  # (B, T, H)
        # R10D F2: SYCL per-head broadcast mul (bitwise vs the torch formula);
        # returns None (and we fall through) on any unsupported case.
        from ltx_core.model.transformer import r8_sycl as _R8  # noqa: PLC0415

        _g = _R8.r10d_gate_mul(attn_out, gates, attn_module.heads, attn_module.dim_head)
        if _g is not None:
            return _g.view(b, t, attn_module.heads * attn_module.dim_head)
        out = attn_out.view(b, t, attn_module.heads, attn_module.dim_head)
        out = out * gates.unsqueeze(-1)  # (B, T, H, D) * (B, T, H, 1)
        return out.view(b, t, attn_module.heads * attn_module.dim_head)
