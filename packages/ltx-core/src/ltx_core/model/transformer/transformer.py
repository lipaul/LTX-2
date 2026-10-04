from dataclasses import dataclass, field, replace

import os

import torch

from ltx_core.model.transformer.adaln import adaln_embedding_coefficient
from ltx_core.model.transformer.attention import (
    Attention,
    AttentionCallable,
    AttentionFunction,
    AttentionOps,
    MaskedAttentionCallable,
    MaskedAttentionFunction,
)
from ltx_core.model.transformer.feed_forward import FeedForward
from ltx_core.model.transformer.ops import (
    AdaZeroCallable,
    GatedAttentionCallable,
    PostSACallable,
    PreAttentionCallable,
    PytorchAdaZeroFunction,
    PytorchGatedAttention,
    PytorchPostSAFunction,
    PytorchPreAttention,
)
from ltx_core.model.transformer.sycl_kernels import (
    lib_ok as _lib_ok,
    ada_zero,
    cross_attention_adaln,
    sycl_enabled as _sycl_enabled,
    post_sa,
    elementwise_addcmul,
    elementwise_fused,
    elementwise_gate_add_exact,
    elementwise_gate_add_inline,
    elementwise_gate_mask_add_inline,
    exact_residual_fuse_enabled,
    gate_inline_av_enabled,
    gate_inline_enabled,
)
from ltx_core.model.transformer.rope import LTXRopeType
from ltx_core.model.transformer.transformer_args import TransformerArgs


def elementwise_mul(x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """廣播運算元為單一元素時的逐元素乘；不支援時走原運算式。"""
    got = elementwise_fused(x, g, None, 1)
    return x * g if got is None else got


@dataclass
class TransformerConfig:
    dim: int
    heads: int
    d_head: int
    context_dim: int
    apply_gated_attention: bool = False
    cross_attention_adaln: bool = False
    ff_bias: bool = True


@dataclass(frozen=True)
class TransformerOpsConfig:
    """Pluggable ops for :class:`BasicAVTransformerBlock`.
    Use :meth:`from_functions` to construct from enum values or partial overrides
    without spelling out a full :class:`AttentionOps`.
    """

    attention_ops: AttentionOps = field(default_factory=AttentionOps)
    ada_zero_function: AdaZeroCallable = field(default_factory=PytorchAdaZeroFunction)
    post_sa_function: PostSACallable = field(default_factory=PytorchPostSAFunction)

    @classmethod
    def from_functions(
        cls,
        attention: AttentionFunction | AttentionCallable = AttentionFunction.AUTOMATIC,
        masked_attention: MaskedAttentionFunction | MaskedAttentionCallable = MaskedAttentionFunction.AUTOMATIC,
        preattention: PreAttentionCallable | None = None,
        gated_attention: GatedAttentionCallable | None = None,
        ada_zero: AdaZeroCallable | None = None,
        post_sa: PostSACallable | None = None,
    ) -> "TransformerOpsConfig":
        """Build a config from individual functions or enums. Each *None* slot
        falls back to the standard PyTorch implementation."""
        attention_callable = attention.to_callable() if isinstance(attention, AttentionFunction) else attention
        masked_callable = (
            masked_attention.to_callable()
            if isinstance(masked_attention, MaskedAttentionFunction)
            else masked_attention
        )
        attention_ops = AttentionOps(
            attention_function=attention_callable,
            masked_attention_function=masked_callable,
            preattention_function=preattention if preattention is not None else PytorchPreAttention(),
            gated_attention_function=(gated_attention if gated_attention is not None else PytorchGatedAttention()),
        )
        return cls(
            attention_ops=attention_ops,
            ada_zero_function=ada_zero if ada_zero is not None else PytorchAdaZeroFunction(),
            post_sa_function=post_sa if post_sa is not None else PytorchPostSAFunction(),
        )


# Frozen, so safe to share as a default argument across callers that want the
# stock PyTorch ops without explicit construction.
DEFAULT_TRANSFORMER_OPS = TransformerOpsConfig()


def _env_on(name, default="1"):
    # Central env helper: LTX_ORIGINAL_PATH=1 routes every path back to plain torch ops.
    if os.environ.get("LTX_ORIGINAL_PATH", "0") == "1":
        return False
    return os.environ.get(name, default) != "0"


def _gate_add_fuse_enabled() -> bool:
    """Whether the text-cross-attention gate multiply is folded into the
    residual add (one addcmul instead of mul-then-add).
    ``LTX_FUSE_GATE_ADD=0`` keeps the two-op form."""
    return _env_on("LTX_FUSE_GATE_ADD")


def _torch_exact_rounding_enabled() -> bool:
    """Whether the fused sites reproduce torch's bf16 rounding exactly.

    With this on, each fused site rounds where the equivalent torch expression
    rounds, so its output matches torch bit for bit.  ``LTX_TORCH_EXACT_ROUNDING=0``
    keeps the cheaper fused rounding; ``LTX_ORIGINAL_PATH=1`` forces the
    unfused path."""
    return _env_on("LTX_TORCH_EXACT_ROUNDING", "1")


def _fuse_adaln_residual() -> bool:
    # Fused AdaLN-residual path (on by default): one MSA+MLP ada fetch, the
    # MLP residual as an addcmul, and the AV gate mask as an addcmul.  Its
    # numerics are correctly rounded (0 ulp against fp64).  Set
    # LTX_FUSE_ADALN_RESIDUAL=0, or LTX_ORIGINAL_PATH=1, for the unfused path.
    return _env_on("LTX_FUSE_ADALN_RESIDUAL")


def _use_fused_kernels(block) -> bool:
    # Whether this block can take the SYCL whole-chain fusion (ada 4->1,
    # cross 3->1, post_sa 3->1).  Requires the SYCL library to be loaded and
    # the block to use the stock ada/post_sa functions.
    return (
        _sycl_enabled()
        and _lib_ok()
        and _env_on("LTX_FUSE_ADALN")
        and _fuse_adaln_residual()
        and isinstance(block.ada_zero_function, PytorchAdaZeroFunction)
        and isinstance(block.post_sa_function, PytorchPostSAFunction)
    )


class BasicAVTransformerBlock(torch.nn.Module):
    def __init__(
        self,
        video: TransformerConfig | None = None,
        audio: TransformerConfig | None = None,
        rope_type: LTXRopeType = LTXRopeType.SPLIT,
        norm_eps: float = 1e-6,
        ops: TransformerOpsConfig | None = None,
    ):
        super().__init__()

        if ops is None:
            ops = TransformerOpsConfig()
        self.ada_zero_function = ops.ada_zero_function
        self.post_sa_function = ops.post_sa_function
        if video is not None:
            self.attn1 = Attention(
                query_dim=video.dim,
                heads=video.heads,
                dim_head=video.d_head,
                context_dim=None,
                rope_type=rope_type,
                norm_eps=norm_eps,
                ops=ops.attention_ops,
                apply_gated_attention=video.apply_gated_attention,
            )
            self.attn2 = Attention(
                query_dim=video.dim,
                context_dim=video.context_dim,
                heads=video.heads,
                dim_head=video.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                ops=ops.attention_ops,
                apply_gated_attention=video.apply_gated_attention,
            )
            self.ff = FeedForward(video.dim, dim_out=video.dim, bias=video.ff_bias)
            video_sst_size = adaln_embedding_coefficient(video.cross_attention_adaln)
            self.scale_shift_table = torch.nn.Parameter(torch.empty(video_sst_size, video.dim))

        if audio is not None:
            self.audio_attn1 = Attention(
                query_dim=audio.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                context_dim=None,
                rope_type=rope_type,
                norm_eps=norm_eps,
                ops=ops.attention_ops,
                apply_gated_attention=audio.apply_gated_attention,
            )
            self.audio_attn2 = Attention(
                query_dim=audio.dim,
                context_dim=audio.context_dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                ops=ops.attention_ops,
                apply_gated_attention=audio.apply_gated_attention,
            )
            self.audio_ff = FeedForward(audio.dim, dim_out=audio.dim, bias=audio.ff_bias)
            audio_sst_size = adaln_embedding_coefficient(audio.cross_attention_adaln)
            self.audio_scale_shift_table = torch.nn.Parameter(torch.empty(audio_sst_size, audio.dim))

        if audio is not None and video is not None:
            # Q: Video, K,V: Audio
            self.audio_to_video_attn = Attention(
                query_dim=video.dim,
                context_dim=audio.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                ops=ops.attention_ops,
                apply_gated_attention=video.apply_gated_attention,
            )

            # Q: Audio, K,V: Video
            self.video_to_audio_attn = Attention(
                query_dim=audio.dim,
                context_dim=video.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                ops=ops.attention_ops,
                apply_gated_attention=audio.apply_gated_attention,
            )

            self.scale_shift_table_a2v_ca_audio = torch.nn.Parameter(torch.empty(5, audio.dim))
            self.scale_shift_table_a2v_ca_video = torch.nn.Parameter(torch.empty(5, video.dim))

        self.cross_attention_adaln = (video is not None and video.cross_attention_adaln) or (
            audio is not None and audio.cross_attention_adaln
        )

        if self.cross_attention_adaln and video is not None:
            self.prompt_scale_shift_table = torch.nn.Parameter(torch.empty(2, video.dim))
        if self.cross_attention_adaln and audio is not None:
            self.audio_prompt_scale_shift_table = torch.nn.Parameter(torch.empty(2, audio.dim))

        self.norm_eps = norm_eps

    def get_ada_values(
        self, scale_shift_table: torch.Tensor, batch_size: int, timestep: torch.Tensor, indices: slice
    ) -> tuple[torch.Tensor, ...]:
        num_ada_params = scale_shift_table.shape[0]

        ada_values = (
            scale_shift_table[indices].unsqueeze(0).unsqueeze(0).to(device=timestep.device, dtype=timestep.dtype)
            + timestep.reshape(batch_size, timestep.shape[1], num_ada_params, -1)[:, :, indices, :]
        ).unbind(dim=2)
        return ada_values

    def get_av_ca_ada_values(
        self,
        scale_shift_table: torch.Tensor,
        batch_size: int,
        scale_shift_timestep: torch.Tensor,
        gate_timestep: torch.Tensor,
        scale_shift_indices: slice,
        num_scale_shift_values: int = 4,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scale_shift_ada_values = self.get_ada_values(
            scale_shift_table[:num_scale_shift_values, :], batch_size, scale_shift_timestep, scale_shift_indices
        )
        gate_ada_values = self.get_ada_values(
            scale_shift_table[num_scale_shift_values:, :], batch_size, gate_timestep, slice(None, None)
        )

        scale, shift = (t.squeeze(2) for t in scale_shift_ada_values)
        (gate,) = (t.squeeze(2) for t in gate_ada_values)

        return scale, shift, gate

    def get_ada_parts(
        self, scale_shift_table: torch.Tensor, batch_size: int, timestep: torch.Tensor, indices: slice
    ) -> list:
        # Un-materialized ada parts [(table_vec, timestep_part)] per param; no compute.
        num_ada_params = scale_shift_table.shape[0]
        table = scale_shift_table[indices].unsqueeze(0).unsqueeze(0).to(device=timestep.device, dtype=timestep.dtype)
        ts = timestep.reshape(batch_size, timestep.shape[1], num_ada_params, -1)[:, :, indices, :]
        return list(zip(table.unbind(dim=2), ts.unbind(dim=2)))

    def get_av_ca_ada_parts(
        self,
        scale_shift_table: torch.Tensor,
        batch_size: int,
        scale_shift_timestep: torch.Tensor,
        gate_timestep: torch.Tensor,
        scale_shift_indices: slice,
        num_scale_shift_values: int = 4,
    ) -> tuple:
        # parts version of get_av_ca_ada_values: (scale, shift, gate) parts.
        scale_shift_parts = self.get_ada_parts(
            scale_shift_table[:num_scale_shift_values, :], batch_size, scale_shift_timestep, scale_shift_indices
        )
        gate_parts = self.get_ada_parts(
            scale_shift_table[num_scale_shift_values:, :], batch_size, gate_timestep, slice(None, None)
        )
        scale = scale_shift_parts[0]
        shift = scale_shift_parts[1]
        (gate,) = gate_parts
        return scale, shift, gate

    def _cross_attention(
        self,
        x_normed: torch.Tensor,
        context: torch.Tensor,
        attn,
        scale_shift_table: torch.Tensor,
        prompt_scale_shift_table: torch.Tensor | None,
        timestep: torch.Tensor,
        prompt_timestep: torch.Tensor | None,
        context_mask: torch.Tensor | None,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # fused cross-attention ada (SYCL; fetch+1p+addcmul 3->1 x2).
        batch_size = x_normed.shape[0]
        _qp = self.get_ada_parts(scale_shift_table, batch_size, timestep, slice(6, 9))
        (qsh_tab, qsh_ts), (qsc_tab, qsc_ts), (g_tab, g_ts) = _qp
        if prompt_timestep is not None:
            _kp = self.get_ada_parts(prompt_scale_shift_table, batch_size, prompt_timestep, slice(0, 2))
            (ksh_tab, ksh_ts), (ksc_tab, ksc_ts) = _kp
        else:
            _kv0 = prompt_scale_shift_table[None, None].to(device=x_normed.device, dtype=x_normed.dtype)
            ksh_tab, ksc_tab = _kv0.unbind(dim=2)
            ksh_ts = ksc_ts = None
        return cross_attention_adaln(
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
            residual=residual,
        )

    def _apply_text_cross_attention(
        self,
        x_normed: torch.Tensor,
        context: torch.Tensor,
        attn: AttentionCallable,
        scale_shift_table: torch.Tensor,
        prompt_scale_shift_table: torch.Tensor | None,
        timestep: torch.Tensor,
        prompt_timestep: torch.Tensor | None,
        context_mask: torch.Tensor | None,
        cross_attention_adaln: bool = False,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply text cross-attention, with optional AdaLN modulation.
        ``x_normed`` is the RMS-normalized self-attention output produced by
        ``post_sa_function`` -- this method does not normalize again.

        ``residual``: when given, the residual add is folded into the gate
        multiply (one addcmul) on the SYCL path; the other paths add it here.
        """
        if cross_attention_adaln:
            if _use_fused_kernels(self):
                return self._cross_attention(
                    x_normed,
                    context,
                    attn,
                    scale_shift_table,
                    prompt_scale_shift_table,
                    timestep,
                    prompt_timestep,
                    context_mask,
                    residual=residual,
                )
            shift_q, scale_q, gate = self.get_ada_values(scale_shift_table, x_normed.shape[0], timestep, slice(6, 9))
            _out = apply_cross_attention_adaln(
                x_normed,
                context,
                attn,
                shift_q,
                scale_q,
                gate,
                prompt_scale_shift_table,
                prompt_timestep,
                context_mask,
            )
            return _out if residual is None else residual + _out
        _out = attn(x_normed, context=context, mask=context_mask)
        return _out if residual is None else residual + _out

    def forward(  # noqa: PLR0915
        self,
        video: TransformerArgs | None,
        audio: TransformerArgs | None,
    ) -> tuple[TransformerArgs | None, TransformerArgs | None]:
        if video is None and audio is None:
            raise ValueError("At least one of video or audio must be provided")

        vx = video.x if video is not None else None
        ax = audio.x if audio is not None else None

        run_vx = video is not None and video.enabled and vx.numel() > 0
        run_ax = audio is not None and audio.enabled and ax.numel() > 0

        run_a2v = run_vx and (audio is not None and ax.numel() > 0)
        run_v2a = run_ax and (video is not None and vx.numel() > 0)

        if run_vx:
            _vx = _use_fused_kernels(self) and self.scale_shift_table.shape[0] >= 6
            if _vx:
                _v06p = self.get_ada_parts(
                    self.scale_shift_table, vx.shape[0], video.timesteps, slice(0, 6)
                )
                # pairs 0:shift_msa 1:scale_msa 2:gate_msa 3:shift_mlp 4:scale_mlp 5:gate_mlp
                norm_vx = ada_zero(vx, self.norm_eps, _v06p[1][0], _v06p[1][1], _v06p[0][0], _v06p[0][1])
                _v_mlp_ready = True
            elif _fuse_adaln_residual() and self.scale_shift_table.shape[0] >= 6:
                _v06 = self.get_ada_values(
                    self.scale_shift_table, vx.shape[0], video.timesteps, slice(0, 6)
                )
                vshift_msa, vscale_msa, vgate_msa = _v06[:3]
                vshift_mlp, vscale_mlp, vgate_mlp = _v06[3:]
                del _v06
                _v_mlp_ready = True
            else:
                vshift_msa, vscale_msa, vgate_msa = self.get_ada_values(
                    self.scale_shift_table, vx.shape[0], video.timesteps, slice(0, 3)
                )
                _v_mlp_ready = False
            if not _vx:
                norm_vx = self.ada_zero_function(vx, self.norm_eps, vscale_msa, vshift_msa)
                del vshift_msa, vscale_msa

            vx_msa_out = self.attn1(
                norm_vx,
                pe=video.positional_embeddings,
                mask=video.self_attention_mask,
                perturbation_mask=video.self_attn_perturbation_mask,
                all_perturbed=video.self_attn_all_perturbed,
            )
            if _vx:
                vx, vx_normed = post_sa(vx, vx_msa_out, _v06p[2][0], _v06p[2][1], None, self.norm_eps)
                del norm_vx, vx_msa_out
            else:
                vx, vx_normed = self.post_sa_function(vx, vx_msa_out, None, self.norm_eps, vgate_msa)
                del vgate_msa, norm_vx, vx_msa_out
            residual_fused = vx if _gate_add_fuse_enabled() else None
            _tca = self._apply_text_cross_attention(
                vx_normed,
                video.context,
                self.attn2,
                self.scale_shift_table,
                getattr(self, "prompt_scale_shift_table", None),
                video.timesteps,
                video.prompt_timestep,
                video.context_mask,
                cross_attention_adaln=self.cross_attention_adaln,
                residual=residual_fused,
            )
            vx = _tca if residual_fused is not None else vx + _tca
            del vx_normed

        if run_ax:
            _ax = _use_fused_kernels(self) and self.audio_scale_shift_table.shape[0] >= 6
            if _ax:
                _a06p = self.get_ada_parts(
                    self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(0, 6)
                )
                norm_ax = ada_zero(ax, self.norm_eps, _a06p[1][0], _a06p[1][1], _a06p[0][0], _a06p[0][1])
                _a_mlp_ready = True
            elif _fuse_adaln_residual() and self.audio_scale_shift_table.shape[0] >= 6:
                _a06 = self.get_ada_values(
                    self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(0, 6)
                )
                ashift_msa, ascale_msa, agate_msa = _a06[:3]
                ashift_mlp, ascale_mlp, agate_mlp = _a06[3:]
                del _a06
                _a_mlp_ready = True
            else:
                ashift_msa, ascale_msa, agate_msa = self.get_ada_values(
                    self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(0, 3)
                )
                _a_mlp_ready = False

            if not _ax:
                norm_ax = self.ada_zero_function(ax, self.norm_eps, ascale_msa, ashift_msa)
                del ashift_msa, ascale_msa
            ax_msa_out = self.audio_attn1(
                norm_ax,
                pe=audio.positional_embeddings,
                mask=audio.self_attention_mask,
                perturbation_mask=audio.self_attn_perturbation_mask,
                all_perturbed=audio.self_attn_all_perturbed,
            )
            if _ax:
                ax, ax_normed = post_sa(ax, ax_msa_out, _a06p[2][0], _a06p[2][1], None, self.norm_eps)
                del norm_ax, ax_msa_out
            else:
                ax, ax_normed = self.post_sa_function(ax, ax_msa_out, None, self.norm_eps, agate_msa)
                del agate_msa, norm_ax, ax_msa_out
            residual_fused = ax if _gate_add_fuse_enabled() else None
            _tca = self._apply_text_cross_attention(
                ax_normed,
                audio.context,
                self.audio_attn2,
                self.audio_scale_shift_table,
                getattr(self, "audio_prompt_scale_shift_table", None),
                audio.timesteps,
                audio.prompt_timestep,
                audio.context_mask,
                cross_attention_adaln=self.cross_attention_adaln,
                residual=residual_fused,
            )
            ax = _tca if residual_fused is not None else ax + _tca
            del ax_normed

        # Audio - Video cross attention.
        if run_a2v or run_v2a:
            # Snapshot vx/ax before A2V mutates vx; V2A's video keys/values must
            # use the pre-A2V state so direction order doesn't bias the result.
            vx_pre_av = vx
            ax_pre_av = ax
            _av = _use_fused_kernels(self)
            _av_gate_inline = _av and _torch_exact_rounding_enabled() and gate_inline_av_enabled()
            if run_a2v and not video.cross_attn_skip_all:
                if _av:
                    _sc_v, _sh_v, _g_v = self.get_av_ca_ada_parts(
                        self.scale_shift_table_a2v_ca_video,
                        vx.shape[0],
                        video.cross_scale_shift_timestep,
                        video.cross_gate_timestep,
                        slice(0, 2),
                    )
                    a2v_vx_scaled = ada_zero(vx_pre_av, self.norm_eps, _sc_v[0], _sc_v[1], _sh_v[0], _sh_v[1])
                    if _av_gate_inline:
                        a2v_gate_tab, a2v_gate_ts = _g_v
                    else:
                        gate_out_a2v = _g_v[0] + _g_v[1]
                    del _sc_v, _sh_v, _g_v
                    _sc_a, _sh_a, _g_a = self.get_av_ca_ada_parts(
                        self.scale_shift_table_a2v_ca_audio,
                        ax.shape[0],
                        audio.cross_scale_shift_timestep,
                        audio.cross_gate_timestep,
                        slice(0, 2),
                    )
                    a2v_ax_scaled = ada_zero(ax_pre_av, self.norm_eps, _sc_a[0], _sc_a[1], _sh_a[0], _sh_a[1])
                    del _sc_a, _sh_a, _g_a
                else:
                    scale_ca_video_a2v, shift_ca_video_a2v, gate_out_a2v = self.get_av_ca_ada_values(
                        self.scale_shift_table_a2v_ca_video,
                        vx.shape[0],
                        video.cross_scale_shift_timestep,
                        video.cross_gate_timestep,
                        slice(0, 2),
                    )
                    a2v_vx_scaled = self.ada_zero_function(vx_pre_av, self.norm_eps, scale_ca_video_a2v, shift_ca_video_a2v)
                    del scale_ca_video_a2v, shift_ca_video_a2v

                    scale_ca_audio_a2v, shift_ca_audio_a2v, _ = self.get_av_ca_ada_values(
                        self.scale_shift_table_a2v_ca_audio,
                        ax.shape[0],
                        audio.cross_scale_shift_timestep,
                        audio.cross_gate_timestep,
                        slice(0, 2),
                    )
                    a2v_ax_scaled = self.ada_zero_function(ax_pre_av, self.norm_eps, scale_ca_audio_a2v, shift_ca_audio_a2v)
                    del scale_ca_audio_a2v, shift_ca_audio_a2v
                if _fuse_adaln_residual() and not _torch_exact_rounding_enabled():
                    a2v_attn_out = self.audio_to_video_attn(
                        a2v_vx_scaled,
                        context=a2v_ax_scaled,
                        pe=video.cross_positional_embeddings,
                        k_pe=audio.cross_positional_embeddings,
                    )
                    if video.cross_attn_perturbation_mask is None:
                        fused_addcmul = elementwise_addcmul(vx, a2v_attn_out, gate_out_a2v)
                        vx = (
                            torch.addcmul(vx, a2v_attn_out, gate_out_a2v)
                            if fused_addcmul is None
                            else fused_addcmul
                        )
                    else:
                        vx = torch.addcmul(
                            vx, a2v_attn_out, gate_out_a2v * video.cross_attn_perturbation_mask
                        )
                    del a2v_attn_out
                elif _av_gate_inline:
                    _a2v = self.audio_to_video_attn(
                        a2v_vx_scaled,
                        context=a2v_ax_scaled,
                        pe=video.cross_positional_embeddings,
                        k_pe=audio.cross_positional_embeddings,
                    )
                    vx = elementwise_gate_mask_add_inline(
                        vx, _a2v, a2v_gate_tab, a2v_gate_ts, video.cross_attn_perturbation_mask
                    )
                    del _a2v, a2v_gate_tab, a2v_gate_ts
                else:
                    _a2v = self.audio_to_video_attn(
                        a2v_vx_scaled,
                        context=a2v_ax_scaled,
                        pe=video.cross_positional_embeddings,
                        k_pe=audio.cross_positional_embeddings,
                    )
                    vx = vx + elementwise_mul(_a2v, gate_out_a2v) * video.cross_attn_perturbation_mask
                if not _av_gate_inline:
                    del gate_out_a2v
                del a2v_vx_scaled, a2v_ax_scaled

            if run_v2a and not audio.cross_attn_skip_all:
                if _av:
                    _sc_va, _sh_va, _g_va = self.get_av_ca_ada_parts(
                        self.scale_shift_table_a2v_ca_audio,
                        ax.shape[0],
                        audio.cross_scale_shift_timestep,
                        audio.cross_gate_timestep,
                        slice(2, 4),
                    )
                    v2a_ax_scaled = ada_zero(ax_pre_av, self.norm_eps, _sc_va[0], _sc_va[1], _sh_va[0], _sh_va[1])
                    if _av_gate_inline:
                        v2a_gate_tab, v2a_gate_ts = _g_va
                    else:
                        gate_out_v2a = _g_va[0] + _g_va[1]
                    del _sc_va, _sh_va, _g_va
                    _sc_vv, _sh_vv, _g_vv = self.get_av_ca_ada_parts(
                        self.scale_shift_table_a2v_ca_video,
                        vx.shape[0],
                        video.cross_scale_shift_timestep,
                        video.cross_gate_timestep,
                        slice(2, 4),
                    )
                    v2a_vx_scaled = ada_zero(vx_pre_av, self.norm_eps, _sc_vv[0], _sc_vv[1], _sh_vv[0], _sh_vv[1])
                    del _sc_vv, _sh_vv, _g_vv
                else:
                    scale_ca_audio_v2a, shift_ca_audio_v2a, gate_out_v2a = self.get_av_ca_ada_values(
                        self.scale_shift_table_a2v_ca_audio,
                        ax.shape[0],
                        audio.cross_scale_shift_timestep,
                        audio.cross_gate_timestep,
                        slice(2, 4),
                    )
                    v2a_ax_scaled = self.ada_zero_function(ax_pre_av, self.norm_eps, scale_ca_audio_v2a, shift_ca_audio_v2a)
                    del scale_ca_audio_v2a, shift_ca_audio_v2a
                    scale_ca_video_v2a, shift_ca_video_v2a, _ = self.get_av_ca_ada_values(
                        self.scale_shift_table_a2v_ca_video,
                        vx.shape[0],
                        video.cross_scale_shift_timestep,
                        video.cross_gate_timestep,
                        slice(2, 4),
                    )
                    v2a_vx_scaled = self.ada_zero_function(vx_pre_av, self.norm_eps, scale_ca_video_v2a, shift_ca_video_v2a)
                    del scale_ca_video_v2a, shift_ca_video_v2a
                if _fuse_adaln_residual() and not _torch_exact_rounding_enabled():
                    v2a_attn_out = self.video_to_audio_attn(
                        v2a_ax_scaled,
                        context=v2a_vx_scaled,
                        pe=audio.cross_positional_embeddings,
                        k_pe=video.cross_positional_embeddings,
                    )
                    if audio.cross_attn_perturbation_mask is None:
                        fused_addcmul = elementwise_addcmul(ax, v2a_attn_out, gate_out_v2a)
                        ax = (
                            torch.addcmul(ax, v2a_attn_out, gate_out_v2a)
                            if fused_addcmul is None
                            else fused_addcmul
                        )
                    else:
                        ax = torch.addcmul(
                            ax, v2a_attn_out, gate_out_v2a * audio.cross_attn_perturbation_mask
                        )
                    del v2a_attn_out
                elif _av_gate_inline:
                    _v2a = self.video_to_audio_attn(
                        v2a_ax_scaled,
                        context=v2a_vx_scaled,
                        pe=audio.cross_positional_embeddings,
                        k_pe=video.cross_positional_embeddings,
                    )
                    ax = elementwise_gate_mask_add_inline(
                        ax, _v2a, v2a_gate_tab, v2a_gate_ts, audio.cross_attn_perturbation_mask
                    )
                    del _v2a, v2a_gate_tab, v2a_gate_ts
                else:
                    _v2a = self.video_to_audio_attn(
                        v2a_ax_scaled,
                        context=v2a_vx_scaled,
                        pe=audio.cross_positional_embeddings,
                        k_pe=video.cross_positional_embeddings,
                    )
                    ax = ax + elementwise_mul(_v2a, gate_out_v2a) * audio.cross_attn_perturbation_mask
                if not _av_gate_inline:
                    del gate_out_v2a
                del v2a_vx_scaled, v2a_ax_scaled
            del vx_pre_av, ax_pre_av

        if run_vx:
            _v_gate_inline = _vx and _torch_exact_rounding_enabled() and gate_inline_enabled()
            if _vx:
                vx_scaled = ada_zero(vx, self.norm_eps, _v06p[4][0], _v06p[4][1], _v06p[3][0], _v06p[3][1])
                if not _v_gate_inline:
                    vgate_mlp = _v06p[5][0] + _v06p[5][1]
            else:
                if not _v_mlp_ready:
                    vshift_mlp, vscale_mlp, vgate_mlp = self.get_ada_values(
                        self.scale_shift_table, vx.shape[0], video.timesteps, slice(3, 6)
                    )
                vx_scaled = self.ada_zero_function(vx, self.norm_eps, vscale_mlp, vshift_mlp)
            if _fuse_adaln_residual() and not _torch_exact_rounding_enabled():
                fused_addcmul = elementwise_addcmul(vx, self.ff(vx_scaled), vgate_mlp)
                vx = torch.addcmul(vx, self.ff(vx_scaled), vgate_mlp) if fused_addcmul is None else fused_addcmul
            elif _v_gate_inline:
                vx = elementwise_gate_add_inline(vx, self.ff(vx_scaled), _v06p[5][0], _v06p[5][1])
            elif _vx and exact_residual_fuse_enabled():
                vx = elementwise_gate_add_exact(vx, self.ff(vx_scaled), vgate_mlp)
            else:
                vx = vx + self.ff(vx_scaled) * vgate_mlp

            if _v_gate_inline:
                del _v06p, vx_scaled
            elif _vx:
                del _v06p, vgate_mlp, vx_scaled
            else:
                del vshift_mlp, vscale_mlp, vgate_mlp, vx_scaled

        if run_ax:
            _a_gate_inline = _ax and _torch_exact_rounding_enabled() and gate_inline_enabled()
            if _ax:
                ax_scaled = ada_zero(ax, self.norm_eps, _a06p[4][0], _a06p[4][1], _a06p[3][0], _a06p[3][1])
                if not _a_gate_inline:
                    agate_mlp = _a06p[5][0] + _a06p[5][1]
            else:
                if not _a_mlp_ready:
                    ashift_mlp, ascale_mlp, agate_mlp = self.get_ada_values(
                        self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(3, 6)
                    )
                ax_scaled = self.ada_zero_function(ax, self.norm_eps, ascale_mlp, ashift_mlp)
            if _fuse_adaln_residual() and not _torch_exact_rounding_enabled():
                fused_addcmul = elementwise_addcmul(ax, self.audio_ff(ax_scaled), agate_mlp)
                ax = torch.addcmul(ax, self.audio_ff(ax_scaled), agate_mlp) if fused_addcmul is None else fused_addcmul
            elif _a_gate_inline:
                ax = elementwise_gate_add_inline(ax, self.audio_ff(ax_scaled), _a06p[5][0], _a06p[5][1])
            elif _ax and exact_residual_fuse_enabled():
                ax = elementwise_gate_add_exact(ax, self.audio_ff(ax_scaled), agate_mlp)
            else:
                ax = ax + self.audio_ff(ax_scaled) * agate_mlp

            if _a_gate_inline:
                del _a06p, ax_scaled
            elif _ax:
                del _a06p, agate_mlp, ax_scaled
            else:
                del ashift_mlp, ascale_mlp, agate_mlp, ax_scaled

        return replace(video, x=vx) if video is not None else None, replace(audio, x=ax) if audio is not None else None


def apply_cross_attention_adaln(
    x_normed: torch.Tensor,
    context: torch.Tensor,
    attn: AttentionCallable,
    q_shift: torch.Tensor,
    q_scale: torch.Tensor,
    q_gate: torch.Tensor,
    prompt_scale_shift_table: torch.Tensor,
    prompt_timestep: torch.Tensor | None,
    context_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply query/key AdaLN modulation then cross-attention.
    ``x_normed`` is already RMS-normalized by ``post_sa_function``; this only
    applies the affine (scale/shift) modulation, so the normalization is not
    repeated here.
    """
    batch_size = x_normed.shape[0]
    # K/V modulation. With the prompt-side AdaLN MLP disabled (use_prompt_adaln_single=False),
    # prompt_timestep is None and only the static per-block table applies, so K/V are
    # timestep-independent and cacheable across denoising/AR steps. Otherwise the timestep-
    # conditioned MLP output is added on top. 
    kv_modulation = prompt_scale_shift_table[None, None].to(device=x_normed.device, dtype=x_normed.dtype)
    if prompt_timestep is not None:
        kv_modulation = kv_modulation + prompt_timestep.reshape(batch_size, prompt_timestep.shape[1], 2, -1)
    shift_kv, scale_kv = kv_modulation.unbind(dim=2)
    # AdaLN fusion: mul+add -> single addcmul kernel, twice.
    # LTX_FUSE_ADALN=0, LTX_ORIGINAL_PATH=1 and torch-exact rounding use the two-op formula.
    if not _env_on("LTX_FUSE_ADALN") or _torch_exact_rounding_enabled():
        attn_input = x_normed * (1 + q_scale) + q_shift
        encoder_hidden_states = context * (1 + scale_kv) + shift_kv
    else:
        attn_input = torch.addcmul(q_shift, x_normed, 1 + q_scale)
        encoder_hidden_states = torch.addcmul(shift_kv, context, 1 + scale_kv)
    return attn(attn_input, context=encoder_hidden_states, mask=context_mask) * q_gate
