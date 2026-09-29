"""R27 main-DiT MLP down-projection residual and gate epilogue."""

from __future__ import annotations

import atexit
import os
from pathlib import Path
from typing import Any

import torch


_OP: Any = None
_CALLS = 0


def r27_mlp_epilogue_on() -> bool:
    """Return whether the R27 fused MLP epilogue is enabled."""
    if os.environ.get("LTX_ALL_ORIG", "0") == "1":
        return False
    value = os.environ.get("LTX_R27_MLP_EPI", "0")
    if value not in {"0", "1"}:
        raise ValueError("LTX_R27_MLP_EPI must be 0 or 1")
    return value == "1"


def _library() -> Any:
    """Load and return the R27 Torch operator namespace."""
    global _OP
    if _OP is not None:
        return _OP
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        raise RuntimeError("R27 requires an available XPU device")
    package_root = Path(__file__).resolve().parents[5]
    library = package_root / "ltx-kernels" / "csrc" / "sycl_r8" / "libr27_mlp.so"
    if not library.is_file():
        raise FileNotFoundError(f"R27 library not found: {library}")
    torch.ops.load_library(str(library))
    _OP = torch.ops.libr27.mlp_epilogue
    return _OP


def _print_stats() -> None:
    if _CALLS:
        print(f"R27_MLP_EPI_STATS calls={_CALLS}", flush=True)


atexit.register(_print_stats)


def r27_mlp_residual_gate(
    feed_forward: torch.nn.Module,
    x: torch.Tensor,
    residual: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    """Run the MLP and return ``residual + down_projection * gate`` in one primitive."""
    global _CALLS
    if not r27_mlp_epilogue_on():
        raise RuntimeError("R27 epilogue called while LTX_R27_MLP_EPI=0")
    if os.environ.get("LTX_R18_SYCLEFF", "0") != "0":
        raise RuntimeError("R27 requires LTX_R18_SYCLEFF=0")
    modules = getattr(feed_forward, "net", None)
    if not isinstance(modules, torch.nn.Sequential) or len(modules) != 3:
        raise TypeError("R27 expects FeedForward.net with three modules")
    if not isinstance(modules[1], torch.nn.Identity):
        raise TypeError("R27 expects Identity after the GELU projection")
    down = modules[2]
    if not isinstance(down, torch.nn.Linear):
        raise TypeError("R27 expects a linear down projection")
    if not (x.is_xpu and residual.is_xpu and gate.is_xpu):
        raise ValueError("R27 requires XPU tensors")
    if x.dtype != torch.bfloat16 or residual.dtype != torch.bfloat16 or gate.dtype != torch.bfloat16:
        raise ValueError("R27 requires bf16 tensors")

    hidden = modules[0](x)
    hidden = modules[1](hidden)
    if not hidden.is_contiguous():
        raise ValueError("R27 requires contiguous MLP hidden output")
    output_dim = int(down.weight.shape[0])
    if tuple(residual.shape) != (*x.shape[:-1], output_dim):
        raise ValueError("R27 residual shape does not match the down projection")
    if gate.numel() != output_dim or gate.shape[-1] != output_dim:
        raise ValueError("R27 gate must broadcast one vector over output channels")

    if down.weight.dtype == torch.float8_e4m3fn:
        from ltx_core.quantization.fp8_cast import _upcast_and_round

        weight = _upcast_and_round(
            down.weight,
            torch.bfloat16,
            down._with_stochastic_rounding,
            down._seed,
        )
        bias = (
            _upcast_and_round(
                down.bias,
                torch.bfloat16,
                down._with_stochastic_rounding,
                down._seed,
            )
            if down.bias is not None
            else None
        )
    elif down.weight.dtype == torch.bfloat16 and (
        down.bias is None or down.bias.dtype == torch.bfloat16
    ):
        weight = down.weight
        bias = down.bias
    else:
        raise TypeError("R27 supports FP8 e4m3fn or bf16 down projections")

    output = _library()(hidden, weight, bias, residual, gate)
    _CALLS += 1
    return output
