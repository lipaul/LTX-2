import torch

from ltx_core.model.transformer import sycl_kernels as _K


class GELUApprox(torch.nn.Module):
    def __init__(self, dim_in: int, dim_out: int, bias: bool = True) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(dim_in, dim_out, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        # SYCL tanh-approximation GELU; returns None on any unsupported case.
        g = _K.gelu_tanh_fast(h)
        if g is not None:
            return g
        return torch.nn.functional.gelu(h, approximate="tanh")
