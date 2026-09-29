import torch

from ltx_core.model.transformer import r8_sycl as _R8


class GELUApprox(torch.nn.Module):
    def __init__(self, dim_in: int, dim_out: int, bias: bool = True) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(dim_in, dim_out, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        # R10D F3: self-written SYCL tanh-approx GELU (bitwise vs torch's
        # GeluTanhFunctor); returns None on any unsupported case -> torch path.
        g = _R8.r10d_gelu(h)
        if g is not None:
            return g
        return torch.nn.functional.gelu(h, approximate="tanh")
