"""Gate1 numeric deep-dive: sign-flip fraction, magnitudes, fp64-truth comparison."""
import os
import sys

import torch

sys.path.insert(0, "/home/simon/ltx2.3-demo-main/LTX-2/packages/ltx-core/src")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from ltx_core.model.transformer import r8_sycl as R8  # noqa: E402
from ltx_core.utils import rms_norm as _rms  # noqa: E402

DEV = "xpu:0"


def sync():
    torch.xpu.synchronize()


def flip_stats(got, ref, tag):
    gf = got.to(torch.float32)
    rf = ref.to(torch.float32)
    sflip = (gf.sign() != rf.sign()) & (gf.abs() > 0) & (rf.abs() > 0)
    n = gf.numel()
    nf = int(sflip.sum().item())
    mag = float(rf[sflip].abs().max().item()) if nf else 0.0
    # fp64 truth comparison
    x64 = None
    print(f"{tag}: elems={n} signflip={nf} ({nf / n * 100:.4f}%) maxflip_mag={mag:.3e}")
    return nf


print("lib_ok:", R8.lib_ok())
torch.manual_seed(0)
rows, C = 4096, 4096
x = torch.randn(rows, C, dtype=torch.bfloat16, device=DEV)
tbs = torch.randn(C, dtype=torch.bfloat16, device=DEV)
tss = torch.randn(rows, C, dtype=torch.bfloat16, device=DEV)
tbh = torch.randn(C, dtype=torch.bfloat16, device=DEV)
tsh = torch.randn(rows, C, dtype=torch.bfloat16, device=DEV)
eps = 1e-6
with torch.inference_mode():
    ref = torch.addcmul(tbh + tsh, _rms(x, eps=eps), 1 + (tbs + tss))
    got = R8.r8_ada_zero(x, eps, tbs, tss, tbh, tsh)
    sync()
    # fp64 truth of the fused formula
    xf = x.to(torch.float64)
    nrm = xf / (xf.pow(2).mean(-1, keepdim=True).add(eps).sqrt())
    sc = tbs.to(torch.float64) + tss.to(torch.float64)
    sh = tbh.to(torch.float64) + tsh.to(torch.float64)
    truth = sh + nrm * (1 + sc)
    def rel(a, b):
        a64 = a.to(torch.float64)
        b64 = b.to(torch.float64) if not isinstance(b, torch.Tensor) or b.dtype != torch.float64 else b
        return float(((a64 - b64).pow(2).sum() / b64.pow(2).sum()).sqrt().item())
    print(f"K1: eager-vs-truth relL2={rel(ref, truth):.3e} sycl-vs-truth relL2={rel(got, truth):.3e}")
flip_stats(got, ref, "K1")

# K2
y = torch.randn(rows, C, dtype=torch.bfloat16, device=DEV)
tbg = torch.randn(C, dtype=torch.bfloat16, device=DEV)
tsg = torch.randn(rows, C, dtype=torch.bfloat16, device=DEV)
with torch.inference_mode():
    gate = tbg + tsg
    fma_ref = torch.addcmul(x, y, gate)
    out_ref = _rms(fma_ref, None, eps=eps)
    fma_got, out_got = R8.r8_post_sa(x, y, tbg, tsg, None, eps)
    sync()
flip_stats(fma_got, fma_ref, "K2-fma")
flip_stats(out_got, out_ref, "K2-out")

# K3
from ltx_core.model.transformer.rope import _apply_split_rotary_emb_negfold as eager_rope  # noqa: E402
B, H, T, D = 1, 32, 4096, 64
Dh = D // 2
xb = torch.randn(B, T, H, D, dtype=torch.bfloat16, device=DEV)
xx = xb.transpose(1, 2)
cb = torch.randn(B, T, H, Dh, dtype=torch.bfloat16, device=DEV)
cos = cb.transpose(1, 2)
sb = torch.randn(B, T, H, Dh, dtype=torch.bfloat16, device=DEV)
sin = sb.transpose(1, 2)
with torch.inference_mode():
    ref = eager_rope(xx, cos, sin)
    got = R8.r8_split_rope(xx, cos, sin)
    sync()
flip_stats(got, ref, "K3")
print("NUMERIC_DONE")
