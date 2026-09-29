"""K3 gate probe: synthetic 4D bf16 XPU tensors through the real gate + callee."""
import os
import sys

import torch

sys.path.insert(0, "/home/simon/ltx2.3-demo-main/LTX-2/packages/ltx-core/src")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from ltx_core.model.transformer import rope as R  # noqa: E402
from ltx_core.model.transformer import r8_sycl as R8  # noqa: E402
from ltx_core.model.transformer.rope import _apply_split_rotary_emb_negfold as eager_rope  # noqa: E402

print("r8_on:", R8.r8_on(), "lib_ok:", R8.lib_ok())
for shape in [(1, 32, 16384, 64), (1, 32, 4096, 128), (2, 8, 1024, 64)]:
    B, H, T, D = shape
    Dh = D // 2
    x = torch.randn(B, H, T, D, dtype=torch.bfloat16, device="xpu:0")
    cos = torch.randn(B, H, T, Dh, dtype=torch.bfloat16, device="xpu:0")
    sin = torch.randn(B, H, T, Dh, dtype=torch.bfloat16, device="xpu:0")
    print(shape, "gate=", R._r8_rope_ok(x, cos, sin))
    try:
        y = R.apply_split_rotary_emb(x, cos, sin)
        torch.xpu.synchronize()
        print("  fired, out shape:", tuple(y.shape), "contig:", y.is_contiguous())
    except Exception as e:
        print("  FAIL:", str(e)[:200])

print("== 3D needs_reshape path (DiT hot path) ==")


def rel_l2(a, b):
    a64 = a.to(torch.float64)
    b64 = b.to(torch.float64)
    return float(((a64 - b64).pow(2).sum() / b64.pow(2).sum().clamp_min(1e-30)).sqrt().item())


import time  # noqa: E402


def bench(fn, iters=20, warm=5):
    for _ in range(warm):
        fn()
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


for B, T, H, D in [(1, 16384, 32, 64), (1, 4096, 32, 64), (2, 1024, 8, 128)]:
    Dh = D // 2
    x = torch.randn(B, T, H * D, dtype=torch.bfloat16, device="xpu:0")
    cos = torch.randn(B, H, T, Dh, dtype=torch.bfloat16, device="xpu:0")
    sin = torch.randn(B, H, T, Dh, dtype=torch.bfloat16, device="xpu:0")
    with torch.inference_mode():
        ref = eager_rope(x, cos, sin)
        got = R.apply_split_rotary_emb(x, cos, sin)
        torch.xpu.synchronize()
    print(f"B={B} T={T} H={H} D={D}: gate={R._r8_rope_ok(x, cos, sin)} relL2={rel_l2(got, ref):.2e}")
    te = bench(lambda: eager_rope(x, cos, sin))
    tg = bench(lambda: R.apply_split_rotary_emb(x, cos, sin))
    print(f"  eager={te:.3f}ms sycl={tg:.3f}ms speedup={te / tg:.2f}x")
print("PROBE_DONE")
