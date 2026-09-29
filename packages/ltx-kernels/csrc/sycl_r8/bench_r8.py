"""R8 standalone microbench: SYCL k1/k2/k3 vs eager bf16 (Gate1 numeric + speedup).

Run on server under flock (GPU). No repo imports beyond ltx_core utils.
"""
import os
import sys
import time

import torch

sys.path.insert(0, "/home/simon/ltx2.3-demo-main/LTX-2/packages/ltx-core/src")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from ltx_core.model.transformer import r8_sycl as R8  # noqa: E402
from ltx_core.utils import rms_norm as _rms  # noqa: E402

DEV = "xpu:0"


def sync():
    torch.xpu.synchronize()


def max_ulp(a: torch.Tensor, b: torch.Tensor) -> int:
    ai = a.to(torch.float32).view(torch.int32)
    bi = b.to(torch.float32).view(torch.int32)
    return int((ai - bi).abs().max().item())


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a64 = a.to(torch.float64)
    b64 = b.to(torch.float64)
    return float(((a64 - b64).pow(2).sum() / b64.pow(2).sum().clamp_min(1e-30)).sqrt().item())


def bench(fn, iters=20, warm=5):
    for _ in range(warm):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync()
    return (time.perf_counter() - t0) / iters * 1e3


def rand(*shape):
    return torch.randn(shape, dtype=torch.bfloat16, device=DEV)


print("lib_ok:", R8.lib_ok())
assert R8.lib_ok(), "SYCL .so not loaded"

print("== K1 ada (rms+fetch+1p+addcmul 4->1) ==")
for rows, C, ts_rows in [(16384, 4096, 16384), (4096, 4096, 4096), (16384, 4096, 1), (4096, 2048, 4096)]:
    x = rand(rows, C)
    tbs = rand(C)
    tss = rand(ts_rows, C)
    tbh = rand(C)
    tsh = rand(ts_rows, C)
    eps = 1e-6
    with torch.inference_mode():
        ref = torch.addcmul(tbh + tsh, _rms(x, eps=eps), 1 + (tbs + tss))
        got = R8.r8_ada_zero(x, eps, tbs, tss, tbh, tsh)
        sync()
    print(f"rows={rows} C={C} tsrows={ts_rows}: relL2={rel_l2(got, ref):.2e} maxULP={max_ulp(got, ref)}")
    te = bench(lambda: torch.addcmul(tbh + tsh, _rms(x, eps=eps), 1 + (tbs + tss)))
    tg = bench(lambda: R8.r8_ada_zero(x, eps, tbs, tss, tbh, tsh))
    print(f"  eager={te:.3f}ms sycl={tg:.3f}ms speedup={te / tg:.2f}x")

print("== K1 affine / cross (fetch+1p+addcmul 3->1) ==")
for rows, C, ts_rows in [(16384, 4096, 16384), (4096, 4096, 1)]:
    x = rand(rows, C)
    tbs, tss, tbh, tsh = rand(C), rand(ts_rows, C), rand(C), rand(ts_rows, C)
    with torch.inference_mode():
        ref = torch.addcmul(tbh + tsh, x, 1 + (tbs + tss))
        got = R8.r8_affine_addcmul(x, tbs, tss, tbh, tsh)
        sync()
    print(f"rows={rows} C={C} tsrows={ts_rows}: relL2={rel_l2(got, ref):.2e} maxULP={max_ulp(got, ref)}")
    te = bench(lambda: torch.addcmul(tbh + tsh, x, 1 + (tbs + tss)))
    tg = bench(lambda: R8.r8_affine_addcmul(x, tbs, tss, tbh, tsh))
    print(f"  eager={te:.3f}ms sycl={tg:.3f}ms speedup={te / tg:.2f}x")

print("== K2 post_sa (gate+addcmul+rms 3->1) ==")
for rows, C, ts_rows in [(16384, 4096, 16384), (4096, 4096, 1), (4096, 2048, 4096)]:
    x, y = rand(rows, C), rand(rows, C)
    tbg, tsg = rand(C), rand(ts_rows, C)
    eps = 1e-6
    with torch.inference_mode():
        gate = tbg + tsg
        fma_ref = torch.addcmul(x, y, gate)
        out_ref = _rms(fma_ref, None, eps=eps)
        fma_got, out_got = R8.r8_post_sa(x, y, tbg, tsg, None, eps)
        sync()
    print(f"rows={rows} C={C} tsrows={ts_rows}: fma relL2={rel_l2(fma_got, fma_ref):.2e} "
          f"ULP={max_ulp(fma_got, fma_ref)} | out relL2={rel_l2(out_got, out_ref):.2e} ULP={max_ulp(out_got, out_ref)}")
    def eager_k2():
        g = tbg + tsg
        f = torch.addcmul(x, y, g)
        return _rms(f, None, eps=eps)
    te = bench(eager_k2)
    tg = bench(lambda: R8.r8_post_sa(x, y, tbg, tsg, None, eps))
    print(f"  eager={te:.3f}ms sycl={tg:.3f}ms speedup={te / tg:.2f}x")

print("== K3 rope negfold (mul+2xaddcmul_ 3->1) ==")
from ltx_core.model.transformer.rope import _apply_split_rotary_emb_negfold as eager_rope  # noqa: E402
for B, H, T, D in [(1, 32, 16384, 64), (1, 32, 4096, 64), (2, 8, 4096, 64)]:
    Dh = D // 2
    xb = torch.randn(B, T, H, D, dtype=torch.bfloat16, device=DEV)
    x = xb.transpose(1, 2)  # non-contiguous view like attention path
    cb = torch.randn(B, T, H, Dh, dtype=torch.bfloat16, device=DEV)
    cos = cb.transpose(1, 2)
    sb = torch.randn(B, T, H, Dh, dtype=torch.bfloat16, device=DEV)
    sin = sb.transpose(1, 2)
    with torch.inference_mode():
        ref = eager_rope(x, cos, sin)
        got = R8.r8_split_rope(x, cos, sin)
        sync()
    print(f"B={B} H={H} T={T} D={D}: contiguous_out={got.is_contiguous()} "
          f"relL2={rel_l2(got, ref):.2e} maxULP={max_ulp(got, ref)}")
    te = bench(lambda: eager_rope(x, cos, sin))
    tg = bench(lambda: R8.r8_split_rope(x, cos, sin))
    print(f"  eager={te:.3f}ms sycl={tg:.3f}ms speedup={te / tg:.2f}x")

print("MICROBENCH_DONE")
