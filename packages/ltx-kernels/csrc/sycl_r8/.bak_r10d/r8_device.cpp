// R8 device kernels (SYCL) v2 — compiled device-only to SPIR-V, launched from
// the host via L0-interop on torch's own queue.
// v2 changes: vector u16x8 loads/stores (manual RNE convert per lane, still no
// __devicelib imports); K2 holds fma in registers across the rstd barrier
// (single x/y/gate read pass: 4R+2W).
#include <sycl/sycl.hpp>

#include "r8_args.h"

using u16 = uint16_t;
using u32 = uint32_t;

constexpr int R8_WG = 256;
constexpr int R8_VEC = 8;
constexpr int R8_MAXSG = 32;
constexpr int R8_MAXIT = 4;  // C <= 8192 => iters <= 4

inline float r8_h2f(u16 h) {
  return sycl::bit_cast<float>((u32)h << 16);
}

inline u16 r8_f2h(float x) {
  u32 u = sycl::bit_cast<u32>(x);
  u32 bias = 0x7FFFu + ((u >> 16) & 1u);
  return (u16)((u + bias) >> 16);
}

inline sycl::vec<u16, 8> r8_vload(const u16 *p, int64_t c0, int64_t C) {
  if (c0 + 8 <= C) return *reinterpret_cast<const sycl::vec<u16, 8> *>(p + c0);
  sycl::vec<u16, 8> v;
  for (int k = 0; k < 8; ++k) v[k] = (c0 + k < C) ? p[c0 + k] : (u16)0;
  return v;
}

inline void r8_vstore(u16 *p, int64_t c0, int64_t C, sycl::vec<u16, 8> v) {
  if (c0 + 8 <= C) {
    *reinterpret_cast<sycl::vec<u16, 8> *>(p + c0) = v;
  } else {
    for (int k = 0; k < 8; ++k)
      if (c0 + k < C) p[c0 + k] = v[k];
  }
}

// Row rstd without SLM: subgroup reduce -> one float per subgroup in global
// scratch -> lid 0 sums -> broadcast (default global+local fence).
inline float r8_row_rstd(sycl::nd_item<1> it, float sumsq, float *scr, float C,
                         float eps, bool do_norm) {
  if (!do_norm) return 1.f;
  sycl::sub_group sg = it.get_sub_group();
  float part = sycl::reduce_over_group(sg, sumsq, sycl::plus<float>());
  if (sg.leader()) scr[sg.get_group_linear_id()] = part;
  it.barrier();
  if (it.get_local_linear_id() == 0) {
    float total = 0.f;
    uint32_t nsg = sg.get_group_linear_range();
    for (uint32_t i = 0; i < nsg; ++i) total += scr[i];
    scr[0] = total;
  }
  it.barrier();
  return sycl::rsqrt(scr[0] / C + eps);
}

template <bool DO_NORM>
inline void k1_body(sycl::nd_item<1> it, R8K1Args A) {
  const int64_t row = (int64_t)it.get_group_linear_id();
  const int64_t lid = (int64_t)it.get_local_linear_id();
  const int64_t C = A.C;
  const int64_t stride = (int64_t)R8_WG * R8_VEC;
  const u16 *xr = static_cast<const u16 *>(A.x) + row * C;
  const u16 *tbs = static_cast<const u16 *>(A.tbs);
  const u16 *tbh = static_cast<const u16 *>(A.tbh);
  const u16 *tss = static_cast<const u16 *>(A.tss);
  const u16 *tsh = static_cast<const u16 *>(A.tsh);
  u16 *orow = static_cast<u16 *>(A.out) + row * C;
  const u16 *tss_r = (A.tss_rs != 0) ? tss + row * A.tss_rs : tss;
  const u16 *tsh_r = (A.tsh_rs != 0) ? tsh + row * A.tsh_rs : tsh;
  float sumsq = 0.f;
  if constexpr (DO_NORM) {
    for (int64_t c0 = lid * R8_VEC; c0 < C; c0 += stride) {
      sycl::vec<u16, 8> vx = r8_vload(xr, c0, C);
      for (int k = 0; k < 8; ++k) {
        float v = r8_h2f(vx[k]);
        sumsq += v * v;
      }
    }
  }
  float *scr = static_cast<float *>(A.scratch) + row * R8_MAXSG;
  const float rstd = r8_row_rstd(it, sumsq, scr, (float)C, A.eps, DO_NORM);
  for (int64_t c0 = lid * R8_VEC; c0 < C; c0 += stride) {
    sycl::vec<u16, 8> vx = r8_vload(xr, c0, C);
    sycl::vec<u16, 8> vs = r8_vload(tbs, c0, C);
    sycl::vec<u16, 8> vh = r8_vload(tbh, c0, C);
    sycl::vec<u16, 8> vss = r8_vload(tss_r, c0, C);
    sycl::vec<u16, 8> vsh = r8_vload(tsh_r, c0, C);
    sycl::vec<u16, 8> vo;
    for (int k = 0; k < 8; ++k) {
      float xn = r8_h2f(vx[k]);
      float sc = r8_h2f(vs[k]) + r8_h2f(vss[k]);
      float sh = r8_h2f(vh[k]) + r8_h2f(vsh[k]);
      float y = sh + (DO_NORM ? xn * rstd : xn) * (1.f + sc);
      vo[k] = r8_f2h(y);
    }
    r8_vstore(orow, c0, C, vo);
  }
}

inline void k2_body(sycl::nd_item<1> it, R8K2Args A) {
  const int64_t row = (int64_t)it.get_group_linear_id();
  const int64_t lid = (int64_t)it.get_local_linear_id();
  const int64_t C = A.C;
  const int64_t stride = (int64_t)R8_WG * R8_VEC;
  const u16 *xr = static_cast<const u16 *>(A.x) + row * C;
  const u16 *yr = static_cast<const u16 *>(A.y) + row * C;
  const u16 *tbg = static_cast<const u16 *>(A.tbg);
  const u16 *tsg = static_cast<const u16 *>(A.tsg);
  const u16 *tsg_r = (A.tsg_rs != 0) ? tsg + row * A.tsg_rs : tsg;
  u16 *fr = static_cast<u16 *>(A.fma) + row * C;
  u16 *orow = static_cast<u16 *>(A.out) + row * C;
  float fm_buf[R8_MAXIT][8];
  int nit = 0;
  float sumsq = 0.f;
  for (int64_t c0 = lid * R8_VEC; c0 < C; c0 += stride) {
    sycl::vec<u16, 8> vx = r8_vload(xr, c0, C);
    sycl::vec<u16, 8> vy = r8_vload(yr, c0, C);
    sycl::vec<u16, 8> vg = r8_vload(tbg, c0, C);
    sycl::vec<u16, 8> vt = r8_vload(tsg_r, c0, C);
    for (int k = 0; k < 8; ++k) {
      float g = r8_h2f(vg[k]) + r8_h2f(vt[k]);
      float fm = r8_h2f(vx[k]) + r8_h2f(vy[k]) * g;
      fm_buf[nit][k] = fm;
      sumsq += fm * fm;
    }
    ++nit;
  }
  float *scr = static_cast<float *>(A.scratch) + row * R8_MAXSG;
  const float rstd = r8_row_rstd(it, sumsq, scr, (float)C, A.eps, true);
  nit = 0;
  for (int64_t c0 = lid * R8_VEC; c0 < C; c0 += stride) {
    sycl::vec<u16, 8> vfm, vno;
    for (int k = 0; k < 8; ++k) {
      float fm = fm_buf[nit][k];
      vfm[k] = (c0 + k < C) ? r8_f2h(fm) : (u16)0;
      vno[k] = (c0 + k < C) ? r8_f2h(fm * rstd) : (u16)0;
    }
    ++nit;
    r8_vstore(fr, c0, C, vfm);
    r8_vstore(orow, c0, C, vno);
  }
}

inline void k3_body(sycl::nd_item<1> it, R8K3Args A) {
  constexpr int K3_WG = 16;
  const int64_t Dh = A.Dh;
  const int64_t D = Dh * 2;
  const int64_t fib = (int64_t)it.get_group_linear_id();
  const int64_t lid = (int64_t)it.get_local_linear_id();
  const int64_t t = fib % A.T;
  const int64_t hb = fib / A.T;
  const int64_t hh = hb % A.H;
  const int64_t b = hb / A.H;
  const u16 *x = static_cast<const u16 *>(A.x);
  const u16 *cos = static_cast<const u16 *>(A.cos);
  const u16 *sin = static_cast<const u16 *>(A.sin);
  const int64_t xo = b * A.xs0 + hh * A.xs1 + t * A.xs2;
  const int64_t co = b * A.cs0 + hh * A.cs1 + t * A.cs2;
  const int64_t so = b * A.ss0 + hh * A.ss1 + t * A.ss2;
  u16 *orow = static_cast<u16 *>(A.out) + fib * D;
  for (int64_t j = lid; j < D; j += K3_WG) {
    const int64_t half = (j < Dh) ? 0 : 1;
    const int64_t k = j - half * Dh;
    float x0 = r8_h2f(x[xo + k * A.xs3]);
    float x1 = r8_h2f(x[xo + (k + Dh) * A.xs3]);
    float c = r8_h2f(cos[co + k * A.cs3]);
    float s = r8_h2f(sin[so + k * A.ss3]);
    orow[j] = r8_f2h((half == 0) ? (x0 * c - x1 * s) : (x0 * s + x1 * c));
  }
}

// R9D: vectorised split-rope (K3). Same math as k3_body, but processes 8 k
// lanes per iteration with u16x8 gathers/stores. Host gates on
// xs3==cs3==ss3==1, Dh%8==0 and 16B alignment of the row starts, so the
// scalar k3_body stays the universal fallback (LTX_R9D_K3V=0 too).
inline void k3_vec_body(sycl::nd_item<1> it, R8K3Args A) {
  const int64_t Dh = A.Dh;
  const int64_t D = Dh * 2;
  const int64_t nvec = Dh / 8;
  const int64_t gid = (int64_t)it.get_global_linear_id();
  const int64_t fib = gid / nvec;
  const int64_t v = gid % nvec;
  if (fib >= A.B * A.H * A.T) return;
  const int64_t t = fib % A.T;
  const int64_t hb = fib / A.T;
  const int64_t hh = hb % A.H;
  const int64_t b = hb / A.H;
  const u16 *x = static_cast<const u16 *>(A.x);
  const u16 *cos = static_cast<const u16 *>(A.cos);
  const u16 *sin = static_cast<const u16 *>(A.sin);
  const int64_t xo = b * A.xs0 + hh * A.xs1 + t * A.xs2;
  const int64_t co = b * A.cs0 + hh * A.cs1 + t * A.cs2;
  const int64_t so = b * A.ss0 + hh * A.ss1 + t * A.ss2;
  u16 *orow = static_cast<u16 *>(A.out) + fib * D;
  const int64_t k = v * 8;
  sycl::vec<u16, 8> vx0 = *reinterpret_cast<const sycl::vec<u16, 8> *>(x + xo + k);
  sycl::vec<u16, 8> vx1 = *reinterpret_cast<const sycl::vec<u16, 8> *>(x + xo + k + Dh);
  sycl::vec<u16, 8> vc = *reinterpret_cast<const sycl::vec<u16, 8> *>(cos + co + k);
  sycl::vec<u16, 8> vs = *reinterpret_cast<const sycl::vec<u16, 8> *>(sin + so + k);
  sycl::vec<u16, 8> o0, o1;
  for (int i = 0; i < 8; ++i) {
    float x0 = r8_h2f(vx0[i]);
    float x1 = r8_h2f(vx1[i]);
    float c = r8_h2f(vc[i]);
    float s = r8_h2f(vs[i]);
    o0[i] = r8_f2h(x0 * c - x1 * s);
    o1[i] = r8_f2h(x0 * s + x1 * c);
  }
  *reinterpret_cast<sycl::vec<u16, 8> *>(orow + k) = o0;
  *reinterpret_cast<sycl::vec<u16, 8> *>(orow + Dh + k) = o1;
}

// Named kernel functors: stable SPIR-V entry names, single POD struct arg.
struct R8K1AdaKernel {
  R8K1Args A;
  void operator()(sycl::nd_item<1> it) const { k1_body<true>(it, A); }
};

struct R8K1AffineKernel {
  R8K1Args A;
  void operator()(sycl::nd_item<1> it) const { k1_body<false>(it, A); }
};

struct R8K2Kernel {
  R8K2Args A;
  void operator()(sycl::nd_item<1> it) const { k2_body(it, A); }
};

struct R8K3Kernel {
  R8K3Args A;
  void operator()(sycl::nd_item<1> it) const { k3_body(it, A); }
};

struct R8K3VecKernel {
  R8K3Args A;
  void operator()(sycl::nd_item<1> it) const { k3_vec_body(it, A); }
};

// Anchor with external linkage so device compilation keeps all kernels.
void r8_device_anchor(sycl::queue &q) {
  R8K1Args a1{};
  R8K2Args a2{};
  R8K3Args a3{};
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::nd_range<1>(R8_WG, R8_WG), R8K1AdaKernel{a1});
  });
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::nd_range<1>(R8_WG, R8_WG), R8K1AffineKernel{a1});
  });
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::nd_range<1>(R8_WG, R8_WG), R8K2Kernel{a2});
  });
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::nd_range<1>(16, 16), R8K3Kernel{a3});
  });
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::nd_range<1>(64, 64), R8K3VecKernel{a3});
  });
}
