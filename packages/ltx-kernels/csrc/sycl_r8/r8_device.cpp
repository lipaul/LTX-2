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
constexpr int64_t R8_K1_MAX_C = 8192;

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

template <int N>
inline sycl::vec<u16, N> r18_vload(const u16 *p, int64_t c0, int64_t C) {
  if (c0 + N <= C) return *reinterpret_cast<const sycl::vec<u16, N> *>(p + c0);
  sycl::vec<u16, N> v;
  for (int k = 0; k < N; ++k) v[k] = (c0 + k < C) ? p[c0 + k] : (u16)0;
  return v;
}

template <int N>
inline void r18_vstore(u16 *p, int64_t c0, int64_t C, sycl::vec<u16, N> v) {
  if (c0 + N <= C) {
    *reinterpret_cast<sycl::vec<u16, N> *>(p + c0) = v;
  } else {
    for (int k = 0; k < N; ++k)
      if (c0 + k < C) p[c0 + k] = v[k];
  }
}

template <typename T>
inline void r18_prefetch(const T *p, size_t bytes) {
  sycl::ext::oneapi::experimental::prefetch(const_cast<T *>(p), bytes);
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

template <bool DO_NORM, bool COMPAT>
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
      float y;
      if constexpr (COMPAT) {
        // R31: round-0 rounding points (each torch bf16 op rounds once, RNE).
        float scb = r8_h2f(r8_f2h(sc));
        float shb = r8_h2f(r8_f2h(sh));
        float nn = DO_NORM ? r8_h2f(r8_f2h(xn * rstd)) : xn;
        float one = r8_h2f(r8_f2h(1.f + scb));
        float pr = r8_h2f(r8_f2h(nn * one));
        y = r8_h2f(r8_f2h(pr + shb));
      } else {
        y = sh + (DO_NORM ? xn * rstd : xn) * (1.f + sc);
      }
      vo[k] = r8_f2h(y);
    }
    r8_vstore(orow, c0, C, vo);
  }
}

// R10D: K1Ada single-pass (reg-hold). Same math as k1_body<true>, but x is
// loaded once and held in registers across the rstd barrier, so the second
// sweep re-reads only the (broadcast/cached) ada tables instead of x.
// Bitwise identical to k1_body<true>: the same values are loaded with the same
// r8_vload padding, and the same float ops run in the same order.
template <bool DO_NORM>
inline void k1_hold_body(sycl::nd_item<1> it, R8K1Args A) {
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
  float xbuf[R8_MAXIT][8];
  float sumsq = 0.f;
  int nit = 0;
  for (int64_t c0 = lid * R8_VEC; c0 < C; c0 += stride) {
    sycl::vec<u16, 8> vx = r8_vload(xr, c0, C);
    for (int k = 0; k < 8; ++k) {
      float v = r8_h2f(vx[k]);
      xbuf[nit][k] = v;
      if constexpr (DO_NORM) sumsq += v * v;
    }
    ++nit;
  }
  float *scr = static_cast<float *>(A.scratch) + row * R8_MAXSG;
  const float rstd = r8_row_rstd(it, sumsq, scr, (float)C, A.eps, DO_NORM);
  nit = 0;
  for (int64_t c0 = lid * R8_VEC; c0 < C; c0 += stride) {
    sycl::vec<u16, 8> vs = r8_vload(tbs, c0, C);
    sycl::vec<u16, 8> vh = r8_vload(tbh, c0, C);
    sycl::vec<u16, 8> vss = r8_vload(tss_r, c0, C);
    sycl::vec<u16, 8> vsh = r8_vload(tsh_r, c0, C);
    sycl::vec<u16, 8> vo;
    for (int k = 0; k < 8; ++k) {
      float xn = xbuf[nit][k];
      float sc = r8_h2f(vs[k]) + r8_h2f(vss[k]);
      float sh = r8_h2f(vh[k]) + r8_h2f(vsh[k]);
      float y = sh + (DO_NORM ? xn * rstd : xn) * (1.f + sc);
      vo[k] = r8_f2h(y);
    }
    ++nit;
    r8_vstore(orow, c0, C, vo);
  }
}

inline float r18_row_rstd_local(sycl::nd_item<1> it, float sumsq,
                                sycl::local_accessor<float, 1> scratch,
                                float C, float eps) {
  sycl::sub_group sg = it.get_sub_group();
  const float part = sycl::reduce_over_group(sg, sumsq, sycl::plus<float>());
  if (sg.leader()) scratch[sg.get_group_linear_id()] = part;
  it.barrier();
  if (it.get_local_linear_id() == 0) {
    float total = 0.f;
    const uint32_t nsg = sg.get_group_linear_range();
    for (uint32_t i = 0; i < nsg; ++i) total += scratch[i];
    scratch[0] = total;
  }
  it.barrier();
  return sycl::rsqrt(scratch[0] / C + eps);
}

// R41: R31-compat ada with the row rstd reduced through shared local memory instead
// of the global scratch pool. The subgroup reduce, the per-subgroup partial write, the
// index-order accumulation by local id 0 and the closing rsqrt are the same sequence
// as r8_row_rstd, so the broadcast rstd is bitwise equal to the global-scratch path;
// only the storage of the 32 partial sums changes, which removes one global-memory
// round trip and one full memory fence from every workgroup. Selected by the host
// when LTX_R41_ADALOCAL=1.
inline void r41_k1_ada_compat_local_body(sycl::nd_item<1> it, R8K1Args A,
                                         sycl::local_accessor<float, 1> scratch) {
  const int64_t row = (int64_t)it.get_group_linear_id();
  const int64_t lid = (int64_t)it.get_local_linear_id();
  const int64_t C = A.C;
  const int64_t wg = (int64_t)it.get_local_range(0);
  const int64_t stride = wg * R8_VEC;
  const u16 *xr = static_cast<const u16 *>(A.x) + row * C;
  const u16 *tbs = static_cast<const u16 *>(A.tbs);
  const u16 *tbh = static_cast<const u16 *>(A.tbh);
  const u16 *tss = static_cast<const u16 *>(A.tss);
  const u16 *tsh = static_cast<const u16 *>(A.tsh);
  u16 *orow = static_cast<u16 *>(A.out) + row * C;
  const u16 *tss_r = (A.tss_rs != 0) ? tss + row * A.tss_rs : tss;
  const u16 *tsh_r = (A.tsh_rs != 0) ? tsh + row * A.tsh_rs : tsh;
  float sumsq = 0.f;
  for (int64_t c0 = lid * R8_VEC; c0 < C; c0 += stride) {
    sycl::vec<u16, R8_VEC> vx = r8_vload(xr, c0, C);
    for (int k = 0; k < R8_VEC; ++k) {
      const float v = r8_h2f(vx[k]);
      sumsq += v * v;
    }
  }
  const float rstd = r18_row_rstd_local(it, sumsq, scratch, (float)C, A.eps);
  for (int64_t c0 = lid * R8_VEC; c0 < C; c0 += stride) {
    sycl::vec<u16, R8_VEC> vx = r8_vload(xr, c0, C);
    sycl::vec<u16, R8_VEC> vs = r8_vload(tbs, c0, C);
    sycl::vec<u16, R8_VEC> vh = r8_vload(tbh, c0, C);
    sycl::vec<u16, R8_VEC> vss = r8_vload(tss_r, c0, C);
    sycl::vec<u16, R8_VEC> vsh = r8_vload(tsh_r, c0, C);
    sycl::vec<u16, R8_VEC> vo;
    for (int k = 0; k < R8_VEC; ++k) {
      const float xn = r8_h2f(vx[k]);
      const float sc = r8_h2f(vs[k]) + r8_h2f(vss[k]);
      const float sh = r8_h2f(vh[k]) + r8_h2f(vsh[k]);
      const float scb = r8_h2f(r8_f2h(sc));
      const float shb = r8_h2f(r8_f2h(sh));
      const float nn = r8_h2f(r8_f2h(xn * rstd));
      const float one = r8_h2f(r8_f2h(1.f + scb));
      const float pr = r8_h2f(r8_f2h(nn * one));
      vo[k] = r8_f2h(pr + shb);
    }
    r8_vstore(orow, c0, C, vo);
  }
}

// R41: row rstd reduction into shared local memory at a caller-chosen offset, so one
// workgroup can reduce several rows from one allocation. The subgroup reduce, the
// per-subgroup partial write, the index-order accumulation by local id 0 and the
// closing rsqrt are the sequence of r8_row_rstd, unchanged.
inline float r41_row_rstd_local_at(sycl::nd_item<1> it, float sumsq,
                                    sycl::local_accessor<float, 1> scratch,
                                    size_t offset, float C, float eps) {
  sycl::sub_group sg = it.get_sub_group();
  const float part = sycl::reduce_over_group(sg, sumsq, sycl::plus<float>());
  if (sg.leader()) scratch[offset + sg.get_group_linear_id()] = part;
  it.barrier();
  if (it.get_local_linear_id() == 0) {
    float total = 0.f;
    const uint32_t nsg = sg.get_group_linear_range();
    for (uint32_t i = 0; i < nsg; ++i) total += scratch[offset + i];
    scratch[offset] = total;
  }
  it.barrier();
  return sycl::rsqrt(scratch[offset] / C + eps);
}

// R41: two-row software-pipelined R31-compat ada.
//
// One workgroup owns two consecutive rows. Both rows' per-token streams (tss, tsh,
// 128 MB each at the in-situ shape) and their x rows are loaded into register buffers;
// the second row's loads are issued before the first row's reduction barrier, so the
// memory system is busy while the barrier drains instead of the whole workgroup
// parking on a late cache line. Halving the workgroup count also halves the number of
// sequential reduction waves the array has to walk through.
//
// The per-element arithmetic, the R31 round-0 rounding chain and the per-row reduction
// sequence are those of k1_body<true,true>, so the result is bitwise equal to
// R8K1AdaCompatLocalKernel. Selected by the host when LTX_R41_ADAPIPE=1.
template <int ITERS>
inline void r41_k1_ada_compat_pipe_body(sycl::nd_item<1> it, R8K1Args A,
                                        sycl::local_accessor<float, 1> scratch) {
  const int64_t wg = (int64_t)it.get_local_range(0);
  const int64_t lid = (int64_t)it.get_local_linear_id();
  const int64_t C = A.C;
  const int64_t stride = wg * R8_VEC;
  const int64_t row0 = (int64_t)it.get_group_linear_id() * 2;
  const int64_t nrows = A.rows;
  const bool two = row0 + 1 < nrows;
  const u16 *tbs = static_cast<const u16 *>(A.tbs);
  const u16 *tbh = static_cast<const u16 *>(A.tbh);

  sycl::vec<u16, R8_VEC> xa[ITERS], tssa[ITERS], tsha[ITERS];
  sycl::vec<u16, R8_VEC> xb[ITERS], tssb[ITERS], tshb[ITERS];

  // prologue: row0 x + per-token tables
  {
    const u16 *xr = static_cast<const u16 *>(A.x) + row0 * C;
    const u16 *tss_r = (A.tss_rs != 0) ? static_cast<const u16 *>(A.tss) + row0 * A.tss_rs
                                       : static_cast<const u16 *>(A.tss);
    const u16 *tsh_r = (A.tsh_rs != 0) ? static_cast<const u16 *>(A.tsh) + row0 * A.tsh_rs
                                       : static_cast<const u16 *>(A.tsh);
#pragma unroll
    for (int j = 0; j < ITERS; ++j) {
      const int64_t c0 = lid * R8_VEC + (int64_t)j * stride;
      xa[j] = r8_vload(xr, c0, C);
      tssa[j] = r8_vload(tss_r, c0, C);
      tsha[j] = r8_vload(tsh_r, c0, C);
    }
  }
  // issue row1's loads before row0's barrier
  if (two) {
    const u16 *xr = static_cast<const u16 *>(A.x) + (row0 + 1) * C;
    const u16 *tss_r = (A.tss_rs != 0) ? static_cast<const u16 *>(A.tss) + (row0 + 1) * A.tss_rs
                                       : static_cast<const u16 *>(A.tss);
    const u16 *tsh_r = (A.tsh_rs != 0) ? static_cast<const u16 *>(A.tsh) + (row0 + 1) * A.tsh_rs
                                       : static_cast<const u16 *>(A.tsh);
#pragma unroll
    for (int j = 0; j < ITERS; ++j) {
      const int64_t c0 = lid * R8_VEC + (int64_t)j * stride;
      xb[j] = r8_vload(xr, c0, C);
      tssb[j] = r8_vload(tss_r, c0, C);
      tshb[j] = r8_vload(tsh_r, c0, C);
    }
  }

  float sa = 0.f, sb = 0.f;
#pragma unroll
  for (int j = 0; j < ITERS; ++j) {
    for (int k = 0; k < R8_VEC; ++k) {
      const float va = r8_h2f(xa[j][k]);
      sa += va * va;
    }
  }
  if (two) {
#pragma unroll
    for (int j = 0; j < ITERS; ++j) {
      for (int k = 0; k < R8_VEC; ++k) {
        const float vb = r8_h2f(xb[j][k]);
        sb += vb * vb;
      }
    }
  }

  const float rstd_a = r41_row_rstd_local_at(it, sa, scratch, 0, (float)C, A.eps);
  if (two) {
    const float rstd_b = r41_row_rstd_local_at(it, sb, scratch, (size_t)wg, (float)C, A.eps);

    u16 *oa = static_cast<u16 *>(A.out) + row0 * C;
    u16 *ob = oa + C;
#pragma unroll
    for (int j = 0; j < ITERS; ++j) {
      const int64_t c0 = lid * R8_VEC + (int64_t)j * stride;
      const sycl::vec<u16, R8_VEC> vs = r8_vload(tbs, c0, C);
      const sycl::vec<u16, R8_VEC> vh = r8_vload(tbh, c0, C);
      sycl::vec<u16, R8_VEC> va_out, vb_out;
#pragma unroll
      for (int k = 0; k < R8_VEC; ++k) {
        const float sc = r8_h2f(vs[k]) + r8_h2f(tssa[j][k]);
        const float sh = r8_h2f(vh[k]) + r8_h2f(tsha[j][k]);
        const float scb = r8_h2f(r8_f2h(sc));
        const float shb = r8_h2f(r8_f2h(sh));
        const float na = r8_h2f(r8_f2h(r8_h2f(xa[j][k]) * rstd_a));
        const float one = r8_h2f(r8_f2h(1.f + scb));
        va_out[k] = r8_f2h(r8_h2f(r8_f2h(na * one)) + shb);

        const float sc2 = r8_h2f(vs[k]) + r8_h2f(tssb[j][k]);
        const float sh2 = r8_h2f(vh[k]) + r8_h2f(tshb[j][k]);
        const float scb2 = r8_h2f(r8_f2h(sc2));
        const float shb2 = r8_h2f(r8_f2h(sh2));
        const float nb = r8_h2f(r8_f2h(r8_h2f(xb[j][k]) * rstd_b));
        const float one2 = r8_h2f(r8_f2h(1.f + scb2));
        vb_out[k] = r8_f2h(r8_h2f(r8_f2h(nb * one2)) + shb2);
      }
      r8_vstore(oa, c0, C, va_out);
      r8_vstore(ob, c0, C, vb_out);
    }
    return;
  }

  u16 *oa = static_cast<u16 *>(A.out) + row0 * C;
#pragma unroll
  for (int j = 0; j < ITERS; ++j) {
    const int64_t c0 = lid * R8_VEC + (int64_t)j * stride;
    const sycl::vec<u16, R8_VEC> vs = r8_vload(tbs, c0, C);
    const sycl::vec<u16, R8_VEC> vh = r8_vload(tbh, c0, C);
    sycl::vec<u16, R8_VEC> vout;
#pragma unroll
    for (int k = 0; k < R8_VEC; ++k) {
      const float sc = r8_h2f(vs[k]) + r8_h2f(tssa[j][k]);
      const float sh = r8_h2f(vh[k]) + r8_h2f(tsha[j][k]);
      const float scb = r8_h2f(r8_f2h(sc));
      const float shb = r8_h2f(r8_f2h(sh));
      const float n = r8_h2f(r8_f2h(r8_h2f(xa[j][k]) * rstd_a));
      const float one = r8_h2f(r8_f2h(1.f + scb));
      vout[k] = r8_f2h(r8_h2f(r8_f2h(n * one)) + shb);
    }
    r8_vstore(oa, c0, C, vout);
  }
}

inline void r19_row_rstd_pair(sycl::nd_item<1> it, float qsumsq, float ksumsq,
                              float *scratch, int64_t row, float C, float eps,
                              float &q_rstd, float &k_rstd) {
  sycl::sub_group sg = it.get_sub_group();
  const uint32_t sgid = sg.get_group_linear_id();
  const float qpart = sycl::reduce_over_group(sg, qsumsq, sycl::plus<float>());
  const float kpart = sycl::reduce_over_group(sg, ksumsq, sycl::plus<float>());
  float *row_scratch = scratch + row * 2 * R8_MAXSG;
  if (sg.leader()) {
    row_scratch[2 * sgid] = qpart;
    row_scratch[2 * sgid + 1] = kpart;
  }
  it.barrier();
  if (it.get_local_linear_id() == 0) {
    float qtotal = 0.f;
    float ktotal = 0.f;
    const uint32_t nsg = sg.get_group_linear_range();
    for (uint32_t i = 0; i < nsg; ++i) {
      qtotal += row_scratch[2 * i];
      ktotal += row_scratch[2 * i + 1];
    }
    row_scratch[0] = qtotal;
    row_scratch[1] = ktotal;
  }
  it.barrier();
  q_rstd = sycl::rsqrt(row_scratch[0] / C + eps);
  k_rstd = sycl::rsqrt(row_scratch[1] / C + eps);
}

inline void r19_norm_rope_body(sycl::nd_item<1> it, R8R19NormRopeArgs A) {
  constexpr int64_t VEC = 4;
  const int64_t row = static_cast<int64_t>(it.get_group_linear_id());
  const int64_t lid = static_cast<int64_t>(it.get_local_linear_id());
  const int64_t b = row / A.T;
  const int64_t t = row % A.T;
  const int64_t C = A.C;
  const int64_t D = A.Dh * 2;
  const int64_t stride = static_cast<int64_t>(it.get_local_range(0)) * VEC;
  const u16 *qrow = static_cast<const u16 *>(A.q) + b * A.q_batch_stride + t * A.q_row_stride;
  const u16 *krow = static_cast<const u16 *>(A.k) + b * A.k_batch_stride + t * A.k_row_stride;
  const u16 *qw = static_cast<const u16 *>(A.q_weight);
  const u16 *kw = static_cast<const u16 *>(A.k_weight);
  const u16 *cos = static_cast<const u16 *>(A.cos);
  const u16 *sin = static_cast<const u16 *>(A.sin);
  u16 *outq = static_cast<u16 *>(A.outq) + row * C;
  u16 *outk = static_cast<u16 *>(A.outk) + row * C;
  float qsumsq = 0.f;
  float ksumsq = 0.f;
  for (int64_t c0 = lid * VEC; c0 < C; c0 += stride) {
    const sycl::vec<u16, VEC> vq = r18_vload<VEC>(qrow, c0, C);
    const sycl::vec<u16, VEC> vk = r18_vload<VEC>(krow, c0, C);
    for (int i = 0; i < VEC; ++i) {
      const float qv = r8_h2f(vq[i]);
      const float kv = r8_h2f(vk[i]);
      qsumsq += qv * qv;
      ksumsq += kv * kv;
    }
  }
  float q_rstd;
  float k_rstd;
  r19_row_rstd_pair(it, qsumsq, ksumsq, static_cast<float *>(A.scratch), row,
                    static_cast<float>(C), A.eps, q_rstd, k_rstd);
  for (int64_t c0 = lid * VEC; c0 < C; c0 += stride) {
    const int64_t h = c0 / D;
    const int64_t local = c0 - h * D;
    const int64_t half = local / A.Dh;
    const int64_t k = local - half * A.Dh;
    const int64_t pair_offset = half == 0 ? A.Dh : -A.Dh;
    const int64_t co = b * A.cos_stride0 + h * A.cos_stride1 + t * A.cos_stride2;
    const int64_t so = b * A.sin_stride0 + h * A.sin_stride1 + t * A.sin_stride2;
    const sycl::vec<u16, VEC> vq = r18_vload<VEC>(qrow, c0, C);
    const sycl::vec<u16, VEC> vk = r18_vload<VEC>(krow, c0, C);
    const sycl::vec<u16, VEC> vq_pair = r18_vload<VEC>(qrow, c0 + pair_offset, C);
    const sycl::vec<u16, VEC> vk_pair = r18_vload<VEC>(krow, c0 + pair_offset, C);
    const sycl::vec<u16, VEC> vqw = r18_vload<VEC>(qw, c0, C);
    const sycl::vec<u16, VEC> vkw = r18_vload<VEC>(kw, c0, C);
    const sycl::vec<u16, VEC> vqw_pair = r18_vload<VEC>(qw, c0 + pair_offset, C);
    const sycl::vec<u16, VEC> vkw_pair = r18_vload<VEC>(kw, c0 + pair_offset, C);
    const sycl::vec<u16, VEC> vc =
        *reinterpret_cast<const sycl::vec<u16, VEC> *>(cos + co + k * A.cos_stride3);
    const sycl::vec<u16, VEC> vs =
        *reinterpret_cast<const sycl::vec<u16, VEC> *>(sin + so + k * A.sin_stride3);
    sycl::vec<u16, VEC> oq, ok;
    for (int i = 0; i < VEC; ++i) {
      const float c = r8_h2f(vc[i]);
      const float s = r8_h2f(vs[i]);
      float q0 = r8_h2f(vq[i]) * q_rstd;
      q0 *= r8_h2f(vqw[i]);
      q0 = r8_h2f(r8_f2h(q0));
      float q1 = r8_h2f(vq_pair[i]) * q_rstd;
      q1 *= r8_h2f(vqw_pair[i]);
      q1 = r8_h2f(r8_f2h(q1));
      float k0 = r8_h2f(vk[i]) * k_rstd;
      k0 *= r8_h2f(vkw[i]);
      k0 = r8_h2f(r8_f2h(k0));
      float k1 = r8_h2f(vk_pair[i]) * k_rstd;
      k1 *= r8_h2f(vkw_pair[i]);
      k1 = r8_h2f(r8_f2h(k1));
      const float qs0 = r8_h2f(r8_f2h(q0 * A.q_scale));
      const float qs1 = r8_h2f(r8_f2h(q1 * A.q_scale));
      const float ks0 = r8_h2f(r8_f2h(k0 * A.k_scale));
      const float ks1 = r8_h2f(r8_f2h(k1 * A.k_scale));
      oq[i] = half == 0 ? r8_f2h(qs0 * c - qs1 * s) : r8_f2h(qs1 * s + qs0 * c);
      ok[i] = half == 0 ? r8_f2h(ks0 * c - ks1 * s) : r8_f2h(ks1 * s + ks0 * c);
    }
    r18_vstore<VEC>(outq, c0, C, oq);
    r18_vstore<VEC>(outk, c0, C, ok);
  }
}

template <int VEC>
inline void r18_k1_local_body(sycl::nd_item<1> it, R8K1Args A,
                              sycl::local_accessor<float, 1> scratch) {
  const int64_t row = (int64_t)it.get_group_linear_id();
  const int64_t lid = (int64_t)it.get_local_linear_id();
  const int64_t C = A.C;
  const int64_t wg = (int64_t)it.get_local_range(0);
  const int64_t stride = wg * VEC;
  const u16 *xr = static_cast<const u16 *>(A.x) + row * C;
  const u16 *tbs = static_cast<const u16 *>(A.tbs);
  const u16 *tbh = static_cast<const u16 *>(A.tbh);
  const u16 *tss = static_cast<const u16 *>(A.tss);
  const u16 *tsh = static_cast<const u16 *>(A.tsh);
  const u16 *tss_r = (A.tss_rs != 0) ? tss + row * A.tss_rs : tss;
  const u16 *tsh_r = (A.tsh_rs != 0) ? tsh + row * A.tsh_rs : tsh;
  u16 *orow = static_cast<u16 *>(A.out) + row * C;
  constexpr int64_t bytes = VEC * (int64_t)sizeof(u16);
  float sumsq = 0.f;
  for (int64_t c0 = lid * VEC; c0 < C; c0 += stride) {
    const int64_t next = c0 + stride;
    if (A.prefetch == 2) {
      r18_prefetch(xr + c0, bytes);
      r18_prefetch(tbs + c0, bytes);
      r18_prefetch(tbh + c0, bytes);
      r18_prefetch(tss_r + c0, bytes);
      r18_prefetch(tsh_r + c0, bytes);
    } else if (A.prefetch == 1 && next < C) {
      r18_prefetch(xr + next, bytes);
    }
    sycl::vec<u16, VEC> vx = r18_vload<VEC>(xr, c0, C);
    for (int k = 0; k < VEC; ++k) {
      const float v = r8_h2f(vx[k]);
      sumsq += v * v;
    }
  }
  const float rstd = r18_row_rstd_local(it, sumsq, scratch, (float)C, A.eps);
  for (int64_t c0 = lid * VEC; c0 < C; c0 += stride) {
    const int64_t next = c0 + stride;
    if (A.prefetch == 1 && next < C) {
      r18_prefetch(tbs + next, bytes);
      r18_prefetch(tbh + next, bytes);
      r18_prefetch(tss_r + next, bytes);
      r18_prefetch(tsh_r + next, bytes);
    }
    sycl::vec<u16, VEC> vx = r18_vload<VEC>(xr, c0, C);
    sycl::vec<u16, VEC> vs = r18_vload<VEC>(tbs, c0, C);
    sycl::vec<u16, VEC> vh = r18_vload<VEC>(tbh, c0, C);
    sycl::vec<u16, VEC> vss = r18_vload<VEC>(tss_r, c0, C);
    sycl::vec<u16, VEC> vsh = r18_vload<VEC>(tsh_r, c0, C);
    sycl::vec<u16, VEC> vo;
    for (int k = 0; k < VEC; ++k) {
      const float xn = r8_h2f(vx[k]);
      const float sc = r8_h2f(vs[k]) + r8_h2f(vss[k]);
      const float sh = r8_h2f(vh[k]) + r8_h2f(vsh[k]);
      const float y = sh + xn * rstd * (1.f + sc);
      vo[k] = r8_f2h(y);
    }
    r18_vstore<VEC>(orow, c0, C, vo);
  }
}

template <int VEC, int ITERS>
inline void r18_k1_reg_body(sycl::nd_item<1> it, R8K1Args A,
                             sycl::local_accessor<float, 1> scratch) {
  const int64_t row = (int64_t)it.get_group_linear_id();
  const int64_t lid = (int64_t)it.get_local_linear_id();
  const int64_t C = A.C;
  const int64_t wg = (int64_t)it.get_local_range(0);
  const int64_t stride = wg * VEC;
  const u16 *xr = static_cast<const u16 *>(A.x) + row * C;
  const u16 *tbs = static_cast<const u16 *>(A.tbs);
  const u16 *tbh = static_cast<const u16 *>(A.tbh);
  const u16 *tss = static_cast<const u16 *>(A.tss);
  const u16 *tsh = static_cast<const u16 *>(A.tsh);
  const u16 *tss_r = (A.tss_rs != 0) ? tss + row * A.tss_rs : tss;
  const u16 *tsh_r = (A.tsh_rs != 0) ? tsh + row * A.tsh_rs : tsh;
  u16 *orow = static_cast<u16 *>(A.out) + row * C;
  constexpr int64_t bytes = VEC * (int64_t)sizeof(u16);
  u16 xbuf[ITERS][VEC];
  u16 ssbuf[ITERS][VEC];
  u16 shbuf[ITERS][VEC];
  int nit = 0;
  float sumsq = 0.f;
  for (int64_t c0 = lid * VEC; c0 < C; c0 += stride) {
    if (A.prefetch == 2) {
      r18_prefetch(xr + c0, bytes);
      r18_prefetch(tbs + c0, bytes);
      r18_prefetch(tbh + c0, bytes);
      r18_prefetch(tss_r + c0, bytes);
      r18_prefetch(tsh_r + c0, bytes);
    } else if (A.prefetch == 1 && c0 + stride < C) {
      r18_prefetch(xr + c0 + stride, bytes);
    }
    sycl::vec<u16, VEC> vx = r18_vload<VEC>(xr, c0, C);
    sycl::vec<u16, VEC> vss = r18_vload<VEC>(tss_r, c0, C);
    sycl::vec<u16, VEC> vsh = r18_vload<VEC>(tsh_r, c0, C);
    for (int k = 0; k < VEC; ++k) {
      xbuf[nit][k] = vx[k];
      ssbuf[nit][k] = vss[k];
      shbuf[nit][k] = vsh[k];
      const float v = r8_h2f(vx[k]);
      sumsq += v * v;
    }
    ++nit;
  }
  const float rstd = r18_row_rstd_local(it, sumsq, scratch, (float)C, A.eps);
  nit = 0;
  for (int64_t c0 = lid * VEC; c0 < C; c0 += stride) {
    const int64_t next = c0 + stride;
    if (A.prefetch != 0 && next < C) {
      r18_prefetch(tbs + next, bytes);
      r18_prefetch(tbh + next, bytes);
    }
    sycl::vec<u16, VEC> vs = r18_vload<VEC>(tbs, c0, C);
    sycl::vec<u16, VEC> vh = r18_vload<VEC>(tbh, c0, C);
    sycl::vec<u16, VEC> vo;
    for (int k = 0; k < VEC; ++k) {
      const float xn = r8_h2f(xbuf[nit][k]);
      const float sc = r8_h2f(vs[k]) + r8_h2f(ssbuf[nit][k]);
      const float sh = r8_h2f(vh[k]) + r8_h2f(shbuf[nit][k]);
      vo[k] = r8_f2h(sh + xn * rstd * (1.f + sc));
    }
    ++nit;
    r8_vstore(orow, c0, C, vo);
  }
}

template <int VEC>
inline void r18_k1_subgroup_body(sycl::nd_item<1> it, R8K1Args A) {
  sycl::sub_group sg = it.get_sub_group();
  const int64_t nsg = (int64_t)it.get_local_range(0) / 32;
  const int64_t row = (int64_t)it.get_group_linear_id() * nsg +
                      (int64_t)sg.get_group_linear_id();
  if (row >= A.rows) return;
  const int64_t C = A.C;
  const int64_t lid = (int64_t)sg.get_local_linear_id();
  const int64_t stride = 32 * VEC;
  const u16 *xr = static_cast<const u16 *>(A.x) + row * C;
  const u16 *tbs = static_cast<const u16 *>(A.tbs);
  const u16 *tbh = static_cast<const u16 *>(A.tbh);
  const u16 *tss = static_cast<const u16 *>(A.tss);
  const u16 *tsh = static_cast<const u16 *>(A.tsh);
  const u16 *tss_r = (A.tss_rs != 0) ? tss + row * A.tss_rs : tss;
  const u16 *tsh_r = (A.tsh_rs != 0) ? tsh + row * A.tsh_rs : tsh;
  u16 *orow = static_cast<u16 *>(A.out) + row * C;
  constexpr int64_t bytes = VEC * (int64_t)sizeof(u16);
  float sumsq = 0.f;
  for (int64_t c0 = lid * VEC; c0 < C; c0 += stride) {
    if (A.prefetch == 2) {
      r18_prefetch(xr + c0, bytes);
      r18_prefetch(tbs + c0, bytes);
      r18_prefetch(tbh + c0, bytes);
      r18_prefetch(tss_r + c0, bytes);
      r18_prefetch(tsh_r + c0, bytes);
    } else if (A.prefetch == 1 && c0 + stride < C) {
      r18_prefetch(xr + c0 + stride, bytes);
    }
    sycl::vec<u16, VEC> vx = r18_vload<VEC>(xr, c0, C);
    for (int k = 0; k < VEC; ++k) {
      const float v = r8_h2f(vx[k]);
      sumsq += v * v;
    }
  }
  const float total = sycl::reduce_over_group(sg, sumsq, sycl::plus<float>());
  const float rstd = sycl::rsqrt(total / (float)C + A.eps);
  for (int64_t c0 = lid * VEC; c0 < C; c0 += stride) {
    const int64_t next = c0 + stride;
    if (A.prefetch == 1 && next < C) {
      r18_prefetch(tbs + next, bytes);
      r18_prefetch(tbh + next, bytes);
      r18_prefetch(tss_r + next, bytes);
      r18_prefetch(tsh_r + next, bytes);
    }
    sycl::vec<u16, VEC> vx = r18_vload<VEC>(xr, c0, C);
    sycl::vec<u16, VEC> vs = r18_vload<VEC>(tbs, c0, C);
    sycl::vec<u16, VEC> vh = r18_vload<VEC>(tbh, c0, C);
    sycl::vec<u16, VEC> vss = r18_vload<VEC>(tss_r, c0, C);
    sycl::vec<u16, VEC> vsh = r18_vload<VEC>(tsh_r, c0, C);
    sycl::vec<u16, VEC> vo;
    for (int k = 0; k < VEC; ++k) {
      const float xn = r8_h2f(vx[k]);
      const float sc = r8_h2f(vs[k]) + r8_h2f(vss[k]);
      const float sh = r8_h2f(vh[k]) + r8_h2f(vsh[k]);
      vo[k] = r8_f2h(sh + xn * rstd * (1.f + sc));
    }
    r18_vstore<VEC>(orow, c0, C, vo);
  }
}

template <int VEC>
inline void r18_k1_group_reduce_body(sycl::nd_item<1> it, R8K1Args A) {
  const int64_t row = (int64_t)it.get_group_linear_id();
  const int64_t lid = (int64_t)it.get_local_linear_id();
  const int64_t C = A.C;
  const int64_t stride = (int64_t)it.get_local_range(0) * VEC;
  const u16 *xr = static_cast<const u16 *>(A.x) + row * C;
  const u16 *tbs = static_cast<const u16 *>(A.tbs);
  const u16 *tbh = static_cast<const u16 *>(A.tbh);
  const u16 *tss = static_cast<const u16 *>(A.tss);
  const u16 *tsh = static_cast<const u16 *>(A.tsh);
  const u16 *tss_r = (A.tss_rs != 0) ? tss + row * A.tss_rs : tss;
  const u16 *tsh_r = (A.tsh_rs != 0) ? tsh + row * A.tsh_rs : tsh;
  u16 *orow = static_cast<u16 *>(A.out) + row * C;
  constexpr int64_t bytes = VEC * (int64_t)sizeof(u16);
  float sumsq = 0.f;
  for (int64_t c0 = lid * VEC; c0 < C; c0 += stride) {
    if (A.prefetch == 2) {
      r18_prefetch(xr + c0, bytes);
      r18_prefetch(tbs + c0, bytes);
      r18_prefetch(tbh + c0, bytes);
      r18_prefetch(tss_r + c0, bytes);
      r18_prefetch(tsh_r + c0, bytes);
    } else if (A.prefetch == 1 && c0 + stride < C) {
      r18_prefetch(xr + c0 + stride, bytes);
    }
    sycl::vec<u16, VEC> vx = r18_vload<VEC>(xr, c0, C);
    for (int k = 0; k < VEC; ++k) {
      const float v = r8_h2f(vx[k]);
      sumsq += v * v;
    }
  }
  const float total = sycl::reduce_over_group(it.get_group(), sumsq, sycl::plus<float>());
  const float rstd = sycl::rsqrt(total / (float)C + A.eps);
  for (int64_t c0 = lid * VEC; c0 < C; c0 += stride) {
    const int64_t next = c0 + stride;
    if (A.prefetch == 1 && next < C) {
      r18_prefetch(tbs + next, bytes);
      r18_prefetch(tbh + next, bytes);
      r18_prefetch(tss_r + next, bytes);
      r18_prefetch(tsh_r + next, bytes);
    }
    sycl::vec<u16, VEC> vx = r18_vload<VEC>(xr, c0, C);
    sycl::vec<u16, VEC> vs = r18_vload<VEC>(tbs, c0, C);
    sycl::vec<u16, VEC> vh = r18_vload<VEC>(tbh, c0, C);
    sycl::vec<u16, VEC> vss = r18_vload<VEC>(tss_r, c0, C);
    sycl::vec<u16, VEC> vsh = r18_vload<VEC>(tsh_r, c0, C);
    sycl::vec<u16, VEC> vo;
    for (int k = 0; k < VEC; ++k) {
      const float xn = r8_h2f(vx[k]);
      const float sc = r8_h2f(vs[k]) + r8_h2f(vss[k]);
      const float sh = r8_h2f(vh[k]) + r8_h2f(vsh[k]);
      vo[k] = r8_f2h(sh + xn * rstd * (1.f + sc));
    }
    r18_vstore<VEC>(orow, c0, C, vo);
  }
}

template <bool COMPAT>
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
      float fm;
      if constexpr (COMPAT) {
        // R31: round-0 `x + y * gate` (mul rounds, then add rounds).
        float gb = r8_h2f(r8_f2h(g));
        float pr = r8_h2f(r8_f2h(r8_h2f(vy[k]) * gb));
        fm = r8_h2f(r8_f2h(r8_h2f(vx[k]) + pr));
      } else {
        fm = r8_h2f(vx[k]) + r8_h2f(vy[k]) * g;
      }
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

inline sycl::vec<u16, 4> r17_vload4(const u16 *p, int64_t c0, int64_t C) {
  if (c0 + 4 <= C) return *reinterpret_cast<const sycl::vec<u16, 4> *>(p + c0);
  sycl::vec<u16, 4> v;
  for (int k = 0; k < 4; ++k) v[k] = (c0 + k < C) ? p[c0 + k] : (u16)0;
  return v;
}

inline void r17_vstore4(u16 *p, int64_t c0, int64_t C, sycl::vec<u16, 4> v) {
  if (c0 + 4 <= C) {
    *reinterpret_cast<sycl::vec<u16, 4> *>(p + c0) = v;
  } else {
    for (int k = 0; k < 4; ++k)
      if (c0 + k < C) p[c0 + k] = v[k];
  }
}

inline void r17_mlp_norm_affine_body(sycl::nd_item<1> it, R8R17MlpNormAffineArgs A) {
  constexpr int64_t R17_SG_SIZE = 32;
  constexpr int64_t R17_ROWS_PER_GROUP = 8;
  constexpr int64_t R17_VECTOR = 8;
  const sycl::sub_group sg = it.get_sub_group();
  const int64_t C = A.C;
  const int64_t lane = (int64_t)sg.get_local_linear_id();
  const int64_t row = (int64_t)it.get_group_linear_id() * R17_ROWS_PER_GROUP +
                      (int64_t)sg.get_group_linear_id();
  const bool active = row < A.rows;
  const int64_t safe_row = active ? row : 0;
  const int64_t c0 = lane * R17_VECTOR;
  const int64_t c_stride = R17_SG_SIZE * R17_VECTOR;
  const u16 *xr = static_cast<const u16 *>(A.x) + safe_row * C;
  const u16 *wr = static_cast<const u16 *>(A.weight);
  const u16 *sr = static_cast<const u16 *>(A.scale);
  const u16 *hr = static_cast<const u16 *>(A.shift);
  u16 *orow = static_cast<u16 *>(A.out) + safe_row * C;

  const sycl::vec<u16, 8> first_vx = r8_vload(xr, c0, C);
  float first_values[8] = {};
  float sumsq = 0.f;
  if (active) {
    for (int k = 0; k < 8; ++k) {
      first_values[k] = r8_h2f(first_vx[k]);
      sumsq += first_values[k] * first_values[k];
    }
    for (int64_t offset = c_stride; offset < C; offset += c_stride) {
      const int64_t cc = c0 + offset;
      if (cc >= C) continue;
      const sycl::vec<u16, 8> vx = r8_vload(xr, cc, C);
      for (int k = 0; k < 8; ++k) {
        const float value = r8_h2f(vx[k]);
        sumsq += value * value;
      }
    }
  }
  const float total = sycl::reduce_over_group(sg, sumsq, sycl::plus<float>());
  if (!active) return;
  const float rstd = sycl::rsqrt(total / (float)C + A.eps);
  for (int64_t offset = 0; offset < C; offset += c_stride) {
    const int64_t cc = c0 + offset;
    if (cc >= C) continue;
    sycl::vec<u16, 8> vx;
    float values[8] = {};
    if (offset == 0) {
      vx = first_vx;
      for (int k = 0; k < 8; ++k) values[k] = first_values[k];
    } else {
      vx = r8_vload(xr, cc, C);
      for (int k = 0; k < 8; ++k) values[k] = r8_h2f(vx[k]);
    }
    const sycl::vec<u16, 8> vw = r8_vload(wr, cc, C);
    const sycl::vec<u16, 8> vs = r8_vload(sr, cc, C);
    const sycl::vec<u16, 8> vh = r8_vload(hr, cc, C);
    sycl::vec<u16, 8> vo;
    for (int k = 0; k < 8; ++k) {
      float norm = values[k] * rstd;
      norm *= r8_h2f(vw[k]);
      norm = r8_h2f(r8_f2h(norm));
      const float scale = r8_h2f(r8_f2h(1.f + r8_h2f(vs[k])));
      float scaled = norm * scale;
      scaled = r8_h2f(r8_f2h(scaled));
      const float shifted = scaled + r8_h2f(vh[k]);
      vo[k] = r8_f2h(shifted);
    }
    r8_vstore(orow, cc, C, vo);
  }
}

template <bool COMPAT>
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
    if constexpr (COMPAT) {
      // R31: round-0 split rope (mul output rounds, then addcmul_ rounds).
      float m0 = r8_h2f(r8_f2h(x0 * c));
      float m1 = r8_h2f(r8_f2h(x1 * c));
      orow[j] = r8_f2h((half == 0) ? (m0 + (-s) * x1) : (m1 + s * x0));
    } else {
      orow[j] = r8_f2h((half == 0) ? (x0 * c - x1 * s) : (x0 * s + x1 * c));
    }
  }
}

// R9D: vectorised split-rope (K3). Same math as k3_body, but processes 8 k
// lanes per iteration with u16x8 gathers/stores. Host gates on
// xs3==cs3==ss3==1, Dh%8==0 and 16B alignment of the row starts, so the
// scalar k3_body stays the universal fallback (LTX_R9D_K3V=0 too).
template <bool COMPAT>
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
    if constexpr (COMPAT) {
      // R31: round-0 split rope (mul output rounds, then addcmul_ rounds).
      float m0 = r8_h2f(r8_f2h(x0 * c));
      float m1 = r8_h2f(r8_f2h(x1 * c));
      o0[i] = r8_f2h(m0 + (-s) * x1);
      o1[i] = r8_f2h(m1 + s * x0);
    } else {
      o0[i] = r8_f2h(x0 * c - x1 * s);
      o1[i] = r8_f2h(x0 * s + x1 * c);
    }
  }
  *reinterpret_cast<sycl::vec<u16, 8> *>(orow + k) = o0;
  *reinterpret_cast<sycl::vec<u16, 8> *>(orow + Dh + k) = o1;
}

inline void k3_vec_dual_body(sycl::nd_item<1> it, R8K3DualArgs A) {
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
  const u16 *xq = static_cast<const u16 *>(A.xq);
  const u16 *xk = static_cast<const u16 *>(A.xk);
  const u16 *cos = static_cast<const u16 *>(A.cos);
  const u16 *sin = static_cast<const u16 *>(A.sin);
  const int64_t qxo = b * A.qxs0 + hh * A.qxs1 + t * A.qxs2;
  const int64_t kxo = b * A.kxs0 + hh * A.kxs1 + t * A.kxs2;
  const int64_t co = b * A.cs0 + hh * A.cs1 + t * A.cs2;
  const int64_t so = b * A.ss0 + hh * A.ss1 + t * A.ss2;
  u16 *oq = static_cast<u16 *>(A.outq) + fib * D;
  u16 *ok = static_cast<u16 *>(A.outk) + fib * D;
  const int64_t k = v * 8;
  if (A.prefetch != 0 && v + 1 < nvec) {
    r18_prefetch(cos + co + k + 8, 16);
    r18_prefetch(sin + so + k + 8, 16);
    r18_prefetch(xq + qxo + k + 8, 16);
    r18_prefetch(xq + qxo + k + Dh + 8, 16);
    r18_prefetch(xk + kxo + k + 8, 16);
    r18_prefetch(xk + kxo + k + Dh + 8, 16);
  }
  const sycl::vec<u16, 8> vq0 =
      *reinterpret_cast<const sycl::vec<u16, 8> *>(xq + qxo + k);
  const sycl::vec<u16, 8> vq1 =
      *reinterpret_cast<const sycl::vec<u16, 8> *>(xq + qxo + k + Dh);
  const sycl::vec<u16, 8> vk0 =
      *reinterpret_cast<const sycl::vec<u16, 8> *>(xk + kxo + k);
  const sycl::vec<u16, 8> vk1 =
      *reinterpret_cast<const sycl::vec<u16, 8> *>(xk + kxo + k + Dh);
  const sycl::vec<u16, 8> vc =
      *reinterpret_cast<const sycl::vec<u16, 8> *>(cos + co + k);
  const sycl::vec<u16, 8> vs =
      *reinterpret_cast<const sycl::vec<u16, 8> *>(sin + so + k);
  sycl::vec<u16, 8> oq0, oq1, ok0, ok1;
  for (int i = 0; i < 8; ++i) {
    const float c = r8_h2f(vc[i]);
    const float s = r8_h2f(vs[i]);
    const float q0 = r8_h2f(vq0[i]);
    const float q1 = r8_h2f(vq1[i]);
    const float k0 = r8_h2f(vk0[i]);
    const float k1 = r8_h2f(vk1[i]);
    oq0[i] = r8_f2h(q0 * c - q1 * s);
    oq1[i] = r8_f2h(q0 * s + q1 * c);
    ok0[i] = r8_f2h(k0 * c - k1 * s);
    ok1[i] = r8_f2h(k0 * s + k1 * c);
  }
  *reinterpret_cast<sycl::vec<u16, 8> *>(oq + k) = oq0;
  *reinterpret_cast<sycl::vec<u16, 8> *>(oq + Dh + k) = oq1;
  *reinterpret_cast<sycl::vec<u16, 8> *>(ok + k) = ok0;
  *reinterpret_cast<sycl::vec<u16, 8> *>(ok + Dh + k) = ok1;
}

// R10D F2: per-head broadcast gate multiply.
//   out[i] = attn[i] * gates[i / dh]   (bf16 in/out, fp32 product, one RNE)
// Bitwise identical to torch's `a.view(B,T,H,Dh) * gates.unsqueeze(-1)` for
// bf16 (TensorIterator opmath is fp32 with a single rounding). Host gates on
// contiguity, dh%8==0 and 16B alignment; anything else keeps the torch path.
inline void gate_mul_body(sycl::nd_item<1> it, R8GateArgs A) {
  const int64_t n = A.n;
  const int64_t dh = A.dh;
  const u16 *a = static_cast<const u16 *>(A.attn);
  const u16 *g = static_cast<const u16 *>(A.gates);
  u16 *o = static_cast<u16 *>(A.out);
  const int64_t gid = (int64_t)it.get_global_linear_id();
  const int64_t i0 = gid * R8_VEC;
  if (i0 >= n) return;
  const u16 gv = g[i0 / dh];
  sycl::vec<u16, 8> va = r8_vload(a, i0, n);
  sycl::vec<u16, 8> vo;
  for (int k = 0; k < 8; ++k) vo[k] = r8_f2h(r8_h2f(va[k]) * r8_h2f(gv));
  r8_vstore(o, i0, n, vo);
}

// R10D F3: tanh-approx GELU. Constants/order copied from torch's
// GeluTanhFunctor so the result is bitwise identical (fp32 opmath, one RNE).
inline float r10d_gelu_tanh(float x) {
  const float kBeta = (float)(1.4142135623730951 * 1.1283791670955126 * 0.5);  // M_SQRT2*M_2_SQRTPI*0.5
  constexpr float kKappa = 0.044715f;
  float x_cube = x * x * x;
  float inner = kBeta * (x + kKappa * x_cube);
  return 0.5f * x * (1.0f + sycl::tanh(inner));
}

inline void gelu_body(sycl::nd_item<1> it, R8GeluArgs A) {
  const int64_t n = A.n;
  const int64_t gid = (int64_t)it.get_global_linear_id();
  const int64_t i0 = gid * R8_VEC;
  if (i0 >= n) return;
  const u16 *x = static_cast<const u16 *>(A.x);
  u16 *o = static_cast<u16 *>(A.out);
  sycl::vec<u16, 8> vx = r8_vload(x, i0, n);
  sycl::vec<u16, 8> vo;
  for (int k = 0; k < 8; ++k) vo[k] = r8_f2h(r10d_gelu_tanh(r8_h2f(vx[k])));
  r8_vstore(o, i0, n, vo);
}

template <int CHUNKS>
inline void silu_mul_body(sycl::nd_item<1> it, R8SiluMulArgs A) {
  constexpr int VEC = 8;
  constexpr int TOTAL = VEC * CHUNKS;
  const int64_t n = A.n;
  const int64_t i0 = (int64_t)it.get_global_linear_id() * TOTAL;
  if (i0 >= n) return;
  const u16 *gate = static_cast<const u16 *>(A.gate);
  const u16 *up = static_cast<const u16 *>(A.up);
  u16 *out = static_cast<u16 *>(A.out);
  for (int c = 0; c < CHUNKS; ++c) {
    const int64_t offset = i0 + c * VEC;
    const sycl::vec<u16, VEC> vg = r8_vload(gate, offset, n);
    const sycl::vec<u16, VEC> vu = r8_vload(up, offset, n);
    sycl::vec<u16, VEC> vo;
    for (int k = 0; k < VEC; ++k) {
      const float g = r8_h2f(vg[k]);
      const float u = r8_h2f(vu[k]);
      const float s_bf16 = r8_h2f(r8_f2h(g / (1.0f + sycl::exp(-g))));
      vo[k] = r8_f2h(s_bf16 * u);
    }
    r8_vstore(out, offset, n, vo);
  }
}

// R21b compat: round-0 torch SiLU/up-product rounding (R31 recipe). Two RNE
// points, same as the eager path: silu_fp32 -> bf16 -> fp32, then
// (silu_bf16 * up_fp32) -> bf16. SiLU uses x*sigmoid(x) opmath to match the
// torch XPU SiluFunctor; NaN lanes canonicalise to 0xFFC1 like torch.
template <int CHUNKS>
inline void silu_mul_compat_body(sycl::nd_item<1> it, R8SiluMulArgs A) {
  constexpr int VEC = 8;
  constexpr int TOTAL = VEC * CHUNKS;
  const int64_t n = A.n;
  const int64_t i0 = (int64_t)it.get_global_linear_id() * TOTAL;
  if (i0 >= n) return;
  const u16 *gate = static_cast<const u16 *>(A.gate);
  const u16 *up = static_cast<const u16 *>(A.up);
  u16 *out = static_cast<u16 *>(A.out);
  for (int c = 0; c < CHUNKS; ++c) {
    const int64_t offset = i0 + c * VEC;
    const sycl::vec<u16, VEC> vg = r8_vload(gate, offset, n);
    const sycl::vec<u16, VEC> vu = r8_vload(up, offset, n);
    sycl::vec<u16, VEC> vo;
    for (int k = 0; k < VEC; ++k) {
      const float g = r8_h2f(vg[k]);
      const float u = r8_h2f(vu[k]);
      // R21b compat: exact-match torch XPU SiLU. The bf16 input 0x40BE /
      // 0x40BF (g = 5.9375) is the one finite bf16 value where a
      // < 2 ULP exp approximation rounds the fp32 SiLU product above the
      // bf16 midpoint while torch rounds it down (0x40BD = 5.90625).
      // Both bf16 keys hold the same fp32 value, so pin once on the fp32
      // value. Every other finite bf16 key is bitwise under exhaustive test.
      float silu;
      if (g == 5.9375f) {
        silu = r8_h2f((u16)0x40BD);
      } else {
        const float sigmoid = 1.0f / (1.0f + sycl::exp(-g));
        silu = r8_h2f(r8_f2h(g * sigmoid));
      }
      const float prod = silu * u;
      vo[k] = sycl::isnan(prod) ? (u16)0xFFC1 : r8_f2h(prod);
    }
    r8_vstore(out, offset, n, vo);
  }
}

// R40: 8-wide elementwise with a stride-broadcast second operand.  One work
// item covers 8 consecutive elements of the innermost contiguous run; the
// broadcast offset is resolved once per work item from the logical strides and
// then advanced by bs4 per lane, so a bs4 of 0 shares one scalar across the
// eight lanes and a bs4 of 1 reads eight consecutive values.  fp32 compute
// with a single round to bf16 keeps this bitwise with ATen's DivFunctor and
// MulFunctor.
inline void r40_bcast_body(sycl::nd_item<1> it, R8BBcastArgs A) {
  const int64_t total = A.d0 * A.d1 * A.d2 * A.d3 * A.inner;
  const int64_t i0 = (int64_t)it.get_global_linear_id() * 8;
  if (i0 >= total) return;
  const int64_t o = i0 / A.inner;
  const int64_t c = i0 - o * A.inner;
  int64_t q = o;
  const int64_t d3 = q % A.d3; q /= A.d3;
  const int64_t d2 = q % A.d2; q /= A.d2;
  const int64_t d1 = q % A.d1;
  const int64_t d0 = q / A.d1;
  const int64_t bo = d0 * A.bs0 + d1 * A.bs1 + d2 * A.bs2 + d3 * A.bs3;
  const u16 *x = static_cast<const u16 *>(A.x);
  const u16 *b = static_cast<const u16 *>(A.b);
  u16 *dst = static_cast<u16 *>(A.out);
  const bool full = (c + 8 <= A.inner);
  sycl::vec<u16, 8> vx;
  if (full) {
    vx = *reinterpret_cast<const sycl::vec<u16, 8> *>(x + i0);
  } else {
    for (int k = 0; k < 8; ++k) vx[k] = (c + k < A.inner) ? x[i0 + k] : (u16)0;
  }
  sycl::vec<u16, 8> vb;
  if (A.bs4 == 0) {
    const u16 sh = b[bo];
    for (int k = 0; k < 8; ++k) vb[k] = sh;
  } else {
    for (int k = 0; k < 8; ++k) vb[k] = b[bo + A.bs4 * (c + k)];
  }
  sycl::vec<u16, 8> vo;
  for (int k = 0; k < 8; ++k) {
    const float a = r8_h2f(vx[k]);
    const float s = r8_h2f(vb[k]);
    const float y = (A.op == 1) ? (a / s) : (a * s);
    vo[k] = r8_f2h(y);
  }
  if (full) {
    *reinterpret_cast<sycl::vec<u16, 8> *>(dst + i0) = vo;
  } else {
    for (int k = 0; k < 8; ++k)
      if (c + k < A.inner) dst[i0 + k] = vo[k];
  }
}


// Strided-x variant of the broadcast kernel: R8BBcastStridedKernel.
//
// Why this exists.  The wired VAE call site is
//     interior = workspace[:, :, 1:-1]
// which is a slice on the channel dimension, so interior.stride(0) is
// T*H*W*(C+2) rather than T*H*W*C.  R8BBcastKernel assumes x is one flat
// contiguous run in (b, t, h, w, c) order, so at that site it would read
// straight across the gap the slice leaves behind -- a genuine out-of-bounds
// read, caught by the host stride check only after the pipeline crashed.
//
// The fix is smaller than it looks, and the shape sweep is what shows it.
// The innermost 8 elements are still contiguous, because the channel dimension
// keeps stride 1 in a channels_last_3d tensor even when the outer strides carry
// a gap.  So the 16-byte vector load is still valid; only the outer addressing
// has to change from a flat index to a real stride sum:
//
//     offset = b*xs0 + t*xs2 + h*xs3 + w*xs4 + c        (xs1 == 1)
//
// Everything else, including the broadcast operand handling, is unchanged.
//
// Every index is bounded by construction: d0..d3 are the caller's extents and
// xs0..xs4 are the caller's own strides, so the largest address reachable is
// the sum over dims of (extent-1)*stride, which the host checks against
// x.numel() before the launch.
//
// This file is the EVOLVE band for the strided variant.  The bounds guard and
// the argument plumbing are fixed and stay outside it.
inline void r40_bcast_strided_body(sycl::nd_item<1> it, R8BBcastStridedArgs A) {
  const int64_t total = A.d0 * A.d1 * A.d2 * A.d3 * A.inner;
  const int64_t i0 = (int64_t)it.get_global_linear_id() * 8;
  if (i0 >= total) return;
  // ====================== EVOLVE_START ================================
  const int64_t o = i0 / A.inner;
  const int64_t c0 = i0 - o * A.inner;
  int64_t q = o;
  const int64_t d3 = q % A.d3; q /= A.d3;
  const int64_t d2 = q % A.d2; q /= A.d2;
  const int64_t d1 = q % A.d1;
  const int64_t d0 = q / A.d1;
  // outer base of this (d0,d1,d2,d3) block, in the caller's stride units
  const int64_t xo = d0 * A.xs0 + d1 * A.xs1 + d2 * A.xs2 + d3 * A.xs3;
  const int64_t bo = d0 * A.bs0 + d1 * A.bs1 + d2 * A.bs2 + d3 * A.bs3;
  const u16 *x = static_cast<const u16 *>(A.x);
  const u16 *b = static_cast<const u16 *>(A.b);
  u16 *dst = static_cast<u16 *>(A.out);
  const int64_t c = c0;
  const bool full = (c0 + 8 <= A.inner);
  sycl::vec<u16, 8> vx;
  if (full) {
    // still a 16-byte contiguous run: the channel dimension has stride 1
    vx = *reinterpret_cast<const sycl::vec<u16, 8> *>(x + xo + c);
  } else {
    for (int k = 0; k < 8; ++k) vx[k] = (c0 + k < A.inner) ? x[xo + c0 + k] : (u16)0;
  }
  sycl::vec<u16, 8> vb;
  if (A.bs4 == 0) {
    const u16 sh = b[bo];
    for (int k = 0; k < 8; ++k) vb[k] = sh;
  } else {
    for (int k = 0; k < 8; ++k) vb[k] = b[bo + A.bs4 * (c0 + k)];
  }
  sycl::vec<u16, 8> vo;
  for (int k = 0; k < 8; ++k) {
    const float a = r8_h2f(vx[k]);
    const float s = r8_h2f(vb[k]);
    const float y = (A.op == 1) ? (a / s) : (a * s);
    vo[k] = r8_f2h(y);
  }
  if (full) {
    *reinterpret_cast<sycl::vec<u16, 8> *>(dst + xo + c) = vo;
  } else {
    for (int k = 0; k < 8; ++k)
      if (c0 + k < A.inner) dst[xo + c0 + k] = vo[k];
  }
  // ======================= EVOLVE_END =================================
}


struct R8BBcastStridedKernel {
  R8BBcastStridedArgs A;
  void operator()(sycl::nd_item<1> it) const { r40_bcast_strided_body(it, A); }
};

struct R8BBcastKernel {
  R8BBcastArgs A;
  void operator()(sycl::nd_item<1> it) const { r40_bcast_body(it, A); }
};

// Named kernel functors: stable SPIR-V entry names, single POD struct arg.
struct R8K1AdaKernel {
  R8K1Args A;
  void operator()(sycl::nd_item<1> it) const { k1_body<true, false>(it, A); }
};

struct R8K1AffineKernel {
  R8K1Args A;
  void operator()(sycl::nd_item<1> it) const { k1_body<false, false>(it, A); }
};

// R31 output-audit compat: round-0 bf16 rounding points (torch single-RNE per
// op). Selected by the host when LTX_R31_COMPAT=1; current math otherwise.
struct R8K1AdaCompatKernel {
  R8K1Args A;
  void operator()(sycl::nd_item<1> it) const { k1_body<true, true>(it, A); }
};

// R41: two-row software-pipelined R31-compat ada.
struct R8K1AdaCompatPipeKernel {
  R8K1Args A;
  void operator()(sycl::nd_item<1> it,
                 sycl::local_accessor<float, 1> scratch) const {
    r41_k1_ada_compat_pipe_body<2>(it, A, scratch);
  }
};

// R41: two-row software-pipelined R31-compat ada, one vector per thread iteration.
struct R8K1AdaCompatPipe1Kernel {
  R8K1Args A;
  void operator()(sycl::nd_item<1> it,
                 sycl::local_accessor<float, 1> scratch) const {
    r41_k1_ada_compat_pipe_body<1>(it, A, scratch);
  }
};

struct R8K1AffineCompatKernel {
  R8K1Args A;
  void operator()(sycl::nd_item<1> it) const { k1_body<false, true>(it, A); }
};

// R41: R31-compat ada with the row rstd reduced through shared local memory.
struct R8K1AdaCompatLocalKernel {
  R8K1Args A;
  void operator()(sycl::nd_item<1> it,
                 sycl::local_accessor<float, 1> scratch) const {
    r41_k1_ada_compat_local_body(it, A, scratch);
  }
};

// R10D: single-pass ada (x held in registers across the rstd barrier).
struct R8K1AdaHoldKernel {
  R8K1Args A;
  void operator()(sycl::nd_item<1> it) const { k1_hold_body<true>(it, A); }
};

struct R8K1AdaLocalKernel {
  R8K1Args A;
  void operator()(sycl::nd_item<1> it,
                 sycl::local_accessor<float, 1> scratch) const {
    r18_k1_local_body<8>(it, A, scratch);
  }
};

struct R8K1AdaLocal16Kernel {
  R8K1Args A;
  void operator()(sycl::nd_item<1> it,
                 sycl::local_accessor<float, 1> scratch) const {
    r18_k1_local_body<16>(it, A, scratch);
  }
};

struct R8K1AdaReg1Kernel {
  R8K1Args A;
  void operator()(sycl::nd_item<1> it,
                 sycl::local_accessor<float, 1> scratch) const {
    r18_k1_reg_body<8, 1>(it, A, scratch);
  }
};

struct R8K1AdaReg2Kernel {
  R8K1Args A;
  void operator()(sycl::nd_item<1> it,
                 sycl::local_accessor<float, 1> scratch) const {
    r18_k1_reg_body<8, 2>(it, A, scratch);
  }
};

struct R8K1AdaSubgroupKernel {
  R8K1Args A;
  void operator()(sycl::nd_item<1> it) const { r18_k1_subgroup_body<8>(it, A); }
};

struct R8K1AdaSubgroup16Kernel {
  R8K1Args A;
  void operator()(sycl::nd_item<1> it) const { r18_k1_subgroup_body<16>(it, A); }
};

struct R8K1AdaGroupReduceKernel {
  R8K1Args A;
  void operator()(sycl::nd_item<1> it) const { r18_k1_group_reduce_body<8>(it, A); }
};

struct R8K1AdaGroupReduce16Kernel {
  R8K1Args A;
  void operator()(sycl::nd_item<1> it) const { r18_k1_group_reduce_body<16>(it, A); }
};

struct R8K2Kernel {
  R8K2Args A;
  void operator()(sycl::nd_item<1> it) const { k2_body<false>(it, A); }
};

// R31 output-audit compat (round-0 `x + y * gate` rounding).
struct R8K2CompatKernel {
  R8K2Args A;
  void operator()(sycl::nd_item<1> it) const { k2_body<true>(it, A); }
};

struct R8R17MlpNormAffineKernel {
  R8R17MlpNormAffineArgs A;
  void operator()(sycl::nd_item<1> it) const { r17_mlp_norm_affine_body(it, A); }
};


struct R8R19NormRopeKernel {
  R8R19NormRopeArgs A;
  void operator()(sycl::nd_item<1> it) const { r19_norm_rope_body(it, A); }
};

struct R8K3Kernel {
  R8K3Args A;
  void operator()(sycl::nd_item<1> it) const { k3_body<false>(it, A); }
};

struct R8K3VecKernel {
  R8K3Args A;
  void operator()(sycl::nd_item<1> it) const { k3_vec_body<false>(it, A); }
};

// R31 output-audit compat (round-0 split-rope mul/addcmul_ rounding).
struct R8K3CompatKernel {
  R8K3Args A;
  void operator()(sycl::nd_item<1> it) const { k3_body<true>(it, A); }
};

struct R8K3VecCompatKernel {
  R8K3Args A;
  void operator()(sycl::nd_item<1> it) const { k3_vec_body<true>(it, A); }
};

struct R8K3VecDualKernel {
  R8K3DualArgs A;
  void operator()(sycl::nd_item<1> it) const { k3_vec_dual_body(it, A); }
};

// R10D F2
struct R8GateKernel {
  R8GateArgs A;
  void operator()(sycl::nd_item<1> it) const { gate_mul_body(it, A); }
};

// R10D F3
struct R8GeluKernel {
  R8GeluArgs A;
  void operator()(sycl::nd_item<1> it) const { gelu_body(it, A); }
};

struct R8SiluMulKernel {
  R8SiluMulArgs A;
  void operator()(sycl::nd_item<1> it) const { silu_mul_body<8>(it, A); }
};

struct R8SiluMulKernelV16 {
  R8SiluMulArgs A;
  void operator()(sycl::nd_item<1> it) const { silu_mul_body<2>(it, A); }
};

struct R8SiluMulKernelV32 {
  R8SiluMulArgs A;
  void operator()(sycl::nd_item<1> it) const { silu_mul_body<4>(it, A); }
};

struct R8SiluMulCompatKernel {
  R8SiluMulArgs A;
  void operator()(sycl::nd_item<1> it) const { silu_mul_compat_body<8>(it, A); }
};

struct R8SiluMulCompatKernelV16 {
  R8SiluMulArgs A;
  void operator()(sycl::nd_item<1> it) const { silu_mul_compat_body<2>(it, A); }
};

struct R8SiluMulCompatKernelV32 {
  R8SiluMulArgs A;
  void operator()(sycl::nd_item<1> it) const { silu_mul_compat_body<4>(it, A); }
};


// R9A: fp8 e4m3fn -> bf16 widening. LUT is bitwise-identical to torch
// float8_e4m3fn.to(bfloat16) for all 256 byte values (NaN canonicalised to
// 0x7FC0), verified exhaustively.
static constexpr u16 R9A_LUT[256] = {
  0x0000, 0x3B00, 0x3B80, 0x3BC0, 0x3C00, 0x3C20, 0x3C40, 0x3C60,
  0x3C80, 0x3C90, 0x3CA0, 0x3CB0, 0x3CC0, 0x3CD0, 0x3CE0, 0x3CF0,
  0x3D00, 0x3D10, 0x3D20, 0x3D30, 0x3D40, 0x3D50, 0x3D60, 0x3D70,
  0x3D80, 0x3D90, 0x3DA0, 0x3DB0, 0x3DC0, 0x3DD0, 0x3DE0, 0x3DF0,
  0x3E00, 0x3E10, 0x3E20, 0x3E30, 0x3E40, 0x3E50, 0x3E60, 0x3E70,
  0x3E80, 0x3E90, 0x3EA0, 0x3EB0, 0x3EC0, 0x3ED0, 0x3EE0, 0x3EF0,
  0x3F00, 0x3F10, 0x3F20, 0x3F30, 0x3F40, 0x3F50, 0x3F60, 0x3F70,
  0x3F80, 0x3F90, 0x3FA0, 0x3FB0, 0x3FC0, 0x3FD0, 0x3FE0, 0x3FF0,
  0x4000, 0x4010, 0x4020, 0x4030, 0x4040, 0x4050, 0x4060, 0x4070,
  0x4080, 0x4090, 0x40A0, 0x40B0, 0x40C0, 0x40D0, 0x40E0, 0x40F0,
  0x4100, 0x4110, 0x4120, 0x4130, 0x4140, 0x4150, 0x4160, 0x4170,
  0x4180, 0x4190, 0x41A0, 0x41B0, 0x41C0, 0x41D0, 0x41E0, 0x41F0,
  0x4200, 0x4210, 0x4220, 0x4230, 0x4240, 0x4250, 0x4260, 0x4270,
  0x4280, 0x4290, 0x42A0, 0x42B0, 0x42C0, 0x42D0, 0x42E0, 0x42F0,
  0x4300, 0x4310, 0x4320, 0x4330, 0x4340, 0x4350, 0x4360, 0x4370,
  0x4380, 0x4390, 0x43A0, 0x43B0, 0x43C0, 0x43D0, 0x43E0, 0xFFC1,
  0x8000, 0xBB00, 0xBB80, 0xBBC0, 0xBC00, 0xBC20, 0xBC40, 0xBC60,
  0xBC80, 0xBC90, 0xBCA0, 0xBCB0, 0xBCC0, 0xBCD0, 0xBCE0, 0xBCF0,
  0xBD00, 0xBD10, 0xBD20, 0xBD30, 0xBD40, 0xBD50, 0xBD60, 0xBD70,
  0xBD80, 0xBD90, 0xBDA0, 0xBDB0, 0xBDC0, 0xBDD0, 0xBDE0, 0xBDF0,
  0xBE00, 0xBE10, 0xBE20, 0xBE30, 0xBE40, 0xBE50, 0xBE60, 0xBE70,
  0xBE80, 0xBE90, 0xBEA0, 0xBEB0, 0xBEC0, 0xBED0, 0xBEE0, 0xBEF0,
  0xBF00, 0xBF10, 0xBF20, 0xBF30, 0xBF40, 0xBF50, 0xBF60, 0xBF70,
  0xBF80, 0xBF90, 0xBFA0, 0xBFB0, 0xBFC0, 0xBFD0, 0xBFE0, 0xBFF0,
  0xC000, 0xC010, 0xC020, 0xC030, 0xC040, 0xC050, 0xC060, 0xC070,
  0xC080, 0xC090, 0xC0A0, 0xC0B0, 0xC0C0, 0xC0D0, 0xC0E0, 0xC0F0,
  0xC100, 0xC110, 0xC120, 0xC130, 0xC140, 0xC150, 0xC160, 0xC170,
  0xC180, 0xC190, 0xC1A0, 0xC1B0, 0xC1C0, 0xC1D0, 0xC1E0, 0xC1F0,
  0xC200, 0xC210, 0xC220, 0xC230, 0xC240, 0xC250, 0xC260, 0xC270,
  0xC280, 0xC290, 0xC2A0, 0xC2B0, 0xC2C0, 0xC2D0, 0xC2E0, 0xC2F0,
  0xC300, 0xC310, 0xC320, 0xC330, 0xC340, 0xC350, 0xC360, 0xC370,
  0xC380, 0xC390, 0xC3A0, 0xC3B0, 0xC3C0, 0xC3D0, 0xC3E0, 0xFFC1,
};

inline void fp8_widen_body(sycl::nd_item<1> it, R8Fp8Args A) {
  const int64_t n = A.n;
  const int64_t i0 = (int64_t)it.get_global_linear_id() * 8;
  if (i0 >= n) return;
  const uint8_t *in = static_cast<const uint8_t *>(A.in);
  u16 *out = static_cast<u16 *>(A.out);
  if (i0 + 8 <= n) {
    sycl::vec<uint8_t, 8> v = *reinterpret_cast<const sycl::vec<uint8_t, 8> *>(in + i0);
    sycl::vec<u16, 8> o;
#pragma unroll
    for (int k = 0; k < 8; ++k) o[k] = R9A_LUT[v[k]];
    *reinterpret_cast<sycl::vec<u16, 8> *>(out + i0) = o;
  } else {
    for (int64_t j = i0; j < n; ++j) out[j] = R9A_LUT[in[j]];
  }
}

struct R8Fp8WidenKernel {
  R8Fp8Args A;
  void operator()(sycl::nd_item<1> it) const { fp8_widen_body(it, A); }
};

inline void fp8_widen_eff_body(sycl::nd_item<1> it, R8Fp8EffArgs A) {
  const int64_t n = A.n;
  const int64_t i0 = (int64_t)it.get_global_linear_id() * 8;
  if (i0 >= n) return;
  const uint8_t *in = static_cast<const uint8_t *>(A.in);
  u16 *out = static_cast<u16 *>(A.out);
  if (A.prefetch != 0 && i0 + 16 <= n) r18_prefetch(in + i0 + 8, 8);
  if (i0 + 8 <= n) {
    const sycl::vec<uint8_t, 8> v =
        *reinterpret_cast<const sycl::vec<uint8_t, 8> *>(in + i0);
    sycl::vec<u16, 8> o;
#pragma unroll
    for (int k = 0; k < 8; ++k) o[k] = R9A_LUT[v[k]];
    *reinterpret_cast<sycl::vec<u16, 8> *>(out + i0) = o;
  } else {
    for (int64_t j = i0; j < n; ++j) out[j] = R9A_LUT[in[j]];
  }
}

struct R8Fp8WidenEffKernel {
  R8Fp8EffArgs A;
  void operator()(sycl::nd_item<1> it) const { fp8_widen_eff_body(it, A); }
};

inline u16 r18_fp8_bits_to_bf16(uint8_t value) {
  const u16 sign = static_cast<u16>((value & 0x80u) << 8);
  const u32 exponent = (value >> 3) & 0x0fu;
  const u32 mantissa = value & 0x07u;
  if (exponent == 0) {
    if (mantissa == 0) return sign;
    if ((mantissa & 0x04u) != 0)
      return sign | (120u << 7) | ((mantissa & 0x03u) << 5);
    if ((mantissa & 0x02u) != 0)
      return sign | (119u << 7) | ((mantissa & 0x01u) << 6);
    return sign | (118u << 7);
  }
  if (exponent == 15 && mantissa == 7) return 0xFFC1u;
  return sign | ((0x3C00u + (exponent << 7)) | (mantissa << 4));
}

inline void fp8_widen_bits_body(sycl::nd_item<1> it, R8Fp8EffArgs A) {
  const int64_t n = A.n;
  const int64_t i0 = (int64_t)it.get_global_linear_id() * 8;
  if (i0 >= n) return;
  const uint8_t *in = static_cast<const uint8_t *>(A.in);
  u16 *out = static_cast<u16 *>(A.out);
  if (A.prefetch != 0 && i0 + 16 <= n) r18_prefetch(in + i0 + 8, 8);
  if (i0 + 8 <= n) {
    const sycl::vec<uint8_t, 8> values =
        *reinterpret_cast<const sycl::vec<uint8_t, 8> *>(in + i0);
    sycl::vec<u16, 8> converted;
#pragma unroll
    for (int k = 0; k < 8; ++k) converted[k] = r18_fp8_bits_to_bf16(values[k]);
    *reinterpret_cast<sycl::vec<u16, 8> *>(out + i0) = converted;
  } else {
    for (int64_t j = i0; j < n; ++j)
      out[j] = r18_fp8_bits_to_bf16(in[j]);
  }
}

struct R8Fp8WidenBitsKernel {
  R8Fp8EffArgs A;
  void operator()(sycl::nd_item<1> it) const { fp8_widen_bits_body(it, A); }
};

inline void fp8_widen_batch_body(sycl::nd_item<1> it, R8Fp8BatchArgs A) {
  const int64_t i0 = (int64_t)it.get_global_linear_id() * 8;
  if (i0 >= A.total) return;
  int64_t local = i0;
  int64_t d = 0;
  for (; d + 1 < A.count; ++d) {
    if (local < A.desc[d].n) break;
    local -= A.desc[d].n;
  }
  const R8Fp8BatchDesc desc = A.desc[d];
  const uint8_t *in = static_cast<const uint8_t *>(desc.in);
  u16 *out = static_cast<u16 *>(desc.out);
  if (A.prefetch != 0 && local + 16 <= desc.n) r18_prefetch(in + local + 8, 8);
  if (local + 8 <= desc.n) {
    const sycl::vec<uint8_t, 8> v =
        *reinterpret_cast<const sycl::vec<uint8_t, 8> *>(in + local);
    sycl::vec<u16, 8> o;
#pragma unroll
    for (int k = 0; k < 8; ++k) o[k] = r18_fp8_bits_to_bf16(v[k]);
    *reinterpret_cast<sycl::vec<u16, 8> *>(out + local) = o;
  } else {
    for (int64_t j = local; j < desc.n; ++j)
      out[j] = r18_fp8_bits_to_bf16(in[j]);
  }
}

struct R8Fp8WidenBatchKernel {
  R8Fp8BatchArgs A;
  void operator()(sycl::nd_item<1> it) const { fp8_widen_batch_body(it, A); }
};


// R9A v2: 16-wide widening (two 8-wide vectors; keeps 16-byte alignment).
inline void fp8_widen16_body(sycl::nd_item<1> it, R8Fp8Args A) {
  const int64_t n = A.n;
  const int64_t i0 = (int64_t)it.get_global_linear_id() * 16;
  if (i0 >= n) return;
  const uint8_t *in = static_cast<const uint8_t *>(A.in);
  u16 *out = static_cast<u16 *>(A.out);
  if (i0 + 16 <= n) {
    sycl::vec<uint8_t, 8> v0 = *reinterpret_cast<const sycl::vec<uint8_t, 8> *>(in + i0);
    sycl::vec<uint8_t, 8> v1 = *reinterpret_cast<const sycl::vec<uint8_t, 8> *>(in + i0 + 8);
    sycl::vec<u16, 8> o0, o1;
#pragma unroll
    for (int k = 0; k < 8; ++k) o0[k] = R9A_LUT[v0[k]];
#pragma unroll
    for (int k = 0; k < 8; ++k) o1[k] = R9A_LUT[v1[k]];
    *reinterpret_cast<sycl::vec<u16, 8> *>(out + i0) = o0;
    *reinterpret_cast<sycl::vec<u16, 8> *>(out + i0 + 8) = o1;
  } else {
    for (int64_t j = i0; j < n; ++j) out[j] = R9A_LUT[in[j]];
  }
}

struct R8Fp8Widen16Kernel {
  R8Fp8Args A;
  void operator()(sycl::nd_item<1> it) const { fp8_widen16_body(it, A); }
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
  R8BBcastArgs a40{};
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::nd_range<1>(R8_WG, R8_WG), R8BBcastKernel{a40});
  });
  R8BBcastStridedArgs a41{};
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::nd_range<1>(R8_WG, R8_WG), R8BBcastStridedKernel{a41});
  });
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::nd_range<1>(R8_WG, R8_WG), R8K1AdaCompatKernel{a1});
  });
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::nd_range<1>(R8_WG, R8_WG), R8K1AffineCompatKernel{a1});
  });
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> scratch(sycl::range<1>(R8_WG), h);
    h.parallel_for<R8K1AdaCompatLocalKernel>(
        sycl::nd_range<1>(R8_WG, R8_WG), [=](sycl::nd_item<1> it) {
          r41_k1_ada_compat_local_body(it, a1, scratch);
        });
  });
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> scratch(sycl::range<1>(2 * R8_WG), h);
    h.parallel_for<R8K1AdaCompatPipeKernel>(
        sycl::nd_range<1>(R8_WG, R8_WG), [=](sycl::nd_item<1> it) {
          r41_k1_ada_compat_pipe_body<2>(it, a1, scratch);
        });
  });
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> scratch(sycl::range<1>(2 * R8_WG), h);
    h.parallel_for<R8K1AdaCompatPipe1Kernel>(
        sycl::nd_range<1>(R8_WG, R8_WG), [=](sycl::nd_item<1> it) {
          r41_k1_ada_compat_pipe_body<1>(it, a1, scratch);
        });
  });
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::nd_range<1>(R8_WG, R8_WG), R8K1AdaHoldKernel{a1});
  });
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> scratch(sycl::range<1>(R8_WG), h);
    h.parallel_for<R8K1AdaLocalKernel>(
        sycl::nd_range<1>(R8_WG, R8_WG), [=](sycl::nd_item<1> it) {
          r18_k1_local_body<8>(it, a1, scratch);
        });
  });
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> scratch(sycl::range<1>(R8_WG), h);
    h.parallel_for<R8K1AdaLocal16Kernel>(
        sycl::nd_range<1>(R8_WG, R8_WG), [=](sycl::nd_item<1> it) {
          r18_k1_local_body<16>(it, a1, scratch);
        });
  });
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> scratch(sycl::range<1>(R8_WG), h);
    h.parallel_for<R8K1AdaReg1Kernel>(
        sycl::nd_range<1>(R8_WG, R8_WG), [=](sycl::nd_item<1> it) {
          r18_k1_reg_body<8, 1>(it, a1, scratch);
        });
  });
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> scratch(sycl::range<1>(R8_WG), h);
    h.parallel_for<R8K1AdaReg2Kernel>(
        sycl::nd_range<1>(R8_WG, R8_WG), [=](sycl::nd_item<1> it) {
          r18_k1_reg_body<8, 2>(it, a1, scratch);
        });
  });
  q.submit([&](sycl::handler &h) {
    h.parallel_for<R8K1AdaSubgroupKernel>(
        sycl::nd_range<1>(R8_WG, R8_WG), [=](sycl::nd_item<1> it) {
          r18_k1_subgroup_body<8>(it, a1);
        });
  });
  q.submit([&](sycl::handler &h) {
    h.parallel_for<R8K1AdaSubgroup16Kernel>(
        sycl::nd_range<1>(R8_WG, R8_WG), [=](sycl::nd_item<1> it) {
          r18_k1_subgroup_body<16>(it, a1);
        });
  });
  q.submit([&](sycl::handler &h) {
    h.parallel_for<R8K1AdaGroupReduceKernel>(
        sycl::nd_range<1>(R8_WG, R8_WG), [=](sycl::nd_item<1> it) {
          r18_k1_group_reduce_body<8>(it, a1);
        });
  });
  q.submit([&](sycl::handler &h) {
    h.parallel_for<R8K1AdaGroupReduce16Kernel>(
        sycl::nd_range<1>(R8_WG, R8_WG), [=](sycl::nd_item<1> it) {
          r18_k1_group_reduce_body<16>(it, a1);
        });
  });
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::nd_range<1>(R8_WG, R8_WG), R8K2Kernel{a2});
  });
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::nd_range<1>(R8_WG, R8_WG), R8K2CompatKernel{a2});
  });
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::nd_range<1>(16, 16), R8K3Kernel{a3});
  });
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::nd_range<1>(64, 64), R8K3VecKernel{a3});
  });
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::nd_range<1>(16, 16), R8K3CompatKernel{a3});
  });
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::nd_range<1>(64, 64), R8K3VecCompatKernel{a3});
  });
  R8K3DualArgs a3d{};
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::nd_range<1>(64, 64), R8K3VecDualKernel{a3d});
  });
  R8R17MlpNormAffineArgs a8{};
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::nd_range<1>(R8_WG, R8_WG), R8R17MlpNormAffineKernel{a8});
  });
  R8R19NormRopeArgs a19{};
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::nd_range<1>(R8_WG, R8_WG), R8R19NormRopeKernel{a19});
  });
  R8GateArgs a4{};
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::nd_range<1>(R8_WG, R8_WG), R8GateKernel{a4});
  });
  R8GeluArgs a5{};
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::nd_range<1>(R8_WG, R8_WG), R8GeluKernel{a5});
  });
  R8SiluMulArgs a5s{};
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::nd_range<1>(R8_WG, R8_WG), R8SiluMulKernel{a5s});
  });
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::nd_range<1>(R8_WG, R8_WG), R8SiluMulKernelV16{a5s});
  });
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::nd_range<1>(R8_WG, R8_WG), R8SiluMulKernelV32{a5s});
  });
  R8SiluMulArgs a5c{};
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::nd_range<1>(R8_WG, R8_WG), R8SiluMulCompatKernel{a5c});
  });
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::nd_range<1>(R8_WG, R8_WG), R8SiluMulCompatKernelV16{a5c});
  });
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::nd_range<1>(R8_WG, R8_WG), R8SiluMulCompatKernelV32{a5c});
  });
  R8Fp8Args a6{};
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::nd_range<1>(R8_WG, R8_WG), R8Fp8WidenKernel{a6});
  });
  R8Fp8EffArgs a6e{};
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::nd_range<1>(R8_WG, R8_WG), R8Fp8WidenEffKernel{a6e});
  });
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::nd_range<1>(R8_WG, R8_WG), R8Fp8WidenBitsKernel{a6e});
  });
  R8Fp8BatchArgs a6b{};
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::nd_range<1>(R8_WG, R8_WG), R8Fp8WidenBatchKernel{a6b});
  });
  R8Fp8Args a7{};
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::nd_range<1>(R8_WG, R8_WG), R8Fp8Widen16Kernel{a7});
  });
}
