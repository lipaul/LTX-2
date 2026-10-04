// Device kernels (SYCL), compiled device-only to SPIR-V and launched from
// the host via L0-interop on torch's own queue.
// Loads and stores are u16x8 vectors with a manual RNE convert per lane and no
// __devicelib imports; the post-attention gate+norm kernel holds fma in
// registers across the rstd barrier (single x/y/gate read pass: 4R+2W).
#include <sycl/sycl.hpp>

#include "kernel_args.h"

using u16 = uint16_t;
using u32 = uint32_t;

constexpr int WG_SIZE = 256;
constexpr int VEC8 = 8;
constexpr int MAX_SUBGROUPS = 32;
constexpr int MAX_ITERS = 4;  // C <= 8192 => iters <= 4
constexpr int64_t ADA_NORM_AFFINE_MAX_C = 8192;

inline float bf16_to_float(u16 h) {
  return sycl::bit_cast<float>((u32)h << 16);
}

inline u16 float_to_bf16(float x) {
  u32 u = sycl::bit_cast<u32>(x);
  if ((u & 0x7FFFFFFFu) > 0x7F800000u) return (u16)0xFFC1u;  // XPU torch canonical bf16 NaN
  u32 bias = 0x7FFFu + ((u >> 16) & 1u);
  return (u16)((u + bias) >> 16);
}

inline sycl::vec<u16, 8> vload8(const u16 *p, int64_t c0, int64_t C) {
  if (c0 + 8 <= C) return *reinterpret_cast<const sycl::vec<u16, 8> *>(p + c0);
  sycl::vec<u16, 8> v;
  for (int k = 0; k < 8; ++k) v[k] = (c0 + k < C) ? p[c0 + k] : (u16)0;
  return v;
}

inline void vstore8(u16 *p, int64_t c0, int64_t C, sycl::vec<u16, 8> v) {
  if (c0 + 8 <= C) {
    *reinterpret_cast<sycl::vec<u16, 8> *>(p + c0) = v;
  } else {
    for (int k = 0; k < 8; ++k)
      if (c0 + k < C) p[c0 + k] = v[k];
  }
}

template <int N>
inline sycl::vec<u16, N> vload_n(const u16 *p, int64_t c0, int64_t C) {
  if (c0 + N <= C) return *reinterpret_cast<const sycl::vec<u16, N> *>(p + c0);
  sycl::vec<u16, N> v;
  for (int k = 0; k < N; ++k) v[k] = (c0 + k < C) ? p[c0 + k] : (u16)0;
  return v;
}

template <int N>
inline void vstore_n(u16 *p, int64_t c0, int64_t C, sycl::vec<u16, N> v) {
  if (c0 + N <= C) {
    *reinterpret_cast<sycl::vec<u16, N> *>(p + c0) = v;
  } else {
    for (int k = 0; k < N; ++k)
      if (c0 + k < C) p[c0 + k] = v[k];
  }
}

template <typename T>
inline void prefetch_range(const T *p, size_t bytes) {
  sycl::ext::oneapi::experimental::prefetch(const_cast<T *>(p), bytes);
}

// Row rstd without SLM: subgroup reduce -> one float per subgroup in global
// scratch -> lid 0 sums -> broadcast (default global+local fence).
inline float row_rstd(sycl::nd_item<1> it, float sumsq, float *scr, float C,
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
inline void ada_norm_affine_body(sycl::nd_item<1> it, AdaNormAffineArgs A) {
  const int64_t row = (int64_t)it.get_group_linear_id();
  const int64_t lid = (int64_t)it.get_local_linear_id();
  const int64_t C = A.C;
  const int64_t stride = (int64_t)WG_SIZE * VEC8;
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
    for (int64_t c0 = lid * VEC8; c0 < C; c0 += stride) {
      sycl::vec<u16, 8> vx = vload8(xr, c0, C);
      for (int k = 0; k < 8; ++k) {
        float v = bf16_to_float(vx[k]);
        sumsq += v * v;
      }
    }
  }
  float *scr = static_cast<float *>(A.scratch) + row * MAX_SUBGROUPS;
  const float rstd = row_rstd(it, sumsq, scr, (float)C, A.eps, DO_NORM);
  for (int64_t c0 = lid * VEC8; c0 < C; c0 += stride) {
    sycl::vec<u16, 8> vx = vload8(xr, c0, C);
    sycl::vec<u16, 8> vs = vload8(tbs, c0, C);
    sycl::vec<u16, 8> vh = vload8(tbh, c0, C);
    sycl::vec<u16, 8> vss = vload8(tss_r, c0, C);
    sycl::vec<u16, 8> vsh = vload8(tsh_r, c0, C);
    sycl::vec<u16, 8> vo;
    for (int k = 0; k < 8; ++k) {
      float xn = bf16_to_float(vx[k]);
      float sc = bf16_to_float(vs[k]) + bf16_to_float(vss[k]);
      float sh = bf16_to_float(vh[k]) + bf16_to_float(vsh[k]);
      float y;
      if constexpr (COMPAT) {
        // Torch reference rounding: each torch bf16 op rounds once (RNE).
        float scb = bf16_to_float(float_to_bf16(sc));
        float shb = bf16_to_float(float_to_bf16(sh));
        float nn = DO_NORM ? bf16_to_float(float_to_bf16(xn * rstd)) : xn;
        float one = bf16_to_float(float_to_bf16(1.f + scb));
        float pr = bf16_to_float(float_to_bf16(nn * one));
        y = bf16_to_float(float_to_bf16(pr + shb));
      } else {
        y = sh + (DO_NORM ? xn * rstd : xn) * (1.f + sc);
      }
      vo[k] = float_to_bf16(y);
    }
    vstore8(orow, c0, C, vo);
  }
}


inline float row_rstd_local(sycl::nd_item<1> it, float sumsq,
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


inline void row_rstd_pair(sycl::nd_item<1> it, float qsumsq, float ksumsq,
                              float *scratch, int64_t row, float C, float eps,
                              float &q_rstd, float &k_rstd) {
  sycl::sub_group sg = it.get_sub_group();
  const uint32_t sgid = sg.get_group_linear_id();
  const float qpart = sycl::reduce_over_group(sg, qsumsq, sycl::plus<float>());
  const float kpart = sycl::reduce_over_group(sg, ksumsq, sycl::plus<float>());
  float *row_scratch = scratch + row * 2 * MAX_SUBGROUPS;
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

template <bool COMPAT>
inline void norm_rope_body(sycl::nd_item<1> it, NormRopeArgs A) {
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
    const sycl::vec<u16, VEC> vq = vload_n<VEC>(qrow, c0, C);
    const sycl::vec<u16, VEC> vk = vload_n<VEC>(krow, c0, C);
    for (int i = 0; i < VEC; ++i) {
      const float qv = bf16_to_float(vq[i]);
      const float kv = bf16_to_float(vk[i]);
      qsumsq += qv * qv;
      ksumsq += kv * kv;
    }
  }
  float q_rstd;
  float k_rstd;
  row_rstd_pair(it, qsumsq, ksumsq, static_cast<float *>(A.scratch), row,
                    static_cast<float>(C), A.eps, q_rstd, k_rstd);
  for (int64_t c0 = lid * VEC; c0 < C; c0 += stride) {
    const int64_t h = c0 / D;
    const int64_t local = c0 - h * D;
    const int64_t half = local / A.Dh;
    const int64_t k = local - half * A.Dh;
    const int64_t pair_offset = half == 0 ? A.Dh : -A.Dh;
    const int64_t co = b * A.cos_stride0 + h * A.cos_stride1 + t * A.cos_stride2;
    const int64_t so = b * A.sin_stride0 + h * A.sin_stride1 + t * A.sin_stride2;
    const sycl::vec<u16, VEC> vq = vload_n<VEC>(qrow, c0, C);
    const sycl::vec<u16, VEC> vk = vload_n<VEC>(krow, c0, C);
    const sycl::vec<u16, VEC> vq_pair = vload_n<VEC>(qrow, c0 + pair_offset, C);
    const sycl::vec<u16, VEC> vk_pair = vload_n<VEC>(krow, c0 + pair_offset, C);
    const sycl::vec<u16, VEC> vqw = vload_n<VEC>(qw, c0, C);
    const sycl::vec<u16, VEC> vkw = vload_n<VEC>(kw, c0, C);
    const sycl::vec<u16, VEC> vqw_pair = vload_n<VEC>(qw, c0 + pair_offset, C);
    const sycl::vec<u16, VEC> vkw_pair = vload_n<VEC>(kw, c0 + pair_offset, C);
    const sycl::vec<u16, VEC> vc =
        *reinterpret_cast<const sycl::vec<u16, VEC> *>(cos + co + k * A.cos_stride3);
    const sycl::vec<u16, VEC> vs =
        *reinterpret_cast<const sycl::vec<u16, VEC> *>(sin + so + k * A.sin_stride3);
    sycl::vec<u16, VEC> oq, ok;
    for (int i = 0; i < VEC; ++i) {
      const float c = bf16_to_float(vc[i]);
      const float s = bf16_to_float(vs[i]);
      float q0 = bf16_to_float(vq[i]) * q_rstd;
      q0 *= bf16_to_float(vqw[i]);
      q0 = bf16_to_float(float_to_bf16(q0));
      float q1 = bf16_to_float(vq_pair[i]) * q_rstd;
      q1 *= bf16_to_float(vqw_pair[i]);
      q1 = bf16_to_float(float_to_bf16(q1));
      float k0 = bf16_to_float(vk[i]) * k_rstd;
      k0 *= bf16_to_float(vkw[i]);
      k0 = bf16_to_float(float_to_bf16(k0));
      float k1 = bf16_to_float(vk_pair[i]) * k_rstd;
      k1 *= bf16_to_float(vkw_pair[i]);
      k1 = bf16_to_float(float_to_bf16(k1));
      const float qs0 = bf16_to_float(float_to_bf16(q0 * A.q_scale));
      const float qs1 = bf16_to_float(float_to_bf16(q1 * A.q_scale));
      const float ks0 = bf16_to_float(float_to_bf16(k0 * A.k_scale));
      const float ks1 = bf16_to_float(float_to_bf16(k1 * A.k_scale));
      if constexpr (COMPAT) {
        // Torch reference rounding for split rope: mul output rounds, then addcmul_ rounds.
        const float qm = bf16_to_float(float_to_bf16(qs0 * c));
        const float km = bf16_to_float(float_to_bf16(ks0 * c));
        oq[i] = float_to_bf16(half == 0 ? (qm + (-s) * qs1) : (qm + s * qs1));
        ok[i] = float_to_bf16(half == 0 ? (km + (-s) * ks1) : (km + s * ks1));
      } else {
        oq[i] = half == 0 ? float_to_bf16(qs0 * c - qs1 * s) : float_to_bf16(qs1 * s + qs0 * c);
        ok[i] = half == 0 ? float_to_bf16(ks0 * c - ks1 * s) : float_to_bf16(ks1 * s + ks0 * c);
      }
    }
    vstore_n<VEC>(outq, c0, C, oq);
    vstore_n<VEC>(outk, c0, C, ok);
  }
}





template <bool COMPAT>
inline void post_attn_gate_norm_body(sycl::nd_item<1> it, PostAttnGateNormArgs A) {
  const int64_t row = (int64_t)it.get_group_linear_id();
  const int64_t lid = (int64_t)it.get_local_linear_id();
  const int64_t C = A.C;
  const int64_t stride = (int64_t)WG_SIZE * VEC8;
  const u16 *xr = static_cast<const u16 *>(A.x) + row * C;
  const u16 *yr = static_cast<const u16 *>(A.y) + row * C;
  const u16 *tbg = static_cast<const u16 *>(A.tbg);
  const u16 *tsg = static_cast<const u16 *>(A.tsg);
  const u16 *tsg_r = (A.tsg_rs != 0) ? tsg + row * A.tsg_rs : tsg;
  u16 *fr = static_cast<u16 *>(A.fma) + row * C;
  u16 *orow = static_cast<u16 *>(A.out) + row * C;
  float fm_buf[MAX_ITERS][8];
  int nit = 0;
  float sumsq = 0.f;
  for (int64_t c0 = lid * VEC8; c0 < C; c0 += stride) {
    sycl::vec<u16, 8> vx = vload8(xr, c0, C);
    sycl::vec<u16, 8> vy = vload8(yr, c0, C);
    sycl::vec<u16, 8> vg = vload8(tbg, c0, C);
    sycl::vec<u16, 8> vt = vload8(tsg_r, c0, C);
    for (int k = 0; k < 8; ++k) {
      float g = bf16_to_float(vg[k]) + bf16_to_float(vt[k]);
      float fm;
      if constexpr (COMPAT) {
        // Torch reference rounding for `x + y * gate`: mul rounds, then add rounds.
        float gb = bf16_to_float(float_to_bf16(g));
        float pr = bf16_to_float(float_to_bf16(bf16_to_float(vy[k]) * gb));
        fm = bf16_to_float(float_to_bf16(bf16_to_float(vx[k]) + pr));
      } else {
        fm = bf16_to_float(vx[k]) + bf16_to_float(vy[k]) * g;
      }
      fm_buf[nit][k] = fm;
      sumsq += fm * fm;
    }
    ++nit;
  }
  float *scr = static_cast<float *>(A.scratch) + row * MAX_SUBGROUPS;
  const float rstd = row_rstd(it, sumsq, scr, (float)C, A.eps, true);
  nit = 0;
  for (int64_t c0 = lid * VEC8; c0 < C; c0 += stride) {
    sycl::vec<u16, 8> vfm, vno;
    for (int k = 0; k < 8; ++k) {
      float fm = fm_buf[nit][k];
      vfm[k] = (c0 + k < C) ? float_to_bf16(fm) : (u16)0;
      vno[k] = (c0 + k < C) ? float_to_bf16(fm * rstd) : (u16)0;
    }
    ++nit;
    vstore8(fr, c0, C, vfm);
    vstore8(orow, c0, C, vno);
  }
}

inline sycl::vec<u16, 4> vload4(const u16 *p, int64_t c0, int64_t C) {
  if (c0 + 4 <= C) return *reinterpret_cast<const sycl::vec<u16, 4> *>(p + c0);
  sycl::vec<u16, 4> v;
  for (int k = 0; k < 4; ++k) v[k] = (c0 + k < C) ? p[c0 + k] : (u16)0;
  return v;
}

inline void vstore4(u16 *p, int64_t c0, int64_t C, sycl::vec<u16, 4> v) {
  if (c0 + 4 <= C) {
    *reinterpret_cast<sycl::vec<u16, 4> *>(p + c0) = v;
  } else {
    for (int k = 0; k < 4; ++k)
      if (c0 + k < C) p[c0 + k] = v[k];
  }
}

inline void mlp_norm_affine_body(sycl::nd_item<1> it, MlpNormAffineArgs A) {
  constexpr int64_t MLP_SG_SIZE = 32;
  constexpr int64_t MLP_ROWS_PER_GROUP = 8;
  constexpr int64_t MLP_VECTOR = 8;
  const sycl::sub_group sg = it.get_sub_group();
  const int64_t C = A.C;
  const int64_t lane = (int64_t)sg.get_local_linear_id();
  const int64_t row = (int64_t)it.get_group_linear_id() * MLP_ROWS_PER_GROUP +
                      (int64_t)sg.get_group_linear_id();
  const bool active = row < A.rows;
  const int64_t safe_row = active ? row : 0;
  const int64_t c0 = lane * MLP_VECTOR;
  const int64_t c_stride = MLP_SG_SIZE * MLP_VECTOR;
  const u16 *xr = static_cast<const u16 *>(A.x) + safe_row * C;
  const u16 *wr = static_cast<const u16 *>(A.weight);
  const u16 *sr = static_cast<const u16 *>(A.scale);
  const u16 *hr = static_cast<const u16 *>(A.shift);
  u16 *orow = static_cast<u16 *>(A.out) + safe_row * C;

  const sycl::vec<u16, 8> first_vx = vload8(xr, c0, C);
  float first_values[8] = {};
  float sumsq = 0.f;
  if (active) {
    for (int k = 0; k < 8; ++k) {
      first_values[k] = bf16_to_float(first_vx[k]);
      sumsq += first_values[k] * first_values[k];
    }
    for (int64_t offset = c_stride; offset < C; offset += c_stride) {
      const int64_t cc = c0 + offset;
      if (cc >= C) continue;
      const sycl::vec<u16, 8> vx = vload8(xr, cc, C);
      for (int k = 0; k < 8; ++k) {
        const float value = bf16_to_float(vx[k]);
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
      vx = vload8(xr, cc, C);
      for (int k = 0; k < 8; ++k) values[k] = bf16_to_float(vx[k]);
    }
    const sycl::vec<u16, 8> vw = vload8(wr, cc, C);
    const sycl::vec<u16, 8> vs = vload8(sr, cc, C);
    const sycl::vec<u16, 8> vh = vload8(hr, cc, C);
    sycl::vec<u16, 8> vo;
    for (int k = 0; k < 8; ++k) {
      float norm = values[k] * rstd;
      norm *= bf16_to_float(vw[k]);
      norm = bf16_to_float(float_to_bf16(norm));
      const float scale = bf16_to_float(float_to_bf16(1.f + bf16_to_float(vs[k])));
      float scaled = norm * scale;
      scaled = bf16_to_float(float_to_bf16(scaled));
      const float shifted = scaled + bf16_to_float(vh[k]);
      vo[k] = float_to_bf16(shifted);
    }
    vstore8(orow, cc, C, vo);
  }
}

template <bool COMPAT>
inline void rope_negfold_body(sycl::nd_item<1> it, RopeNegFoldArgs A) {
  constexpr int ROPE_NEGFOLD_WG = 16;
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
  for (int64_t j = lid; j < D; j += ROPE_NEGFOLD_WG) {
    const int64_t half = (j < Dh) ? 0 : 1;
    const int64_t k = j - half * Dh;
    float x0 = bf16_to_float(x[xo + k * A.xs3]);
    float x1 = bf16_to_float(x[xo + (k + Dh) * A.xs3]);
    float c = bf16_to_float(cos[co + k * A.cs3]);
    float s = bf16_to_float(sin[so + k * A.ss3]);
    if constexpr (COMPAT) {
      // Torch reference rounding for split rope: mul output rounds, then addcmul_ rounds.
      float m0 = bf16_to_float(float_to_bf16(x0 * c));
      float m1 = bf16_to_float(float_to_bf16(x1 * c));
      orow[j] = float_to_bf16((half == 0) ? (m0 + (-s) * x1) : (m1 + s * x0));
    } else {
      orow[j] = float_to_bf16((half == 0) ? (x0 * c - x1 * s) : (x0 * s + x1 * c));
    }
  }
}

// Vectorised split-rope negative fold. Same math as rope_negfold_body, but
// processes 8 k lanes per iteration with u16x8 gathers/stores. The host selects
// it when xs3==cs3==ss3==1, Dh%8==0 and the row starts are 16B aligned; every
// other case, and LTX_SYCL_ROPE_NEGFOLD_VEC=0, uses the scalar rope_negfold_body.
template <bool COMPAT>
inline void rope_negfold_vec_body(sycl::nd_item<1> it, RopeNegFoldArgs A) {
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
    float x0 = bf16_to_float(vx0[i]);
    float x1 = bf16_to_float(vx1[i]);
    float c = bf16_to_float(vc[i]);
    float s = bf16_to_float(vs[i]);
    if constexpr (COMPAT) {
      // Torch reference rounding for split rope: mul output rounds, then addcmul_ rounds.
      float m0 = bf16_to_float(float_to_bf16(x0 * c));
      float m1 = bf16_to_float(float_to_bf16(x1 * c));
      o0[i] = float_to_bf16(m0 + (-s) * x1);
      o1[i] = float_to_bf16(m1 + s * x0);
    } else {
      o0[i] = float_to_bf16(x0 * c - x1 * s);
      o1[i] = float_to_bf16(x0 * s + x1 * c);
    }
  }
  *reinterpret_cast<sycl::vec<u16, 8> *>(orow + k) = o0;
  *reinterpret_cast<sycl::vec<u16, 8> *>(orow + Dh + k) = o1;
}


// Per-head broadcast gate multiply.
//   out[i] = attn[i] * gates[i / dh]   (bf16 in/out, fp32 product, one RNE)
// Bitwise identical to torch's `a.view(B,T,H,Dh) * gates.unsqueeze(-1)` for
// bf16 (TensorIterator opmath is fp32 with a single rounding). Host gates on
// contiguity, dh%8==0 and 16B alignment; anything else keeps the torch path.
inline void gate_mul_body(sycl::nd_item<1> it, GateBroadcastArgs A) {
  const int64_t n = A.n;
  const int64_t dh = A.dh;
  const u16 *a = static_cast<const u16 *>(A.attn);
  const u16 *g = static_cast<const u16 *>(A.gates);
  u16 *o = static_cast<u16 *>(A.out);
  const int64_t gid = (int64_t)it.get_global_linear_id();
  const int64_t i0 = gid * VEC8;
  if (i0 >= n) return;
  const u16 gv = g[i0 / dh];
  sycl::vec<u16, 8> va = vload8(a, i0, n);
  sycl::vec<u16, 8> vo;
  for (int k = 0; k < 8; ++k) vo[k] = float_to_bf16(bf16_to_float(va[k]) * bf16_to_float(gv));
  vstore8(o, i0, n, vo);
}


/* tanh-approx GELU。EXACT 版本與 torch 的 GeluTanhFunctor 同式（libm tanh），輸出逐位相同；
   快速版本以 x * sigmoid(2u) = x / (1 + exp2(-2u * log2(e))) 計算，exp 為 native exp2，
   在 u 為大負值時沒有 1 + tanh 的相消，進位點與 torch 不同。 */
template <bool EXACT>
inline float gelu_tanh_value(float x) {
  const float kBeta = (float)(1.4142135623730951 * 1.1283791670955126 * 0.5);
  constexpr float kKappa = 0.044715f;
  const float x_cube = x * x * x;
  const float inner = kBeta * (x + kKappa * x_cube);
  if constexpr (EXACT) {
    return 0.5f * x * (1.0f + sycl::tanh(inner));
  } else {
    const float e = sycl::native::exp2(-inner * 2.885390081777927f);
    return x / (1.0f + e);
  }
}

template <bool EXACT>
inline void gelu_fast_body(sycl::nd_item<1> it, GeluArgs A) {
  const int64_t n = A.n;
  const int64_t gid = (int64_t)it.get_global_linear_id();
  const int64_t i0 = gid * VEC8;
  if (i0 >= n) return;
  const u16 *x = static_cast<const u16 *>(A.x);
  u16 *o = static_cast<u16 *>(A.out);
  sycl::vec<u16, 8> vx = vload8(x, i0, n);
  sycl::vec<u16, 8> vo;
  for (int k = 0; k < 8; ++k) vo[k] = float_to_bf16(gelu_tanh_value<EXACT>(bf16_to_float(vx[k])));
  vstore8(o, i0, n, vo);
}

template <int CHUNKS>
inline void silu_mul_body(sycl::nd_item<1> it, SiluMulArgs A) {
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
    const sycl::vec<u16, VEC> vg = vload8(gate, offset, n);
    const sycl::vec<u16, VEC> vu = vload8(up, offset, n);
    sycl::vec<u16, VEC> vo;
    for (int k = 0; k < VEC; ++k) {
      const float g = bf16_to_float(vg[k]);
      const float u = bf16_to_float(vu[k]);
      const float s_bf16 = bf16_to_float(float_to_bf16(g / (1.0f + sycl::exp(-g))));
      vo[k] = float_to_bf16(s_bf16 * u);
    }
    vstore8(out, offset, n, vo);
  }
}

// Torch reference rounding for the SiLU/up product. Two RNE
// points, same as the eager path: silu_fp32 -> bf16 -> fp32, then
// (silu_bf16 * up_fp32) -> bf16. SiLU uses x*sigmoid(x) opmath to match the
// torch XPU SiluFunctor; NaN lanes canonicalise to 0xFFC1 like torch.
template <int CHUNKS>
inline void silu_mul_compat_body(sycl::nd_item<1> it, SiluMulArgs A) {
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
    const sycl::vec<u16, VEC> vg = vload8(gate, offset, n);
    const sycl::vec<u16, VEC> vu = vload8(up, offset, n);
    sycl::vec<u16, VEC> vo;
    for (int k = 0; k < VEC; ++k) {
      const float g = bf16_to_float(vg[k]);
      const float u = bf16_to_float(vu[k]);
      // compat: exact-match torch XPU SiLU. The bf16 input 0x40BE /
      // 0x40BF (g = 5.9375) is the one finite bf16 value where a
      // < 2 ULP exp approximation rounds the fp32 SiLU product above the
      // bf16 midpoint while torch rounds it down (0x40BD = 5.90625).
      // Both bf16 keys hold the same fp32 value, so pin once on the fp32
      // value. Every other finite bf16 key is bitwise under exhaustive test.
      float silu;
      if (g == 5.9375f) {
        silu = bf16_to_float((u16)0x40BD);
      } else {
        const float sigmoid = 1.0f / (1.0f + sycl::exp(-g));
        silu = bf16_to_float(float_to_bf16(g * sigmoid));
      }
      const float prod = silu * u;
      vo[k] = sycl::isnan(prod) ? (u16)0xFFC1 : float_to_bf16(prod);
    }
    vstore8(out, offset, n, vo);
  }
}

// strided broadcast: 8-wide elementwise with a stride-broadcast second
// operand.  The host fuses the five logical axes (b, t, h, w, c) into at most
// three groups for x and b jointly; group 0 is the innermost, where x advances
// by 1 per element and b by A.g0b.  The two outer groups each cost one
// multiply-add, so the body carries no integer division.  fp32 compute with a
// single round to bf16 keeps this bitwise with ATen's DivFunctor and
// MulFunctor.
inline void bcast_strided_body(sycl::nd_item<3> it, BcastStridedArgs A) {
  const int64_t c0 = (int64_t)it.get_global_id(2) * 8;
  if (c0 >= A.e0) return;
  const int64_t xo =
      c0 + (int64_t)it.get_global_id(1) * A.g1x + (int64_t)it.get_global_id(0) * A.g2x;
  const int64_t bo =
      c0 * A.g0b + (int64_t)it.get_global_id(1) * A.g1b + (int64_t)it.get_global_id(0) * A.g2b;
  const u16 *x = static_cast<const u16 *>(A.x);
  const u16 *b = static_cast<const u16 *>(A.b);
  u16 *dst = static_cast<u16 *>(A.out);
  const bool full = (c0 + 8 <= A.e0);
  // The vector read of b is only valid on a full block: a partial block may
  // reach past the operand's last element.
  sycl::vec<u16, 8> vb;
  if (A.g0b == 0) {
    const u16 sh = b[bo];
    for (int k = 0; k < 8; ++k) vb[k] = sh;
  } else if (A.vec_b != 0 && full) {
    vb = *reinterpret_cast<const sycl::vec<u16, 8> *>(b + bo);
  } else {
    for (int k = 0; k < 8; ++k) vb[k] = b[bo + A.g0b * k];
  }
  // op 4 copies b into x: the bf16 payload moves verbatim, so the float
  // round-trip is skipped and x is not read at all.
  if (A.op == 4) {
    if (full) {
      *reinterpret_cast<sycl::vec<u16, 8> *>(dst + xo) = vb;
    } else {
      for (int k = 0; k < 8; ++k)
        if (c0 + k < A.e0) dst[xo + k] = vb[k];
    }
    return;
  }
  sycl::vec<u16, 8> vx;
  if (full) {
    vx = *reinterpret_cast<const sycl::vec<u16, 8> *>(x + xo);
  } else {
    for (int k = 0; k < 8; ++k) vx[k] = (c0 + k < A.e0) ? x[xo + k] : (u16)0;
  }
  sycl::vec<u16, 8> vo;
  for (int k = 0; k < 8; ++k) {
    const float a = bf16_to_float(vx[k]);
    const float s = bf16_to_float(vb[k]);
    float y;
    if (A.op == 2) {
      // SiLU：與 F.silu 一致，在 float 算完再進 bf16 一次。
      y = a / (1.0f + sycl::exp(-a));
    } else if (A.op == 3) {
      y = a + s;
    } else {
      y = (A.op == 1) ? (a / s) : (a * s);
    }
    vo[k] = float_to_bf16(y);
  }
  if (full) {
    *reinterpret_cast<sycl::vec<u16, 8> *>(dst + xo) = vo;
  } else {
    for (int k = 0; k < 8; ++k)
      if (c0 + k < A.e0) dst[xo + k] = vo[k];
  }
}

struct BBcastStridedKernel {
  BcastStridedArgs A;
  void operator()(sycl::nd_item<3> it) const { bcast_strided_body(it, A); }
};

// 從運算元取 VEC 個元素：inner 為真時連續讀取（不足一個區塊時以 0 補），
// 否則是每列共用的單一值。
template <int VEC>
inline sycl::vec<u16, VEC> vec_load(const u16 *p, int64_t off, bool inner, bool full, int64_t lim,
                                    int64_t c0) {
  sycl::vec<u16, VEC> v;
  if (inner) {
    if (full) {
      v = *reinterpret_cast<const sycl::vec<u16, VEC> *>(p + off);
    } else {
      for (int k = 0; k < VEC; ++k) v[k] = (c0 + k < lim) ? p[off + k] : (u16)0;
    }
  } else {
    const u16 s = p[off];
    for (int k = 0; k < VEC; ++k) v[k] = s;
  }
  return v;
}

// 向量化 elementwise：一個工作項處理 VEC 個連續元素，位址只有列乘加。
// op 2 先乘後加，兩次都以 bf16 進位，與 torch 的 mul 再接 add 逐位元相同。
// op 4：a + rnd(b * rnd(g + t))，g 為 (1, d) 表列，t 為時間步列（第 row 列讀 t + (row / tdiv) * trs）。
// op 5：a + rnd(rnd(b * rnd(g + t)) * m[row / mrows])。rnd = bf16 RNE。
template <int VEC>
inline void elem_body(sycl::nd_item<2> it, ElemArgs A) {
  const int64_t c0 = (int64_t)it.get_global_id(1) * VEC;
  if (c0 >= A.d) return;
  const int64_t row = (int64_t)it.get_global_id(0);
  const int64_t o1 = row * A.d + c0;
  const int64_t o2 = row * A.bs + (A.bvec != 0 ? c0 : 0);
  const int64_t o3 = row * A.gs + (A.gvec != 0 ? c0 : 0);
  const u16 *pa = static_cast<const u16 *>(A.a);
  const u16 *pb = static_cast<const u16 *>(A.b);
  const u16 *pg = static_cast<const u16 *>(A.g);
  u16 *out = static_cast<u16 *>(A.out);
  const bool full = (c0 + VEC <= A.d);
  const sycl::vec<u16, VEC> v1 = vec_load<VEC>(pa, o1, true, full, A.d, c0);
  const sycl::vec<u16, VEC> v2 = vec_load<VEC>(pb, o2, A.bvec != 0, full, A.d, c0);
  const sycl::vec<u16, VEC> v3 =
      A.op >= 2 ? vec_load<VEC>(pg, o3, A.gvec != 0, full, A.d, c0) : sycl::vec<u16, VEC>{};
  sycl::vec<u16, VEC> v4{};
  float mk = 1.f;
  if (A.op >= 4) {
    const int64_t o4 = (int64_t)((uint32_t)row / (uint32_t)A.tdiv) * A.trs + c0;
    v4 = vec_load<VEC>(static_cast<const u16 *>(A.t), o4, true, full, A.d, c0);
    if (A.op == 5)
      mk = bf16_to_float(static_cast<const u16 *>(A.m)[(uint32_t)row / (uint32_t)A.mrows]);
  }
  sycl::vec<u16, VEC> vo;
  for (int k = 0; k < VEC; ++k) {
    const float a = bf16_to_float(v1[k]);
    const float b = bf16_to_float(v2[k]);
    float y;
    if (A.op == 0) {
      y = a + b;
    } else if (A.op == 1) {
      y = a * b;
    } else if (A.op == 3) {
      // 單次進位：與 torch.addcmul 相同。
      y = a + b * bf16_to_float(v3[k]);
    } else if (A.op >= 4) {
      // gate = rnd(g + t)；乘積 rnd(b * gate)；op 5 再乘 mask 並進位；最後加 a 並進位。
      const float gate = bf16_to_float(float_to_bf16(bf16_to_float(v3[k]) + bf16_to_float(v4[k])));
      float q = bf16_to_float(float_to_bf16(b * gate));
      if (A.op == 5) q = bf16_to_float(float_to_bf16(q * mk));
      y = a + q;
    } else {
      // 先乘並進 bf16，再加並進 bf16。
      y = a + bf16_to_float(float_to_bf16(b * bf16_to_float(v3[k])));
    }
    vo[k] = float_to_bf16(y);
  }
  if (full) {
    *reinterpret_cast<sycl::vec<u16, VEC> *>(out + o1) = vo;
  } else {
    for (int k = 0; k < VEC; ++k)
      if (c0 + k < A.d) out[o1 + k] = vo[k];
  }
}

struct ElemKernel {
  ElemArgs A;
  void operator()(sycl::nd_item<2> it) const { elem_body<8>(it, A); }
};

struct ElemKernelV16 {
  ElemArgs A;
  void operator()(sycl::nd_item<2> it) const { elem_body<16>(it, A); }
};
// Named kernel functors: stable SPIR-V entry names, single POD struct arg.
struct AdaNormAffineKernel {
  AdaNormAffineArgs A;
  void operator()(sycl::nd_item<1> it) const { ada_norm_affine_body<true, false>(it, A); }
};

struct AffineAddcmulKernel {
  AdaNormAffineArgs A;
  void operator()(sycl::nd_item<1> it) const { ada_norm_affine_body<false, false>(it, A); }
};

// Torch reference rounding: the bf16 rounding points of the torch expression
// (single RNE per op). Selected by the host when LTX_TORCH_EXACT_ROUNDING=1;
// the fused math otherwise.
struct AdaNormAffineTorchRoundKernel {
  AdaNormAffineArgs A;
  void operator()(sycl::nd_item<1> it) const { ada_norm_affine_body<true, true>(it, A); }
};



struct AffineAddcmulTorchRoundKernel {
  AdaNormAffineArgs A;
  void operator()(sycl::nd_item<1> it) const { ada_norm_affine_body<false, true>(it, A); }
};











struct PostAttnGateNormKernel {
  PostAttnGateNormArgs A;
  void operator()(sycl::nd_item<1> it) const { post_attn_gate_norm_body<false>(it, A); }
};

// Torch reference rounding for `x + y * gate`.
struct PostAttnGateNormTorchRoundKernel {
  PostAttnGateNormArgs A;
  void operator()(sycl::nd_item<1> it) const { post_attn_gate_norm_body<true>(it, A); }
};

struct MlpNormAffineKernel {
  MlpNormAffineArgs A;
  void operator()(sycl::nd_item<1> it) const { mlp_norm_affine_body(it, A); }
};


struct NormRopeKernel {
  NormRopeArgs A;
  void operator()(sycl::nd_item<1> it) const { norm_rope_body<false>(it, A); }
};

// Torch reference rounding for q/k RMSNorm followed by split rope.
struct NormRopeTorchRoundKernel {
  NormRopeArgs A;
  void operator()(sycl::nd_item<1> it) const { norm_rope_body<true>(it, A); }
};

struct RopeNegFoldKernel {
  RopeNegFoldArgs A;
  void operator()(sycl::nd_item<1> it) const { rope_negfold_body<false>(it, A); }
};

struct RopeNegFoldVecKernel {
  RopeNegFoldArgs A;
  void operator()(sycl::nd_item<1> it) const { rope_negfold_vec_body<false>(it, A); }
};

// Torch reference rounding for split rope (mul, then addcmul_).
struct RopeNegFoldTorchRoundKernel {
  RopeNegFoldArgs A;
  void operator()(sycl::nd_item<1> it) const { rope_negfold_body<true>(it, A); }
};

struct RopeNegFoldVecTorchRoundKernel {
  RopeNegFoldArgs A;
  void operator()(sycl::nd_item<1> it) const { rope_negfold_vec_body<true>(it, A); }
};


struct GateBroadcastKernel {
  GateBroadcastArgs A;
  void operator()(sycl::nd_item<1> it) const { gate_mul_body(it, A); }
};


struct GeluFastKernel {
  GeluArgs A;
  void operator()(sycl::nd_item<1> it) const { gelu_fast_body<false>(it, A); }
};

// Same expression as torch's GeluTanhFunctor (bitwise equal output).
struct GeluTanhExactKernel {
  GeluArgs A;
  void operator()(sycl::nd_item<1> it) const { gelu_fast_body<true>(it, A); }
};








// fp8 e4m3fn -> bf16 widening. LUT is bitwise-identical to the XPU torch
// float8_e4m3fn.to(bfloat16) for all 256 byte values (both NaN codes map to
// 0xFFC1 on XPU; the CPU conversion yields 0x7FC0), verified exhaustively on XPU.
static constexpr u16 FP8_LUT[256] = {
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

inline void fp8_widen_body(sycl::nd_item<1> it, Fp8Args A) {
  const int64_t n = A.n;
  const int64_t i0 = (int64_t)it.get_global_linear_id() * 8;
  if (i0 >= n) return;
  const uint8_t *in = static_cast<const uint8_t *>(A.in);
  u16 *out = static_cast<u16 *>(A.out);
  if (i0 + 8 <= n) {
    sycl::vec<uint8_t, 8> v = *reinterpret_cast<const sycl::vec<uint8_t, 8> *>(in + i0);
    sycl::vec<u16, 8> o;
#pragma unroll
    for (int k = 0; k < 8; ++k) o[k] = FP8_LUT[v[k]];
    *reinterpret_cast<sycl::vec<u16, 8> *>(out + i0) = o;
  } else {
    for (int64_t j = i0; j < n; ++j) out[j] = FP8_LUT[in[j]];
  }
}

struct Fp8WidenKernel {
  Fp8Args A;
  void operator()(sycl::nd_item<1> it) const { fp8_widen_body(it, A); }
};









// v2: 16-wide widening (two 8-wide vectors; keeps 16-byte alignment).
inline void fp8_widen16_body(sycl::nd_item<1> it, Fp8Args A) {
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
    for (int k = 0; k < 8; ++k) o0[k] = FP8_LUT[v0[k]];
#pragma unroll
    for (int k = 0; k < 8; ++k) o1[k] = FP8_LUT[v1[k]];
    *reinterpret_cast<sycl::vec<u16, 8> *>(out + i0) = o0;
    *reinterpret_cast<sycl::vec<u16, 8> *>(out + i0 + 8) = o1;
  } else {
    for (int64_t j = i0; j < n; ++j) out[j] = FP8_LUT[in[j]];
  }
}


// Anchor with external linkage so device compilation keeps all kernels.
void device_kernel_anchor(sycl::queue &q) {
  AdaNormAffineArgs ada_args{};
  PostAttnGateNormArgs post_attn_args{};
  RopeNegFoldArgs rope_args{};
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::nd_range<1>(WG_SIZE, WG_SIZE), AdaNormAffineKernel{ada_args});
  });
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::nd_range<1>(WG_SIZE, WG_SIZE), AffineAddcmulKernel{ada_args});
  });
  BcastStridedArgs bcast_args{};
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, WG_SIZE), sycl::range<3>(1, 1, WG_SIZE)),
                   BBcastStridedKernel{bcast_args});
  });
  ElemArgs elem_args{};
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::nd_range<2>(sycl::range<2>(1, WG_SIZE), sycl::range<2>(1, WG_SIZE)),
                   ElemKernel{elem_args});
  });
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::nd_range<2>(sycl::range<2>(1, WG_SIZE), sycl::range<2>(1, WG_SIZE)),
                   ElemKernelV16{elem_args});
  });
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::nd_range<1>(WG_SIZE, WG_SIZE), AdaNormAffineTorchRoundKernel{ada_args});
  });
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::nd_range<1>(WG_SIZE, WG_SIZE), AffineAddcmulTorchRoundKernel{ada_args});
  });
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::nd_range<1>(WG_SIZE, WG_SIZE), PostAttnGateNormKernel{post_attn_args});
  });
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::nd_range<1>(WG_SIZE, WG_SIZE), PostAttnGateNormTorchRoundKernel{post_attn_args});
  });
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::nd_range<1>(16, 16), RopeNegFoldKernel{rope_args});
  });
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::nd_range<1>(64, 64), RopeNegFoldVecKernel{rope_args});
  });
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::nd_range<1>(16, 16), RopeNegFoldTorchRoundKernel{rope_args});
  });
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::nd_range<1>(64, 64), RopeNegFoldVecTorchRoundKernel{rope_args});
  });
  MlpNormAffineArgs mlp_args{};
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::nd_range<1>(WG_SIZE, WG_SIZE), MlpNormAffineKernel{mlp_args});
  });
  NormRopeArgs norm_rope_args{};
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::nd_range<1>(WG_SIZE, WG_SIZE), NormRopeKernel{norm_rope_args});
  });
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::nd_range<1>(WG_SIZE, WG_SIZE), NormRopeTorchRoundKernel{norm_rope_args});
  });
  GateBroadcastArgs gate_args{};
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::nd_range<1>(WG_SIZE, WG_SIZE), GateBroadcastKernel{gate_args});
  });
  GeluArgs gelu_args{};
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::nd_range<1>(WG_SIZE, WG_SIZE), GeluFastKernel{gelu_args});
  });
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::nd_range<1>(WG_SIZE, WG_SIZE), GeluTanhExactKernel{gelu_args});
  });
  Fp8Args fp8_args{};
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::nd_range<1>(WG_SIZE, WG_SIZE), Fp8WidenKernel{fp8_args});
  });
}
