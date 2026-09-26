#include <sycl/sycl.hpp>

#include "r26_args.h"

using u16 = uint16_t;
using u32 = uint32_t;

constexpr int R26_NORM_WG = 256;
constexpr int R26_NORM_ROWS_PER_WG = 16;
constexpr int R26_QK_WG = 512;
constexpr int R26_QK_ROWS_PER_WG = 32;

inline float r26_h2f(u16 h) {
  return sycl::bit_cast<float>((u32)h << 16);
}

inline u16 r26_f2h(float x) {
  u32 u = sycl::bit_cast<u32>(x);
  u32 bias = 0x7FFFu + ((u >> 16) & 1u);
  return (u16)((u + bias) >> 16);
}

inline float r26_round(float x) {
  return r26_h2f(r26_f2h(x));
}

inline sycl::vec<u16, 8> r26_load8(const u16 *p, int64_t c0, int64_t C) {
  if (c0 + 8 <= C) return *reinterpret_cast<const sycl::vec<u16, 8> *>(p + c0);
  sycl::vec<u16, 8> v;
  for (int k = 0; k < 8; ++k) v[k] = (c0 + k < C) ? p[c0 + k] : (u16)0;
  return v;
}

inline void r26_store8(u16 *p, int64_t c0, int64_t C, sycl::vec<u16, 8> v) {
  if (c0 + 8 <= C) {
    *reinterpret_cast<sycl::vec<u16, 8> *>(p + c0) = v;
  } else {
    for (int k = 0; k < 8; ++k)
      if (c0 + k < C) p[c0 + k] = v[k];
  }
}

template <bool Precomputed>
inline void r26_norm_affine_body(sycl::nd_item<1> it, R26NormAffineArgs A) {
  const sycl::sub_group sg = it.get_sub_group();
  const int64_t lane = static_cast<int64_t>(sg.get_local_linear_id());
  const int64_t c0 = lane * 8;
  const int64_t C = A.C;
  const bool full_channel = C == 256;
  const u16 *wr = static_cast<const u16 *>(A.weight);
  const u16 *sr = static_cast<const u16 *>(A.scale);
  const u16 *hr = static_cast<const u16 *>(A.shift);
  const sycl::vec<u16, 8> vw = full_channel
      ? *reinterpret_cast<const sycl::vec<u16, 8> *>(wr + c0)
      : r26_load8(wr, c0, C);
  const sycl::vec<u16, 8> vs = full_channel
      ? *reinterpret_cast<const sycl::vec<u16, 8> *>(sr + c0)
      : r26_load8(sr, c0, C);
  const sycl::vec<u16, 8> vh = full_channel
      ? *reinterpret_cast<const sycl::vec<u16, 8> *>(hr + c0)
      : r26_load8(hr, c0, C);
  const int64_t row_base = static_cast<int64_t>(it.get_group_linear_id()) *
                              R26_NORM_ROWS_PER_WG +
                          static_cast<int64_t>(sg.get_group_linear_id()) * 2;
  for (int64_t row_offset = 0; row_offset < 2; ++row_offset) {
    const int64_t row = row_base + row_offset;
    if (row >= A.rows) continue;
    const u16 *xr = static_cast<const u16 *>(A.x) + row * C;
    u16 *out = static_cast<u16 *>(A.out) + row * C;
    const sycl::vec<u16, 8> vx = full_channel
        ? *reinterpret_cast<const sycl::vec<u16, 8> *>(xr + c0)
        : r26_load8(xr, c0, C);
    float value[8];
    for (int k = 0; k < 8; ++k) value[k] = r26_h2f(vx[k]);
    const float p0 = value[0] * value[0] + value[1] * value[1];
    const float p1 = value[2] * value[2] + value[3] * value[3];
    const float p2 = value[4] * value[4] + value[5] * value[5];
    const float p3 = value[6] * value[6] + value[7] * value[7];
    const float sumsq = (p0 + p1) + (p2 + p3);
    const float total = sycl::reduce_over_group(sg, sumsq, sycl::plus<float>());
    const float mean = full_channel ? total * 0.00390625f
                                     : total / static_cast<float>(C);
    const float rstd = sycl::rsqrt(mean + A.eps);
    sycl::vec<u16, 8> result;
    for (int k = 0; k < 8; ++k) {
      if (!full_channel && c0 + k >= C) {
        result[k] = 0;
        continue;
      }
      float norm = value[k] * rstd;
      norm *= r26_h2f(vw[k]);
      norm = r26_round(norm);
      const float affine_scale = Precomputed
          ? r26_h2f(vs[k])
          : r26_round(1.f + r26_h2f(vs[k]));
      const float shifted = r26_round(norm * affine_scale) + r26_h2f(vh[k]);
      result[k] = r26_f2h(shifted);
    }
    if (full_channel) {
      *reinterpret_cast<sycl::vec<u16, 8> *>(out + c0) = result;
    } else {
      r26_store8(out, c0, C, result);
    }
  }
}

inline void r26_norm_rope_body(sycl::nd_item<1> it, R26NormRopeArgs A) {
  constexpr int64_t HALF_LANES = 16;
  const sycl::sub_group sg = it.get_sub_group();
  const int64_t group = (int64_t)it.get_group_linear_id();
  const int64_t sgid = (int64_t)sg.get_group_linear_id();
  const int64_t lane = (int64_t)sg.get_local_linear_id();
  const int64_t half = lane / HALF_LANES;
  const int64_t lane_local = lane % HALF_LANES;
  const int64_t row = group * R26_QK_ROWS_PER_WG + sgid * 2 + half;
  const bool active = row < A.rows;
  const int64_t DH = A.DH;
  const int64_t c0 = lane_local * 4;
  const int64_t hw = A.H * A.W * A.NH;
  const int64_t b = row / (A.T * hw);
  const int64_t rem = row - b * A.T * hw;
  const int64_t t = rem / hw;
  const int64_t hw0 = rem - t * hw;
  const int64_t h = hw0 / (A.W * A.NH);
  const int64_t hw1 = hw0 - h * A.W * A.NH;
  const int64_t w = hw1 / A.NH;
  const int64_t nh = hw1 - w * A.NH;
  (void)b;
  (void)nh;

  sycl::vec<u16, 4> vq;
  sycl::vec<u16, 4> vk;
  float qv[4] = {};
  float kv[4] = {};
  if (active) {
    const u16 *qrow = static_cast<const u16 *>(A.q) + row * DH;
    const u16 *krow = static_cast<const u16 *>(A.k) + row * DH;
    vq = *reinterpret_cast<const sycl::vec<u16, 4> *>(qrow + c0);
    vk = *reinterpret_cast<const sycl::vec<u16, 4> *>(krow + c0);
    for (int j = 0; j < 4; ++j) {
      qv[j] = r26_h2f(vq[j]);
      kv[j] = r26_h2f(vk[j]);
    }
  }

  float qsumsq = 0.f;
  float ksumsq = 0.f;
  if (active) {
    qsumsq = qv[0] * qv[0] + qv[1] * qv[1];
    ksumsq = kv[0] * kv[0] + kv[1] * kv[1];
    qsumsq += qv[2] * qv[2] + qv[3] * qv[3];
    ksumsq += kv[2] * kv[2] + kv[3] * kv[3];
  }
#pragma unroll
  for (int step = 8; step >= 1; step >>= 1) {
    const auto peer = static_cast<decltype(sg.get_local_linear_id())>(
        static_cast<size_t>(lane) ^ static_cast<size_t>(step));
    qsumsq += sycl::select_from_group(sg, qsumsq, peer);
    ksumsq += sycl::select_from_group(sg, ksumsq, peer);
  }
  if (!active) return;

  const float q_rstd = sycl::rsqrt(qsumsq / static_cast<float>(DH) + A.eps);
  const float k_rstd = sycl::rsqrt(ksumsq / static_cast<float>(DH) + A.eps);
  const sycl::vec<u16, 4> vqw = *reinterpret_cast<const sycl::vec<u16, 4> *>(
      static_cast<const u16 *>(A.q_weight) + c0);
  const sycl::vec<u16, 4> vkw = *reinterpret_cast<const sycl::vec<u16, 4> *>(
      static_cast<const u16 *>(A.k_weight) + c0);
  sycl::vec<u16, 4> outvq;
  sycl::vec<u16, 4> outvk;
  const float *cp = static_cast<const float *>(A.cos);
  const float *sp = static_cast<const float *>(A.sin);
  const int64_t dim = c0;
  const float *pair_cos = cp;
  const float *pair_sin = sp;
  if (dim < A.d_t) {
    pair_cos += t * A.t_pairs + dim / 2;
    pair_sin += t * A.t_pairs + dim / 2;
  } else if (dim < A.d_t + A.d_h) {
    const int64_t local_pair = (dim - A.d_t) / 2;
    pair_cos += A.T * A.t_pairs + h * A.h_pairs + local_pair;
    pair_sin += A.T * A.t_pairs + h * A.h_pairs + local_pair;
  } else {
    const int64_t local_pair = (dim - A.d_t - A.d_h) / 2;
    const int64_t offset = A.T * A.t_pairs + A.H * A.h_pairs;
    pair_cos += offset + w * A.w_pairs + local_pair;
    pair_sin += offset + w * A.w_pairs + local_pair;
  }
  const sycl::vec<float, 2> cos_pair =
      *reinterpret_cast<const sycl::vec<float, 2> *>(pair_cos);
  const sycl::vec<float, 2> sin_pair =
      *reinterpret_cast<const sycl::vec<float, 2> *>(pair_sin);
  for (int j = 0; j < 4; j += 2) {
    float qa = qv[j] * q_rstd;
    float qb = qv[j + 1] * q_rstd;
    float ka = kv[j] * k_rstd;
    float kb = kv[j + 1] * k_rstd;
    qa = r26_round(qa * r26_h2f(vqw[j]));
    qb = r26_round(qb * r26_h2f(vqw[j + 1]));
    ka = r26_round(ka * r26_h2f(vkw[j]));
    kb = r26_round(kb * r26_h2f(vkw[j + 1]));
    qa = r26_round(qa * A.q_scale);
    qb = r26_round(qb * A.q_scale);
    ka = r26_round(ka * A.k_scale);
    kb = r26_round(kb * A.k_scale);
    const float c = cos_pair[j >> 1];
    const float s = sin_pair[j >> 1];
    outvq[j] = r26_f2h(qa * c - qb * s);
    outvq[j + 1] = r26_f2h(qa * s + qb * c);
    outvk[j] = r26_f2h(ka * c - kb * s);
    outvk[j + 1] = r26_f2h(ka * s + kb * c);
  }
  u16 *outq = static_cast<u16 *>(A.outq) + row * DH;
  u16 *outk = static_cast<u16 *>(A.outk) + row * DH;
  *reinterpret_cast<sycl::vec<u16, 4> *>(outq + c0) = outvq;
  *reinterpret_cast<sycl::vec<u16, 4> *>(outk + c0) = outvk;
}

struct R8R26VaeNormAffineKernel {
  R26NormAffineArgs A;
  void operator()(sycl::nd_item<1> it) const { r26_norm_affine_body<false>(it, A); }
};

struct R8R26VaeNormAffinePrecomputedKernel {
  R26NormAffineArgs A;
  void operator()(sycl::nd_item<1> it) const { r26_norm_affine_body<true>(it, A); }
};

struct R8R26VaeNormRopeKernel {
  R26NormRopeArgs A;
  void operator()(sycl::nd_item<1> it) const { r26_norm_rope_body(it, A); }
};

void r26_device_anchor(sycl::queue &q) {
  R26NormAffineArgs a{};
  R26NormRopeArgs b{};
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::nd_range<1>(R26_NORM_WG, R26_NORM_WG),
                   R8R26VaeNormAffineKernel{a});
  });
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::nd_range<1>(R26_NORM_WG, R26_NORM_WG),
                   R8R26VaeNormAffinePrecomputedKernel{a});
  });
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::nd_range<1>(R26_QK_WG, R26_QK_WG),
                   R8R26VaeNormRopeKernel{b});
  });
}
