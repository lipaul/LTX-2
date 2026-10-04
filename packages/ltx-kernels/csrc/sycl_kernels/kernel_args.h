// kernel argument structs — shared by device (device_kernels.cpp) and host
// (host_ops.cpp). Passed BY VALUE as the single kernel argument, so there is
// no capture-ordering risk. Standard layout, 8-byte aligned members first.
#pragma once

#include <cstdint>

struct AdaNormAffineArgs {
  const void *x;
  const void *tbs;
  const void *tss;
  const void *tbh;
  const void *tsh;
  void *out;
  void *scratch;
  int64_t tss_rs;
  int64_t tsh_rs;
  int64_t rows;
  int64_t C;
  float eps;
  float pad;
  int64_t prefetch;
};

struct PostAttnGateNormArgs {
  const void *x;
  const void *y;
  const void *tbg;
  const void *tsg;
  void *fma;
  void *out;
  void *scratch;
  int64_t tsg_rs;
  int64_t rows;
  int64_t C;
  float eps;
  float pad;
};

struct MlpNormAffineArgs {
  const void *x;
  const void *weight;
  const void *scale;
  const void *shift;
  void *out;
  void *scratch;
  int64_t rows;
  int64_t C;
  float eps;
  float pad;
};

struct NormRopeArgs {
  const void *q;
  const void *k;
  const void *q_weight;
  const void *k_weight;
  const void *cos;
  const void *sin;
  void *scratch;
  void *outq;
  void *outk;
  int64_t q_batch_stride;
  int64_t q_row_stride;
  int64_t k_batch_stride;
  int64_t k_row_stride;
  int64_t cos_stride0;
  int64_t cos_stride1;
  int64_t cos_stride2;
  int64_t cos_stride3;
  int64_t sin_stride0;
  int64_t sin_stride1;
  int64_t sin_stride2;
  int64_t sin_stride3;
  int64_t B;
  int64_t H;
  int64_t T;
  int64_t Dh;
  int64_t C;
  float eps;
  float q_scale;
  float k_scale;
};

struct RopeNegFoldArgs {
  const void *x;
  const void *cos;
  const void *sin;
  void *out;
  int64_t xs0;
  int64_t xs1;
  int64_t xs2;
  int64_t xs3;
  int64_t cs0;
  int64_t cs1;
  int64_t cs2;
  int64_t cs3;
  int64_t ss0;
  int64_t ss1;
  int64_t ss2;
  int64_t ss3;
  int64_t B;
  int64_t H;
  int64_t T;
  int64_t Dh;
};


// Per-head broadcast gate multiply  out[i] = attn[i] * gates[i / dh]
struct GateBroadcastArgs {
  const void *attn;
  const void *gates;
  void *out;
  int64_t n;
  int64_t dh;
};

// tanh-approximation GELU (matches torch's GeluTanhFunctor exactly)
struct GeluArgs {
  const void *x;
  void *out;
  int64_t n;
};

struct SiluMulArgs {
  const void *gate;
  const void *up;
  void *out;
  int64_t n;
};


// fp8 e4m3fn -> bf16 widening (vectorized, integer LUT).
struct Fp8Args {
  const void *in;
  void *out;
  int64_t n;
};




// 向量化 elementwise：把 (rows, d) 的 bf16 張量逐元素合併。運算元可為
// (rows, d)、(rows, 1)（每列共用一個值）、(1, d)（列廣播，列步幅 0）或 (1, 1)。
// op 0 = a+b、1 = a*b、2 = 相容 addcmul：先乘後加的兩次 bf16 進位。
struct ElemArgs {
  const void *a;
  const void *b;
  const void *g;
  void *out;
  int64_t rows;
  int64_t d;
  int64_t bs;
  int64_t gs;
  int64_t bvec;
  int64_t gvec;
  int32_t op;
  int32_t pad;
  // op 4 / op 5 operands.  t: bf16 timestep rows, row r reads the d elements at
  // t + (r / tdiv) * trs.  m: bf16 mask, row r reads m[r / mrows].
  const void *t;
  const void *m;
  int64_t trs;
  int64_t tdiv;
  int64_t mrows;
};

// Strided-x broadcast args.  The five logical axes (b, t, h, w, c) are fused on
// the host into at most three groups, jointly for x and b, so that the kernel
// needs no integer division.  Group 0 is the innermost group: x advances by 1
// per element there and b by ``g0b``.  ``vec_b`` is set when b is contiguous
// inside group 0 and every group-0 block start is 16-byte aligned in b.
struct BcastStridedArgs {
  const void *x;
  const void *b;
  void *out;
  int64_t e0;
  int64_t g0b;
  int64_t e1;
  int64_t g1x;
  int64_t g1b;
  int64_t e2;
  int64_t g2x;
  int64_t g2b;
  int32_t vec_b;
  int32_t op;
};
