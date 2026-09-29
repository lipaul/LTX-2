// R8 kernel argument structs — shared by device (r8_device.cpp) and host
// (r8_host.cpp). Passed BY VALUE as the single kernel argument, so there is
// no capture-ordering risk. Standard layout, 8-byte aligned members first.
#pragma once

#include <cstdint>

// R40: 8-wide elementwise over the contiguous innermost run with a
// stride-broadcast second operand.  ``x`` and ``out`` are contiguous; the
// broadcast operand is addressed through the logical strides ``bs0..bs4`` of
// its broadcast shape, so a 0 entry means that dimension is broadcast.  The
// flat layout of x is NHWC: the innermost extent ``inner`` has stride 1 and the
// outer index enumerates (d0, d1, d2, d3).
struct R8BBcastArgs {
  const void *x;
  const void *b;
  void *out;
  int64_t d0;
  int64_t d1;
  int64_t d2;
  int64_t d3;
  int64_t inner;
  int64_t bs0;
  int64_t bs1;
  int64_t bs2;
  int64_t bs3;
  int64_t bs4;
  int32_t op;
  int32_t pad;
};

struct R8K1Args {
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

struct R8K2Args {
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

struct R8R17MlpNormAffineArgs {
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

struct R8R19NormRopeArgs {
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

struct R8K3Args {
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

struct R8K3DualArgs {
  const void *xq;
  const void *xk;
  const void *cos;
  const void *sin;
  void *outq;
  void *outk;
  int64_t qxs0;
  int64_t qxs1;
  int64_t qxs2;
  int64_t qxs3;
  int64_t kxs0;
  int64_t kxs1;
  int64_t kxs2;
  int64_t kxs3;
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
  int64_t prefetch;
};

// R10D F2: per-head broadcast gate multiply  out[i] = attn[i] * gates[i / dh]
struct R8GateArgs {
  const void *attn;
  const void *gates;
  void *out;
  int64_t n;
  int64_t dh;
};

// R10D F3: tanh-approximation GELU (matches torch's GeluTanhFunctor exactly)
struct R8GeluArgs {
  const void *x;
  void *out;
  int64_t n;
};

struct R8SiluMulArgs {
  const void *gate;
  const void *up;
  void *out;
  int64_t n;
};


// R9A: fp8 e4m3fn -> bf16 widening (vectorized, integer LUT).
struct R8Fp8Args {
  const void *in;
  void *out;
  int64_t n;
};

struct R8Fp8EffArgs {
  const void *in;
  void *out;
  int64_t n;
  int64_t prefetch;
};

struct R8Fp8BatchDesc {
  const void *in;
  void *out;
  int64_t n;
};

struct R8Fp8BatchArgs {
  R8Fp8BatchDesc desc[4];
  int64_t count;
  int64_t total;
  int64_t prefetch;
};

// Strided-x broadcast args.  Identical to R8BBcastArgs except that x carries its
// own four outer strides, because a view sliced on the channel dimension has a
// stride gap and the flat-run assumption is false for it.
struct R8BBcastStridedArgs {
  const void *x;
  const void *b;
  void *out;
  int64_t d0;
  int64_t d1;
  int64_t d2;
  int64_t d3;
  int64_t inner;
  int64_t xs0;
  int64_t xs1;
  int64_t xs2;
  int64_t xs3;
  int64_t bs0;
  int64_t bs1;
  int64_t bs2;
  int64_t bs3;
  int64_t bs4;
  int32_t op;
  int32_t pad;
};
