#pragma once

#include <cstdint>

struct R26NormAffineArgs {
  const void *x;
  const void *weight;
  const void *scale;
  const void *shift;
  void *out;
  int64_t rows;
  int64_t C;
  float eps;
  float pad0;
  float pad1;
  float pad2;
  float pad3;
};

struct R26NormRopeArgs {
  const void *q;
  const void *k;
  const void *q_weight;
  const void *k_weight;
  const void *cos;
  const void *sin;
  void *outq;
  void *outk;
  int64_t B;
  int64_t T;
  int64_t H;
  int64_t W;
  int64_t NH;
  int64_t DH;
  int64_t d_t;
  int64_t d_h;
  int64_t t_pairs;
  int64_t h_pairs;
  int64_t w_pairs;
  int64_t rows;
  float eps;
  float q_scale;
  float k_scale;
  float pad0;
  float pad1;
  float pad2;
  float pad3;
};
