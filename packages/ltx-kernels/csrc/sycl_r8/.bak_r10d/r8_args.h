// R8 kernel argument structs — shared by device (r8_device.cpp) and host
// (r8_host.cpp). Passed BY VALUE as the single kernel argument, so there is
// no capture-ordering risk. Standard layout, 8-byte aligned members first.
#pragma once

#include <cstdint>

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
