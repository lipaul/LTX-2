// R8 host side: TORCH_LIBRARY ops launching prebuilt SPIR-V kernels
// (r8_kernels.spv) on torch's own XPU queue via L0 interop.
//
// Why interop: torch's XPU runtime is pinned to libsycl.so.8 (2025.3 ABI)
// while only a 2026.1 compiler exists here. Submitting through sycl::queue
// compiled against 2026 headers references .so.9-only symbols. Instead:
//   device kernels -> SPIR-V (icpx -fsycl, headers don't matter for device),
//   host compiled WITHOUT -fsycl against the venv 2025.3 headers (.so.8 ABI),
//   kernels wrapped with make_kernel_bundle/make_kernel and launched with
//   handler::parallel_for(nd_range, kernel) on torch's queue. Same queue =>
//   ordering with eager kernels needs no extra sync.
#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>

#include <level_zero/ze_api.h>

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <torch/library.h>
#include <torch/torch.h>

#include <dlfcn.h>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>
#include <cstdlib>
#include <cstdint>

#include "r8_args.h"

namespace {

using u16 = uint16_t;

constexpr int64_t R8_MAX_C = 8192;
constexpr size_t R8_SCRATCH_FLOATS = 1 << 24;  // 64MB: R17 row stride 272
constexpr int R8_WG = 256;

constexpr int R17_NORM_WG = 256;
constexpr int R17_NORM_ROWS_PER_GROUP = 8;

int64_t r18_env_int(const char *name, int64_t default_value, int64_t minimum,
                    int64_t maximum) {
  const char *text = std::getenv(name);
  if (text == nullptr) return default_value;
  char *end = nullptr;
  const long long value = std::strtoll(text, &end, 10);
  TORCH_CHECK(end != nullptr && end != text && *end == '\0' && value >= minimum &&
                  value <= maximum,
              "r18: invalid ", name, "=", text);
  return static_cast<int64_t>(value);
}

int64_t r18_env_flag(const char *name, int64_t default_value) {
  const int64_t value = r18_env_int(name, default_value, 0, 1);
  return value;
}

int64_t r18_env_flag_with_common(const char *specific, const char *common,
                                  int64_t default_value) {
  if (std::getenv(specific) != nullptr) return r18_env_flag(specific, default_value);
  if (std::getenv(common) != nullptr) return r18_env_flag(common, default_value);
  return default_value;
}

int64_t r19_env_int(const char *name, int64_t default_value, int64_t minimum,
                    int64_t maximum) {
  const char *text = std::getenv(name);
  if (text == nullptr) return default_value;
  char *end = nullptr;
  const long long value = std::strtoll(text, &end, 10);
  TORCH_CHECK(end != nullptr && end != text && *end == '\0' && value >= minimum &&
                  value <= maximum,
              "r19: invalid ", name, "=", text);
  return static_cast<int64_t>(value);
}

int64_t r21_env_int(const char *name, int64_t default_value, int64_t minimum,
                    int64_t maximum) {
  const char *text = std::getenv(name);
  if (text == nullptr) return default_value;
  char *end = nullptr;
  const long long value = std::strtoll(text, &end, 10);
  TORCH_CHECK(end != nullptr && end != text && *end == '\0' && value >= minimum &&
                  value <= maximum,
              "r21: invalid ", name, "=", text);
  return static_cast<int64_t>(value);
}

// R31 output-audit compat: reproduce round-0 bf16 rounding at the fused
// R3/R6/R8/R10D sites. Default on (METHOD §7); explicit LTX_R31_COMPAT=0
// falls back to the fused math; LTX_ALL_ORIG=1 forces off.
bool r8_r31_compat_on() {
  const char *all = std::getenv("LTX_ALL_ORIG");
  if (all != nullptr && all[0] == '1' && all[1] == '\0') return false;
  const char *v = std::getenv("LTX_R31_COMPAT");
  if (v == nullptr) return true;
  return !(v[0] == '0' && v[1] == '\0');
}

static const char *mangled_strided() {
  // Itanium mangling of an internal-linkage type is _ZTS<len><name>.
  // The length is computed rather than hand-counted so a rename
  // cannot silently break the kernel lookup.
  static const char n[] = "R8BBcastStridedKernel";
  static std::string m = "_ZTS" + std::to_string(sizeof(n) - 1) + n;
  return m.c_str();
}

struct R8State {
  bool ready = false;
  std::optional<sycl::queue> q;
  std::optional<sycl::kernel> k1a;
  std::optional<sycl::kernel> k1b;
  std::optional<sycl::kernel> k1a_c;
  std::optional<sycl::kernel> k1a_c_local;
  std::optional<sycl::kernel> k1a_pipe2;
  std::optional<sycl::kernel> k1a_pipe1;
  std::optional<sycl::kernel> k1b_c;
  std::optional<sycl::kernel> k1h;
  std::optional<sycl::kernel> k1al;
  std::optional<sycl::kernel> k1al16;
  std::optional<sycl::kernel> k1reg1;
  std::optional<sycl::kernel> k1reg2;
  std::optional<sycl::kernel> k1sub;
  std::optional<sycl::kernel> k1sub16;
  std::optional<sycl::kernel> k1group;
  std::optional<sycl::kernel> k1group16;
  std::optional<sycl::kernel> k2;
  std::optional<sycl::kernel> k2_c;
  std::optional<sycl::kernel> r17;
  std::optional<sycl::kernel> r19;
  std::optional<sycl::kernel> k3;
  std::optional<sycl::kernel> k3v;
  std::optional<sycl::kernel> k3_c;
  std::optional<sycl::kernel> k3v_c;
  std::optional<sycl::kernel> k3d;
  std::optional<sycl::kernel> gate;
  std::optional<sycl::kernel> gelu;
  std::optional<sycl::kernel> silu_mul;
  std::optional<sycl::kernel> silu_mul16;
  std::optional<sycl::kernel> silu_mul32;
  std::optional<sycl::kernel> silu_mul_c;
  std::optional<sycl::kernel> silu_mul16_c;
  std::optional<sycl::kernel> silu_mul32_c;
  std::optional<sycl::kernel> fp8w;
  std::optional<sycl::kernel> fp8we;
  std::optional<sycl::kernel> fp8wbits;
  std::optional<sycl::kernel> fp8wb;
  std::optional<sycl::kernel> fp8w16;
  std::optional<sycl::kernel> bbcast;
  std::optional<sycl::kernel> bbcast_strided;
  float *scratch = nullptr;
};

R8State &r8_state() {
  static R8State s;
  return s;
}
std::mutex &r8_mutex() {
  static std::mutex m;
  return m;
}

std::string r8_dir() {
  Dl_info info{};
  if (dladdr((const void *)&r8_state, &info) && info.dli_fname)
    return std::string(info.dli_fname);
  return {};
}

std::string r8_spv_path() {
  std::string d = r8_dir();
  auto pos = d.find_last_of('/');
  std::string dir = (pos == std::string::npos) ? "." : d.substr(0, pos);
  return dir + "/r8_kernels.spv";
}

void r8_init_locked() {
  R8State &s = r8_state();
  if (s.ready) return;
  sycl::queue &q = c10::xpu::getCurrentXPUStream().queue();
  TORCH_CHECK(q.is_in_order(), "r8: torch XPU queue must be in-order");
  s.q = q;
  sycl::context sctx = q.get_context();
  sycl::device sdev = q.get_device();
  constexpr auto L0BE = sycl::backend::ext_oneapi_level_zero;
  auto zctx = sycl::get_native<L0BE>(sctx);
  auto zdev = sycl::get_native<L0BE>(sdev);

  std::string path = r8_spv_path();
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  TORCH_CHECK(f.good(), "r8: cannot open ", path);
  size_t n = (size_t)f.tellg();
  f.seekg(0);
  std::vector<uint8_t> il(n);
  TORCH_CHECK(f.read((char *)il.data(), n).good(), "r8: cannot read ", path);

  ze_module_desc_t mdesc{};
  mdesc.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
  mdesc.format = ZE_MODULE_FORMAT_IL_SPIRV;
  mdesc.inputSize = n;
  mdesc.pInputModule = il.data();
  ze_module_handle_t zmod = nullptr;
  TORCH_CHECK(zeModuleCreate(zctx, zdev, &mdesc, &zmod, nullptr) == ZE_RESULT_SUCCESS,
              "r8: zeModuleCreate failed");

  using sycl::ext::oneapi::level_zero::ownership;
  auto bundle = sycl::make_kernel_bundle<L0BE, sycl::bundle_state::executable>(
      {zmod, ownership::keep}, sctx);

  auto wrap = [&](const char *name) {
    ze_kernel_desc_t kdesc{};
    kdesc.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
    kdesc.pKernelName = name;
    ze_kernel_handle_t zk = nullptr;
    ze_result_t zr = zeKernelCreate(zmod, &kdesc, &zk);
    TORCH_CHECK(zr == ZE_RESULT_SUCCESS, "r8: zeKernelCreate ", name, " zr=", (int)zr);
    return sycl::make_kernel<L0BE>({bundle, zk, ownership::transfer}, sctx);
  };
  s.k1a = wrap("_ZTS13R8K1AdaKernel");
  s.k1b = wrap("_ZTS16R8K1AffineKernel");
  s.k1a_c = wrap("_ZTS19R8K1AdaCompatKernel");
  s.k1a_c_local = wrap("_ZTS24R8K1AdaCompatLocalKernel");
  s.k1a_pipe2 = wrap("_ZTS23R8K1AdaCompatPipeKernel");
  s.k1a_pipe1 = wrap("_ZTS24R8K1AdaCompatPipe1Kernel");
  s.k1b_c = wrap("_ZTS22R8K1AffineCompatKernel");
  s.k1h = wrap("_ZTS17R8K1AdaHoldKernel");
  s.k1al = wrap("_ZTS18R8K1AdaLocalKernel");
  s.k1al16 = wrap("_ZTS20R8K1AdaLocal16Kernel");
  s.k1reg1 = wrap("_ZTS17R8K1AdaReg1Kernel");
  s.k1reg2 = wrap("_ZTS17R8K1AdaReg2Kernel");
  s.k1sub = wrap("_ZTS21R8K1AdaSubgroupKernel");
  s.k1sub16 = wrap("_ZTS23R8K1AdaSubgroup16Kernel");
  s.k1group = wrap("_ZTS24R8K1AdaGroupReduceKernel");
  s.k1group16 = wrap("_ZTS26R8K1AdaGroupReduce16Kernel");
  s.k2 = wrap("_ZTS10R8K2Kernel");
  s.k2_c = wrap("_ZTS16R8K2CompatKernel");
  s.r17 = wrap("_ZTS24R8R17MlpNormAffineKernel");
  s.r19 = wrap("_ZTS19R8R19NormRopeKernel");
  s.k3 = wrap("_ZTS10R8K3Kernel");
  s.k3v = wrap("_ZTS13R8K3VecKernel");
  s.k3_c = wrap("_ZTS16R8K3CompatKernel");
  s.k3v_c = wrap("_ZTS19R8K3VecCompatKernel");
  s.k3d = wrap("_ZTS17R8K3VecDualKernel");
  s.gate = wrap("_ZTS12R8GateKernel");
  s.gelu = wrap("_ZTS12R8GeluKernel");
  s.silu_mul = wrap("_ZTS15R8SiluMulKernel");
  s.silu_mul16 = wrap("_ZTS18R8SiluMulKernelV16");
  s.silu_mul32 = wrap("_ZTS18R8SiluMulKernelV32");
  s.silu_mul_c = wrap("_ZTS21R8SiluMulCompatKernel");
  s.silu_mul16_c = wrap("_ZTS24R8SiluMulCompatKernelV16");
  s.silu_mul32_c = wrap("_ZTS24R8SiluMulCompatKernelV32");
  s.fp8w = wrap("_ZTS16R8Fp8WidenKernel");
  s.fp8we = wrap("_ZTS19R8Fp8WidenEffKernel");
  s.fp8wbits = wrap("_ZTS20R8Fp8WidenBitsKernel");
  s.fp8wb = wrap("_ZTS21R8Fp8WidenBatchKernel");
  s.fp8w16 = wrap("_ZTS18R8Fp8Widen16Kernel");
  s.bbcast = wrap("_ZTS14R8BBcastKernel");
  s.bbcast_strided = wrap(mangled_strided());
  s.scratch = sycl::malloc_device<float>(R8_SCRATCH_FLOATS, q);
  TORCH_CHECK(s.scratch, "r8: scratch alloc failed");
  s.ready = true;
}

R8State &r8_ensure() {
  std::lock_guard<std::mutex> lk(r8_mutex());
  r8_init_locked();
  return r8_state();
}

bool r8_tensors_ok(const torch::Tensor &t) {
  return t.is_xpu() && t.dtype() == torch::kBFloat16;
}

void r8_check_row_major(const torch::Tensor &x, int64_t C) {
  TORCH_CHECK(r8_tensors_ok(x), "r8: x must be XPU bf16");
  TORCH_CHECK(x.stride(-1) == 1, "r8: x innermost stride must be 1");
  TORCH_CHECK(x.is_contiguous(), "r8: x must be contiguous");
  TORCH_CHECK(C >= 64 && C <= R8_MAX_C, "r8: C out of range");
}

void r8_check_part(int64_t C, const torch::Tensor &tab, const torch::Tensor &ts) {
  TORCH_CHECK(r8_tensors_ok(tab) && tab.numel() == C, "r8: table must be XPU bf16 numel==C");
  TORCH_CHECK(r8_tensors_ok(ts) && ts.size(-1) == C && ts.stride(-1) == 1,
              "r8: ts must be XPU bf16 matching last dim, stride 1");
}

torch::Tensor k1_impl(torch::Tensor x, torch::Tensor tbs, torch::Tensor tss, int64_t tss_rs,
                      torch::Tensor tbh, torch::Tensor tsh, int64_t tsh_rs, double eps,
                      bool do_norm, bool hold) {
  const int64_t C = x.size(-1);
  r8_check_row_major(x, C);
  r8_check_part(C, tbs, tss);
  r8_check_part(C, tbh, tsh);
  torch::Tensor out = at::empty_like(x);
  const int64_t rows = x.numel() / C;
  TORCH_CHECK((size_t)rows * 32 <= R8_SCRATCH_FLOATS, "r8: too many rows for scratch");
  R8State &s = r8_ensure();
  R8K1Args A{};
  A.x = x.data_ptr();
  A.tbs = tbs.data_ptr();
  A.tss = tss.data_ptr();
  A.tbh = tbh.data_ptr();
  A.tsh = tsh.data_ptr();
  A.out = out.data_ptr();
  A.scratch = s.scratch;
  A.tss_rs = tss_rs;
  A.tsh_rs = tsh_rs;
  A.rows = rows;
  A.C = C;
  A.eps = (float)eps;
  sycl::kernel k = do_norm ? (hold ? *s.k1h : (r8_r31_compat_on() ? *s.k1a_c : *s.k1a))
                           : (r8_r31_compat_on() ? *s.k1b_c : *s.k1b);
  s.q->submit([&](sycl::handler &h) {
    h.set_arg(0, A);
    h.parallel_for(sycl::nd_range<1>((size_t)rows * R8_WG, R8_WG), k);
  });
  return out;
}

torch::Tensor k1_compat_local_impl(torch::Tensor x, torch::Tensor tbs, torch::Tensor tss,
                                   int64_t tss_rs, torch::Tensor tbh, torch::Tensor tsh,
                                   int64_t tsh_rs, double eps, int64_t wg) {
  const int64_t C = x.size(-1);
  r8_check_row_major(x, C);
  r8_check_part(C, tbs, tss);
  r8_check_part(C, tbh, tsh);
  TORCH_CHECK(wg >= 32 && wg <= 1024 && wg % 32 == 0,
              "r41: compat-local WG must be a multiple of 32 in [32,1024]");
  TORCH_CHECK(C % 8 == 0, "r41: compat-local vector width is 8");
  TORCH_CHECK((size_t)wg * 32 <= R8_SCRATCH_FLOATS,
              "r41: compat-local scratch exceeds the device scratch allocation");
  const auto aligned = [](const torch::Tensor &t) {
    return (reinterpret_cast<uintptr_t>(t.data_ptr()) & 15u) == 0;
  };
  TORCH_CHECK(aligned(x) && aligned(tbs) && aligned(tbh) && aligned(tss) && aligned(tsh),
              "r41: compat-local tensors must be 16-byte aligned");
  if (tss_rs != 0) TORCH_CHECK(tss_rs % 8 == 0, "r41: tss_rs must be 8-element aligned");
  if (tsh_rs != 0) TORCH_CHECK(tsh_rs % 8 == 0, "r41: tsh_rs must be 8-element aligned");
  const int64_t rows = x.numel() / C;
  torch::Tensor out = at::empty_like(x);
  R8State &s = r8_ensure();
  R8K1Args A{};
  A.x = x.data_ptr();
  A.tbs = tbs.data_ptr();
  A.tss = tss.data_ptr();
  A.tbh = tbh.data_ptr();
  A.tsh = tsh.data_ptr();
  A.out = out.data_ptr();
  A.scratch = s.scratch;
  A.tss_rs = tss_rs;
  A.tsh_rs = tsh_rs;
  A.rows = rows;
  A.C = C;
  A.eps = (float)eps;
  A.prefetch = 0;
  sycl::kernel k = *s.k1a_c_local;
  s.q->submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> scratch(
        sycl::range<1>(static_cast<size_t>(wg)), h);
    h.set_arg(0, A);
    h.set_arg(1, scratch);
    h.parallel_for(sycl::nd_range<1>((size_t)rows * wg, wg), k);
  });
  return out;
}

torch::Tensor k1_compat_pipe_impl(torch::Tensor x, torch::Tensor tbs, torch::Tensor tss,
                                  int64_t tss_rs, torch::Tensor tbh, torch::Tensor tsh,
                                  int64_t tsh_rs, double eps, int64_t wg) {
  const int64_t C = x.size(-1);
  r8_check_row_major(x, C);
  r8_check_part(C, tbs, tss);
  r8_check_part(C, tbh, tsh);
  TORCH_CHECK(wg >= 32 && wg <= 1024 && wg % 32 == 0,
              "r41: compat-pipe WG must be a multiple of 32 in [32,1024]");
  TORCH_CHECK(C % 8 == 0, "r41: compat-pipe vector width is 8");
  TORCH_CHECK(C == wg * 8 || C == 2 * wg * 8,
              "r41: compat-pipe requires C/wg/8 in {1,2}");
  const auto aligned = [](const torch::Tensor &t) {
    return (reinterpret_cast<uintptr_t>(t.data_ptr()) & 15u) == 0;
  };
  TORCH_CHECK(aligned(x) && aligned(tbs) && aligned(tbh) && aligned(tss) && aligned(tsh),
              "r41: compat-pipe tensors must be 16-byte aligned");
  if (tss_rs != 0) TORCH_CHECK(tss_rs % 8 == 0, "r41: tss_rs must be 8-element aligned");
  if (tsh_rs != 0) TORCH_CHECK(tsh_rs % 8 == 0, "r41: tsh_rs must be 8-element aligned");
  const int64_t rows = x.numel() / C;
  torch::Tensor out = at::empty_like(x);
  R8State &s = r8_ensure();
  R8K1Args A{};
  A.x = x.data_ptr();
  A.tbs = tbs.data_ptr();
  A.tss = tss.data_ptr();
  A.tbh = tbh.data_ptr();
  A.tsh = tsh.data_ptr();
  A.out = out.data_ptr();
  A.scratch = s.scratch;
  A.tss_rs = tss_rs;
  A.tsh_rs = tsh_rs;
  A.rows = rows;
  A.C = C;
  A.eps = (float)eps;
  A.prefetch = 0;
  sycl::kernel k = (C == wg * 8) ? *s.k1a_pipe1 : *s.k1a_pipe2;
  const int64_t tasks = (rows + 1) / 2;
  s.q->submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> scratch(
        sycl::range<1>(static_cast<size_t>(2 * wg)), h);
    h.set_arg(0, A);
    h.set_arg(1, scratch);
    h.parallel_for(sycl::nd_range<1>((size_t)tasks * wg, wg), k);
  });
  return out;
}

torch::Tensor k1_local_impl(torch::Tensor x, torch::Tensor tbs, torch::Tensor tss,
                           int64_t tss_rs, torch::Tensor tbh, torch::Tensor tsh,
                           int64_t tsh_rs, double eps, int64_t wg, int64_t vec,
                           int64_t prefetch) {
  const int64_t C = x.size(-1);
  r8_check_row_major(x, C);
  r8_check_part(C, tbs, tss);
  r8_check_part(C, tbh, tsh);
  TORCH_CHECK(wg >= 32 && wg <= 1024 && wg % 32 == 0,
              "r18: K1 local WG must be a multiple of 32 in [32,1024]");
  TORCH_CHECK((vec == 8 || vec == 16), "r18: K1 local vector width must be 8 or 16");
  TORCH_CHECK(prefetch >= 0 && prefetch <= 2, "r18: K1 prefetch must be 0, 1, or 2");
  TORCH_CHECK(C % vec == 0, "r18: K1 local vector width must divide C");
  const int64_t iterations = (C + wg * vec - 1) / (wg * vec);
  TORCH_CHECK(iterations <= 4, "r18: K1 local register-hold shape exceeds four vectors");
  const int64_t rows = x.numel() / C;
  auto aligned = [](const torch::Tensor &t, uintptr_t bytes) {
    return (reinterpret_cast<uintptr_t>(t.data_ptr()) & (bytes - 1u)) == 0;
  };
  const uintptr_t required = vec == 16 ? 32u : 16u;
  TORCH_CHECK(aligned(x, required) && aligned(tbs, required) && aligned(tbh, required) &&
                  aligned(tss, required) && aligned(tsh, required),
              "r18: K1 local tensors must be vector aligned");
  if (tss_rs != 0) TORCH_CHECK(tss_rs % vec == 0, "r18: tss_rs must match vector width");
  if (tsh_rs != 0) TORCH_CHECK(tsh_rs % vec == 0, "r18: tsh_rs must match vector width");
  torch::Tensor out = at::empty_like(x);
  R8State &s = r8_ensure();
  R8K1Args A{};
  A.x = x.data_ptr();
  A.tbs = tbs.data_ptr();
  A.tss = tss.data_ptr();
  A.tbh = tbh.data_ptr();
  A.tsh = tsh.data_ptr();
  A.out = out.data_ptr();
  A.scratch = nullptr;
  A.tss_rs = tss_rs;
  A.tsh_rs = tsh_rs;
  A.rows = rows;
  A.C = C;
  A.eps = (float)eps;
  A.prefetch = prefetch;
  sycl::kernel k = vec == 16 ? *s.k1al16 : *s.k1al;
  s.q->submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> scratch(
        sycl::range<1>(static_cast<size_t>(wg)), h);
    h.set_arg(0, A);
    h.set_arg(1, scratch);
    h.parallel_for(sycl::nd_range<1>((size_t)rows * wg, wg), k);
  });
  return out;
}

torch::Tensor k1_reg_impl(torch::Tensor x, torch::Tensor tbs, torch::Tensor tss,
                          int64_t tss_rs, torch::Tensor tbh, torch::Tensor tsh,
                          int64_t tsh_rs, double eps, int64_t wg,
                          int64_t prefetch) {
  const int64_t C = x.size(-1);
  r8_check_row_major(x, C);
  r8_check_part(C, tbs, tss);
  r8_check_part(C, tbh, tsh);
  TORCH_CHECK(wg >= 32 && wg <= 1024 && wg % 32 == 0,
              "r18: K1 register WG must be a multiple of 32 in [32,1024]");
  TORCH_CHECK(prefetch >= 0 && prefetch <= 2, "r18: K1 prefetch must be 0, 1, or 2");
  TORCH_CHECK(C % 8 == 0, "r18: K1 register vector width is 8");
  const int64_t iterations = (C + wg * 8 - 1) / (wg * 8);
  TORCH_CHECK(iterations <= 2, "r18: K1 register shape exceeds two vectors");
  const auto aligned = [](const torch::Tensor &t) {
    return (reinterpret_cast<uintptr_t>(t.data_ptr()) & 15u) == 0;
  };
  TORCH_CHECK(aligned(x) && aligned(tbs) && aligned(tbh) && aligned(tss) && aligned(tsh),
              "r18: K1 register tensors must be 16-byte aligned");
  if (tss_rs != 0) TORCH_CHECK(tss_rs % 8 == 0, "r18: tss_rs must be 8-element aligned");
  if (tsh_rs != 0) TORCH_CHECK(tsh_rs % 8 == 0, "r18: tsh_rs must be 8-element aligned");
  const int64_t rows = x.numel() / C;
  torch::Tensor out = at::empty_like(x);
  R8State &s = r8_ensure();
  R8K1Args A{};
  A.x = x.data_ptr();
  A.tbs = tbs.data_ptr();
  A.tss = tss.data_ptr();
  A.tbh = tbh.data_ptr();
  A.tsh = tsh.data_ptr();
  A.out = out.data_ptr();
  A.scratch = nullptr;
  A.tss_rs = tss_rs;
  A.tsh_rs = tsh_rs;
  A.rows = rows;
  A.C = C;
  A.eps = (float)eps;
  A.prefetch = prefetch;
  sycl::kernel k = iterations == 1 ? *s.k1reg1 : *s.k1reg2;
  s.q->submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> scratch(
        sycl::range<1>(static_cast<size_t>(wg)), h);
    h.set_arg(0, A);
    h.set_arg(1, scratch);
    h.parallel_for(sycl::nd_range<1>((size_t)rows * wg, wg), k);
  });
  return out;
}

torch::Tensor k1_subgroup_impl(torch::Tensor x, torch::Tensor tbs, torch::Tensor tss,
                               int64_t tss_rs, torch::Tensor tbh, torch::Tensor tsh,
                               int64_t tsh_rs, double eps, int64_t wg, int64_t vec,
                               int64_t prefetch) {
  const int64_t C = x.size(-1);
  r8_check_row_major(x, C);
  r8_check_part(C, tbs, tss);
  r8_check_part(C, tbh, tsh);
  TORCH_CHECK(wg >= 32 && wg <= 1024 && wg % 32 == 0,
              "r18: K1 subgroup WG must be a multiple of 32 in [32,1024]");
  TORCH_CHECK((vec == 8 || vec == 16), "r18: K1 subgroup vector width must be 8 or 16");
  TORCH_CHECK(prefetch >= 0 && prefetch <= 2, "r18: K1 prefetch must be 0, 1, or 2");
  TORCH_CHECK(C % vec == 0, "r18: K1 subgroup vector width must divide C");
  const auto aligned = [](const torch::Tensor &t, uintptr_t bytes) {
    return (reinterpret_cast<uintptr_t>(t.data_ptr()) & (bytes - 1u)) == 0;
  };
  const uintptr_t required = vec == 16 ? 32u : 16u;
  TORCH_CHECK(aligned(x, required) && aligned(tbs, required) && aligned(tbh, required) &&
                  aligned(tss, required) && aligned(tsh, required),
              "r18: K1 subgroup tensors must be vector aligned");
  if (tss_rs != 0) TORCH_CHECK(tss_rs % vec == 0, "r18: tss_rs must match vector width");
  if (tsh_rs != 0) TORCH_CHECK(tsh_rs % vec == 0, "r18: tsh_rs must match vector width");
  const int64_t rows = x.numel() / C;
  torch::Tensor out = at::empty_like(x);
  R8State &s = r8_ensure();
  R8K1Args A{};
  A.x = x.data_ptr();
  A.tbs = tbs.data_ptr();
  A.tss = tss.data_ptr();
  A.tbh = tbh.data_ptr();
  A.tsh = tsh.data_ptr();
  A.out = out.data_ptr();
  A.scratch = nullptr;
  A.tss_rs = tss_rs;
  A.tsh_rs = tsh_rs;
  A.rows = rows;
  A.C = C;
  A.eps = (float)eps;
  A.prefetch = prefetch;
  const int64_t subgroups = wg / 32;
  const int64_t groups = (rows + subgroups - 1) / subgroups;
  sycl::kernel k = vec == 16 ? *s.k1sub16 : *s.k1sub;
  s.q->submit([&](sycl::handler &h) {
    h.set_arg(0, A);
    h.parallel_for(sycl::nd_range<1>((size_t)groups * wg, wg), k);
  });
  return out;
}

torch::Tensor k1_group_impl(torch::Tensor x, torch::Tensor tbs, torch::Tensor tss,
                            int64_t tss_rs, torch::Tensor tbh, torch::Tensor tsh,
                            int64_t tsh_rs, double eps, int64_t wg, int64_t vec,
                            int64_t prefetch) {
  const int64_t C = x.size(-1);
  r8_check_row_major(x, C);
  r8_check_part(C, tbs, tss);
  r8_check_part(C, tbh, tsh);
  TORCH_CHECK(wg >= 32 && wg <= 1024 && wg % 32 == 0,
              "r18: K1 group WG must be a multiple of 32 in [32,1024]");
  TORCH_CHECK((vec == 8 || vec == 16), "r18: K1 group vector width must be 8 or 16");
  TORCH_CHECK(prefetch >= 0 && prefetch <= 2, "r18: K1 prefetch must be 0, 1, or 2");
  TORCH_CHECK(C % vec == 0, "r18: K1 group vector width must divide C");
  const auto aligned = [](const torch::Tensor &t, uintptr_t bytes) {
    return (reinterpret_cast<uintptr_t>(t.data_ptr()) & (bytes - 1u)) == 0;
  };
  const uintptr_t required = vec == 16 ? 32u : 16u;
  TORCH_CHECK(aligned(x, required) && aligned(tbs, required) && aligned(tbh, required) &&
                  aligned(tss, required) && aligned(tsh, required),
              "r18: K1 group tensors must be vector aligned");
  if (tss_rs != 0) TORCH_CHECK(tss_rs % vec == 0, "r18: tss_rs must match vector width");
  if (tsh_rs != 0) TORCH_CHECK(tsh_rs % vec == 0, "r18: tsh_rs must match vector width");
  const int64_t rows = x.numel() / C;
  torch::Tensor out = at::empty_like(x);
  R8State &s = r8_ensure();
  R8K1Args A{};
  A.x = x.data_ptr();
  A.tbs = tbs.data_ptr();
  A.tss = tss.data_ptr();
  A.tbh = tbh.data_ptr();
  A.tsh = tsh.data_ptr();
  A.out = out.data_ptr();
  A.scratch = nullptr;
  A.tss_rs = tss_rs;
  A.tsh_rs = tsh_rs;
  A.rows = rows;
  A.C = C;
  A.eps = (float)eps;
  A.prefetch = prefetch;
  sycl::kernel k = vec == 16 ? *s.k1group16 : *s.k1group;
  s.q->submit([&](sycl::handler &h) {
    h.set_arg(0, A);
    h.parallel_for(sycl::nd_range<1>((size_t)rows * wg, wg), k);
  });
  return out;
}

std::vector<torch::Tensor> k2_impl(torch::Tensor x, torch::Tensor y, torch::Tensor tbg,
                                   torch::Tensor tsg, int64_t tsg_rs, double eps) {
  const int64_t C = x.size(-1);
  r8_check_row_major(x, C);
  TORCH_CHECK(y.sizes() == x.sizes() && r8_tensors_ok(y) && y.is_contiguous(),
              "r8: y must match x (XPU bf16 contiguous)");
  r8_check_part(C, tbg, tsg);
  torch::Tensor fma = at::empty_like(x);
  torch::Tensor out = at::empty_like(x);
  const int64_t rows = x.numel() / C;
  TORCH_CHECK((size_t)rows * 32 <= R8_SCRATCH_FLOATS, "r8: too many rows for scratch");
  R8State &s = r8_ensure();
  R8K2Args A{};
  A.x = x.data_ptr();
  A.y = y.data_ptr();
  A.tbg = tbg.data_ptr();
  A.tsg = tsg.data_ptr();
  A.fma = fma.data_ptr();
  A.out = out.data_ptr();
  A.scratch = s.scratch;
  A.tsg_rs = tsg_rs;
  A.rows = rows;
  A.C = C;
  A.eps = (float)eps;
  s.q->submit([&](sycl::handler &h) {
    h.set_arg(0, A);
    h.parallel_for(sycl::nd_range<1>((size_t)rows * R8_WG, R8_WG),
                   r8_r31_compat_on() ? *s.k2_c : *s.k2);
  });
  return {fma, out};
}

void r8_check_r17_norm_affine(const torch::Tensor &x, int64_t C,
                                 const torch::Tensor &norm_weight,
                                 const torch::Tensor &scale, const torch::Tensor &shift,
                                 const torch::Tensor &out) {
  TORCH_CHECK(r8_tensors_ok(norm_weight) && norm_weight.numel() == C &&
                  norm_weight.is_contiguous() && norm_weight.stride(-1) == 1,
              "r17: norm_weight must be contiguous XPU bf16 numel==C");
  TORCH_CHECK(r8_tensors_ok(scale) && scale.numel() == C && scale.is_contiguous() &&
                  scale.stride(-1) == 1,
              "r17: scale must be contiguous XPU bf16 numel==C");
  TORCH_CHECK(r8_tensors_ok(shift) && shift.numel() == C && shift.is_contiguous() &&
                  shift.stride(-1) == 1,
              "r17: shift must be contiguous XPU bf16 numel==C");
  TORCH_CHECK(out.sizes() == x.sizes() && r8_tensors_ok(out) && out.is_contiguous() &&
                  out.stride(-1) == 1,
              "r17: out must match x and be contiguous XPU bf16");
  TORCH_CHECK(norm_weight.device() == x.device() && scale.device() == x.device() &&
                  shift.device() == x.device() && out.device() == x.device(),
              "r17: all tensors must be on the same XPU");
  const auto aligned = [](const torch::Tensor &t) {
    return (reinterpret_cast<uintptr_t>(t.data_ptr()) & 15u) == 0;
  };
  TORCH_CHECK(aligned(x) && aligned(norm_weight) && aligned(scale) && aligned(shift) && aligned(out),
              "r17: all tensors must be 16-byte aligned");
}

torch::Tensor r17_mlp_norm_affine_impl(torch::Tensor x, torch::Tensor norm_weight,
                                       torch::Tensor scale, torch::Tensor shift,
                                       torch::Tensor out, double eps) {
  const int64_t C = x.size(-1);
  r8_check_row_major(x, C);
  r8_check_r17_norm_affine(x, C, norm_weight, scale, shift, out);
  const int64_t rows = x.numel() / C;
  TORCH_CHECK(rows > 0, "r17: empty norm input");
  R8State &s = r8_ensure();
  R8R17MlpNormAffineArgs A{};
  A.x = x.data_ptr();
  A.weight = norm_weight.data_ptr();
  A.scale = scale.data_ptr();
  A.shift = shift.data_ptr();
  A.out = out.data_ptr();
  A.scratch = nullptr;
  A.rows = rows;
  A.C = C;
  A.eps = (float)eps;
  const int64_t groups = (rows + R17_NORM_ROWS_PER_GROUP - 1) / R17_NORM_ROWS_PER_GROUP;
  s.q->submit([&](sycl::handler &h) {
    h.set_arg(0, A);
    h.parallel_for(
        sycl::nd_range<1>((size_t)groups * R17_NORM_WG, R17_NORM_WG), *s.r17);
  });
  return out;
}

std::vector<torch::Tensor> r19_norm_rope_impl(
    torch::Tensor q, torch::Tensor k, torch::Tensor q_weight, torch::Tensor k_weight,
    torch::Tensor cos, torch::Tensor sin, int64_t B, int64_t H, int64_t T,
    int64_t Dh, double eps, double q_scale, double k_scale) {
  TORCH_CHECK(q.is_xpu() && k.is_xpu(), "r19: q/k must be XPU tensors");
  TORCH_CHECK(q.scalar_type() == torch::kBFloat16 && k.scalar_type() == torch::kBFloat16,
              "r19: q/k must be bf16");
  TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && q.sizes() == k.sizes(),
              "r19: q/k must be matching rank-3 tensors");
  TORCH_CHECK(B > 0 && H > 0 && T > 0 && Dh > 0 && B == q.size(0) && T == q.size(1),
              "r19: invalid B/H/T metadata");
  const int64_t C = H * Dh * 2;
  TORCH_CHECK(q.size(2) == C && C >= 64 && C <= R8_MAX_C && C % 8 == 0 && Dh % 8 == 0,
              "r19: unsupported q/k channel shape");
  TORCH_CHECK(q.stride(2) == 1 && k.stride(2) == 1,
              "r19: q/k innermost stride must be one");
  TORCH_CHECK(q.stride(1) % 8 == 0 && q.stride(0) % 8 == 0 &&
                  k.stride(1) % 8 == 0 && k.stride(0) % 8 == 0,
              "r19: q/k outer strides must be 8-element aligned");
  TORCH_CHECK(q_weight.is_xpu() && k_weight.is_xpu() &&
                  q_weight.scalar_type() == torch::kBFloat16 &&
                  k_weight.scalar_type() == torch::kBFloat16 &&
                  q_weight.numel() == C && k_weight.numel() == C &&
                  q_weight.is_contiguous() && k_weight.is_contiguous(),
              "r19: q/k weights must be contiguous XPU bf16 vectors");
  TORCH_CHECK(q_weight.device() == q.device() && k_weight.device() == q.device(),
              "r19: q/k weights must be on the q device");
  TORCH_CHECK(cos.is_xpu() && sin.is_xpu() && cos.dim() == 4 && sin.dim() == 4 &&
                  cos.sizes() == sin.sizes() && cos.scalar_type() == torch::kBFloat16 &&
                  sin.scalar_type() == torch::kBFloat16,
              "r19: cos/sin must be matching XPU bf16 rank-4 tensors");
  TORCH_CHECK((cos.size(0) == 1 || cos.size(0) == B) && cos.size(1) == H &&
                  cos.size(2) == T && cos.size(3) == Dh,
              "r19: cos/sin shape metadata mismatch");
  TORCH_CHECK(cos.stride(3) == 1 && sin.stride(3) == 1 &&
                  cos.stride(0) % 8 == 0 && cos.stride(1) % 8 == 0 &&
                  cos.stride(2) % 8 == 0 && sin.stride(0) % 8 == 0 &&
                  sin.stride(1) % 8 == 0 && sin.stride(2) % 8 == 0,
              "r19: cos/sin strides are not vector compatible");
  const auto aligned = [](const torch::Tensor &t) {
    return (reinterpret_cast<uintptr_t>(t.data_ptr()) & 15u) == 0;
  };
  TORCH_CHECK(aligned(q) && aligned(k) && aligned(q_weight) && aligned(k_weight) &&
                  aligned(cos) && aligned(sin),
              "r19: all tensors must be 16-byte aligned");
  const int64_t default_wg = C / 4 < 128 ? (C / 4 < 32 ? 32 : C / 4) : 128;
  const int64_t wg = r19_env_int("LTX_R19_WG", default_wg, 32, 1024);
  TORCH_CHECK(wg % 32 == 0, "r19: LTX_R19_WG must be a multiple of 32");
  const int64_t rows = B * T;
  TORCH_CHECK(rows * 2 * 32 <= static_cast<int64_t>(R8_SCRATCH_FLOATS),
              "r19: row scratch exceeds the device scratch allocation");
  torch::Tensor outq = at::empty(q.sizes(), q.options());
  torch::Tensor outk = at::empty(k.sizes(), k.options());
  R8State &s = r8_ensure();
  R8R19NormRopeArgs A{};
  A.q = q.data_ptr();
  A.k = k.data_ptr();
  A.q_weight = q_weight.data_ptr();
  A.k_weight = k_weight.data_ptr();
  A.cos = cos.data_ptr();
  A.sin = sin.data_ptr();
  A.scratch = s.scratch;
  A.outq = outq.data_ptr();
  A.outk = outk.data_ptr();
  A.q_batch_stride = q.stride(0);
  A.q_row_stride = q.stride(1);
  A.k_batch_stride = k.stride(0);
  A.k_row_stride = k.stride(1);
  A.cos_stride0 = cos.stride(0);
  A.cos_stride1 = cos.stride(1);
  A.cos_stride2 = cos.stride(2);
  A.cos_stride3 = cos.stride(3);
  A.sin_stride0 = sin.stride(0);
  A.sin_stride1 = sin.stride(1);
  A.sin_stride2 = sin.stride(2);
  A.sin_stride3 = sin.stride(3);
  A.B = B;
  A.H = H;
  A.T = T;
  A.Dh = Dh;
  A.C = C;
  A.eps = static_cast<float>(eps);
  A.q_scale = static_cast<float>(q_scale);
  A.k_scale = static_cast<float>(k_scale);
  s.q->submit([&](sycl::handler &h) {
    h.set_arg(0, A);
    h.parallel_for(sycl::nd_range<1>(static_cast<size_t>(rows * wg), wg), *s.r19);
  });
  return {outq, outk};
}

torch::Tensor k3_impl_common(torch::Tensor x, torch::IntArrayRef xs, torch::Tensor cos,
                              torch::IntArrayRef cs, torch::Tensor sin, torch::IntArrayRef ss,
                              int64_t B, int64_t H, int64_t T, int64_t Dh, bool vec) {
  TORCH_CHECK(r8_tensors_ok(x) && r8_tensors_ok(cos) && r8_tensors_ok(sin),
              "r8: k3 tensors must be XPU bf16");
  TORCH_CHECK(xs.size() == 4 && cs.size() == 4 && ss.size() == 4, "r8: k3 needs 4 strides");
  torch::Tensor out = at::empty({B, H, T, Dh * 2}, x.options());
  const int64_t fibers = B * H * T;
  R8State &s = r8_ensure();
  R8K3Args A{};
  A.x = x.data_ptr();
  A.cos = cos.data_ptr();
  A.sin = sin.data_ptr();
  A.out = out.data_ptr();
  A.xs0 = xs[0];
  A.xs1 = xs[1];
  A.xs2 = xs[2];
  A.xs3 = xs[3];
  A.cs0 = cs[0];
  A.cs1 = cs[1];
  A.cs2 = cs[2];
  A.cs3 = cs[3];
  A.ss0 = ss[0];
  A.ss1 = ss[1];
  A.ss2 = ss[2];
  A.ss3 = ss[3];
  A.B = B;
  A.H = H;
  A.T = T;
  A.Dh = Dh;
  if (vec) {
    const int64_t nvec = Dh / 8;
    TORCH_CHECK(nvec > 0, "r9d: Dh must be a multiple of 8");
    const int64_t tasks = fibers * nvec;
    constexpr int64_t K3V_WG = 64;
    const int64_t groups = (tasks + K3V_WG - 1) / K3V_WG;
    s.q->submit([&](sycl::handler &h) {
      h.set_arg(0, A);
      h.parallel_for(sycl::nd_range<1>((size_t)groups * K3V_WG, K3V_WG),
                     r8_r31_compat_on() ? *s.k3v_c : *s.k3v);
    });
  } else {
    s.q->submit([&](sycl::handler &h) {
      h.set_arg(0, A);
      h.parallel_for(sycl::nd_range<1>((size_t)fibers * 16, 16),
                     r8_r31_compat_on() ? *s.k3_c : *s.k3);
    });
  }
  return out;
}

torch::Tensor k3_impl(torch::Tensor x, torch::IntArrayRef xs, torch::Tensor cos,
                      torch::IntArrayRef cs, torch::Tensor sin, torch::IntArrayRef ss,
                      int64_t B, int64_t H, int64_t T, int64_t Dh) {
  return k3_impl_common(x, xs, cos, cs, sin, ss, B, H, T, Dh, false);
}

torch::Tensor k3_vec_impl(torch::Tensor x, torch::IntArrayRef xs, torch::Tensor cos,
                          torch::IntArrayRef cs, torch::Tensor sin, torch::IntArrayRef ss,
                          int64_t B, int64_t H, int64_t T, int64_t Dh) {
  return k3_impl_common(x, xs, cos, cs, sin, ss, B, H, T, Dh, true);
}

std::vector<torch::Tensor> k3_dual_impl(
    torch::Tensor xq, torch::IntArrayRef qxs, torch::Tensor xk,
    torch::IntArrayRef kxs, torch::Tensor cos, torch::IntArrayRef cs,
    torch::Tensor sin, torch::IntArrayRef ss, int64_t B, int64_t H, int64_t T,
    int64_t Dh) {
  TORCH_CHECK(r8_tensors_ok(xq) && r8_tensors_ok(xk) && r8_tensors_ok(cos) &&
                  r8_tensors_ok(sin),
              "r18: K3 dual tensors must be XPU bf16");
  TORCH_CHECK(xq.sizes() == xk.sizes(), "r18: K3 dual q/k shapes must match");
  TORCH_CHECK(qxs.size() == 4 && kxs.size() == 4 && cs.size() == 4 && ss.size() == 4,
              "r18: K3 dual needs four strides per tensor");
  TORCH_CHECK(Dh > 0 && Dh % 8 == 0, "r18: K3 dual Dh must be a multiple of 8");
  const int64_t fibers = B * H * T;
  const int64_t nvec = Dh / 8;
  const int64_t tasks = fibers * nvec;
  const int64_t wg = r18_env_int("LTX_R18_K3_WG", 64, 32, 128);
  TORCH_CHECK(wg % 32 == 0, "r18: K3 dual WG must be a multiple of 32");
  const int64_t prefetch =
      r18_env_flag_with_common("LTX_R18_K3_PREFETCH", "LTX_R18_PREFETCH", 1);
  const auto aligned = [](const torch::Tensor &t) {
    return (reinterpret_cast<uintptr_t>(t.data_ptr()) & 15u) == 0;
  };
  TORCH_CHECK(aligned(xq) && aligned(xk) && aligned(cos) && aligned(sin),
              "r18: K3 dual tensors must be 16-byte aligned");
  for (int i = 0; i < 3; ++i) TORCH_CHECK(qxs[i] % 8 == 0, "r18: q outer stride must be 8-element aligned");
  for (int i = 0; i < 3; ++i) TORCH_CHECK(kxs[i] % 8 == 0, "r18: k outer stride must be 8-element aligned");
  for (int i = 0; i < 3; ++i) TORCH_CHECK(cs[i] % 8 == 0, "r18: cos outer stride must be 8-element aligned");
  for (int i = 0; i < 3; ++i) TORCH_CHECK(ss[i] % 8 == 0, "r18: sin outer stride must be 8-element aligned");
  torch::Tensor outq = at::empty({B, H, T, Dh * 2}, xq.options());
  torch::Tensor outk = at::empty({B, H, T, Dh * 2}, xk.options());
  R8State &s = r8_ensure();
  R8K3DualArgs A{};
  A.xq = xq.data_ptr();
  A.xk = xk.data_ptr();
  A.cos = cos.data_ptr();
  A.sin = sin.data_ptr();
  A.outq = outq.data_ptr();
  A.outk = outk.data_ptr();
  A.qxs0 = qxs[0]; A.qxs1 = qxs[1]; A.qxs2 = qxs[2]; A.qxs3 = qxs[3];
  A.kxs0 = kxs[0]; A.kxs1 = kxs[1]; A.kxs2 = kxs[2]; A.kxs3 = kxs[3];
  A.cs0 = cs[0]; A.cs1 = cs[1]; A.cs2 = cs[2]; A.cs3 = cs[3];
  A.ss0 = ss[0]; A.ss1 = ss[1]; A.ss2 = ss[2]; A.ss3 = ss[3];
  A.B = B; A.H = H; A.T = T; A.Dh = Dh; A.prefetch = prefetch;
  const int64_t groups = (tasks + wg - 1) / wg;
  s.q->submit([&](sycl::handler &h) {
    h.set_arg(0, A);
    h.parallel_for(sycl::nd_range<1>((size_t)groups * wg, wg), *s.k3d);
  });
  return {outq, outk};
}

// R10D F2: out = attn * gates[i / dh]  (bf16, per-head broadcast).
torch::Tensor gate_mul_impl(torch::Tensor attn, torch::Tensor gates, int64_t dh) {
  const int64_t n = attn.numel();
  TORCH_CHECK(r8_tensors_ok(attn) && attn.is_contiguous() && attn.stride(-1) == 1,
              "r10d: attn must be XPU bf16 contiguous");
  TORCH_CHECK(r8_tensors_ok(gates) && gates.is_contiguous(),
              "r10d: gates must be XPU bf16 contiguous");
  TORCH_CHECK(dh > 0 && dh % 8 == 0 && n % 8 == 0, "r10d: bad dh/n");
  TORCH_CHECK(gates.numel() == n / dh, "r10d: gates numel must be n/dh");
  torch::Tensor out = at::empty_like(attn);
  R8State &s = r8_ensure();
  R8GateArgs A{};
  A.attn = attn.data_ptr();
  A.gates = gates.data_ptr();
  A.out = out.data_ptr();
  A.n = n;
  A.dh = dh;
  const int64_t tasks = (n / 8 + R8_WG - 1) / R8_WG;
  s.q->submit([&](sycl::handler &h) {
    h.set_arg(0, A);
    h.parallel_for(sycl::nd_range<1>((size_t)tasks * R8_WG, R8_WG), *s.gate);
  });
  return out;
}

// R40: in-place bf16 mul/div where the second operand is broadcast, vectorised
// over the contiguous innermost run.  ``outer`` holds the four outer extents in
// the flat order the kernel walks and ``inner`` the innermost contiguous extent;
// ``bstrides`` are the broadcast operand's element strides in that same order,
// with 0 marking a broadcast dimension.
torch::Tensor bcast_inplace_impl(torch::Tensor x, torch::Tensor b,
                                 std::vector<int64_t> outer, int64_t inner,
                                 std::vector<int64_t> bstrides, int64_t op) {
  TORCH_CHECK(r8_tensors_ok(x) && r8_tensors_ok(b), "r40: XPU bf16 tensors required");
  TORCH_CHECK(x.dim() == 5, "r40: x must be 5-D");
  // Layout is validated by the explicit stride checks below, which is the
  // channels_last_3d form; is_contiguous() would reject it.
  TORCH_CHECK(b.numel() > 0, "r40: b must be non-empty");
  // The device index math divides and takes a modulus by every outer extent, so
  // none of them may be zero.
  for (int i = 0; i < 4; ++i) {
    TORCH_CHECK(outer[i] >= 1, "r40: outer extent ", i, " must be >= 1");
  }
  // The vectorised path moves 16 bytes per work item, so the base must be
  // 16-byte aligned; b is read as u16 so it needs 2-byte alignment.
  TORCH_CHECK(((uintptr_t)x.data_ptr() % 16) == 0, "r40: x must be 16-byte aligned");
  TORCH_CHECK(((uintptr_t)b.data_ptr() % 2) == 0, "r40: b must be 2-byte aligned");
  TORCH_CHECK(outer.size() == 4 && bstrides.size() == 5, "r40: 4 outer sizes and 5 strides required");
  TORCH_CHECK(op == 0 || op == 1, "r40: op must be 0 (mul) or 1 (div)");
  TORCH_CHECK(inner > 0, "r40: inner must be positive");
  // channels_last_3d: logical (B, C, T, H, W) is stored flat as (b, t, h, w, c),
  // so the kernel's roles (d0, d1, d2, d3, inner) map onto x.stride(0), (2), (3),
  // (4) and (1).  Each must match the size implied by the caller's outer extents.
  TORCH_CHECK(x.stride(1) == 1, "r40: x needs channels_last_3d (logical dim 1 innermost)");
  TORCH_CHECK(x.stride(4) == inner, "r40: x stride 4 is ", x.stride(4), " but inner is ", inner);
  TORCH_CHECK(x.stride(3) == outer[3] * inner, "r40: x stride 3 does not match outer[3]*inner");
  TORCH_CHECK(x.stride(2) == outer[2] * outer[3] * inner,
              "r40: x stride 2 does not match outer[2]*outer[3]*inner");
  TORCH_CHECK(x.stride(0) == outer[1] * outer[2] * outer[3] * inner,
              "r40: x stride 0 does not match outer[1]*outer[2]*outer[3]*inner");
  const int64_t total = outer[0] * outer[1] * outer[2] * outer[3] * inner;
  TORCH_CHECK(total == x.numel(), "r40: outer*inner must equal x.numel");
  TORCH_CHECK(total % 8 == 0, "r40: numel must be a multiple of 8");
  // Bound the broadcast operand: the largest offset the kernel can form is the
  // sum over dims of (extent - 1) * stride, and it must stay inside b.
  const int64_t ext[5] = {outer[0], outer[1], outer[2], outer[3], inner};
  for (int i = 0; i < 5; ++i) {
    TORCH_CHECK(bstrides[i] >= 0, "r40: broadcast stride ", i, " must be >= 0");
  }
  int64_t bmax = 0;
  for (int i = 0; i < 5; ++i) bmax += (ext[i] - 1) * bstrides[i];
  TORCH_CHECK(bmax < b.numel(), "r40: broadcast max offset ", bmax,
              " is outside the operand's ", b.numel(), " elements");
  R8State &s = r8_ensure();
  R8BBcastArgs A{};
  A.x = x.data_ptr();
  A.b = b.data_ptr();
  A.out = x.data_ptr();
  A.d0 = outer[0];
  A.d1 = outer[1];
  A.d2 = outer[2];
  A.d3 = outer[3];
  A.inner = inner;
  A.bs0 = bstrides[0];
  A.bs1 = bstrides[1];
  A.bs2 = bstrides[2];
  A.bs3 = bstrides[3];
  A.bs4 = bstrides[4];
  A.op = static_cast<int32_t>(op);
  A.pad = 0;
  const int64_t tasks = (total / 8 + R8_WG - 1) / R8_WG;
  s.q->submit([&](sycl::handler &h) {
    h.set_arg(0, A);
    h.parallel_for(sycl::nd_range<1>((size_t)tasks * R8_WG, R8_WG), *s.bbcast);
  });
  return x;
}

// ---- strided-x broadcast: for x views that are sliced on the channel dim ----
torch::Tensor bcast_strided_inplace_impl(torch::Tensor x, torch::Tensor b,
                                          std::vector<int64_t> outer, int64_t inner,
                                          std::vector<int64_t> xstrides,
                                          std::vector<int64_t> bstrides, int64_t op) {
  TORCH_CHECK(r8_tensors_ok(x) && r8_tensors_ok(b), "r40s: XPU bf16 tensors required");
  TORCH_CHECK(x.dim() == 5, "r40s: x must be 5-D");
  TORCH_CHECK(b.numel() > 0, "r40s: b must be non-empty");
  TORCH_CHECK(outer.size() == 4, "r40s: 4 outer extents required");
  TORCH_CHECK(xstrides.size() == 4 && bstrides.size() == 5,
              "r40s: 4 x strides and 5 broadcast strides required");
  for (int i = 0; i < 4; ++i) {
    TORCH_CHECK(outer[i] >= 1, "r40s: outer extent ", i, " must be >= 1");
    TORCH_CHECK(xstrides[i] >= 0, "r40s: x stride ", i, " must be >= 0");
    TORCH_CHECK(bstrides[i] >= 0, "r40s: broadcast stride ", i, " must be >= 0");
  }
  TORCH_CHECK(bstrides[4] >= 0, "r40s: broadcast stride 4 must be >= 0");
  TORCH_CHECK(inner > 0, "r40s: inner must be positive");
  TORCH_CHECK(op == 0 || op == 1, "r40s: op must be 0 (mul) or 1 (div)");
  // The vectorised path moves 16 bytes per run, so the channel stride must be 1
  // and the base must be 16-byte aligned.
  TORCH_CHECK(x.stride(1) == 1, "r40s: channel stride must be 1 for the vector run");
  TORCH_CHECK(((uintptr_t)x.data_ptr() % 16) == 0, "r40s: x must be 16-byte aligned");
  TORCH_CHECK(((uintptr_t)b.data_ptr() % 2) == 0, "r40s: b must be 2-byte aligned");
  // The caller's extents must cover the tensor.
  TORCH_CHECK(outer[0] * outer[1] * outer[2] * outer[3] * inner == x.numel(),
              "r40s: outer*inner must equal x.numel");
  TORCH_CHECK(x.numel() % 8 == 0, "r40s: numel must be a multiple of 8");
  // The whole point of this kernel is that x may be strided, so the bound is a
  // real sum over the dims rather than the contiguous numel-1.  Without this
  // check a wrong extent/stride pair reads out of bounds, which is the failure
  // that produced this morning's engine resets.
  const int64_t ext[5] = {outer[0], outer[1], outer[2], outer[3], inner};
  const int64_t xs[5] = {xstrides[0], xstrides[1], xstrides[2], xstrides[3], 1};
  int64_t xmax = 0;
  for (int i = 0; i < 5; ++i) xmax += (ext[i] - 1) * xs[i];
  // The bound is the underlying storage, not x.numel().  x is deliberately a
  // strided view here, so its elements are scattered across a larger parent
  // buffer and the largest address legitimately exceeds the view's own element
  // count.  Checking against numel would reject every legitimate call site,
  // which is worse than not checking: it turns the flag into a silent no-op.
  const int64_t storage =
      static_cast<int64_t>(x.storage().nbytes() / x.element_size());
  TORCH_CHECK(storage >= x.numel(), "r40s: storage smaller than the view");
  TORCH_CHECK(xmax < storage, "r40s: strided max offset ", xmax,
              " is outside the ", storage, " element storage");
  int64_t bmax = 0;
  for (int i = 0; i < 5; ++i) bmax += (ext[i] - 1) * bstrides[i];
  TORCH_CHECK(bmax < b.numel(), "r40s: broadcast max offset ", bmax,
              " is outside the operand's ", b.numel(), " elements");
  // The broadcast operand's strides are *assumed* to describe its real layout, and
  // bmax < numel cannot catch a wrong assumption: if the caller passes strides that
  // are self-consistent but do not match the tensor, every index stays in bounds
  // and the kernel silently reads the wrong elements.  That is a wrong-answer bug
  // with no bounds violation to trip on, so the layout is checked here instead.
  // The operand has extent 1 on the channel axis, so its logical dim 1 stride is
  // irrelevant and dims 0,2,3,4 must match bstrides[0..3].
  TORCH_CHECK(b.dim() == 5, "r40s: operand must be 5-D");
  TORCH_CHECK(b.stride(0) == bstrides[0] && b.stride(2) == bstrides[1] &&
                  b.stride(3) == bstrides[2] && b.stride(4) == bstrides[3],
              "r40s: operand strides ", b.stride(0), ",", b.stride(2), ",", b.stride(3),
              ",", b.stride(4), " do not match the given broadcast strides ",
              bstrides[0], ",", bstrides[1], ",", bstrides[2], ",", bstrides[3]);
  R8State &s = r8_ensure();
  R8BBcastStridedArgs A{};
  A.x = x.data_ptr();
  A.b = b.data_ptr();
  A.out = x.data_ptr();
  A.d0 = outer[0];
  A.d1 = outer[1];
  A.d2 = outer[2];
  A.d3 = outer[3];
  A.inner = inner;
  A.xs0 = xstrides[0];
  A.xs1 = xstrides[1];
  A.xs2 = xstrides[2];
  A.xs3 = xstrides[3];
  A.bs0 = bstrides[0];
  A.bs1 = bstrides[1];
  A.bs2 = bstrides[2];
  A.bs3 = bstrides[3];
  A.bs4 = bstrides[4];
  A.op = (int32_t)op;
  A.pad = 0;
  const int64_t tasks = (x.numel() / 8 + R8_WG - 1) / R8_WG;
  s.q->submit([&](sycl::handler &h) {
    h.set_arg(0, A);
    h.parallel_for(sycl::nd_range<1>((size_t)tasks * R8_WG, R8_WG), *s.bbcast_strided);
  });
  return x;
}

// R10D F3: tanh-approx GELU (bitwise vs torch's GeluTanhFunctor).
torch::Tensor gelu_impl(torch::Tensor x) {
  const int64_t n = x.numel();
  TORCH_CHECK(r8_tensors_ok(x) && x.is_contiguous() && x.stride(-1) == 1,
              "r10d: gelu input must be XPU bf16 contiguous");
  TORCH_CHECK(n % 8 == 0, "r10d: gelu numel must be a multiple of 8");
  torch::Tensor out = at::empty_like(x);
  R8State &s = r8_ensure();
  R8GeluArgs A{};
  A.x = x.data_ptr();
  A.out = out.data_ptr();
  A.n = n;
  const int64_t tasks = (n / 8 + R8_WG - 1) / R8_WG;
  s.q->submit([&](sycl::handler &h) {
    h.set_arg(0, A);
    h.parallel_for(sycl::nd_range<1>((size_t)tasks * R8_WG, R8_WG), *s.gelu);
  });
  return out;
}

// R21: in-place BF16 SiLU(up) product for the DiffVAE SwiGLU chain.
torch::Tensor silu_mul_impl(torch::Tensor gate, torch::Tensor up) {
  const int64_t n = gate.numel();
  TORCH_CHECK(r8_tensors_ok(gate) && gate.is_contiguous() && gate.stride(-1) == 1,
              "r21: gate must be XPU bf16 contiguous");
  TORCH_CHECK(r8_tensors_ok(up) && up.is_contiguous() && up.stride(-1) == 1,
              "r21: up must be XPU bf16 contiguous");
  TORCH_CHECK(gate.sizes() == up.sizes() && gate.device() == up.device(),
              "r21: gate/up shape and device must match");
  TORCH_CHECK(n % 8 == 0, "r21: numel must be a multiple of 8");
  const int64_t requested_vec = r21_env_int("LTX_R21_SILU_MUL_VEC", 8, 8, 32);
  TORCH_CHECK(requested_vec == 8 || requested_vec == 16 || requested_vec == 32,
              "r21: LTX_R21_SILU_MUL_VEC must be 8, 16, or 32");
  const uintptr_t gate_address = reinterpret_cast<uintptr_t>(gate.data_ptr());
  const uintptr_t up_address = reinterpret_cast<uintptr_t>(up.data_ptr());
  TORCH_CHECK(gate_address % 16 == 0 && up_address % 16 == 0,
              "r21: gate/up pointers must be 16-byte aligned");
  const bool align16 = gate_address % 32 == 0 && up_address % 32 == 0;
  const bool align32 = gate_address % 64 == 0 && up_address % 64 == 0;
  const int64_t vec = requested_vec == 32 && n % 32 == 0 && align32 ? 32
                    : requested_vec >= 16 && n % 16 == 0 && align16 ? 16 : 8;
  const int64_t wg = r21_env_int("LTX_R21_SILU_MUL_WG", 256, 32, 512);
  TORCH_CHECK(wg % 32 == 0, "r21: LTX_R21_SILU_MUL_WG must be a multiple of 32");
  R8State &s = r8_ensure();
  R8SiluMulArgs A{};
  A.gate = gate.data_ptr();
  A.up = up.data_ptr();
  A.out = gate.data_ptr();
  A.n = n;
  const int64_t tasks = (n / vec + wg - 1) / wg;
  s.q->submit([&](sycl::handler &h) {
    h.set_arg(0, A);
    if (vec == 32) {
      h.parallel_for(sycl::nd_range<1>((size_t)tasks * wg, wg), *s.silu_mul32);
    } else if (vec == 16) {
      h.parallel_for(sycl::nd_range<1>((size_t)tasks * wg, wg), *s.silu_mul16);
    } else {
      h.parallel_for(sycl::nd_range<1>((size_t)tasks * wg, wg), *s.silu_mul);
    }
  });
  return gate;
}

// R21b: compat SiLU/up-product (round-0 torch rounding). Additive; unset = current.
bool r21b_on() {
  const char *all = std::getenv("LTX_ALL_ORIG");
  if (all != nullptr && all[0] == '1' && all[1] == '\0') return false;
  const char *v = std::getenv("LTX_R21B_ELEMWISE");
  if (v == nullptr) return true;
  if (v[0] == '1' && v[1] == '\0') return true;
  TORCH_CHECK(v[0] == '0' && v[1] == '\0', "r21b: LTX_R21B_ELEMWISE must be 0 or 1");
  return false;
}

torch::Tensor silu_mul_compat_impl(torch::Tensor gate, torch::Tensor up) {
  const int64_t n = gate.numel();
  TORCH_CHECK(r8_tensors_ok(gate) && gate.is_contiguous() && gate.stride(-1) == 1,
              "r21b: gate must be XPU bf16 contiguous");
  TORCH_CHECK(r8_tensors_ok(up) && up.is_contiguous() && up.stride(-1) == 1,
              "r21b: up must be XPU bf16 contiguous");
  TORCH_CHECK(gate.sizes() == up.sizes() && gate.device() == up.device(),
              "r21b: gate/up shape and device must match");
  TORCH_CHECK(n % 8 == 0, "r21b: numel must be a multiple of 8");
  const int64_t requested_vec = r21_env_int("LTX_R21B_SILU_MUL_VEC", 16, 8, 32);
  TORCH_CHECK(requested_vec == 8 || requested_vec == 16 || requested_vec == 32,
              "r21b: LTX_R21B_SILU_MUL_VEC must be 8, 16, or 32");
  const uintptr_t gate_address = reinterpret_cast<uintptr_t>(gate.data_ptr());
  const uintptr_t up_address = reinterpret_cast<uintptr_t>(up.data_ptr());
  TORCH_CHECK(gate_address % 16 == 0 && up_address % 16 == 0,
              "r21b: gate/up pointers must be 16-byte aligned");
  const bool align16 = gate_address % 32 == 0 && up_address % 32 == 0;
  const bool align32 = gate_address % 64 == 0 && up_address % 64 == 0;
  const int64_t vec = requested_vec == 32 && n % 32 == 0 && align32 ? 32
                    : requested_vec >= 16 && n % 16 == 0 && align16 ? 16 : 8;
  const int64_t wg = r21_env_int("LTX_R21B_SILU_MUL_WG", 128, 32, 512);
  TORCH_CHECK(wg % 32 == 0, "r21b: LTX_R21B_SILU_MUL_WG must be a multiple of 32");
  R8State &s = r8_ensure();
  R8SiluMulArgs A{};
  A.gate = gate.data_ptr();
  A.up = up.data_ptr();
  A.out = gate.data_ptr();
  A.n = n;
  const int64_t tasks = (n / vec + wg - 1) / wg;
  s.q->submit([&](sycl::handler &h) {
    h.set_arg(0, A);
    if (vec == 32) {
      h.parallel_for(sycl::nd_range<1>((size_t)tasks * wg, wg), *s.silu_mul32_c);
    } else if (vec == 16) {
      h.parallel_for(sycl::nd_range<1>((size_t)tasks * wg, wg), *s.silu_mul16_c);
    } else {
      h.parallel_for(sycl::nd_range<1>((size_t)tasks * wg, wg), *s.silu_mul_c);
    }
  });
  return gate;
}

// R9A: fp8 e4m3fn -> bf16 widening (vectorized, integer LUT).
torch::Tensor fp8_widen_impl(torch::Tensor x) {
  TORCH_CHECK(x.is_xpu(), "r9a: input must be XPU");
  TORCH_CHECK(x.scalar_type() == at::kFloat8_e4m3fn, "r9a: input must be float8_e4m3fn");
  TORCH_CHECK(x.is_contiguous(), "r9a: input must be contiguous");
  const int64_t n = x.numel();
  TORCH_CHECK(n % 8 == 0, "r9a: numel must be a multiple of 8");
  torch::Tensor out = at::empty(x.sizes(), x.options().dtype(at::kBFloat16));
  R8State &s = r8_ensure();
  R8Fp8Args A{};
  A.in = x.data_ptr();
  A.out = out.data_ptr();
  A.n = n;
  const int64_t tasks = (n / 8 + R8_WG - 1) / R8_WG;
  s.q->submit([&](sycl::handler &h) {
    h.set_arg(0, A);
    h.parallel_for(sycl::nd_range<1>((size_t)tasks * R8_WG, R8_WG), *s.fp8w);
  });
  return out;
}

torch::Tensor fp8_widen_eff_impl(torch::Tensor x) {
  TORCH_CHECK(x.is_xpu(), "r18: input must be XPU");
  TORCH_CHECK(x.scalar_type() == at::kFloat8_e4m3fn,
              "r18: input must be float8_e4m3fn");
  TORCH_CHECK(x.is_contiguous(), "r18: input must be contiguous");
  const int64_t n = x.numel();
  TORCH_CHECK(n % 8 == 0, "r18: numel must be a multiple of 8");
  const int64_t wg = r18_env_int("LTX_R18_FP8_WG", 256, 32, 1024);
  TORCH_CHECK(wg % 32 == 0, "r18: FP8 WG must be a multiple of 32");
  const int64_t prefetch =
      r18_env_flag_with_common("LTX_R18_FP8_PREFETCH", "LTX_R18_PREFETCH", 0);
  const auto aligned = [](const torch::Tensor &t) {
    return (reinterpret_cast<uintptr_t>(t.data_ptr()) & 15u) == 0;
  };
  TORCH_CHECK(aligned(x), "r18: FP8 input must be 16-byte aligned");
  torch::Tensor out = at::empty(x.sizes(), x.options().dtype(at::kBFloat16));
  R8State &s = r8_ensure();
  R8Fp8EffArgs A{};
  A.in = x.data_ptr();
  A.out = out.data_ptr();
  A.n = n;
  A.prefetch = prefetch;
  const int64_t tasks = (n / 8 + wg - 1) / wg;
  s.q->submit([&](sycl::handler &h) {
    h.set_arg(0, A);
    h.parallel_for(sycl::nd_range<1>((size_t)tasks * wg, wg), *s.fp8we);
  });
  return out;
}

torch::Tensor fp8_widen_bits_impl(torch::Tensor x) {
  TORCH_CHECK(x.is_xpu(), "r18: input must be XPU");
  TORCH_CHECK(x.scalar_type() == at::kFloat8_e4m3fn,
              "r18: input must be float8_e4m3fn");
  TORCH_CHECK(x.is_contiguous(), "r18: input must be contiguous");
  const int64_t n = x.numel();
  TORCH_CHECK(n % 8 == 0, "r18: input numel must be a multiple of 8");
  const int64_t wg = r18_env_int("LTX_R18_FP8_WG", 256, 32, 1024);
  TORCH_CHECK(wg % 32 == 0, "r18: FP8 WG must be a multiple of 32");
  const int64_t prefetch =
      r18_env_flag_with_common("LTX_R18_FP8_PREFETCH", "LTX_R18_PREFETCH", 0);
  const auto aligned = [](const torch::Tensor &t) {
    return (reinterpret_cast<uintptr_t>(t.data_ptr()) & 15u) == 0;
  };
  TORCH_CHECK(aligned(x), "r18: FP8 input must be 16-byte aligned");
  torch::Tensor out = at::empty(x.sizes(), x.options().dtype(at::kBFloat16));
  R8State &s = r8_ensure();
  R8Fp8EffArgs A{};
  A.in = x.data_ptr();
  A.out = out.data_ptr();
  A.n = n;
  A.prefetch = prefetch;
  const int64_t tasks = (n / 8 + wg - 1) / wg;
  s.q->submit([&](sycl::handler &h) {
    h.set_arg(0, A);
    h.parallel_for(sycl::nd_range<1>((size_t)tasks * wg, wg), *s.fp8wbits);
  });
  return out;
}

std::vector<torch::Tensor> fp8_widen_batch_impl(
    const std::vector<torch::Tensor> &inputs) {
  TORCH_CHECK(!inputs.empty() && inputs.size() <= 4,
              "r18: FP8 batch descriptor count must be in [1,4]");
  const int64_t wg = r18_env_int("LTX_R18_FP8_WG", 256, 32, 1024);
  TORCH_CHECK(wg % 32 == 0, "r18: FP8 WG must be a multiple of 32");
  const int64_t prefetch =
      r18_env_flag_with_common("LTX_R18_FP8_PREFETCH", "LTX_R18_PREFETCH", 0);
  std::vector<torch::Tensor> outputs;
  outputs.reserve(inputs.size());
  R8State &s = r8_ensure();
  R8Fp8BatchArgs A{};
  A.count = static_cast<int64_t>(inputs.size());
  int64_t total = 0;
  for (size_t i = 0; i < inputs.size(); ++i) {
    const torch::Tensor &x = inputs[i];
    TORCH_CHECK(x.is_xpu() && x.scalar_type() == at::kFloat8_e4m3fn &&
                    x.is_contiguous(),
                "r18: FP8 batch inputs must be contiguous XPU float8_e4m3fn");
    const int64_t n = x.numel();
    TORCH_CHECK(n > 0 && n % 8 == 0, "r18: FP8 batch numel must be a positive multiple of 8");
    TORCH_CHECK((reinterpret_cast<uintptr_t>(x.data_ptr()) & 15u) == 0,
                "r18: FP8 batch inputs must be 16-byte aligned");
    torch::Tensor out = at::empty(x.sizes(), x.options().dtype(at::kBFloat16));
    A.desc[i].in = x.data_ptr();
    A.desc[i].out = out.data_ptr();
    A.desc[i].n = n;
    outputs.push_back(out);
    total += n;
  }
  A.total = total;
  A.prefetch = prefetch;
  const int64_t tasks = (total / 8 + wg - 1) / wg;
  s.q->submit([&](sycl::handler &h) {
    h.set_arg(0, A);
    h.parallel_for(sycl::nd_range<1>((size_t)tasks * wg, wg), *s.fp8wb);
  });
  return outputs;
}

// R9A v2: 16-wide widening, WG from env LTX_R9A_WG (default 256).
torch::Tensor fp8_widen16_impl(torch::Tensor x) {
  TORCH_CHECK(x.is_xpu(), "r9a: input must be XPU");
  TORCH_CHECK(x.scalar_type() == at::kFloat8_e4m3fn, "r9a: input must be float8_e4m3fn");
  TORCH_CHECK(x.is_contiguous(), "r9a: input must be contiguous");
  const int64_t n = x.numel();
  TORCH_CHECK(n % 16 == 0, "r9a: numel must be a multiple of 16");
  torch::Tensor out = at::empty(x.sizes(), x.options().dtype(at::kBFloat16));
  R8State &s = r8_ensure();
  R8Fp8Args A{};
  A.in = x.data_ptr();
  A.out = out.data_ptr();
  A.n = n;
  const char *wg = std::getenv("LTX_R9A_WG");
  const int64_t WG = wg ? std::atoll(wg) : 256;
  const int64_t tasks = (n / 16 + WG - 1) / WG;
  s.q->submit([&](sycl::handler &h) {
    h.set_arg(0, A);
    h.parallel_for(sycl::nd_range<1>((size_t)tasks * WG, WG), *s.fp8w16);
  });
  return out;
}

}  // namespace
TORCH_LIBRARY(libr8, m) {
  m.def("k1_ada(Tensor x, Tensor tbs, Tensor tss, int tss_rs, Tensor tbh, Tensor tsh, int tsh_rs, float eps) -> Tensor");
  m.impl("k1_ada", [](torch::Tensor x, torch::Tensor tbs, torch::Tensor tss, int64_t tss_rs,
                      torch::Tensor tbh, torch::Tensor tsh, int64_t tsh_rs,
                      double eps) { return k1_impl(x, tbs, tss, tss_rs, tbh, tsh, tsh_rs, eps, true, false); });
  m.def("k1_ada_hold(Tensor x, Tensor tbs, Tensor tss, int tss_rs, Tensor tbh, Tensor tsh, int tsh_rs, float eps) -> Tensor");
  m.impl("k1_ada_hold", [](torch::Tensor x, torch::Tensor tbs, torch::Tensor tss, int64_t tss_rs,
                           torch::Tensor tbh, torch::Tensor tsh, int64_t tsh_rs,
                           double eps) { return k1_impl(x, tbs, tss, tss_rs, tbh, tsh, tsh_rs, eps, true, true); });
  m.def("k1_ada_compat_local(Tensor x, Tensor tbs, Tensor tss, int tss_rs, Tensor tbh, Tensor tsh, int tsh_rs, float eps, int wg) -> Tensor");
  m.impl("k1_ada_compat_local", [](torch::Tensor x, torch::Tensor tbs, torch::Tensor tss, int64_t tss_rs,
                                   torch::Tensor tbh, torch::Tensor tsh, int64_t tsh_rs,
                                   double eps, int64_t wg) {
    return k1_compat_local_impl(x, tbs, tss, tss_rs, tbh, tsh, tsh_rs, eps, wg);
  });
  m.def("k1_ada_compat_pipe(Tensor x, Tensor tbs, Tensor tss, int tss_rs, Tensor tbh, Tensor tsh, int tsh_rs, float eps, int wg) -> Tensor");
  m.impl("k1_ada_compat_pipe", [](torch::Tensor x, torch::Tensor tbs, torch::Tensor tss, int64_t tss_rs,
                                  torch::Tensor tbh, torch::Tensor tsh, int64_t tsh_rs,
                                  double eps, int64_t wg) {
    return k1_compat_pipe_impl(x, tbs, tss, tss_rs, tbh, tsh, tsh_rs, eps, wg);
  });
  m.def("k1_ada_local(Tensor x, Tensor tbs, Tensor tss, int tss_rs, Tensor tbh, Tensor tsh, int tsh_rs, float eps, int wg, int vec, int prefetch) -> Tensor");
  m.impl("k1_ada_local", [](torch::Tensor x, torch::Tensor tbs, torch::Tensor tss, int64_t tss_rs,
                           torch::Tensor tbh, torch::Tensor tsh, int64_t tsh_rs,
                           double eps, int64_t wg, int64_t vec, int64_t prefetch) {
    return k1_local_impl(x, tbs, tss, tss_rs, tbh, tsh, tsh_rs, eps, wg, vec, prefetch);
  });
  m.def("k1_ada_reg_local(Tensor x, Tensor tbs, Tensor tss, int tss_rs, Tensor tbh, Tensor tsh, int tsh_rs, float eps, int wg, int prefetch) -> Tensor");
  m.impl("k1_ada_reg_local", [](torch::Tensor x, torch::Tensor tbs, torch::Tensor tss, int64_t tss_rs,
                                torch::Tensor tbh, torch::Tensor tsh, int64_t tsh_rs,
                                double eps, int64_t wg, int64_t prefetch) {
    return k1_reg_impl(x, tbs, tss, tss_rs, tbh, tsh, tsh_rs, eps, wg, prefetch);
  });
  m.def("k1_ada_subgroup(Tensor x, Tensor tbs, Tensor tss, int tss_rs, Tensor tbh, Tensor tsh, int tsh_rs, float eps, int wg, int vec, int prefetch) -> Tensor");
  m.impl("k1_ada_subgroup", [](torch::Tensor x, torch::Tensor tbs, torch::Tensor tss, int64_t tss_rs,
                                torch::Tensor tbh, torch::Tensor tsh, int64_t tsh_rs,
                                double eps, int64_t wg, int64_t vec, int64_t prefetch) {
    return k1_subgroup_impl(x, tbs, tss, tss_rs, tbh, tsh, tsh_rs, eps, wg, vec, prefetch);
  });
  m.def("k1_ada_group_reduce(Tensor x, Tensor tbs, Tensor tss, int tss_rs, Tensor tbh, Tensor tsh, int tsh_rs, float eps, int wg, int vec, int prefetch) -> Tensor");
  m.impl("k1_ada_group_reduce", [](torch::Tensor x, torch::Tensor tbs, torch::Tensor tss, int64_t tss_rs,
                                    torch::Tensor tbh, torch::Tensor tsh, int64_t tsh_rs,
                                    double eps, int64_t wg, int64_t vec, int64_t prefetch) {
    return k1_group_impl(x, tbs, tss, tss_rs, tbh, tsh, tsh_rs, eps, wg, vec, prefetch);
  });
  m.def("k1_affine(Tensor x, Tensor tbs, Tensor tss, int tss_rs, Tensor tbh, Tensor tsh, int tsh_rs) -> Tensor");
  m.impl("k1_affine", [](torch::Tensor x, torch::Tensor tbs, torch::Tensor tss, int64_t tss_rs,
                         torch::Tensor tbh, torch::Tensor tsh,
                         int64_t tsh_rs) { return k1_impl(x, tbs, tss, tss_rs, tbh, tsh, tsh_rs, 1e-6, false, false); });
  m.def("k2_postsa(Tensor x, Tensor y, Tensor tbg, Tensor tsg, int tsg_rs, float eps) -> Tensor[]");
  m.impl("k2_postsa", [](torch::Tensor x, torch::Tensor y, torch::Tensor tbg, torch::Tensor tsg,
                         int64_t tsg_rs, double eps) { return k2_impl(x, y, tbg, tsg, tsg_rs, eps); });
  m.def("r17_mlp_norm_affine(Tensor x, Tensor norm_weight, Tensor scale, Tensor shift, Tensor out, float eps) -> Tensor");
  m.impl("r17_mlp_norm_affine", [](torch::Tensor x, torch::Tensor norm_weight, torch::Tensor scale,
                                    torch::Tensor shift, torch::Tensor out,
                                    double eps) { return r17_mlp_norm_affine_impl(x, norm_weight, scale, shift, out, eps); });
  m.def("r19_norm_rope(Tensor q, Tensor k, Tensor q_weight, Tensor k_weight, Tensor cos, Tensor sin, int B, int H, int T, int Dh, float eps, float q_scale, float k_scale) -> Tensor[]");
  m.impl("r19_norm_rope", [](torch::Tensor q, torch::Tensor k, torch::Tensor q_weight,
                             torch::Tensor k_weight, torch::Tensor cos, torch::Tensor sin,
                             int64_t B, int64_t H, int64_t T, int64_t Dh, double eps,
                             double q_scale, double k_scale) {
    return r19_norm_rope_impl(q, k, q_weight, k_weight, cos, sin, B, H, T, Dh, eps,
                              q_scale, k_scale);
  });
  m.def("k3_rope(Tensor x, int[] xs, Tensor cos, int[] cs, Tensor sin, int[] ss, int B, int H, int T, int Dh) -> Tensor");
  m.impl("k3_rope", [](torch::Tensor x, torch::IntArrayRef xs, torch::Tensor cos,
                       torch::IntArrayRef cs, torch::Tensor sin, torch::IntArrayRef ss,
                       int64_t B, int64_t H, int64_t T,
                       int64_t Dh) { return k3_impl(x, xs, cos, cs, sin, ss, B, H, T, Dh); });
  m.def("k3_rope_vec(Tensor x, int[] xs, Tensor cos, int[] cs, Tensor sin, int[] ss, int B, int H, int T, int Dh) -> Tensor");  m.impl("k3_rope_vec", [](torch::Tensor x, torch::IntArrayRef xs, torch::Tensor cos,
                           torch::IntArrayRef cs, torch::Tensor sin, torch::IntArrayRef ss,
                           int64_t B, int64_t H, int64_t T,
                           int64_t Dh) { return k3_vec_impl(x, xs, cos, cs, sin, ss, B, H, T, Dh); });
  m.def("k3_rope_dual(Tensor xq, int[] qxs, Tensor xk, int[] kxs, Tensor cos, int[] cs, Tensor sin, int[] ss, int B, int H, int T, int Dh) -> Tensor[]");
  m.impl("k3_rope_dual", [](torch::Tensor xq, torch::IntArrayRef qxs, torch::Tensor xk,
                            torch::IntArrayRef kxs, torch::Tensor cos, torch::IntArrayRef cs,
                            torch::Tensor sin, torch::IntArrayRef ss, int64_t B, int64_t H,
                            int64_t T, int64_t Dh) {
    return k3_dual_impl(xq, qxs, xk, kxs, cos, cs, sin, ss, B, H, T, Dh);
  });
  m.def(
      "r40_bcast_inplace(Tensor x, Tensor b, int[] outer, int inner, int[] bstrides, int op)"
      " -> Tensor");
  m.impl("r40_bcast_inplace", [](torch::Tensor x, torch::Tensor b, std::vector<int64_t> outer,
                                 int64_t inner, std::vector<int64_t> bstrides, int64_t op) {
    return bcast_inplace_impl(x, b, outer, inner, bstrides, op);
  });
  m.def("r10d_gelu(Tensor x) -> Tensor");
  m.impl("r10d_gelu", [](torch::Tensor x) { return gelu_impl(x); });
  m.def("r21_swiglu_silu_mul(Tensor gate, Tensor up) -> Tensor");
  m.impl("r21_swiglu_silu_mul", [](torch::Tensor gate, torch::Tensor up) {
    return silu_mul_impl(gate, up);
  });
  m.def("r21b_swiglu_silu_mul(Tensor gate, Tensor up) -> Tensor");
  m.impl("r21b_swiglu_silu_mul", [](torch::Tensor gate, torch::Tensor up) {
    return silu_mul_compat_impl(gate, up);
  });
  m.def("r10d_gate_mul(Tensor attn, Tensor gates, int dh) -> Tensor");
  m.impl("r10d_gate_mul", [](torch::Tensor attn, torch::Tensor gates,
                             int64_t dh) { return gate_mul_impl(attn, gates, dh); });
  m.def("r9a_fp8_widen(Tensor x) -> Tensor");
  m.impl("r9a_fp8_widen", [](torch::Tensor x) { return fp8_widen_impl(x); });
  m.def("r9a_fp8_widen_eff(Tensor x) -> Tensor");
  m.impl("r9a_fp8_widen_eff", [](torch::Tensor x) { return fp8_widen_eff_impl(x); });
  m.def("r9a_fp8_widen_bits(Tensor x) -> Tensor");
  m.impl("r9a_fp8_widen_bits", [](torch::Tensor x) { return fp8_widen_bits_impl(x); });
  m.def("r9a_fp8_widen_batch(Tensor[] inputs) -> Tensor[]");
  m.impl("r9a_fp8_widen_batch", [](std::vector<torch::Tensor> inputs) {
    return fp8_widen_batch_impl(inputs);
  });
  m.def("r9a_fp8_widen16(Tensor x) -> Tensor");
  m.impl("r9a_fp8_widen16", [](torch::Tensor x) { return fp8_widen16_impl(x); });

  m.def("r40_bcast_strided_inplace(Tensor x, Tensor b, int[] outer, int inner, "
        "int[] xstrides, int[] bstrides, int op) -> Tensor");
  m.impl("r40_bcast_strided_inplace",
         [](torch::Tensor x, torch::Tensor b, std::vector<int64_t> outer, int64_t inner,
            std::vector<int64_t> xstrides, std::vector<int64_t> bstrides, int64_t op) {
           return bcast_strided_inplace_impl(x, b, outer, inner, xstrides, bstrides, op);
         });
}
