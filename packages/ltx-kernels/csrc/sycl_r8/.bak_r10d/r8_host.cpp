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

#include "r8_args.h"

namespace {

constexpr int64_t R8_MAX_C = 8192;
constexpr size_t R8_SCRATCH_FLOATS = 1 << 20;  // 4MB: 32768 rows x 32
constexpr int R8_WG = 256;

struct R8State {
  bool ready = false;
  std::optional<sycl::queue> q;
  std::optional<sycl::kernel> k1a;
  std::optional<sycl::kernel> k1b;
  std::optional<sycl::kernel> k2;
  std::optional<sycl::kernel> k3;
  std::optional<sycl::kernel> k3v;
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
  s.k2 = wrap("_ZTS10R8K2Kernel");
  s.k3 = wrap("_ZTS10R8K3Kernel");
  s.k3v = wrap("_ZTS13R8K3VecKernel");
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
                      bool do_norm) {
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
  sycl::kernel k = do_norm ? *s.k1a : *s.k1b;
  s.q->submit([&](sycl::handler &h) {
    h.set_arg(0, A);
    h.parallel_for(sycl::nd_range<1>((size_t)rows * R8_WG, R8_WG), k);
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
    h.parallel_for(sycl::nd_range<1>((size_t)rows * R8_WG, R8_WG), *s.k2);
  });
  return {fma, out};
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
      h.parallel_for(sycl::nd_range<1>((size_t)groups * K3V_WG, K3V_WG), *s.k3v);
    });
  } else {
    s.q->submit([&](sycl::handler &h) {
      h.set_arg(0, A);
      h.parallel_for(sycl::nd_range<1>((size_t)fibers * 16, 16), *s.k3);
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

}  // namespace

TORCH_LIBRARY(libr8, m) {
  m.def("k1_ada(Tensor x, Tensor tbs, Tensor tss, int tss_rs, Tensor tbh, Tensor tsh, int tsh_rs, float eps) -> Tensor");
  m.impl("k1_ada", [](torch::Tensor x, torch::Tensor tbs, torch::Tensor tss, int64_t tss_rs,
                      torch::Tensor tbh, torch::Tensor tsh, int64_t tsh_rs,
                      double eps) { return k1_impl(x, tbs, tss, tss_rs, tbh, tsh, tsh_rs, eps, true); });
  m.def("k1_affine(Tensor x, Tensor tbs, Tensor tss, int tss_rs, Tensor tbh, Tensor tsh, int tsh_rs) -> Tensor");
  m.impl("k1_affine", [](torch::Tensor x, torch::Tensor tbs, torch::Tensor tss, int64_t tss_rs,
                         torch::Tensor tbh, torch::Tensor tsh,
                         int64_t tsh_rs) { return k1_impl(x, tbs, tss, tss_rs, tbh, tsh, tsh_rs, 1e-6, false); });
  m.def("k2_postsa(Tensor x, Tensor y, Tensor tbg, Tensor tsg, int tsg_rs, float eps) -> Tensor[]");
  m.impl("k2_postsa", [](torch::Tensor x, torch::Tensor y, torch::Tensor tbg, torch::Tensor tsg,
                         int64_t tsg_rs, double eps) { return k2_impl(x, y, tbg, tsg, tsg_rs, eps); });
  m.def("k3_rope(Tensor x, int[] xs, Tensor cos, int[] cs, Tensor sin, int[] ss, int B, int H, int T, int Dh) -> Tensor");
  m.impl("k3_rope", [](torch::Tensor x, torch::IntArrayRef xs, torch::Tensor cos,
                       torch::IntArrayRef cs, torch::Tensor sin, torch::IntArrayRef ss,
                       int64_t B, int64_t H, int64_t T,
                       int64_t Dh) { return k3_impl(x, xs, cos, cs, sin, ss, B, H, T, Dh); });
  m.def("k3_rope_vec(Tensor x, int[] xs, Tensor cos, int[] cs, Tensor sin, int[] ss, int B, int H, int T, int Dh) -> Tensor");
  m.impl("k3_rope_vec", [](torch::Tensor x, torch::IntArrayRef xs, torch::Tensor cos,
                           torch::IntArrayRef cs, torch::Tensor sin, torch::IntArrayRef ss,
                           int64_t B, int64_t H, int64_t T,
                           int64_t Dh) { return k3_vec_impl(x, xs, cos, cs, sin, ss, B, H, T, Dh); });
}
