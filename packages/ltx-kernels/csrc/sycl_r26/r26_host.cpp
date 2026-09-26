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
#include <optional>
#include <string>
#include <vector>

#include "r26_args.h"

namespace {

constexpr int R26_NORM_WG = 256;
constexpr int R26_NORM_ROWS_PER_WG = 16;
constexpr int R26_QK_WG = 512;
constexpr int R26_QK_ROWS_PER_WG = 32;
constexpr int64_t R26_NORM_MAX_C = 256;
constexpr int64_t R26_QK_HEAD_DIM = 64;

struct R26State {
  bool ready = false;
  std::optional<sycl::queue> q;
  std::optional<sycl::kernel> norm;
  std::optional<sycl::kernel> norm_precomputed;
  std::optional<sycl::kernel> norm_rope;
};

R26State &r26_state() {
  static R26State state;
  return state;
}

std::mutex &r26_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::string r26_dir() {
  Dl_info info{};
  if (dladdr((const void *)&r26_state, &info) && info.dli_fname)
    return std::string(info.dli_fname);
  return {};
}

std::string r26_spv_path() {
  const std::string dir = r26_dir();
  const auto pos = dir.find_last_of('/');
  return (pos == std::string::npos ? std::string(".") : dir.substr(0, pos)) +
         "/r26_kernels.spv";
}

void r26_init_locked() {
  R26State &state = r26_state();
  if (state.ready) return;
  sycl::queue &queue = c10::xpu::getCurrentXPUStream().queue();
  TORCH_CHECK(queue.is_in_order(), "r26: torch XPU queue must be in-order");
  state.q = queue;
  sycl::context context = queue.get_context();
  sycl::device device = queue.get_device();
  constexpr auto L0BE = sycl::backend::ext_oneapi_level_zero;
  auto zctx = sycl::get_native<L0BE>(context);
  auto zdev = sycl::get_native<L0BE>(device);

  const std::string path = r26_spv_path();
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  TORCH_CHECK(file.good(), "r26: cannot open ", path);
  const size_t size = static_cast<size_t>(file.tellg());
  file.seekg(0);
  std::vector<uint8_t> image(size);
  TORCH_CHECK(file.read(reinterpret_cast<char *>(image.data()), size).good(),
              "r26: cannot read ", path);

  ze_module_desc_t module_desc{};
  module_desc.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
  module_desc.format = ZE_MODULE_FORMAT_IL_SPIRV;
  module_desc.inputSize = size;
  module_desc.pInputModule = image.data();
  ze_module_handle_t zmodule = nullptr;
  TORCH_CHECK(zeModuleCreate(zctx, zdev, &module_desc, &zmodule, nullptr) ==
                  ZE_RESULT_SUCCESS,
              "r26: zeModuleCreate failed");

  using sycl::ext::oneapi::level_zero::ownership;
  auto bundle = sycl::make_kernel_bundle<L0BE, sycl::bundle_state::executable>(
      {zmodule, ownership::keep}, context);
  auto wrap = [&](const char *name) {
    ze_kernel_desc_t desc{};
    desc.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
    desc.pKernelName = name;
    ze_kernel_handle_t zkernel = nullptr;
    const ze_result_t result = zeKernelCreate(zmodule, &desc, &zkernel);
    TORCH_CHECK(result == ZE_RESULT_SUCCESS, "r26: zeKernelCreate ", name,
                " result=", static_cast<int>(result));
    return sycl::make_kernel<L0BE>({bundle, zkernel, ownership::transfer}, context);
  };
  state.norm = wrap("_ZTS24R8R26VaeNormAffineKernel");
  state.norm_precomputed = wrap("_ZTS35R8R26VaeNormAffinePrecomputedKernel");
  state.norm_rope = wrap("_ZTS22R8R26VaeNormRopeKernel");
  state.ready = true;
}

R26State &r26_ensure() {
  std::lock_guard<std::mutex> lock(r26_mutex());
  r26_init_locked();
  return r26_state();
}

bool r26_bf16_xpu(const torch::Tensor &tensor) {
  return tensor.is_xpu() && tensor.scalar_type() == torch::kBFloat16;
}

bool r26_aligned(const torch::Tensor &tensor, uintptr_t bytes) {
  return (reinterpret_cast<uintptr_t>(tensor.data_ptr()) & (bytes - 1)) == 0;
}

void r26_check_norm_affine(const torch::Tensor &x, int64_t C,
                           const torch::Tensor &weight, const torch::Tensor &scale,
                           const torch::Tensor &shift, const torch::Tensor &out) {
  TORCH_CHECK(r26_bf16_xpu(x) && x.is_contiguous() && x.stride(-1) == 1,
              "r26: x must be contiguous XPU bf16");
  TORCH_CHECK(C >= 64 && C <= R26_NORM_MAX_C && C % 8 == 0,
              "r26: norm channel count must be in [64,256] and divisible by 8");
  TORCH_CHECK(r26_bf16_xpu(weight) && weight.numel() == C && weight.is_contiguous(),
              "r26: weight must be contiguous XPU bf16 with C elements");
  TORCH_CHECK(r26_bf16_xpu(scale) && scale.numel() == C && scale.is_contiguous(),
              "r26: scale must be contiguous XPU bf16 with C elements");
  TORCH_CHECK(r26_bf16_xpu(shift) && shift.numel() == C && shift.is_contiguous(),
              "r26: shift must be contiguous XPU bf16 with C elements");
  TORCH_CHECK(out.sizes() == x.sizes() && r26_bf16_xpu(out) && out.is_contiguous() &&
                  out.stride(-1) == 1,
              "r26: out must match x and be contiguous XPU bf16");
  TORCH_CHECK(weight.device() == x.device() && scale.device() == x.device() &&
                  shift.device() == x.device() && out.device() == x.device(),
              "r26: all tensors must be on the same XPU");
  TORCH_CHECK(r26_aligned(x, 16) && r26_aligned(weight, 16) &&
                  r26_aligned(scale, 16) && r26_aligned(shift, 16) &&
                  r26_aligned(out, 16),
              "r26: bf16 tensors must be 16-byte aligned");
}

torch::Tensor r26_norm_affine_impl(torch::Tensor x, torch::Tensor weight,
                                   torch::Tensor scale, torch::Tensor shift,
                                   torch::Tensor out, double eps) {
  const int64_t C = x.size(-1);
  r26_check_norm_affine(x, C, weight, scale, shift, out);
  const int64_t rows = x.numel() / C;
  TORCH_CHECK(rows > 0, "r26: empty norm input");
  R26State &state = r26_ensure();
  R26NormAffineArgs args{};
  args.x = x.data_ptr();
  args.weight = weight.data_ptr();
  args.scale = scale.data_ptr();
  args.shift = shift.data_ptr();
  args.out = out.data_ptr();
  args.rows = rows;
  args.C = C;
  args.eps = static_cast<float>(eps);
  const int64_t groups = (rows + R26_NORM_ROWS_PER_WG - 1) / R26_NORM_ROWS_PER_WG;
  state.q->submit([&](sycl::handler &handler) {
    handler.set_arg(0, args);
    handler.parallel_for(
        sycl::nd_range<1>(static_cast<size_t>(groups) * R26_NORM_WG, R26_NORM_WG),
        *state.norm);
  });
  return out;
}

torch::Tensor r26_norm_affine_precomputed_impl(
    torch::Tensor x, torch::Tensor weight, torch::Tensor affine_scale,
    torch::Tensor shift, torch::Tensor out, double eps) {
  const int64_t C = x.size(-1);
  r26_check_norm_affine(x, C, weight, affine_scale, shift, out);
  const int64_t rows = x.numel() / C;
  TORCH_CHECK(rows > 0, "r26: empty norm input");
  R26State &state = r26_ensure();
  R26NormAffineArgs args{};
  args.x = x.data_ptr();
  args.weight = weight.data_ptr();
  args.scale = affine_scale.data_ptr();
  args.shift = shift.data_ptr();
  args.out = out.data_ptr();
  args.rows = rows;
  args.C = C;
  args.eps = static_cast<float>(eps);
  const int64_t groups = (rows + R26_NORM_ROWS_PER_WG - 1) / R26_NORM_ROWS_PER_WG;
  state.q->submit([&](sycl::handler &handler) {
    handler.set_arg(0, args);
    handler.parallel_for(
        sycl::nd_range<1>(static_cast<size_t>(groups) * R26_NORM_WG, R26_NORM_WG),
        *state.norm_precomputed);
  });
  return out;
}

void r26_check_norm_rope(const torch::Tensor &q, const torch::Tensor &k,
                         const torch::Tensor &q_weight, const torch::Tensor &k_weight,
                         const torch::Tensor &cos, const torch::Tensor &sin,
                         int64_t B, int64_t T, int64_t H, int64_t W, int64_t NH,
                         int64_t DH, int64_t d_t, int64_t d_h) {
  TORCH_CHECK(q.dim() == 6 && k.dim() == 6 && q.sizes() == k.sizes(),
              "r26: q/k must be matching rank-6 tensors");
  TORCH_CHECK(r26_bf16_xpu(q) && r26_bf16_xpu(k) && q.is_contiguous() &&
                  k.is_contiguous(),
              "r26: q/k must be contiguous XPU bf16");
  TORCH_CHECK(B > 0 && T > 0 && H > 0 && W > 0 && NH > 0 && DH == R26_QK_HEAD_DIM,
              "r26: unsupported q/k shape");
  TORCH_CHECK(q.size(0) == B && q.size(1) == T && q.size(2) == H &&
                  q.size(3) == W && q.size(4) == NH && q.size(5) == DH,
              "r26: q/k shape metadata mismatch");
  TORCH_CHECK(d_t > 0 && d_h > 0 && d_t % 2 == 0 && d_h % 2 == 0 &&
                  d_t + d_h < DH && (DH - d_t - d_h) % 2 == 0,
              "r26: invalid RoPE dimension split");
  TORCH_CHECK(r26_bf16_xpu(q_weight) && r26_bf16_xpu(k_weight) &&
                  q_weight.numel() == DH && k_weight.numel() == DH &&
                  q_weight.is_contiguous() && k_weight.is_contiguous(),
              "r26: q/k weights must be contiguous XPU bf16 vectors");
  TORCH_CHECK(cos.is_xpu() && sin.is_xpu() && cos.scalar_type() == torch::kFloat32 &&
                  sin.scalar_type() == torch::kFloat32 && cos.dim() == 1 &&
                  sin.dim() == 1 && cos.sizes() == sin.sizes() &&
                  cos.is_contiguous() && sin.is_contiguous(),
              "r26: cos/sin must be contiguous XPU fp32 vectors");
  const int64_t t_pairs = d_t / 2;
  const int64_t h_pairs = d_h / 2;
  const int64_t w_pairs = (DH - d_t - d_h) / 2;
  const int64_t table_size = T * t_pairs + H * h_pairs + W * w_pairs;
  TORCH_CHECK(cos.numel() == table_size && sin.numel() == table_size,
              "r26: cos/sin table size mismatch");
  TORCH_CHECK(q_weight.device() == q.device() && k_weight.device() == q.device() &&
                  cos.device() == q.device() && sin.device() == q.device(),
              "r26: all tensors must be on the same XPU");
  TORCH_CHECK(r26_aligned(q, 16) && r26_aligned(k, 16) &&
                  r26_aligned(q_weight, 16) && r26_aligned(k_weight, 16) &&
                  r26_aligned(cos, 4) && r26_aligned(sin, 4),
              "r26: tensor alignment mismatch");
}

std::vector<torch::Tensor> r26_norm_rope_impl(
    torch::Tensor q, torch::Tensor k, torch::Tensor q_weight, torch::Tensor k_weight,
    torch::Tensor cos, torch::Tensor sin, int64_t B, int64_t T, int64_t H,
    int64_t W, int64_t NH, int64_t DH, int64_t d_t, int64_t d_h, double eps,
    double q_scale, double k_scale) {
  r26_check_norm_rope(q, k, q_weight, k_weight, cos, sin, B, T, H, W, NH, DH, d_t,
                      d_h);
  const int64_t t_pairs = d_t / 2;
  const int64_t h_pairs = d_h / 2;
  const int64_t w_pairs = (DH - d_t - d_h) / 2;
  const int64_t rows = B * T * H * W * NH;
  torch::Tensor outq = at::empty(q.sizes(), q.options());
  torch::Tensor outk = at::empty(k.sizes(), k.options());
  R26State &state = r26_ensure();
  R26NormRopeArgs args{};
  args.q = q.data_ptr();
  args.k = k.data_ptr();
  args.q_weight = q_weight.data_ptr();
  args.k_weight = k_weight.data_ptr();
  args.cos = cos.data_ptr();
  args.sin = sin.data_ptr();
  args.outq = outq.data_ptr();
  args.outk = outk.data_ptr();
  args.B = B;
  args.T = T;
  args.H = H;
  args.W = W;
  args.NH = NH;
  args.DH = DH;
  args.d_t = d_t;
  args.d_h = d_h;
  args.t_pairs = t_pairs;
  args.h_pairs = h_pairs;
  args.w_pairs = w_pairs;
  args.rows = rows;
  args.eps = static_cast<float>(eps);
  args.q_scale = static_cast<float>(q_scale);
  args.k_scale = static_cast<float>(k_scale);
  const int64_t groups = (rows + R26_QK_ROWS_PER_WG - 1) / R26_QK_ROWS_PER_WG;
  state.q->submit([&](sycl::handler &handler) {
    handler.set_arg(0, args);
    handler.parallel_for(
        sycl::nd_range<1>(static_cast<size_t>(groups) * R26_QK_WG, R26_QK_WG),
        *state.norm_rope);
  });
  return {outq, outk};
}

}  // namespace

TORCH_LIBRARY(libr26_vae, m) {
  m.def("r26_norm_affine(Tensor x, Tensor weight, Tensor scale, Tensor shift, Tensor out, float eps) -> Tensor");
  m.impl("r26_norm_affine", [](torch::Tensor x, torch::Tensor weight,
                                torch::Tensor scale, torch::Tensor shift,
                                torch::Tensor out, double eps) {
    return r26_norm_affine_impl(x, weight, scale, shift, out, eps);
  });
  m.def("r26_norm_affine_precomputed(Tensor x, Tensor weight, Tensor affine_scale, Tensor shift, Tensor out, float eps) -> Tensor");
  m.impl("r26_norm_affine_precomputed", [](torch::Tensor x, torch::Tensor weight,
                                           torch::Tensor affine_scale,
                                           torch::Tensor shift,
                                           torch::Tensor out, double eps) {
    return r26_norm_affine_precomputed_impl(x, weight, affine_scale, shift, out, eps);
  });
  m.def("r26_norm_rope(Tensor q, Tensor k, Tensor q_weight, Tensor k_weight, Tensor cos, Tensor sin, int B, int T, int H, int W, int NH, int DH, int d_t, int d_h, float eps, float q_scale, float k_scale) -> Tensor[]");
  m.impl("r26_norm_rope", [](torch::Tensor q, torch::Tensor k,
                              torch::Tensor q_weight, torch::Tensor k_weight,
                              torch::Tensor cos, torch::Tensor sin, int64_t B,
                              int64_t T, int64_t H, int64_t W, int64_t NH,
                              int64_t DH, int64_t d_t, int64_t d_h, double eps,
                              double q_scale, double k_scale) {
    return r26_norm_rope_impl(q, k, q_weight, k_weight, cos, sin, B, T, H, W,
                              NH, DH, d_t, d_h, eps, q_scale, k_scale);
  });
}
