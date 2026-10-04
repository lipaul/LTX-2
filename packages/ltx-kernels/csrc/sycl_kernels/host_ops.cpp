// host side: TORCH_LIBRARY ops launching prebuilt SPIR-V kernels
// (kernels.spv) on torch's own XPU queue via L0 interop.
//
// Device kernels are compiled to SPIR-V with icpx -fsycl (oneAPI 2026.1); the
// host side is plain C++ linked against the venv libsycl.so.9 and launches the
// kernels, wrapped with make_kernel_bundle/make_kernel, through
// handler::parallel_for(nd_range, kernel) on torch's queue. The queue is the
// current torch XPU stream; every op checks it is the queue captured at
// initialisation, because the shared scratch buffer and the in-order
// guarantee hold for that queue only.
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

#include "kernel_args.h"

namespace {

using u16 = uint16_t;

constexpr int64_t MAX_C = 8192;
constexpr size_t SCRATCH_FLOATS = 1 << 24;  // 64MB of floats; each row uses MAX_SUBGROUPS (32) floats
constexpr int WG_SIZE = 256;
constexpr int VEC8 = 8;

constexpr int MLP_NORM_WG = 256;
constexpr int MLP_ROWS_PER_GROUP = 8;

int64_t env_int(const char *name, int64_t default_value, int64_t minimum,
                int64_t maximum) {
  const char *text = std::getenv(name);
  if (text == nullptr) return default_value;
  char *end = nullptr;
  const long long value = std::strtoll(text, &end, 10);
  TORCH_CHECK(end != nullptr && end != text && *end == '\0' && value >= minimum &&
                  value <= maximum,
              "norm_rope: invalid ", name, "=", text);
  return static_cast<int64_t>(value);
}



// Torch-exact rounding: the fused sites reproduce the bf16 rounding of the
// torch expression. On by default; LTX_TORCH_EXACT_ROUNDING=0 selects the fused
// math; LTX_ORIGINAL_PATH=1 forces off.
bool torch_exact_rounding_on() {
  const char *all = std::getenv("LTX_ORIGINAL_PATH");
  if (all != nullptr && all[0] == '1' && all[1] == '\0') return false;
  const char *v = std::getenv("LTX_TORCH_EXACT_ROUNDING");
  if (v == nullptr) return true;
  return !(v[0] == '0' && v[1] == '\0');
}

static const char *mangled_type(const char *n) {
  // Itanium mangling of an internal-linkage type is _ZTS<len><name>.
  // The length is computed rather than hand-counted so a rename
  // cannot silently break the kernel lookup.
  static std::string m;
  m = "_ZTS" + std::to_string(std::strlen(n)) + n;
  return m.c_str();
}

static const char *unused_mangled_strided() {
  return mangled_type("BBcastStridedKernel");
}

struct SyclState {
  bool ready = false;
  std::optional<sycl::queue> q;
  // Ada-zero (norm+affine) and affine-only variants, plus torch-rounding forms.
  std::optional<sycl::kernel> ada_norm_affine;
  std::optional<sycl::kernel> affine_addcmul;
  std::optional<sycl::kernel> ada_norm_affine_torch_round;
  std::optional<sycl::kernel> affine_addcmul_torch_round;
  // Post-self-attention gate plus RMSNorm, dual output.
  std::optional<sycl::kernel> post_attn_gate_norm;
  std::optional<sycl::kernel> post_attn_gate_norm_torch_round;
  // Fused MLP pre-norm/affine and q-k norm+rope.
  std::optional<sycl::kernel> mlp_norm;
  std::optional<sycl::kernel> norm_rope;
  std::optional<sycl::kernel> norm_rope_torch_round;
  // Split-rope negative fold, scalar and vectorised, plus torch-rounding forms.
  std::optional<sycl::kernel> rope_negfold;
  std::optional<sycl::kernel> rope_negfold_vec;
  std::optional<sycl::kernel> rope_negfold_torch_round;
  std::optional<sycl::kernel> rope_negfold_vec_torch_round;
  // Per-head gate broadcast multiply and tanh-approximation GELU.
  std::optional<sycl::kernel> gate;
  std::optional<sycl::kernel> gelu_fast;
  std::optional<sycl::kernel> gelu_exact;
  // fp8(e4m3fn) -> bf16 widening.
  std::optional<sycl::kernel> fp8w;
  // Strided and flat-run broadcast in-place ops.
  std::optional<sycl::kernel> bbcast_strided;
  std::optional<sycl::kernel> elem;
  std::optional<sycl::kernel> elem16;
  float *scratch = nullptr;
};

SyclState &sycl_state() {
  static SyclState s;
  return s;
}
std::mutex &sycl_mutex() {
  static std::mutex m;
  return m;
}

std::string spv_dir() {
  Dl_info info{};
  if (dladdr((const void *)&sycl_state, &info) && info.dli_fname)
    return std::string(info.dli_fname);
  return {};
}

std::string spv_path() {
  std::string d = spv_dir();
  auto pos = d.find_last_of('/');
  std::string dir = (pos == std::string::npos) ? "." : d.substr(0, pos);
  return dir + "/kernels.spv";
}

void sycl_init_locked() {
  SyclState &s = sycl_state();
  if (s.ready) return;
  sycl::queue &q = c10::xpu::getCurrentXPUStream().queue();
  TORCH_CHECK(q.is_in_order(), "ltx_sycl: torch XPU queue must be in-order");
  s.q = q;
  sycl::context sctx = q.get_context();
  sycl::device sdev = q.get_device();
  constexpr auto L0BE = sycl::backend::ext_oneapi_level_zero;
  auto zctx = sycl::get_native<L0BE>(sctx);
  auto zdev = sycl::get_native<L0BE>(sdev);

  std::string path = spv_path();
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  TORCH_CHECK(f.good(), "ltx_sycl: cannot open ", path);
  size_t n = (size_t)f.tellg();
  f.seekg(0);
  std::vector<uint8_t> il(n);
  TORCH_CHECK(f.read((char *)il.data(), n).good(), "ltx_sycl: cannot read ", path);

  ze_module_desc_t mdesc{};
  mdesc.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
  mdesc.format = ZE_MODULE_FORMAT_IL_SPIRV;
  mdesc.inputSize = n;
  mdesc.pInputModule = il.data();
  ze_module_handle_t zmod = nullptr;
  TORCH_CHECK(zeModuleCreate(zctx, zdev, &mdesc, &zmod, nullptr) == ZE_RESULT_SUCCESS,
              "ltx_sycl: zeModuleCreate failed");

  using sycl::ext::oneapi::level_zero::ownership;
  auto bundle = sycl::make_kernel_bundle<L0BE, sycl::bundle_state::executable>(
      {zmod, ownership::keep}, sctx);

  auto wrap = [&](const char *name) {
    ze_kernel_desc_t kdesc{};
    kdesc.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
    kdesc.pKernelName = name;
    ze_kernel_handle_t zk = nullptr;
    ze_result_t zr = zeKernelCreate(zmod, &kdesc, &zk);
    TORCH_CHECK(zr == ZE_RESULT_SUCCESS, "ltx_sycl: zeKernelCreate ", name, " zr=", (int)zr);
    return sycl::make_kernel<L0BE>({bundle, zk, ownership::transfer}, sctx);
  };
  // Kernel names are the Itanium mangling of the device-side kernel type, so
  // they are built with mangled_type() rather than hand-counted string literals.
  s.ada_norm_affine = wrap(mangled_type("AdaNormAffineKernel"));
  s.affine_addcmul = wrap(mangled_type("AffineAddcmulKernel"));
  s.ada_norm_affine_torch_round = wrap(mangled_type("AdaNormAffineTorchRoundKernel"));
  s.affine_addcmul_torch_round = wrap(mangled_type("AffineAddcmulTorchRoundKernel"));
  s.post_attn_gate_norm = wrap(mangled_type("PostAttnGateNormKernel"));
  s.post_attn_gate_norm_torch_round = wrap(mangled_type("PostAttnGateNormTorchRoundKernel"));
  s.mlp_norm = wrap(mangled_type("MlpNormAffineKernel"));
  s.norm_rope = wrap(mangled_type("NormRopeKernel"));
  s.norm_rope_torch_round = wrap(mangled_type("NormRopeTorchRoundKernel"));
  s.rope_negfold = wrap(mangled_type("RopeNegFoldKernel"));
  s.rope_negfold_vec = wrap(mangled_type("RopeNegFoldVecKernel"));
  s.rope_negfold_torch_round = wrap(mangled_type("RopeNegFoldTorchRoundKernel"));
  s.rope_negfold_vec_torch_round = wrap(mangled_type("RopeNegFoldVecTorchRoundKernel"));
  s.gate = wrap(mangled_type("GateBroadcastKernel"));
  s.gelu_fast = wrap(mangled_type("GeluFastKernel"));
  s.gelu_exact = wrap(mangled_type("GeluTanhExactKernel"));
  s.fp8w = wrap(mangled_type("Fp8WidenKernel"));
  s.bbcast_strided = wrap(mangled_type("BBcastStridedKernel"));
  s.elem = wrap(mangled_type("ElemKernel"));
  s.elem16 = wrap(mangled_type("ElemKernelV16"));
  s.scratch = sycl::malloc_device<float>(SCRATCH_FLOATS, q);
  TORCH_CHECK(s.scratch, "ltx_sycl: scratch alloc failed");
  s.ready = true;
}

SyclState &sycl_ensure() {
  std::lock_guard<std::mutex> lk(sycl_mutex());
  sycl_init_locked();
  SyclState &s = sycl_state();
  TORCH_CHECK(c10::xpu::getCurrentXPUStream().queue() == *s.q,
              "ltx_sycl: called on a different XPU stream or device than the one captured at initialisation");
  return s;
}

bool tensors_ok(const torch::Tensor &t) {
  return t.is_xpu() && t.dtype() == torch::kBFloat16;
}

void check_row_major(const torch::Tensor &x, int64_t C) {
  TORCH_CHECK(tensors_ok(x), "ltx_sycl: x must be XPU bf16");
  TORCH_CHECK(x.stride(-1) == 1, "ltx_sycl: x innermost stride must be 1");
  TORCH_CHECK(x.is_contiguous(), "ltx_sycl: x must be contiguous");
  TORCH_CHECK(C >= 64 && C <= MAX_C, "ltx_sycl: C out of range");
}

void check_part(int64_t C, const torch::Tensor &tab, const torch::Tensor &ts) {
  TORCH_CHECK(tensors_ok(tab) && tab.numel() == C, "ltx_sycl: table must be XPU bf16 numel==C");
  TORCH_CHECK(tensors_ok(ts) && ts.size(-1) == C && ts.stride(-1) == 1,
              "ltx_sycl: ts must be XPU bf16 matching last dim, stride 1");
}

torch::Tensor ada_impl(torch::Tensor x, torch::Tensor tbs, torch::Tensor tss, int64_t tss_rs,
                      torch::Tensor tbh, torch::Tensor tsh, int64_t tsh_rs, double eps,
                      bool do_norm) {
  const int64_t C = x.size(-1);
  check_row_major(x, C);
  check_part(C, tbs, tss);
  check_part(C, tbh, tsh);
  torch::Tensor out = at::empty_like(x);
  const int64_t rows = x.numel() / C;
  TORCH_CHECK((size_t)rows * 32 <= SCRATCH_FLOATS, "ltx_sycl: too many rows for scratch");
  SyclState &s = sycl_ensure();
  AdaNormAffineArgs A{};
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
  sycl::kernel k = do_norm ? (torch_exact_rounding_on() ? *s.ada_norm_affine_torch_round : *s.ada_norm_affine)
                           : (torch_exact_rounding_on() ? *s.affine_addcmul_torch_round : *s.affine_addcmul);
  s.q->submit([&](sycl::handler &h) {
    h.set_arg(0, A);
    h.parallel_for(sycl::nd_range<1>((size_t)rows * WG_SIZE, WG_SIZE), k);
  });
  return out;
}









std::vector<torch::Tensor> post_sa_impl(torch::Tensor x, torch::Tensor y, torch::Tensor tbg,
                                   torch::Tensor tsg, int64_t tsg_rs, double eps) {
  const int64_t C = x.size(-1);
  check_row_major(x, C);
  TORCH_CHECK(y.sizes() == x.sizes() && tensors_ok(y) && y.is_contiguous(),
              "ltx_sycl: y must match x (XPU bf16 contiguous)");
  check_part(C, tbg, tsg);
  torch::Tensor fma = at::empty_like(x);
  torch::Tensor out = at::empty_like(x);
  const int64_t rows = x.numel() / C;
  TORCH_CHECK((size_t)rows * 32 <= SCRATCH_FLOATS, "ltx_sycl: too many rows for scratch");
  SyclState &s = sycl_ensure();
  PostAttnGateNormArgs A{};
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
    h.parallel_for(sycl::nd_range<1>((size_t)rows * WG_SIZE, WG_SIZE),
                   torch_exact_rounding_on() ? *s.post_attn_gate_norm_torch_round : *s.post_attn_gate_norm);
  });
  return {fma, out};
}

void check_mlp_norm_affine(const torch::Tensor &x, int64_t C,
                                 const torch::Tensor &norm_weight,
                                 const torch::Tensor &scale, const torch::Tensor &shift,
                                 const torch::Tensor &out) {
  TORCH_CHECK(tensors_ok(norm_weight) && norm_weight.numel() == C &&
                  norm_weight.is_contiguous() && norm_weight.stride(-1) == 1,
              "mlp_norm: norm_weight must be contiguous XPU bf16 numel==C");
  TORCH_CHECK(tensors_ok(scale) && scale.numel() == C && scale.is_contiguous() &&
                  scale.stride(-1) == 1,
              "mlp_norm: scale must be contiguous XPU bf16 numel==C");
  TORCH_CHECK(tensors_ok(shift) && shift.numel() == C && shift.is_contiguous() &&
                  shift.stride(-1) == 1,
              "mlp_norm: shift must be contiguous XPU bf16 numel==C");
  TORCH_CHECK(out.sizes() == x.sizes() && tensors_ok(out) && out.is_contiguous() &&
                  out.stride(-1) == 1,
              "mlp_norm: out must match x and be contiguous XPU bf16");
  TORCH_CHECK(norm_weight.device() == x.device() && scale.device() == x.device() &&
                  shift.device() == x.device() && out.device() == x.device(),
              "mlp_norm: all tensors must be on the same XPU");
  const auto aligned = [](const torch::Tensor &t) {
    return (reinterpret_cast<uintptr_t>(t.data_ptr()) & 15u) == 0;
  };
  TORCH_CHECK(aligned(x) && aligned(norm_weight) && aligned(scale) && aligned(shift) && aligned(out),
              "mlp_norm: all tensors must be 16-byte aligned");
}

torch::Tensor mlp_norm_affine_impl(torch::Tensor x, torch::Tensor norm_weight,
                                       torch::Tensor scale, torch::Tensor shift,
                                       torch::Tensor out, double eps) {
  const int64_t C = x.size(-1);
  check_row_major(x, C);
  check_mlp_norm_affine(x, C, norm_weight, scale, shift, out);
  const int64_t rows = x.numel() / C;
  TORCH_CHECK(rows > 0, "mlp_norm: empty norm input");
  SyclState &s = sycl_ensure();
  MlpNormAffineArgs A{};
  A.x = x.data_ptr();
  A.weight = norm_weight.data_ptr();
  A.scale = scale.data_ptr();
  A.shift = shift.data_ptr();
  A.out = out.data_ptr();
  A.scratch = nullptr;
  A.rows = rows;
  A.C = C;
  A.eps = (float)eps;
  const int64_t groups = (rows + MLP_ROWS_PER_GROUP - 1) / MLP_ROWS_PER_GROUP;
  s.q->submit([&](sycl::handler &h) {
    h.set_arg(0, A);
    h.parallel_for(
        sycl::nd_range<1>((size_t)groups * MLP_NORM_WG, MLP_NORM_WG), *s.mlp_norm);
  });
  return out;
}

std::vector<torch::Tensor> norm_rope_impl(
    torch::Tensor q, torch::Tensor k, torch::Tensor q_weight, torch::Tensor k_weight,
    torch::Tensor cos, torch::Tensor sin, int64_t B, int64_t H, int64_t T,
    int64_t Dh, double eps, double q_scale, double k_scale) {
  TORCH_CHECK(q.is_xpu() && k.is_xpu(), "norm_rope: q/k must be XPU tensors");
  TORCH_CHECK(q.scalar_type() == torch::kBFloat16 && k.scalar_type() == torch::kBFloat16,
              "norm_rope: q/k must be bf16");
  TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && q.sizes() == k.sizes(),
              "norm_rope: q/k must be matching rank-3 tensors");
  TORCH_CHECK(B > 0 && H > 0 && T > 0 && Dh > 0 && B == q.size(0) && T == q.size(1),
              "norm_rope: invalid B/H/T metadata");
  const int64_t C = H * Dh * 2;
  TORCH_CHECK(q.size(2) == C && C >= 64 && C <= MAX_C && C % 8 == 0 && Dh % 8 == 0,
              "norm_rope: unsupported q/k channel shape");
  TORCH_CHECK(q.stride(2) == 1 && k.stride(2) == 1,
              "norm_rope: q/k innermost stride must be one");
  TORCH_CHECK(q.stride(1) % 8 == 0 && q.stride(0) % 8 == 0 &&
                  k.stride(1) % 8 == 0 && k.stride(0) % 8 == 0,
              "norm_rope: q/k outer strides must be 8-element aligned");
  TORCH_CHECK(q_weight.is_xpu() && k_weight.is_xpu() &&
                  q_weight.scalar_type() == torch::kBFloat16 &&
                  k_weight.scalar_type() == torch::kBFloat16 &&
                  q_weight.numel() == C && k_weight.numel() == C &&
                  q_weight.is_contiguous() && k_weight.is_contiguous(),
              "norm_rope: q/k weights must be contiguous XPU bf16 vectors");
  TORCH_CHECK(q_weight.device() == q.device() && k_weight.device() == q.device(),
              "norm_rope: q/k weights must be on the q device");
  TORCH_CHECK(cos.is_xpu() && sin.is_xpu() && cos.dim() == 4 && sin.dim() == 4 &&
                  cos.sizes() == sin.sizes() && cos.scalar_type() == torch::kBFloat16 &&
                  sin.scalar_type() == torch::kBFloat16,
              "norm_rope: cos/sin must be matching XPU bf16 rank-4 tensors");
  TORCH_CHECK((cos.size(0) == B || (cos.stride(0) == 0 && sin.stride(0) == 0)) && cos.size(1) == H &&
                  cos.size(2) == T && cos.size(3) == Dh,
              "norm_rope: cos/sin shape metadata mismatch (batch must equal B or be broadcast with stride 0)");
  TORCH_CHECK(cos.stride(3) == 1 && sin.stride(3) == 1 &&
                  cos.stride(0) % 8 == 0 && cos.stride(1) % 8 == 0 &&
                  cos.stride(2) % 8 == 0 && sin.stride(0) % 8 == 0 &&
                  sin.stride(1) % 8 == 0 && sin.stride(2) % 8 == 0,
              "norm_rope: cos/sin strides are not vector compatible");
  const auto aligned = [](const torch::Tensor &t) {
    return (reinterpret_cast<uintptr_t>(t.data_ptr()) & 15u) == 0;
  };
  TORCH_CHECK(aligned(q) && aligned(k) && aligned(q_weight) && aligned(k_weight) &&
                  aligned(cos) && aligned(sin),
              "norm_rope: all tensors must be 16-byte aligned");
  const int64_t default_wg = C / 4 < 128 ? (C / 4 < 32 ? 32 : C / 4) : 128;
  const int64_t wg = env_int("LTX_SYCL_NORM_ROPE_WG", default_wg, 32, 1024);
  TORCH_CHECK(wg % 32 == 0, "norm_rope: LTX_SYCL_NORM_ROPE_WG must be a multiple of 32");
  const int64_t rows = B * T;
  TORCH_CHECK(rows * 2 * 32 <= static_cast<int64_t>(SCRATCH_FLOATS),
              "norm_rope: row scratch exceeds the device scratch allocation");
  torch::Tensor outq = at::empty(q.sizes(), q.options());
  torch::Tensor outk = at::empty(k.sizes(), k.options());
  SyclState &s = sycl_ensure();
  NormRopeArgs A{};
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
    h.parallel_for(sycl::nd_range<1>(static_cast<size_t>(rows * wg), wg),
                   torch_exact_rounding_on() ? *s.norm_rope_torch_round : *s.norm_rope);
  });
  return {outq, outk};
}

torch::Tensor rope_negfold_impl_common(torch::Tensor x, torch::IntArrayRef xs, torch::Tensor cos,
                              torch::IntArrayRef cs, torch::Tensor sin, torch::IntArrayRef ss,
                              int64_t B, int64_t H, int64_t T, int64_t Dh, bool vec) {
  TORCH_CHECK(tensors_ok(x) && tensors_ok(cos) && tensors_ok(sin),
              "rope_negfold: tensors must be XPU bf16");
  TORCH_CHECK(xs.size() == 4 && cs.size() == 4 && ss.size() == 4, "rope_negfold: needs 4 strides");
  TORCH_CHECK(B == 1 || ((cos.size(0) == B || cs[0] == 0) && (sin.size(0) == B || ss[0] == 0)),
              "rope_negfold: cos/sin batch must equal B or be broadcast with stride 0");
  torch::Tensor out = at::empty({B, H, T, Dh * 2}, x.options());
  const int64_t fibers = B * H * T;
  SyclState &s = sycl_ensure();
  RopeNegFoldArgs A{};
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
    TORCH_CHECK(nvec > 0, "rope_negfold: Dh must be a multiple of 8");
    const int64_t tasks = fibers * nvec;
    constexpr int64_t ROPE_NEGFOLD_VEC_WG = 64;
    const int64_t groups = (tasks + ROPE_NEGFOLD_VEC_WG - 1) / ROPE_NEGFOLD_VEC_WG;
    s.q->submit([&](sycl::handler &h) {
      h.set_arg(0, A);
      h.parallel_for(sycl::nd_range<1>((size_t)groups * ROPE_NEGFOLD_VEC_WG, ROPE_NEGFOLD_VEC_WG),
                     torch_exact_rounding_on() ? *s.rope_negfold_vec_torch_round : *s.rope_negfold_vec);
    });
  } else {
    s.q->submit([&](sycl::handler &h) {
      h.set_arg(0, A);
      h.parallel_for(sycl::nd_range<1>((size_t)fibers * 16, 16),
                     torch_exact_rounding_on() ? *s.rope_negfold_torch_round : *s.rope_negfold);
    });
  }
  return out;
}

torch::Tensor rope_negfold_impl(torch::Tensor x, torch::IntArrayRef xs, torch::Tensor cos,
                      torch::IntArrayRef cs, torch::Tensor sin, torch::IntArrayRef ss,
                      int64_t B, int64_t H, int64_t T, int64_t Dh) {
  return rope_negfold_impl_common(x, xs, cos, cs, sin, ss, B, H, T, Dh, false);
}

torch::Tensor rope_vec_impl(torch::Tensor x, torch::IntArrayRef xs, torch::Tensor cos,
                          torch::IntArrayRef cs, torch::Tensor sin, torch::IntArrayRef ss,
                          int64_t B, int64_t H, int64_t T, int64_t Dh) {
  return rope_negfold_impl_common(x, xs, cos, cs, sin, ss, B, H, T, Dh, true);
}



// out = attn * gates[i / dh]  (bf16, per-head broadcast).
torch::Tensor gate_broadcast_mul_impl(torch::Tensor attn, torch::Tensor gates, int64_t dh) {
  const int64_t n = attn.numel();
  TORCH_CHECK(tensors_ok(attn) && attn.is_contiguous() && attn.stride(-1) == 1,
              "ltx_sycl: attn must be XPU bf16 contiguous");
  TORCH_CHECK(tensors_ok(gates) && gates.is_contiguous(),
              "ltx_sycl: gates must be XPU bf16 contiguous");
  TORCH_CHECK(dh > 0 && dh % 8 == 0 && n % 8 == 0, "ltx_sycl: bad dh/n");
  TORCH_CHECK(gates.numel() == n / dh, "ltx_sycl: gates numel must be n/dh");
  torch::Tensor out = at::empty_like(attn);
  SyclState &s = sycl_ensure();
  GateBroadcastArgs A{};
  A.attn = attn.data_ptr();
  A.gates = gates.data_ptr();
  A.out = out.data_ptr();
  A.n = n;
  A.dh = dh;
  const int64_t tasks = (n / 8 + WG_SIZE - 1) / WG_SIZE;
  s.q->submit([&](sycl::handler &h) {
    h.set_arg(0, A);
    h.parallel_for(sycl::nd_range<1>((size_t)tasks * WG_SIZE, WG_SIZE), *s.gate);
  });
  return out;
}

// 跨步廣播：把五個邏輯軸 (b, t, h, w, c) 融合成最多三組。
//
// 組＝一段相鄰軸；只有當這一段在 x 與 b 上都成連續鏈時才有效，因為 kernel 只
// 用一個線性索引走這一組（x 偏移 = 組內索引 × 組的步幅）。元素數為 1 的外軸先
// 移除，其偏移貢獻為 0；c 軸永遠保留，它是向量通道。回傳 false 代表需要四組。
struct BBcastGroups {
  int64_t e[3];
  int64_t xs[3];
  int64_t bs[3];
  int n;
};

static bool bcast_fuse(const int64_t outer[4], int64_t inner, const int64_t xs_in[4],
                           const int64_t bs_in[5], BBcastGroups &g) {
  int64_t ex[5];
  int64_t xs[5];
  int64_t bs[5];
  int m = 0;
  for (int i = 0; i < 4; ++i) {
    if (outer[i] == 1) continue;
    ex[m] = outer[i];
    xs[m] = xs_in[i];
    bs[m] = bs_in[i];
    ++m;
  }
  ex[m] = inner;
  xs[m] = 1;
  bs[m] = bs_in[4];
  ++m;
  int k = 0;
  g.e[0] = ex[m - 1];
  g.xs[0] = xs[m - 1];
  g.bs[0] = bs[m - 1];
  for (int i = m - 2; i >= 0; --i) {
    if (xs[i] == g.xs[k] * g.e[k] && bs[i] == g.bs[k] * g.e[k]) {
      g.e[k] *= ex[i];
      continue;
    }
    if (k == 2) return false;
    ++k;
    g.e[k] = ex[i];
    g.xs[k] = xs[i];
    g.bs[k] = bs[i];
  }
  for (int i = k + 1; i < 3; ++i) {
    g.e[i] = 1;
    g.xs[i] = 0;
    g.bs[i] = 0;
  }
  g.n = k + 1;
  return true;
}

// 以三群組座標啟動 BBcastStridedKernel：grid 為 (e2, e1, e0 的 8 元素區塊)，
// 最內群組由 dim 2 走。vec_b 只在 b 於最內群組連續、且每個區塊起點 16 位元組
// 對齊時開啟；否則 kernel 以純量收集讀取 b。
static void bcast_launch(torch::Tensor &x, const torch::Tensor &b, const int64_t outer[4],
                             int64_t inner, const int64_t xs_in[4], const int64_t bs_in[5],
                             int64_t op) {
  BBcastGroups g{};
  TORCH_CHECK(bcast_fuse(outer, inner, xs_in, bs_in, g),
              "bcast: x strides ", xs_in[0], ",", xs_in[1], ",", xs_in[2], ",", xs_in[3],
              " with broadcast strides ", bs_in[0], ",", bs_in[1], ",", bs_in[2], ",",
              bs_in[3], ",", bs_in[4], " need more than three fused groups");
  SyclState &s = sycl_ensure();
  BcastStridedArgs A{};
  A.x = x.data_ptr();
  A.b = b.data_ptr();
  A.out = x.data_ptr();
  A.e0 = g.e[0];
  A.g0b = g.bs[0];
  A.e1 = g.e[1];
  A.g1x = g.xs[1];
  A.g1b = g.bs[1];
  A.e2 = g.e[2];
  A.g2x = g.xs[2];
  A.g2b = g.bs[2];
  A.vec_b = (g.bs[0] == 1 && (reinterpret_cast<uintptr_t>(b.data_ptr()) % 16) == 0 &&
             g.bs[1] % 8 == 0 && g.bs[2] % 8 == 0)
                ? 1
                : 0;
  A.op = static_cast<int32_t>(op);
  const int64_t blocks = (g.e[0] + VEC8 - 1) / VEC8;
  const int64_t grid2 = ((blocks + WG_SIZE - 1) / WG_SIZE) * WG_SIZE;
  s.q->submit([&](sycl::handler &h) {
    h.set_arg(0, A);
    h.parallel_for(sycl::nd_range<3>(
                       sycl::range<3>((size_t)g.e[2], (size_t)g.e[1], (size_t)grid2),
                       sycl::range<3>(1, 1, (size_t)WG_SIZE)),
                   *s.bbcast_strided);
  });
}

// 運算元的列步幅 0 代表該運算元在列方向廣播；元素步幅 0（內維尺寸 1）代表每個
// 區塊共用一個值。op 0 = a+b、1 = a*b、2 = 相容 addcmul（先乘後加，兩次 bf16
// 進位）。vec 為每個工作項處理的元素數（8 或 16）。
torch::Tensor elem_impl(torch::Tensor a, torch::Tensor b, c10::optional<torch::Tensor> g,
                           int64_t op, int64_t vec) {
  TORCH_CHECK(tensors_ok(a) && tensors_ok(b), "elementwise: XPU bf16 tensors required");
  TORCH_CHECK(op >= 0 && op <= 3,
              "elementwise: op must be 0 (add), 1 (mul), 2 (compat addcmul) or 3 (fused addcmul)");
  TORCH_CHECK(vec == 8 || vec == 16, "elementwise: vec must be 8 or 16");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && a.is_contiguous() && b.is_contiguous(),
              "elementwise: a and b must be contiguous 2-D tensors");
  TORCH_CHECK(((uintptr_t)a.data_ptr() % 16) == 0 && ((uintptr_t)b.data_ptr() % 16) == 0,
              "elementwise: a and b must be 16-byte aligned");
  const int64_t rows = a.size(0);
  const int64_t d = a.size(1);
  TORCH_CHECK(rows >= 1 && d >= 1, "elementwise: empty operand");
  TORCH_CHECK(b.size(1) == d || b.size(1) == 1,
              "elementwise: b inner size ", b.size(1), " must be ", d, " or 1");
  TORCH_CHECK(b.size(0) == rows || b.size(0) == 1,
              "elementwise: b rows ", b.size(0), " must be ", rows, " or 1");
  if (op >= 2) {
    TORCH_CHECK(g.has_value() && g->defined(), "elementwise: op 2 and 3 need the third operand");
    TORCH_CHECK(tensors_ok(*g) && g->dim() == 2 && g->is_contiguous(),
                "elementwise: g must be a contiguous 2-D bf16 tensor");
    TORCH_CHECK(g->size(1) == d || g->size(1) == 1,
                "elementwise: g inner size ", g->size(1), " must be ", d, " or 1");
    TORCH_CHECK(g->size(0) == rows || g->size(0) == 1,
                "elementwise: g must be (rows,d), (rows,1), (1,d) or (1,1)");
    TORCH_CHECK(((uintptr_t)g->data_ptr() % 16) == 0, "elementwise: g must be 16-byte aligned");
  }
  SyclState &s = sycl_ensure();
  ElemArgs A{};
  A.a = a.data_ptr();
  A.b = b.data_ptr();
  A.g = op >= 2 ? g->data_ptr() : a.data_ptr();
  torch::Tensor out = at::empty({rows, d}, a.options());
  A.out = out.data_ptr();
  A.rows = rows;
  A.d = d;
  A.bs = b.size(0) == 1 ? 0 : d;
  A.gs = (op >= 2 && g->size(0) != 1) ? d : 0;
  A.bvec = b.size(1) == d ? 1 : 0;
  A.gvec = (op >= 2 && g->size(1) == d) ? 1 : 0;
  A.op = static_cast<int32_t>(op);
  const int64_t blocks = (d + vec - 1) / vec;
  const int64_t grid1 = ((blocks + WG_SIZE - 1) / WG_SIZE) * WG_SIZE;
  s.q->submit([&](sycl::handler &h) {
    h.set_arg(0, A);
    sycl::kernel *k = vec == 16 ? &*s.elem16 : &*s.elem;
    h.parallel_for(sycl::nd_range<2>(sycl::range<2>((size_t)rows, (size_t)grid1),
                                     sycl::range<2>(1, (size_t)WG_SIZE)),
                   *k);
  });
  return out;
}

// ElemKernel op 4 / op 5 (gate inline), bf16:
//   op 4: out = a + rnd(b * rnd(tab + ts))
//   op 5: out = a + rnd(rnd(b * rnd(tab + ts)) * m[row / mrows])
// a, b: contiguous (rows, d), 16-byte aligned, d % 8 == 0.
// tab: d contiguous elements, 16-byte aligned.
// ts: row r starts at ts.data_ptr() + (r / ts_div) * ts_rs elements, unit inner
//     stride, 16-byte aligned, ts_rs % 8 == 0, rows % ts_div == 0.
// m (op 5): contiguous bf16, numel == rows / mrows, rows % mrows == 0.
// vec: elements per work item (8 or 16).  Returns a new (rows, d) tensor.
torch::Tensor elem_gate_impl(torch::Tensor a, torch::Tensor b, torch::Tensor tab, torch::Tensor ts,
                             int64_t ts_rs, int64_t ts_div, c10::optional<torch::Tensor> m,
                             int64_t mrows, int64_t vec) {
  TORCH_CHECK_VALUE(tensors_ok(a) && tensors_ok(b) && tensors_ok(tab) && tensors_ok(ts),
                    "elem_gate: XPU bf16 tensors required");
  TORCH_CHECK_VALUE(vec == 8 || vec == 16, "elem_gate: vec must be 8 or 16");
  TORCH_CHECK_VALUE(a.dim() == 2 && a.is_contiguous() && b.sizes() == a.sizes() && b.is_contiguous(),
                    "elem_gate: a and b must be contiguous (rows, d) tensors of equal shape");
  const int64_t rows = a.size(0);
  const int64_t d = a.size(1);
  TORCH_CHECK_VALUE(rows >= 1 && rows <= INT32_MAX && d >= 8 && d % 8 == 0,
                    "elem_gate: rows in [1, 2^31) and d % 8 == 0 required, got rows=", rows, " d=", d);
  const auto aligned = [](const torch::Tensor &t) {
    return (reinterpret_cast<uintptr_t>(t.data_ptr()) & 15u) == 0;
  };
  TORCH_CHECK_VALUE(aligned(a) && aligned(b) && aligned(tab) && aligned(ts),
                    "elem_gate: a, b, tab and ts must be 16-byte aligned");
  TORCH_CHECK_VALUE(tab.numel() == d && tab.is_contiguous(), "elem_gate: tab must hold d contiguous elements");
  TORCH_CHECK_VALUE(ts.size(-1) == d && ts.stride(-1) == 1, "elem_gate: ts inner size d with unit stride required");
  TORCH_CHECK_VALUE(ts_rs >= 0 && ts_rs % 8 == 0, "elem_gate: ts_rs must be a non-negative multiple of 8");
  TORCH_CHECK_VALUE(ts_div >= 1 && rows % ts_div == 0, "elem_gate: rows must be a multiple of ts_div, got rows=",
                    rows, " ts_div=", ts_div);
  const int64_t ts_avail =
      static_cast<int64_t>(ts.storage().nbytes() / ts.element_size()) - ts.storage_offset();
  const int64_t ts_need = (rows / ts_div - 1) * ts_rs + d;
  TORCH_CHECK_VALUE(ts_need <= ts_avail, "elem_gate: ts rows reach past the storage (", ts_need, " > ",
                    ts_avail, " elements)");
  const bool masked = m.has_value() && m->defined();
  if (masked) {
    TORCH_CHECK_VALUE(tensors_ok(*m) && m->is_contiguous(), "elem_gate: m must be contiguous XPU bf16");
    TORCH_CHECK_VALUE(mrows >= 1 && rows % mrows == 0 && m->numel() == rows / mrows,
                      "elem_gate: m numel must equal rows / mrows, got numel=", m->numel(), " rows=", rows,
                      " mrows=", mrows);
  }
  SyclState &s = sycl_ensure();
  ElemArgs A{};
  A.a = a.data_ptr();
  A.b = b.data_ptr();
  A.g = tab.data_ptr();
  torch::Tensor out = at::empty({rows, d}, a.options());
  A.out = out.data_ptr();
  A.rows = rows;
  A.d = d;
  A.bs = d;
  A.gs = 0;
  A.bvec = 1;
  A.gvec = 1;
  A.op = masked ? 5 : 4;
  A.t = ts.data_ptr();
  A.m = masked ? m->data_ptr() : nullptr;
  A.trs = ts_rs;
  A.tdiv = ts_div;
  A.mrows = masked ? mrows : 1;
  const int64_t blocks = (d + vec - 1) / vec;
  const int64_t grid1 = ((blocks + WG_SIZE - 1) / WG_SIZE) * WG_SIZE;
  s.q->submit([&](sycl::handler &h) {
    h.set_arg(0, A);
    sycl::kernel *k = vec == 16 ? &*s.elem16 : &*s.elem;
    h.parallel_for(sycl::nd_range<2>(sycl::range<2>((size_t)rows, (size_t)grid1),
                                     sycl::range<2>(1, (size_t)WG_SIZE)),
                   *k);
  });
  return out;
}

// In-place bf16 mul/div where the second operand is broadcast, vectorised
// over the contiguous innermost run.  ``outer`` holds the four outer extents in
// the flat order the kernel walks and ``inner`` the innermost contiguous extent;
// ``bstrides`` are the broadcast operand's element strides in that same order,
// with 0 marking a broadcast dimension.
torch::Tensor bcast_inplace_impl(torch::Tensor x, torch::Tensor b,
                                 std::vector<int64_t> outer, int64_t inner,
                                 std::vector<int64_t> bstrides, int64_t op) {
  TORCH_CHECK(tensors_ok(x) && tensors_ok(b), "bcast: XPU bf16 tensors required");
  TORCH_CHECK(x.dim() == 5, "bcast: x must be 5-D");
  // Layout is validated by the explicit stride checks below, which is the
  // channels_last_3d form; is_contiguous() would reject it.
  TORCH_CHECK(b.numel() > 0, "bcast: b must be non-empty");
  // The device index math divides and takes a modulus by every outer extent, so
  // none of them may be zero.
  for (int i = 0; i < 4; ++i) {
    TORCH_CHECK(outer[i] >= 1, "bcast: outer extent ", i, " must be >= 1");
  }
  // The vectorised path moves 16 bytes per work item, so the base must be
  // 16-byte aligned; b is read as u16 so it needs 2-byte alignment.
  TORCH_CHECK(((uintptr_t)x.data_ptr() % 16) == 0, "bcast: x must be 16-byte aligned");
  TORCH_CHECK(((uintptr_t)b.data_ptr() % 2) == 0, "bcast: b must be 2-byte aligned");
  TORCH_CHECK(outer.size() == 4 && bstrides.size() == 5, "bcast: 4 outer sizes and 5 strides required");
  TORCH_CHECK(op == 0 || op == 1, "bcast: op must be 0 (mul) or 1 (div)");
  TORCH_CHECK(inner > 0, "bcast: inner must be positive");
  // channels_last_3d: logical (B, C, T, H, W) is stored flat as (b, t, h, w, c),
  // so the kernel's roles (d0, d1, d2, d3, inner) map onto x.stride(0), (2), (3),
  // (4) and (1).  Each must match the size implied by the caller's outer extents.
  TORCH_CHECK(x.stride(1) == 1, "bcast: x needs channels_last_3d (logical dim 1 innermost)");
  TORCH_CHECK(x.stride(4) == inner, "bcast: x stride 4 is ", x.stride(4), " but inner is ", inner);
  TORCH_CHECK(x.stride(3) == outer[3] * inner, "bcast: x stride 3 does not match outer[3]*inner");
  TORCH_CHECK(x.stride(2) == outer[2] * outer[3] * inner,
              "bcast: x stride 2 does not match outer[2]*outer[3]*inner");
  TORCH_CHECK(x.stride(0) == outer[1] * outer[2] * outer[3] * inner,
              "bcast: x stride 0 does not match outer[1]*outer[2]*outer[3]*inner");
  const int64_t total = outer[0] * outer[1] * outer[2] * outer[3] * inner;
  TORCH_CHECK(total == x.numel(), "bcast: outer*inner must equal x.numel");
  TORCH_CHECK(total % 8 == 0, "bcast: numel must be a multiple of 8");
  // Bound the broadcast operand: the largest offset the kernel can form is the
  // sum over dims of (extent - 1) * stride, and it must stay inside b.
  const int64_t ext[5] = {outer[0], outer[1], outer[2], outer[3], inner};
  for (int i = 0; i < 5; ++i) {
    TORCH_CHECK(bstrides[i] >= 0, "bcast: broadcast stride ", i, " must be >= 0");
  }
  int64_t bmax = 0;
  for (int i = 0; i < 5; ++i) bmax += (ext[i] - 1) * bstrides[i];
  TORCH_CHECK(bmax < b.numel(), "bcast: broadcast max offset ", bmax,
              " is outside the operand's ", b.numel(), " elements");
  // x is one flat run in (b, t, h, w, c) order, so its four outer strides are
  // implied by the caller's extents.
  const int64_t xs[4] = {outer[1] * outer[2] * outer[3] * inner, outer[2] * outer[3] * inner,
                         outer[3] * inner, inner};
  bcast_launch(x, b, outer.data(), inner, xs, bstrides.data(), op);
  return x;
}

// ---- strided-x broadcast: for x views that are sliced on the channel dim ----
torch::Tensor bcast_strided_inplace_impl(torch::Tensor x, torch::Tensor b,
                                          std::vector<int64_t> outer, int64_t inner,
                                          std::vector<int64_t> xstrides,
                                          std::vector<int64_t> bstrides, int64_t op) {
  TORCH_CHECK(tensors_ok(x) && tensors_ok(b), "bcast_strided: XPU bf16 tensors required");
  TORCH_CHECK(x.dim() == 5, "bcast_strided: x must be 5-D");
  TORCH_CHECK(b.numel() > 0, "bcast_strided: b must be non-empty");
  TORCH_CHECK(outer.size() == 4, "bcast_strided: 4 outer extents required");
  TORCH_CHECK(xstrides.size() == 4 && bstrides.size() == 5,
              "bcast_strided: 4 x strides and 5 broadcast strides required");
  for (int i = 0; i < 4; ++i) {
    TORCH_CHECK(outer[i] >= 1, "bcast_strided: outer extent ", i, " must be >= 1");
    TORCH_CHECK(xstrides[i] >= 0, "bcast_strided: x stride ", i, " must be >= 0");
    TORCH_CHECK(bstrides[i] >= 0, "bcast_strided: broadcast stride ", i, " must be >= 0");
  }
  TORCH_CHECK(bstrides[4] >= 0, "bcast_strided: broadcast stride 4 must be >= 0");
  TORCH_CHECK(inner > 0, "bcast_strided: inner must be positive");
  TORCH_CHECK(op >= 0 && op <= 4,
                "bcast_strided: op must be 0 (mul), 1 (div), 2 (silu), 3 (add) or 4 (copy)");
  // The vectorised path moves 16 bytes per run, so the channel stride must be 1
  // and the base must be 16-byte aligned.
  TORCH_CHECK(x.stride(1) == 1, "bcast_strided: channel stride must be 1 for the vector run");
  TORCH_CHECK(((uintptr_t)x.data_ptr() % 16) == 0, "bcast_strided: x must be 16-byte aligned");
  TORCH_CHECK(((uintptr_t)b.data_ptr() % 2) == 0, "bcast_strided: b must be 2-byte aligned");
  // The caller's extents must cover the tensor.
  TORCH_CHECK(outer[0] * outer[1] * outer[2] * outer[3] * inner == x.numel(),
              "bcast_strided: outer*inner must equal x.numel");
  TORCH_CHECK(x.numel() % 8 == 0, "bcast_strided: numel must be a multiple of 8");
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
  TORCH_CHECK(storage >= x.numel(), "bcast_strided: storage smaller than the view");
  TORCH_CHECK(xmax < storage, "bcast_strided: strided max offset ", xmax,
              " is outside the ", storage, " element storage");
  int64_t bmax = 0;
  for (int i = 0; i < 5; ++i) bmax += (ext[i] - 1) * bstrides[i];
  TORCH_CHECK(bmax < b.numel(), "bcast_strided: broadcast max offset ", bmax,
              " is outside the operand's ", b.numel(), " elements");
  // The broadcast operand's strides are *assumed* to describe its real layout, and
  // bmax < numel cannot catch a wrong assumption: if the caller passes strides that
  // are self-consistent but do not match the tensor, every index stays in bounds
  // and the kernel silently reads the wrong elements.  That is a wrong-answer bug
  // with no bounds violation to trip on, so the layout is checked here instead.
  // The operand has extent 1 on the channel axis, so its logical dim 1 stride is
  // irrelevant and dims 0,2,3,4 must match bstrides[0..3].
  TORCH_CHECK(b.dim() == 5, "bcast_strided: operand must be 5-D");
  TORCH_CHECK(b.stride(0) == bstrides[0] && b.stride(2) == bstrides[1] &&
                  b.stride(3) == bstrides[2] && b.stride(4) == bstrides[3],
              "bcast_strided: operand strides ", b.stride(0), ",", b.stride(2), ",", b.stride(3),
              ",", b.stride(4), " do not match the given broadcast strides ",
              bstrides[0], ",", bstrides[1], ",", bstrides[2], ",", bstrides[3]);
  bcast_launch(x, b, outer.data(), inner, xs, bstrides.data(), op);
  return x;
}

// tanh-approx GELU (bitwise vs torch's GeluTanhFunctor).
torch::Tensor gelu_tanh_fast_impl(torch::Tensor x) {
  const int64_t n = x.numel();
  TORCH_CHECK(tensors_ok(x) && x.is_contiguous() && x.stride(-1) == 1,
              "gelu_tanh_fast input must be XPU bf16 contiguous");
  TORCH_CHECK(n % 8 == 0, "gelu_tanh_fast numel must be a multiple of 8");
  torch::Tensor out = at::empty_like(x);
  SyclState &s = sycl_ensure();
  GeluArgs A{};
  A.x = x.data_ptr();
  A.out = out.data_ptr();
  A.n = n;
  const int64_t tasks = (n / 8 + WG_SIZE - 1) / WG_SIZE;
  s.q->submit([&](sycl::handler &h) {
    h.set_arg(0, A);
    h.parallel_for(sycl::nd_range<1>((size_t)tasks * WG_SIZE, WG_SIZE),
                   torch_exact_rounding_on() ? *s.gelu_exact : *s.gelu_fast);
  });
  return out;
}


// fp8 e4m3fn -> bf16 widening (vectorised, integer LUT).
torch::Tensor fp8_widen_impl(torch::Tensor x) {
  TORCH_CHECK(x.is_xpu(), "fp8_widen: input must be XPU");
  TORCH_CHECK(x.scalar_type() == at::kFloat8_e4m3fn, "fp8_widen: input must be float8_e4m3fn");
  TORCH_CHECK(x.is_contiguous(), "fp8_widen: input must be contiguous");
  const int64_t n = x.numel();
  TORCH_CHECK(n % 8 == 0, "fp8_widen: numel must be a multiple of 8");
  torch::Tensor out = at::empty(x.sizes(), x.options().dtype(at::kBFloat16));
  SyclState &s = sycl_ensure();
  Fp8Args A{};
  A.in = x.data_ptr();
  A.out = out.data_ptr();
  A.n = n;
  const int64_t tasks = (n / 8 + WG_SIZE - 1) / WG_SIZE;
  s.q->submit([&](sycl::handler &h) {
    h.set_arg(0, A);
    h.parallel_for(sycl::nd_range<1>((size_t)tasks * WG_SIZE, WG_SIZE), *s.fp8w);
  });
  return out;
}





}  // namespace
TORCH_LIBRARY(ltx_sycl, m) {
  m.def("ada_zero(Tensor x, Tensor tbs, Tensor tss, int tss_rs, Tensor tbh, Tensor tsh, int tsh_rs, float eps) -> Tensor");
  m.impl("ada_zero", [](torch::Tensor x, torch::Tensor tbs, torch::Tensor tss, int64_t tss_rs,
                      torch::Tensor tbh, torch::Tensor tsh, int64_t tsh_rs,
                      double eps) { return ada_impl(x, tbs, tss, tss_rs, tbh, tsh, tsh_rs, eps, true); });
  m.def("affine_addcmul(Tensor x, Tensor tbs, Tensor tss, int tss_rs, Tensor tbh, Tensor tsh, int tsh_rs) -> Tensor");
  m.impl("affine_addcmul", [](torch::Tensor x, torch::Tensor tbs, torch::Tensor tss, int64_t tss_rs,
                         torch::Tensor tbh, torch::Tensor tsh,
                         int64_t tsh_rs) { return ada_impl(x, tbs, tss, tss_rs, tbh, tsh, tsh_rs, 1e-6, false); });
  m.def("post_sa(Tensor x, Tensor y, Tensor tbg, Tensor tsg, int tsg_rs, float eps) -> Tensor[]");
  m.impl("post_sa", [](torch::Tensor x, torch::Tensor y, torch::Tensor tbg, torch::Tensor tsg,
                         int64_t tsg_rs, double eps) { return post_sa_impl(x, y, tbg, tsg, tsg_rs, eps); });
  m.def("mlp_norm_affine(Tensor x, Tensor norm_weight, Tensor scale, Tensor shift, Tensor out, float eps) -> Tensor");
  m.impl("mlp_norm_affine", [](torch::Tensor x, torch::Tensor norm_weight, torch::Tensor scale,
                                    torch::Tensor shift, torch::Tensor out,
                                    double eps) { return mlp_norm_affine_impl(x, norm_weight, scale, shift, out, eps); });
  m.def("norm_rope(Tensor q, Tensor k, Tensor q_weight, Tensor k_weight, Tensor cos, Tensor sin, int B, int H, int T, int Dh, float eps, float q_scale, float k_scale) -> Tensor[]");
  m.impl("norm_rope", [](torch::Tensor q, torch::Tensor k, torch::Tensor q_weight,
                             torch::Tensor k_weight, torch::Tensor cos, torch::Tensor sin,
                             int64_t B, int64_t H, int64_t T, int64_t Dh, double eps,
                             double q_scale, double k_scale) {
    return norm_rope_impl(q, k, q_weight, k_weight, cos, sin, B, H, T, Dh, eps,
                              q_scale, k_scale);
  });
  m.def("rope_negfold(Tensor x, int[] xs, Tensor cos, int[] cs, Tensor sin, int[] ss, int B, int H, int T, int Dh) -> Tensor");
  m.impl("rope_negfold", [](torch::Tensor x, torch::IntArrayRef xs, torch::Tensor cos,
                       torch::IntArrayRef cs, torch::Tensor sin, torch::IntArrayRef ss,
                       int64_t B, int64_t H, int64_t T,
                       int64_t Dh) { return rope_negfold_impl(x, xs, cos, cs, sin, ss, B, H, T, Dh); });
  m.def("rope_negfold_vec(Tensor x, int[] xs, Tensor cos, int[] cs, Tensor sin, int[] ss, int B, int H, int T, int Dh) -> Tensor");  m.impl("rope_negfold_vec", [](torch::Tensor x, torch::IntArrayRef xs, torch::Tensor cos,
                           torch::IntArrayRef cs, torch::Tensor sin, torch::IntArrayRef ss,
                           int64_t B, int64_t H, int64_t T,
                           int64_t Dh) { return rope_vec_impl(x, xs, cos, cs, sin, ss, B, H, T, Dh); });
  m.def(
      "bcast_inplace(Tensor x, Tensor b, int[] outer, int inner, int[] bstrides, int op)"
      " -> Tensor");
  m.impl("bcast_inplace", [](torch::Tensor x, torch::Tensor b, std::vector<int64_t> outer,
                                 int64_t inner, std::vector<int64_t> bstrides, int64_t op) {
    return bcast_inplace_impl(x, b, outer, inner, bstrides, op);
  });
  m.def("elem(Tensor a, Tensor b, int op, int vec) -> Tensor");
  m.impl("elem", [](torch::Tensor a, torch::Tensor b, int64_t op, int64_t vec) {
    return elem_impl(a, b, c10::optional<torch::Tensor>(), op, vec);
  });
  m.def("elem3(Tensor a, Tensor b, Tensor g, int op, int vec) -> Tensor");
  m.impl("elem3", [](torch::Tensor a, torch::Tensor b, torch::Tensor g, int64_t op,
                        int64_t vec) {
    return elem_impl(a, b, c10::optional<torch::Tensor>(g), op, vec);
  });
  m.def("elem_gate(Tensor a, Tensor b, Tensor tab, Tensor ts, int ts_rs, int ts_div, Tensor? m, int mrows, "
        "int vec) -> Tensor");
  m.impl("elem_gate", [](torch::Tensor a, torch::Tensor b, torch::Tensor tab, torch::Tensor ts,
                         int64_t ts_rs, int64_t ts_div, c10::optional<torch::Tensor> mk, int64_t mrows,
                         int64_t vec) {
    return elem_gate_impl(a, b, tab, ts, ts_rs, ts_div, mk, mrows, vec);
  });
  m.def("gelu_tanh_fast(Tensor x) -> Tensor");
  m.impl("gelu_tanh_fast", [](torch::Tensor x) { return gelu_tanh_fast_impl(x); });
  m.def("gate_broadcast_mul(Tensor attn, Tensor gates, int dh) -> Tensor");
  m.impl("gate_broadcast_mul", [](torch::Tensor attn, torch::Tensor gates,
                             int64_t dh) { return gate_broadcast_mul_impl(attn, gates, dh); });
  m.def("fp8_widen(Tensor x) -> Tensor");
  m.impl("fp8_widen", [](torch::Tensor x) { return fp8_widen_impl(x); });

  m.def("bcast_strided_inplace(Tensor x, Tensor b, int[] outer, int inner, "
        "int[] xstrides, int[] bstrides, int op) -> Tensor");
  m.impl("bcast_strided_inplace",
         [](torch::Tensor x, torch::Tensor b, std::vector<int64_t> outer, int64_t inner,
            std::vector<int64_t> xstrides, std::vector<int64_t> bstrides, int64_t op) {
           return bcast_strided_inplace_impl(x, b, outer, inner, xstrides, bstrides, op);
         });
}
