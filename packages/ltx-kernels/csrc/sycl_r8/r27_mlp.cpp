#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/library.h>
#include <torch/torch.h>

#include <dlfcn.h>
#include <elf.h>
#include <fcntl.h>
#include <link.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <sycl/sycl.hpp>

#include <oneapi/dnnl/dnnl.h>
#include <oneapi/dnnl/dnnl_sycl.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

struct TorchXpuImage {
  uintptr_t base = 0;
  std::string path;
};

int locate_torch_xpu(dl_phdr_info *info, size_t, void *data) {
  const char *name = info->dlpi_name;
  if (name == nullptr) return 0;
  const char *slash = std::strrchr(name, '/');
  const char *base_name = slash == nullptr ? name : slash + 1;
  if (std::strcmp(base_name, "libtorch_xpu.so") != 0) return 0;
  auto *out = static_cast<TorchXpuImage *>(data);
  out->base = static_cast<uintptr_t>(info->dlpi_addr);
  out->path = name;
  return 1;
}

class ElfSymbols {
 public:
  explicit ElfSymbols(const std::string &path) {
    file_ = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    TORCH_CHECK(file_ >= 0, "r27: cannot open ", path);
    struct stat st {};
    TORCH_CHECK(fstat(file_, &st) == 0, "r27: fstat failed for ", path);
    size_ = static_cast<size_t>(st.st_size);
    map_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, file_, 0);
    TORCH_CHECK(map_ != MAP_FAILED, "r27: mmap failed for ", path);
  }

  ~ElfSymbols() {
    if (map_ != MAP_FAILED) munmap(map_, size_);
    if (file_ >= 0) close(file_);
  }

  uintptr_t value(const char *name) const {
    const auto *base = static_cast<const uint8_t *>(map_);
    const auto *elf = reinterpret_cast<const Elf64_Ehdr *>(base);
    TORCH_CHECK(elf->e_shoff + static_cast<size_t>(elf->e_shnum) * elf->e_shentsize <= size_,
                "r27: invalid ELF section table");
    const auto *sections = reinterpret_cast<const Elf64_Shdr *>(base + elf->e_shoff);
    for (uint16_t i = 0; i < elf->e_shnum; ++i) {
      if (sections[i].sh_type != SHT_SYMTAB && sections[i].sh_type != SHT_DYNSYM) continue;
      const Elf64_Shdr &strings = sections[sections[i].sh_link];
      TORCH_CHECK(strings.sh_offset + strings.sh_size <= size_, "r27: invalid ELF string table");
      const auto *symbols = reinterpret_cast<const Elf64_Sym *>(base + sections[i].sh_offset);
      const size_t count = sections[i].sh_size / sizeof(Elf64_Sym);
      for (size_t j = 0; j < count; ++j) {
        if (symbols[j].st_shndx == SHN_UNDEF) continue;
        const auto *strings_data = reinterpret_cast<const char *>(base + strings.sh_offset);
        if (std::strcmp(strings_data + symbols[j].st_name, name) == 0) {
          return static_cast<uintptr_t>(symbols[j].st_value);
        }
      }
    }
    TORCH_CHECK(false, "r27: embedded oneDNN symbol missing: ", name);
  }

 private:
  int file_ = -1;
  void *map_ = MAP_FAILED;
  size_t size_ = 0;
};

struct DnnlApi {
  decltype(&dnnl_version) version = nullptr;
  decltype(&dnnl_engine_create) engine_create = nullptr;
  decltype(&dnnl_engine_destroy) engine_destroy = nullptr;
  decltype(&dnnl_memory_desc_create_with_tag) memory_desc_create_with_tag = nullptr;
  decltype(&dnnl_memory_desc_destroy) memory_desc_destroy = nullptr;
  decltype(&dnnl_memory_create) memory_create = nullptr;
  decltype(&dnnl_memory_destroy) memory_destroy = nullptr;
  decltype(&dnnl_primitive_attr_create) primitive_attr_create = nullptr;
  decltype(&dnnl_primitive_attr_set_post_ops) primitive_attr_set_post_ops = nullptr;
  decltype(&dnnl_primitive_attr_destroy) primitive_attr_destroy = nullptr;
  decltype(&dnnl_post_ops_create) post_ops_create = nullptr;
  decltype(&dnnl_post_ops_append_binary) post_ops_append_binary = nullptr;
  decltype(&dnnl_post_ops_destroy) post_ops_destroy = nullptr;
  decltype(&dnnl_matmul_primitive_desc_create) matmul_pd_create = nullptr;
  decltype(&dnnl_primitive_desc_destroy) primitive_desc_destroy = nullptr;
  decltype(&dnnl_primitive_create) primitive_create = nullptr;
  decltype(&dnnl_primitive_execute) primitive_execute = nullptr;
  decltype(&dnnl_primitive_destroy) primitive_destroy = nullptr;
  decltype(&dnnl_sycl_interop_stream_create) stream_create = nullptr;
  decltype(&dnnl_stream_destroy) stream_destroy = nullptr;
};

class EmbeddedDnnl {
 public:
  static EmbeddedDnnl &get() {
    static EmbeddedDnnl instance;
    return instance;
  }

  const DnnlApi &api() const { return api_; }

  void check(dnnl_status_t status, const char *operation) const {
    TORCH_CHECK(status == dnnl_success, "r27: oneDNN ", operation, " failed with status ",
                static_cast<int>(status));
  }

 private:
  EmbeddedDnnl() {
    TorchXpuImage image;
    dl_iterate_phdr(locate_torch_xpu, &image);
    TORCH_CHECK(image.base != 0 && !image.path.empty(), "r27: libtorch_xpu.so is not loaded");
    ElfSymbols symbols(image.path);
    auto resolve = [&](const char *name) { return image.base + symbols.value(name); };
#define R27_RESOLVE(field) api_.field = reinterpret_cast<decltype(api_.field)>(resolve("dnnl_" #field))
    R27_RESOLVE(version);
    R27_RESOLVE(engine_create);
    R27_RESOLVE(engine_destroy);
    R27_RESOLVE(memory_desc_create_with_tag);
    R27_RESOLVE(memory_desc_destroy);
    R27_RESOLVE(memory_create);
    R27_RESOLVE(memory_destroy);
    R27_RESOLVE(primitive_attr_create);
    R27_RESOLVE(primitive_attr_set_post_ops);
    R27_RESOLVE(primitive_attr_destroy);
    R27_RESOLVE(post_ops_create);
    R27_RESOLVE(post_ops_append_binary);
    R27_RESOLVE(post_ops_destroy);
    api_.matmul_pd_create = reinterpret_cast<decltype(api_.matmul_pd_create)>(
        resolve("dnnl_matmul_primitive_desc_create"));
    R27_RESOLVE(primitive_desc_destroy);
    R27_RESOLVE(primitive_create);
    R27_RESOLVE(primitive_execute);
    R27_RESOLVE(primitive_destroy);
    api_.stream_create = reinterpret_cast<decltype(api_.stream_create)>(
        resolve("dnnl_sycl_interop_stream_create"));
    R27_RESOLVE(stream_destroy);
#undef R27_RESOLVE
    const dnnl_version_t *v = api_.version();
    TORCH_CHECK(v->major == 3 && v->minor == 11,
                "r27: embedded oneDNN 3.11 required, got ", v->major, ".", v->minor);
  }

  DnnlApi api_;
};

struct PostOpsGuard {
  ~PostOpsGuard() {
    if (value != nullptr) EmbeddedDnnl::get().api().post_ops_destroy(value);
  }
  dnnl_post_ops_t value = nullptr;
};

struct PrimitiveAttrGuard {
  ~PrimitiveAttrGuard() {
    if (value != nullptr) EmbeddedDnnl::get().api().primitive_attr_destroy(value);
  }
  dnnl_primitive_attr_t value = nullptr;
};

struct PrimitiveDescGuard {
  ~PrimitiveDescGuard() {
    if (value != nullptr) EmbeddedDnnl::get().api().primitive_desc_destroy(value);
  }
  dnnl_primitive_desc_t value = nullptr;
};

struct MemoryGuard {
  ~MemoryGuard() {
    if (value != nullptr) EmbeddedDnnl::get().api().memory_destroy(value);
  }
  dnnl_memory_t value = nullptr;
};

struct PrimitiveEntry {
  ~PrimitiveEntry() {
    const DnnlApi &api = EmbeddedDnnl::get().api();
    if (primitive != nullptr) api.primitive_destroy(primitive);
    if (bias_md != nullptr) api.memory_desc_destroy(bias_md);
    if (dst_md != nullptr) api.memory_desc_destroy(dst_md);
    if (residual_md != nullptr) api.memory_desc_destroy(residual_md);
    if (gate_md != nullptr) api.memory_desc_destroy(gate_md);
    if (weights_md != nullptr) api.memory_desc_destroy(weights_md);
    if (src_md != nullptr) api.memory_desc_destroy(src_md);
  }

  int64_t M = 0;
  int64_t N = 0;
  int64_t K = 0;
  bool has_bias = false;
  dnnl_memory_desc_t src_md = nullptr;
  dnnl_memory_desc_t weights_md = nullptr;
  dnnl_memory_desc_t bias_md = nullptr;
  dnnl_memory_desc_t dst_md = nullptr;
  dnnl_memory_desc_t gate_md = nullptr;
  dnnl_memory_desc_t residual_md = nullptr;
  dnnl_primitive_t primitive = nullptr;
};

struct State {
  std::mutex mutex;
  dnnl_engine_t engine = nullptr;
  dnnl_stream_t stream = nullptr;
  sycl::queue *queue = nullptr;
  std::unordered_map<uint64_t, std::unique_ptr<PrimitiveEntry>> cache;
};

State &state() {
  static State *state = new State();
  return *state;
}

uint64_t key_for(int64_t M, int64_t N, int64_t K, bool has_bias) {
  uint64_t key = static_cast<uint64_t>(M);
  key = key * 1000003ull + static_cast<uint64_t>(N);
  key = key * 1000003ull + static_cast<uint64_t>(K);
  return (key << 1) | static_cast<uint64_t>(has_bias);
}

PrimitiveEntry &get_primitive(State &s, int64_t M, int64_t N, int64_t K, bool has_bias) {
  const uint64_t key = key_for(M, N, K, has_bias);
  auto it = s.cache.find(key);
  if (it != s.cache.end()) return *it->second;
  const DnnlApi &api = EmbeddedDnnl::get().api();
  auto entry = std::make_unique<PrimitiveEntry>();
  entry->M = M;
  entry->N = N;
  entry->K = K;
  entry->has_bias = has_bias;
  dnnl_dims_t src_dims = {M, K};
  dnnl_dims_t weight_dims = {K, N};
  dnnl_dims_t bias_dims = {1, N};
  dnnl_dims_t dst_dims = {M, N};
  dnnl_dims_t gate_dims = {1, N};
  dnnl_dims_t residual_dims = {M, N};
  EmbeddedDnnl::get().check(
      api.memory_desc_create_with_tag(&entry->src_md, 2, src_dims, dnnl_bf16, dnnl_ab),
      "src_desc");
  EmbeddedDnnl::get().check(
      api.memory_desc_create_with_tag(&entry->weights_md, 2, weight_dims, dnnl_bf16, dnnl_ba),
      "weights_desc");
  EmbeddedDnnl::get().check(
      api.memory_desc_create_with_tag(&entry->dst_md, 2, dst_dims, dnnl_bf16, dnnl_ab), "dst_desc");
  EmbeddedDnnl::get().check(
      api.memory_desc_create_with_tag(&entry->gate_md, 2, gate_dims, dnnl_bf16, dnnl_ab), "gate_desc");
  EmbeddedDnnl::get().check(
      api.memory_desc_create_with_tag(&entry->residual_md, 2, residual_dims, dnnl_bf16, dnnl_ab),
      "residual_desc");
  if (has_bias) {
    EmbeddedDnnl::get().check(
        api.memory_desc_create_with_tag(&entry->bias_md, 2, bias_dims, dnnl_bf16, dnnl_ab),
        "bias_desc");
  }
  PostOpsGuard post_ops;
  PrimitiveAttrGuard attr;
  PrimitiveDescGuard pd;
  EmbeddedDnnl::get().check(api.post_ops_create(&post_ops.value), "post_ops_create");
  EmbeddedDnnl::get().check(
      api.post_ops_append_binary(post_ops.value, dnnl_binary_mul, entry->gate_md),
      "append_binary_mul");
  EmbeddedDnnl::get().check(
      api.post_ops_append_binary(post_ops.value, dnnl_binary_add, entry->residual_md),
      "append_binary_add");
  EmbeddedDnnl::get().check(api.primitive_attr_create(&attr.value), "attr_create");
  EmbeddedDnnl::get().check(
      api.primitive_attr_set_post_ops(attr.value, post_ops.value), "attr_set_post_ops");
  EmbeddedDnnl::get().check(
      api.matmul_pd_create(&pd.value, s.engine, entry->src_md, entry->weights_md, entry->bias_md,
                           entry->dst_md, attr.value),
      "matmul_pd_create");
  EmbeddedDnnl::get().check(
      api.primitive_create(&entry->primitive, pd.value), "primitive_create");
  auto *result = entry.get();
  s.cache.emplace(key, std::move(entry));
  return *result;
}

void ensure_engine_and_stream(State &s) {
  const DnnlApi &api = EmbeddedDnnl::get().api();
  if (s.engine == nullptr) {
    EmbeddedDnnl::get().check(api.engine_create(&s.engine, dnnl_gpu, 0), "engine_create");
  }
  sycl::queue &queue = c10::xpu::getCurrentXPUStream().queue();
  TORCH_CHECK(queue.is_in_order(), "r27: the current XPU queue must be in-order");
  if (s.queue != &queue) {
    if (s.stream != nullptr) api.stream_destroy(s.stream);
    s.stream = nullptr;
    EmbeddedDnnl::get().check(api.stream_create(&s.stream, s.engine, &queue), "stream_create");
    s.queue = &queue;
  }
}

void memory_from_tensor(State &s, dnnl_memory_desc_t md, void *pointer, dnnl_memory_t *memory) {
  EmbeddedDnnl::get().check(EmbeddedDnnl::get().api().memory_create(memory, md, s.engine, pointer),
                             "memory_create");
}

torch::Tensor r27_mlp_epilogue(torch::Tensor input, torch::Tensor weight,
                               c10::optional<torch::Tensor> bias_opt, torch::Tensor residual,
                               torch::Tensor gate) {
  TORCH_CHECK(input.is_xpu() && weight.is_xpu() && residual.is_xpu() && gate.is_xpu(),
              "r27: all tensors must be XPU");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16 && weight.scalar_type() == at::kBFloat16 &&
                  residual.scalar_type() == at::kBFloat16 && gate.scalar_type() == at::kBFloat16,
              "r27: all tensors must be bf16");
  TORCH_CHECK(input.dim() >= 2 && weight.dim() == 2, "r27: invalid input/weight rank");
  TORCH_CHECK(input.is_contiguous() && weight.is_contiguous() && residual.is_contiguous() &&
                  gate.is_contiguous(),
              "r27: all tensors must be contiguous");
  const int64_t K = input.size(-1);
  const int64_t N = weight.size(0);
  const int64_t M = input.numel() / K;
  TORCH_CHECK(weight.size(1) == K, "r27: K mismatch");
  TORCH_CHECK(residual.numel() == M * N, "r27: residual shape mismatch");
  TORCH_CHECK(gate.numel() == N, "r27: gate must contain N values");
  torch::Tensor bias_tensor;
  const bool has_bias = bias_opt.has_value();
  if (has_bias) {
    bias_tensor = *bias_opt;
    TORCH_CHECK(bias_tensor.is_xpu() && bias_tensor.scalar_type() == at::kBFloat16 &&
                    bias_tensor.is_contiguous() && bias_tensor.numel() == N,
                "r27: invalid bias");
  }
  auto output_shape = input.sizes().vec();
  output_shape.back() = N;
  torch::Tensor output = at::empty(output_shape, input.options());
  State &s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  ensure_engine_and_stream(s);
  PrimitiveEntry &entry = get_primitive(s, M, N, K, has_bias);
  const DnnlApi &api = EmbeddedDnnl::get().api();
  MemoryGuard src;
  MemoryGuard weights;
  MemoryGuard bias_mem;
  MemoryGuard dst;
  MemoryGuard gate_mem;
  MemoryGuard residual_mem;
  memory_from_tensor(s, entry.src_md, input.data_ptr(), &src.value);
  memory_from_tensor(s, entry.weights_md, weight.data_ptr(), &weights.value);
  memory_from_tensor(s, entry.dst_md, output.data_ptr(), &dst.value);
  memory_from_tensor(s, entry.gate_md, gate.data_ptr(), &gate_mem.value);
  memory_from_tensor(s, entry.residual_md, residual.data_ptr(), &residual_mem.value);
  if (has_bias) memory_from_tensor(s, entry.bias_md, bias_tensor.data_ptr(), &bias_mem.value);
  std::vector<dnnl_exec_arg_t> args;
  args.reserve(6);
  args.push_back({DNNL_ARG_SRC, src.value});
  args.push_back({DNNL_ARG_WEIGHTS, weights.value});
  if (has_bias) args.push_back({DNNL_ARG_BIAS, bias_mem.value});
  args.push_back({DNNL_ARG_DST, dst.value});
  args.push_back({DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, gate_mem.value});
  args.push_back({DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1, residual_mem.value});
  EmbeddedDnnl::get().check(
      api.primitive_execute(entry.primitive, s.stream, static_cast<int>(args.size()), args.data()),
      "primitive_execute");
  return output;
}

}  // namespace

TORCH_LIBRARY(libr27, m) {
  m.def(
      "mlp_epilogue(Tensor input, Tensor weight, Tensor? bias, Tensor residual, Tensor gate) "
      "-> Tensor");
  m.impl("mlp_epilogue", [](torch::Tensor input, torch::Tensor weight,
                            c10::optional<torch::Tensor> bias, torch::Tensor residual,
                            torch::Tensor gate) {
    return r27_mlp_epilogue(input, weight, bias, residual, gate);
  });
}
