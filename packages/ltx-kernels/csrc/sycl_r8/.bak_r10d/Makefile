# R8 SYCL kernel library, interop build.
# Device kernels (SYCL) -> SPIR-V; host (plain C++, no -fsycl) launches them on
# torch's own XPU queue via L0 interop. See r8_host.cpp for the ABI rationale.
ICPX ?= /opt/intel/oneapi/compiler/latest/bin/icpx
BUNDLER ?= /opt/intel/oneapi/compiler/latest/bin/compiler/clang-offload-bundler
POSTLINK ?= /opt/intel/oneapi/compiler/latest/bin/compiler/sycl-post-link
TOSPIRV ?= /opt/intel/oneapi/compiler/latest/bin/compiler/llvm-spirv
REPO ?= /home/simon/ltx2.3-demo-main
VENV := $(REPO)/.venv
TORCH_INC := $(VENV)/lib/python3.12/site-packages/torch/include
TORCH_LIB := $(VENV)/lib/python3.12/site-packages/torch/lib
PY_INC := /home/simon/.local/share/uv/python/cpython-3.12.14-linux-x86_64-gnu/include/python3.12
VENV_INC := $(VENV)/include

# Host: plain C++ (NO -fsycl), venv 2025.3 SYCL headers matching libsycl.so.8.
HOSTFLAGS := -O2 -fPIC -std=c++17 -DSYCL_DISABLE_FSYCL_SYCLHPP_WARNING \
  -I$(TORCH_INC) -I$(TORCH_INC)/torch/csrc/api/include \
  -I$(PY_INC) -I$(VENV_INC) -I.
HOSTLINK := -shared \
  -L$(VENV)/lib -L$(TORCH_LIB) -lsycl -lze_loader \
  -ltorch -ltorch_cpu -ltorch_xpu -ltorch_python -lc10 -lc10_xpu \
  -Wl,-rpath,$(VENV)/lib -Wl,-rpath,$(TORCH_LIB)

all: libr8_sycl.so

r8_kernels.spv: r8_device.cpp r8_args.h
	$(ICPX) -fsycl -fsycl-targets=spir64 -O2 -std=c++17 -c r8_device.cpp -o r8_device.o
	$(BUNDLER) -unbundle -input=r8_device.o -output=r8_dev.bc -type=o -targets=sycl-spir64-unknown-unknown
	$(POSTLINK) --ir-output-only --spec-const=emulation -o r8_p.bc r8_dev.bc
	$(TOSPIRV) -o r8_kernels.spv r8_p.bc

libr8_sycl.so: r8_host.cpp r8_args.h r8_kernels.spv
	$(ICPX) $(HOSTFLAGS) r8_host.cpp -o $@ $(HOSTLINK)

clean:
	rm -f libr8_sycl.so r8_device.o r8_dev.bc r8_dev.spv r8_dev.table r8_dev_0.ll
	rm -f r8_p.bc r8_kernels.spv r8_back.ll r8_back.err r8b.table r8b_0.ll r8c.table r8c_0.ll

.PHONY: all clean
