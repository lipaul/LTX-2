ICPX ?= /opt/intel/oneapi/compiler/latest/bin/icpx
BUNDLER ?= /opt/intel/oneapi/compiler/latest/bin/compiler/clang-offload-bundler
POSTLINK ?= /opt/intel/oneapi/compiler/latest/bin/compiler/sycl-post-link
TOSPIRV ?= /opt/intel/oneapi/compiler/latest/bin/compiler/llvm-spirv
REPO ?= /home/simon/ltx2.5-demo-main
VENV := $(REPO)/.venv
TORCH_INC := $(VENV)/lib/python3.12/site-packages/torch/include
TORCH_LIB := $(VENV)/lib/python3.12/site-packages/torch/lib
PY_INC := /home/simon/.local/share/uv/python/cpython-3.12.14-linux-x86_64-gnu/include/python3.12
VENV_INC := $(VENV)/include

HOSTFLAGS := -O2 -fPIC -std=c++17 -DSYCL_DISABLE_FSYCL_SYCLHPP_WARNING \
  -I$(TORCH_INC) -I$(TORCH_INC)/torch/csrc/api/include \
  -I$(PY_INC) -I$(VENV_INC) -I.
HOSTLINK := -shared \
  -L$(VENV)/lib -L$(TORCH_LIB) -lsycl -lze_loader \
  -ltorch -ltorch_cpu -ltorch_xpu -ltorch_python -lc10 -lc10_xpu \
  -Wl,-rpath,$(VENV)/lib -Wl,-rpath,$(TORCH_LIB)

all: libr26_vae.so

r26_kernels.spv: r26_device.cpp r26_args.h
	$(ICPX) -fsycl -fsycl-targets=spir64 -O3 -std=c++17 -c r26_device.cpp -o r26_device.o
	$(BUNDLER) -unbundle -input=r26_device.o -output=r26_dev.bc -type=o -targets=sycl-spir64-unknown-unknown
	$(POSTLINK) --ir-output-only --spec-const=emulation -o r26_p.bc r26_dev.bc
	$(TOSPIRV) -spirv-ext=+SPV_INTEL_subgroups -o r26_kernels.spv r26_p.bc

libr26_vae.so: r26_host.cpp r26_args.h r26_kernels.spv
	$(ICPX) $(HOSTFLAGS) r26_host.cpp -o $@ $(HOSTLINK)

clean:
	rm -f libr26_vae.so r26_device.o r26_dev.bc r26_kernels.spv r26_p.bc

.PHONY: all clean
