# SYCL kernel library for the LTX XPU build (L0-interop).
#
# Device kernels are compiled from device_kernels.cpp into SPIR-V
# (kernels.spv).  The host side in host_ops.cpp is plain C++ without -fsycl and
# launches those kernels on torch's own XPU queue; see host_ops.cpp for the ABI
# rationale.  Kernel argument structs are shared through kernel_args.h.
ICPX      ?= $(shell command -v icpx 2>/dev/null || echo /opt/intel/oneapi/compiler/2026.1/bin/icpx)
BUNDLER   ?= $(shell command -v clang-offload-bundler 2>/dev/null || echo /opt/intel/oneapi/compiler/2026.1/bin/compiler/clang-offload-bundler)
POSTLINK  ?= $(shell command -v sycl-post-link 2>/dev/null || echo /opt/intel/oneapi/compiler/2026.1/bin/compiler/sycl-post-link)
TOSPIRV   ?= $(shell command -v llvm-spirv 2>/dev/null || echo /opt/intel/oneapi/compiler/2026.1/bin/compiler/llvm-spirv)

# 路径全部从本文件位置推导，不再烧死某台机器的绝对路径。
# 之前 REPO / PY_INC 写死 /home/simon/...，换机器后 make 会因为产物时间戳说
# "Nothing to be done"，但 RUNPATH 仍指向那个不存在的 venv，
# torch.ops.load_library 随后失败 —— 一个很隐蔽的坑。
#
#   csrc/sycl_kernels -> csrc -> ltx-kernels -> packages -> LTX-2
SYCL_DIR  := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
LTX2_ROOT := $(abspath $(SYCL_DIR)/../../../..)

# venv 默认在 LTX-2 的同级（ltx2.5-demo/.venv）；换布局就用 VENV=... 覆盖。
VENV      ?= $(LTX2_ROOT)/../.venv
VENV_PY   := $(VENV)/bin/python
TORCH_INC := $(VENV)/lib/python3.12/site-packages/torch/include
TORCH_LIB := $(VENV)/lib/python3.12/site-packages/torch/lib
VENV_INC  := $(VENV)/include

# uv 建的 venv 里没有 include/python3.12（头文件在 uv 托管的基础解释器下），
# 所以 PY_INC 必须问 venv 自己的 python 要，不能猜路径。
PY_INC    ?= $(shell $(VENV_PY) -c 'import sysconfig;print(sysconfig.get_paths()["include"])' 2>/dev/null)

# Host: plain C++ (no -fsycl).  torch headers need C++20; the link target is the
# libsycl.so.9 shipped in the venv (oneAPI 2026.1).
HOSTFLAGS := -O2 -fPIC -std=c++20 -DSYCL_DISABLE_FSYCL_SYCLHPP_WARNING \
  -I$(TORCH_INC) -I$(TORCH_INC)/torch/csrc/api/include \
  -I$(PY_INC) -I$(VENV_INC) -I.
HOSTLINK := -shared \
  -L$(VENV)/lib -L$(TORCH_LIB) -lsycl -lze_loader \
  -ltorch -ltorch_cpu -ltorch_xpu -ltorch_python -lc10 -lc10_xpu \
  -Wl,-rpath,$(VENV)/lib -Wl,-rpath,$(TORCH_LIB)

all: libltx_sycl.so

kernels.spv: device_kernels.cpp kernel_args.h
	$(ICPX) -fsycl -fsycl-targets=spir64 -O2 -std=c++17 -c device_kernels.cpp -o device_kernels.o
	$(BUNDLER) -unbundle -input=device_kernels.o -output=device.bc -type=o -targets=sycl-spir64-unknown-unknown
	$(POSTLINK) --ir-output-only --spec-const=emulation -o postlink.bc device.bc
	$(TOSPIRV) -o kernels.spv postlink.bc

libltx_sycl.so: host_ops.cpp kernel_args.h kernels.spv
	$(ICPX) $(HOSTFLAGS) host_ops.cpp -o $@ $(HOSTLINK)

clean:
	rm -f libltx_sycl.so kernels.spv device_kernels.o device.bc postlink.bc

.PHONY: all clean
