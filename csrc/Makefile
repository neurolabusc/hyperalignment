UNAME := $(shell uname -s)

CC = cc
CFLAGS = -std=c11 -O2 -Wall -Wextra -Wpedantic -fPIC
LDFLAGS =

ifeq ($(UNAME),Darwin)
	LDFLAGS += -framework Accelerate -framework Metal \
	           -framework MetalPerformanceShaders -framework Foundation
	METAL_SRCS = ha_metal.m
	METAL_OBJS = $(METAL_SRCS:.m=.o)
	SHARED_LIB = libhyperalignment.dylib
	SHARED_FLAGS = -dynamiclib
else
	# MKL on Linux (Anaconda install).  libmkl_rt provides BLAS + LAPACK.
	# mkl_cblas.h and mkl_lapack.h live under CONDA_PREFIX/include.
	# _GNU_SOURCE exposes POSIX extensions (M_PI etc.) under strict -std=c11.
	MKL_PREFIX ?= $(HOME)/anaconda3
	CFLAGS  += -D_GNU_SOURCE -I$(MKL_PREFIX)/include
	LDFLAGS += -L$(MKL_PREFIX)/lib -lmkl_rt -lm -Wl,-rpath,$(MKL_PREFIX)/lib
	METAL_OBJS =
	SHARED_LIB = libhyperalignment.so
	SHARED_FLAGS = -shared
endif

ifdef OPENMP
	ifeq ($(UNAME),Darwin)
		CFLAGS += -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include
		LDFLAGS += -L/opt/homebrew/opt/libomp/lib -lomp
	else
		CFLAGS += -fopenmp
		LDFLAGS += -fopenmp
	endif
endif

# CUDA backend (Linux only, NVIDIA GPU required)
# Usage: make CUDA=1  or  make CUDA=1 OPENMP=1
ifdef CUDA
	ifneq ($(UNAME),Darwin)
		NVCC = nvcc
		# RTX 4090 is compute capability 8.9; sm_80 is also compatible for broader use
		CUDA_ARCH = sm_89
		NVCC_FLAGS = -O2 -arch=$(CUDA_ARCH) --compiler-options "-fPIC" \
		             -DHA_CUDA_ENABLED -D_GNU_SOURCE \
		             -I$(MKL_PREFIX)/include
		# Pass OpenMP flag through nvcc if enabled
		ifdef OPENMP
			NVCC_FLAGS += --compiler-options "-fopenmp"
		endif
		CFLAGS += -DHA_CUDA_ENABLED
		CUDA_LIB_DIR = /usr/lib/x86_64-linux-gnu
		LDFLAGS += -L$(CUDA_LIB_DIR) -lcudart -lcublas -lcusolver \
		           -Wl,-rpath,$(CUDA_LIB_DIR)
		CUDA_SRCS = ha_cuda.cu
		CUDA_OBJS = ha_cuda.o
	endif
endif

CUDA_OBJS ?=

ifdef DEBUG
	CFLAGS = -std=c11 -g -O0 -Wall -Wextra -Wpedantic -fsanitize=address
	ifeq ($(UNAME),Darwin)
		LDFLAGS = -framework Accelerate -framework Metal \
		          -framework MetalPerformanceShaders -framework Foundation \
		          -fsanitize=address
	else
		MKL_PREFIX ?= $(HOME)/anaconda3
		CFLAGS += -I$(MKL_PREFIX)/include
		LDFLAGS = -L$(MKL_PREFIX)/lib -lmkl_rt -lm -fsanitize=address -Wl,-rpath,$(MKL_PREFIX)/lib
	endif
endif

SRCS = ha_linalg.c ha_procrustes.c ha_ridge.c ha_sparse.c \
       ha_searchlight.c ha_template.c ha_ensemble.c
OBJS = $(SRCS:.c=.o) $(METAL_OBJS) $(CUDA_OBJS)
LIB = libhyperalignment.a
TEST = ha_test

all: $(LIB) $(SHARED_LIB)

$(LIB): $(OBJS)
	ar rcs $@ $^

$(SHARED_LIB): $(OBJS)
	$(CC) $(SHARED_FLAGS) $^ $(LDFLAGS) -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.m
	$(CC) -O2 -Wall -Wextra -fobjc-arc -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

test: $(TEST)
	./$(TEST)

$(TEST): ha_test.c $(LIB)
	$(CC) $(CFLAGS) $< -L. -lhyperalignment $(LDFLAGS) -Wl,-rpath,. -o $@

clean:
	rm -f $(OBJS) $(LIB) $(SHARED_LIB) $(TEST)

.PHONY: all test clean
