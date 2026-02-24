# CLAUDE.md

## Project Overview

Hyperalignment library: aligns fMRI brain data across subjects by finding optimal transformations that maximize representational similarity. The C implementation in `csrc/` supports three compute backends: CPU FP64, CPU FP32, and Metal GPU FP32 (macOS). A Python ctypes wrapper (`hyperalignment_c.py`) exposes the C library for benchmarking against the Python reference.

The Python code in `src/hyperalignment/` is the reference implementation (also installed as a package). The C port lives in `csrc/`. The `dcm/` directory contains reference C code for style conventions and build patterns but is not part of the hyperalignment library.

## Repository Layout

```
src/hyperalignment/       Python reference implementation
    linalg.py             safe_svd (LAPACK driver fallback), svd_pca
    procrustes.py         Orthogonal Procrustes: minimize ||XT - Y||_F
    ridge.py              Ridge regression via SVD, grid search, ensemble CV
    sparse.py             Sparse matrix init for searchlight sparsity patterns
    searchlight.py        Searchlight alignment loop, weight computation
    local_template.py     Template construction (Procrustes, GPA, PCA variants)
    ensemble.py           Temporal cross-validation, ensemble searchlight
tests/                    Python tests (pytest)
csrc/                     C library (CPU FP64/FP32 + Metal GPU FP32)
hyperalignment_c.py       Python ctypes wrapper for C shared library
benchmark_neuroboros.py   Benchmark on Forrest dataset (--backend python|c64|c32|metal)
dcm/                      Reference C code (style guide, not part of library)
```

## C Coding Conventions

Follow the style in `dcm/` (dcm2niix codebase):

### Naming
- Functions: `snake_case` with module prefix (e.g., `ha_procrustes`, `ha_svd`, `mat_mul`)
- Constants: `k` prefix, camelCase (e.g., `kMaxSearchlightSize`, `kDefaultAlpha`)
- Types: `T` prefix for typedef'd structs (e.g., `TMat`, `TVec`, `TSparseCSC`)
- Booleans: `is` prefix (e.g., `isReflection`, `isScaling`)
- Loop variables: `i`, `j`, `k` for indices

### Formatting
- Tabs for indentation
- K&R brace style (opening brace on same line)
- C++ style comments (`//`) for inline, `/* */` for block/file headers
- No line length limit, but break long expressions at operators

### Types
- `int32_t`, `int64_t`, `size_t` from `<stdint.h>` for sizes and indices
- `double` for all FP64 computation (matching Python float64)
- `float` for FP32 compute paths (CPU FP32, Metal GPU) — FP32 types use `TMatF` and `*_f32` function suffixes
- `bool` from `<stdbool.h>`
- `ha_lapack_int` for LAPACK integer arguments (`__LAPACK_int` on macOS, `int` on Linux)

### Memory
- `malloc`/`free` exclusively (no C++ `new`/`delete`)
- Always check allocation: `if (!ptr) { ... return NULL; }`
- Caller owns returned pointers unless documented otherwise
- Large buffers always heap-allocated (brain data can be hundreds of MB)

### Error Handling
- Return `NULL` for pointer-returning functions on failure
- Return `int` error codes: `kHaSuccess` (0), `kHaErrorAlloc` (-1), `kHaErrorSvd` (-2), `kHaErrorArg` (-3), `kHaErrorInternal` (-4)
- Print errors to stderr via `fprintf(stderr, ...)`
- No `exit()` calls in library code; only in CLI main

## Core Data Structures

```c
// Dense matrix — FP64 (row-major, data[i * cols + j])
typedef struct {
    double *data;
    int32_t rows, cols;
} TMat;

// Dense matrix — FP32 (for CPU FP32 and Metal backends)
typedef struct {
    float *data;
    int32_t rows, cols;
} TMatF;

// Sparse CSC matrix (matches scipy.sparse.csc_matrix)
typedef struct {
    double *data;
    int32_t *indices;   // row indices
    int32_t *indptr;    // column pointers (length cols+1)
    int32_t rows, cols;
    int64_t nnz;
} TSparseCSC;

// Searchlight definition (pointer-of-pointers, used by sparse API)
typedef struct {
    int32_t **indices;
    double **dists;     // NULL for uniform weighting
    int32_t *sizes;
    int32_t count;
    double radius;
} TSearchlights;

// Backend selection (runtime, no recompilation)
typedef enum {
    kHaBackendCPU64 = 0,   // CPU FP64 (default)
    kHaBackendCPU32 = 1,   // CPU FP32
    kHaBackendMetal = 2    // Metal GPU FP32 (macOS only)
} THaBackend;
```

Precision conversion helpers: `ha_mat_to_float()` (TMat->TMatF) and `ha_matf_to_double()` (TMatF->TMat) in `ha_common.h`.

## Algorithm-to-Function Mapping

| Python | C Function | Core Operation |
|--------|-----------|---------------|
| `safe_svd(X)` | `ha_svd()` / `ha_svd_f32()` | LAPACK `dgesdd`/`sgesdd`, fallback `dgesvd`/`sgesvd` |
| `svd_pca(X)` | `ha_pca()` | SVD then `U * s` |
| `procrustes(X, Y)` | `ha_procrustes()` / `ha_procrustes_f32()` | `A = X'Y`, SVD of A, `T = U @ Vt` |
| — | `ha_polar_newton()` | FP32 polar decomposition via Newton iteration (Metal path) |
| `ridge(X, Y, alpha)` | `ha_ridge()` | SVD of X, damped solve |
| `ridge_grid(...)` | `ha_ridge_grid()` | Vectorized over alpha/npc grid |
| `compute_searchlight_weights(sls)` | `ha_searchlight_weights()` | Distance-based weighting |
| `searchlight_procrustes(...)` | **`ha_searchlight_procrustes_dense()`** | **Primary entry point**: flat arrays, dense output, OMP atomic scatter-add |
| `searchlight_procrustes(...)` | `ha_searchlight_procrustes()` | Legacy sparse API: TSearchlights input, sparse CSC output |
| `searchlight_ridge(...)` | `ha_searchlight_ridge()` | Loop: local ridge + sparse accumulate |
| `compute_template(dss)` | `ha_template()` | Iterative Procrustes/GPA/PCA |
| `compute_ensemble_indices(nt)` | `ha_ensemble_indices()` | Block permutation CV splits |

## Build System

Uses Make with platform detection. Links against Accelerate on macOS, system LAPACK/BLAS on Linux.

```
csrc/
    Makefile              Platform-detecting build system (static + shared library)
    hyperalignment.h      Umbrella header (includes all modules + ha_metal.h)
    ha_common.h           Types, alloc helpers, extract/scatter utilities
    ha_linalg.h/c         SVD FP64/FP32, PCA, z-score
    ha_procrustes.h/c     Procrustes FP64/FP32 + polar Newton (CPU)
    ha_ridge.h/c          Ridge regression, grid search, ensemble ridge
    ha_sparse.h/c         CSC sparse matrix init, scatter-add (legacy path)
    ha_searchlight.h/c    Searchlight weights, dense+sparse alignment loops, workspace structs
    ha_template.h/c       Template construction (Procrustes, GPA, PCA)
    ha_ensemble.h/c       Cross-validation index generation
    ha_metal.h            Metal API header (inline stubs on non-Apple)
    ha_metal.m            Metal/MPS implementation (macOS only, Objective-C)
    ha_test.c             Test harness (2621 tests)
    bench_*.c             Microbenchmarks (SVD, pipeline, extraction, column-major)
```

### Build commands
```bash
cd csrc
make                       # Release build -> libhyperalignment.a + .dylib/.so
make test                  # Build and run tests
make DEBUG=1 test          # Build with AddressSanitizer and run tests
make OPENMP=1              # Build with OpenMP searchlight parallelism
make OPENMP=1 test         # Build with OpenMP and run tests
make CUDA=1 OPENMP=1       # Build with CUDA + OpenMP (Linux, requires NVIDIA GPU)
make CUDA=1 OPENMP=1 test  # Build with CUDA + OpenMP and run tests
make clean
```

**Linux prerequisites**: Anaconda with `mkl-devel` (`conda install mkl-devel`). CUDA libraries at `/usr/lib/x86_64-linux-gnu` (standard Ubuntu CUDA package). OpenMP via `gcc -fopenmp` (no extra package needed). Set `MKL_NUM_THREADS=1` at runtime to prevent MKL from spawning internal BLAS threads that conflict with OpenMP.

**macOS prerequisites**: Xcode Command Line Tools, Homebrew `libomp`. Metal backend requires macOS and Apple Silicon.

### Compiler flags
- `-std=c11 -O2 -Wall -Wextra -Wpedantic -fPIC` for release
- `-std=c11 -g -O0 -fsanitize=address` for debug
- `-framework Accelerate -framework Metal -framework MetalPerformanceShaders -framework Foundation` on macOS
- `-lmkl_rt -lm -L$(HOME)/anaconda3/lib` on Linux (Anaconda MKL)
- OpenMP opt-in via `OPENMP=1` flag
  - macOS: `-Xpreprocessor -fopenmp` + homebrew libomp
  - Linux: `-fopenmp`
- CUDA opt-in via `CUDA=1` (Linux only): links `-lcudart -lcublas -lcusolver` from `/usr/lib/x86_64-linux-gnu`

## LAPACK/BLAS Functions Used

FP64: `dgesdd_`/`dgesvd_` (SVD), `cblas_dgemm` (GEMM), `dgemv_` (matvec), `dnrm2_`, `cblas_dscal`, `dcopy_`, `dgetrf_` (LU, for det sign check), `cblas_dger` (rank-1 update, for reflection correction)
FP32: `sgesdd_`/`sgesvd_` (SVD), `cblas_sgemm` (GEMM), `sgetrf_`/`sgetri_` (LU inverse, for Metal Newton and det sign), `cblas_sger`, `cblas_sscal`

Both `CblasRowMajor` and `CblasColMajor` layouts are used — see "Column-Major Optimization" below.

### Row-major SVD trick
Row-major `X(M,N)` is column-major `X^T(N,M)` to LAPACK. We call `dgesdd("S", N, M, ...)` and swap the U/Vt buffer assignments: LAPACK's "U" output becomes our Vt, LAPACK's "Vt" output becomes our U. No explicit transpose needed. Used by `procrustes_ws_fp64()` and `procrustes_ws_fp32()`.

### Column-major (native LAPACK) path
When `col_major=true`, data is already in LAPACK's native column-major layout. No U/Vt swap trick needed — pass `dgesdd("S", N, N, ...)` with standard U/Vt ordering. CBLAS calls use `CblasColMajor`. Used by `procrustes_ws_fp64_cm()` and `procrustes_ws_fp32_cm()`. This is the primary path used by the Python wrapper.

## Primary C Entry Point: `ha_searchlight_procrustes_dense`

This is the optimized entry point used by the Python wrapper. It replaces the older sparse API with a single-call dense-output design.

```c
int ha_searchlight_procrustes_dense(
    const double *X_data, const double *Y_data,    // nt x nv matrices
    int32_t nt, int32_t nv,
    const int32_t *sl_indices,   // flat concatenated vertex indices
    const int32_t *sl_offsets,   // offsets into sl_indices, length count+1
    const double *sl_dists,      // flat concatenated distances (NULL = uniform)
    int32_t count,               // number of searchlights
    double radius,
    double *T_out,               // pre-allocated nv x nv, caller zero-inits
    bool isReflection, bool isScaling,
    THaBackend backend,
    bool col_major);             // true = Fortran order input (preferred)
```

### Design decisions
- **Flat arrays** (`sl_indices` + `sl_offsets`) instead of pointer-of-pointers (`TSearchlights`). Built via `np.concatenate` + `np.cumsum` in Python — no per-searchlight Python loop.
- **Dense output** (`T_out` nv x nv) instead of sparse CSC. Scatter-add uses `#pragma omp atomic` per element — lock-free, no contention at typical sparsity levels.
- **Column-major input** (`col_major=true`): Python data from `np.concatenate` is naturally F-contiguous. Column-major extraction is contiguous `memcpy` per column vs scattered reads in row-major. This alone gives a **2x speedup** (see Performance Notes).
- **Internal weight computation**: Two-pass over flat arrays to compute normalized weights. No per-searchlight malloc.
- **Per-thread workspace**: `WorkspaceFP64`/`WorkspaceFP32` structs pre-allocate all SVD/Procrustes scratch buffers once per thread at max searchlight size. Eliminates ~310K malloc/free pairs per hemisphere.

## Per-Thread Workspace Pattern

Defined in `ha_searchlight.c`. Each thread gets a pre-allocated workspace sized for the largest searchlight:

```c
typedef struct {
    double *local_X;         // nt x max_sz
    double *local_Y;         // nt x max_sz
    double *T;               // max_sz x max_sz
    double *A;               // max_sz x max_sz (cross-correlation, destroyed by SVD)
    double *U, *s, *Vt;     // SVD outputs
    double *svd_backup;      // max_sz x max_sz (dgesdd fallback copy)
    double *svd_work;        // LAPACK workspace (queried at alloc time)
    ha_lapack_int *svd_iwork; // 8 * max_sz
    ha_lapack_int svd_lwork;
    double *det_tmp;         // max_sz x max_sz (LU for reflection check)
    ha_lapack_int *det_ipiv; // max_sz
} WorkspaceFP64;
```

Allocated once per thread before the OMP parallel loop, freed after. LAPACK workspace size is queried via `dgesdd` with `lwork=-1` at allocation time to get the optimal size. `WorkspaceFP32` is the float equivalent.

Inline Procrustes functions (`procrustes_ws_fp64`, `procrustes_ws_fp64_cm`, `procrustes_ws_fp32`, `procrustes_ws_fp32_cm`) operate directly on workspace buffers — no TMat structs, no malloc, no free in the hot loop.

## Actual Searchlight Dimensions (Forrest Dataset)

These are the real numbers from the benchmark dataset (StudyForrest, radius=20mm, `neuroboros` package):

| Property | Value |
|----------|-------|
| Vertices per hemisphere | ~9,675 |
| Searchlights per hemisphere | ~9,675 (one per vertex) |
| Timepoints (4 training runs) | 1,818 |
| Median searchlight size | 121 vertices |
| Max searchlight size | ~170 vertices |
| Local matrices | ~121 x 121 (median) |
| Dense output matrix | 9,675 x 9,675 = ~749 MB |
| Searchlight indices | Highly scattered: mean gap ~80, span ~9,500/9,675, only 1.2% consecutive |

Previous CLAUDE.md references to "~200x200 matrices" or "~19K searchlights" were incorrect. The per-searchlight SVD is on ~121x121 matrices.

## Performance: Current State of the Art

### Benchmark Results (Apple M4 Pro, Forrest Dataset, Median of 3)

Hardware: Apple M4 Pro (10 performance + 4 efficiency cores), 48 GB unified memory, macOS.

| Backend | Threads | L hemi (s) | R hemi (s) | Total (s) | vs Python |
|---------|---------|-----------|-----------|-----------|-----------|
| Python (numpy/Accelerate) | 1 | 15.1 | 15.0 | 30.1 | 1.0x |
| Python (numpy/Accelerate) | 10 | 15.4 | 15.4 | 30.8 | 0.98x |
| C CPU FP64 | 1 | 14.4 | 14.6 | 28.9 | 1.04x |
| **C CPU FP64** | **10** | **2.2** | **2.2** | **4.3** | **7.0x** |
| C CPU FP32 | 1 | 9.3 | 9.2 | 18.5 | 1.6x |
| **C CPU FP32** | **10** | **1.25** | **1.24** | **2.5** | **12.0x** |
| C Metal GPU FP32 | 10 | 8.3 | 8.4 | 16.8 | 1.8x |

All backends produce numerically identical percentile distributions at 4 decimal places. FP32 backends match FP64 because weight normalization and scatter-add accumulation are done in FP64 regardless of local Procrustes precision.

Python "Threads" column: 1 = default, 10 = `VECLIB_MAXIMUM_THREADS=10`. No measurable effect because Accelerate internal threading provides negligible benefit for ~121x121 per-searchlight SVDs.

### OpenMP Scaling (M4 Pro, 10 performance cores)

| Backend | 1 thread | 10 threads | Speedup |
|---------|----------|------------|---------|
| C CPU FP64 | 28.9s | 4.3s | 6.7x |
| C CPU FP32 | 18.5s | 2.5s | 7.4x |

Sub-linear scaling (6.7-7.4x on 10 cores) due to memory bandwidth contention and Accelerate-internal threading within individual LAPACK calls.

### Benchmark Results (AMD Threadripper 7995WX + RTX 4090, Forrest Dataset, Median of 3)

Hardware: AMD Threadripper 7995WX (96 cores / 192 threads), NVIDIA RTX 4090 (24 GB GDDR6X), Ubuntu 24.04. MKL_NUM_THREADS=1 to suppress BLAS internal threading.

| Backend | Threads | L hemi (s) | R hemi (s) | Total (s) | vs Python |
|---------|---------|-----------|-----------|-----------|-----------|
| Python (numpy/MKL) | 1 | 59.8 | 59.5 | 119.5 | 1.0x |
| C CPU FP64 | 1 | 25.6 | 25.6 | 51.2 | 2.3x |
| C CPU FP32 | 1 | 17.2 | 17.1 | 34.3 | 3.5x |
| C CPU FP64 | 32 | 1.6 | 1.7 | 3.3 | 36x |
| C CPU FP32 | 32 | 1.1 | 1.1 | 2.3 | 52x |
| **C CPU FP64** | **96** | **1.09** | **1.05** | **2.1** | **57x** |
| **C CPU FP32** | **96** | **0.64** | **0.59** | **1.2** | **100x** |
| C CUDA GPU FP32 | — | 21.1 | 21.1 | 42.3 | 2.8x |

All backends produce numerically identical percentile distributions. Python is 4x slower on Threadripper than M4 Pro (single-threaded MKL vs Accelerate with AMX).

### OpenMP Scaling (Threadripper 7995WX, 96 physical cores)

| Backend | 1 thread | 32 threads | 96 threads | 32→96 ratio |
|---------|----------|------------|------------|------------|
| C CPU FP64 | 51.2s | 3.3s | 2.1s | 0.64x |
| C CPU FP32 | 34.3s | 2.3s | 1.2s | 0.52x |

Scaling from 32→96 threads is ~0.5-0.6x (not linear) due to NUMA topology. The Threadripper 7995WX has 4 NUMA nodes; threads on remote nodes pay higher memory latency. Setting `OMP_PROC_BIND=close` may improve this. C CPU FP32 at 96 threads achieves **100x speedup** vs Python and **1.2s total** for both hemispheres.

### CUDA Backend Analysis

The RTX 4090 CUDA backend (**42.3s**) is significantly slower than CPU multi-threading (**1.2s at 96 threads FP32**). This is expected and has a known explanation:

**Root cause**: `cusolverDnSgesvdaStridedBatched` — our only option for batched SVD of 121×121 matrices — computes an approximate SVD using iterative polar decomposition (Jacobi sweeps). For matrices this small, the iterative algorithm requires many sweeps to converge, and the 256-batch GPU execution doesn't provide enough parallelism to amortize the overhead. The effective GPU utilization during SVD is low.

**Why not use other cuSOLVER routines**:
- `cusolverDnSgesvdjBatched`: hard limit of 32×32 — not usable at 121×121
- `cusolverDnSgesvdj` (non-batched): no size limit but effectively serializes 9,675 SVDs
- `cusolverDnSgesvd` (QR-based): synchronous, no batching

**What would help** (future work):
- Custom one-sided Jacobi SVD kernel: true parallel batched SVD at any size, GPU thread per matrix element
- Hybrid: GPU GEMM for `A = X^T @ Y`, download A, CPU SVD (MKL 96 threads), upload U/V, GPU GEMM for `T = U @ V^T`. The CPU SVD bottleneck is ~18s FP32 single-threaded / ~0.2s at 96 threads — fast enough to be negligible.
- For `gesvdaStridedBatched`: the "approximate" in the name means it uses `rank=k` truncated SVD. We need `k=max_sz` (full rank) which is most expensive.

### Key Optimizations (in order of impact)

#### 1. Column-Major Input (2x single-threaded speedup)

**The single most impactful optimization.** The Python `hyperalignment` package produces Fortran-contiguous (column-major) data from `np.concatenate`. The C wrapper calls `np.asfortranarray()` and passes `col_major=true`.

Why it matters: searchlight indices are highly scattered (mean gap ~80 across ~9,675 vertices). With row-major data, extracting columns for one searchlight requires ~121 x 1,818 = ~220K scattered reads across 75KB-wide rows, thrashing the CPU cache. With column-major data, each column is contiguous in memory — extraction is `memcpy` per column.

Measured impact: single-threaded C went from ~54s (row-major) to ~29s (column-major), matching Python's ~30s.

Column-major extraction:
```c
// Col-major: column j is contiguous at X_data[sl[j]*nt], length nt
for (int32_t j = 0; j < sz; j++) {
    memcpy(&ws->local_X[j * nt], &X_data[(size_t)sl[j] * nt],
           (size_t)nt * sizeof(double));
}
```

#### 2. Dense Output with OMP Atomic (~2x multi-threaded speedup)

Replaced sparse CSC output (`ha_sparse_init` qsort + binary-search scatter-add + `#pragma omp critical`) with dense output (direct indexed scatter-add + `#pragma omp atomic`). The sparse infrastructure was the dominant bottleneck at high thread counts due to the global critical section serializing all scatter-adds.

```c
for (int32_t i = 0; i < sz; i++) {
    for (int32_t j = 0; j < sz; j++) {
        double val = ws->T[j * sz + i] * w[i];  // col-major T
        #pragma omp atomic
        T_out[sl[i] * nv + sl[j]] += val;
    }
}
```

#### 3. Per-Thread Workspace Pre-allocation

Eliminated ~310K malloc/free pairs per hemisphere by pre-allocating all SVD/Procrustes scratch buffers per thread. Each `WorkspaceFP64`/`WorkspaceFP32` is allocated once at the max searchlight size before the OMP parallel loop. LAPACK workspace size is queried via `lwork=-1` at allocation time.

Note: a previous attempt at workspace pre-allocation was reverted because it was tested without the column-major optimization. With column-major input, the workspace approach is strictly beneficial since the allocation savings are on top of the already-fast extraction.

#### 4. Flat-Array API (eliminates Python loop)

The dense entry point takes flat concatenated arrays (`sl_indices` + `sl_offsets`) instead of pointer-of-pointers. Python builds these with vectorized numpy ops:
```python
sizes = np.array([len(s) for s in sls], dtype=np.int32)
offsets = np.zeros(len(sls) + 1, dtype=np.int32)
np.cumsum(sizes, out=offsets[1:])
all_indices = np.concatenate(sls).astype(np.int32)
```
This replaces a 9,675-iteration Python loop with per-element ctypes calls.

## Metal GPU Backend (macOS, Completed)

The Metal backend (`ha_metal.m`) computes Procrustes alignment via **polar decomposition using Newton iteration**, avoiding SVD entirely:

```
X_0 = X^T @ Y                          (GPU GEMM via MPSMatrixMultiplication)
X_{k+1} = (X_k + X_k^{-T}) / 2        (CPU LAPACK inverse + transpose, ~6-10 iterations)
T = X_converged                          (orthogonal polar factor)
```

**Architecture**: GPU GEMM for the initial `X^T @ Y` multiplication, CPU LAPACK (`sgetrf_`/`sgetri_`) for Newton iteration inverse (~121x121 matrices where GPU dispatch overhead dominates). Falls back to `ha_procrustes_f32` (CPU FP32 SVD) if singular.

**Implementation detail**: `matrix_inverse_cpu` passes row-major data to column-major LAPACK, so it returns `A^{-1}` (not `A^{-T}`). An explicit `transpose_inplace()` is needed after the inverse to get `A^{-T}` for Newton iteration correctness.

**Batching and pipelining** (implemented): 256 searchlights per GPU command buffer (`METAL_BATCH_DENSE`). While the GPU computes the next batch of `X^T @ Y` products, the CPU runs Newton iterations for the previous batch. OpenMP parallelizes the CPU Newton stage.

**Performance**: 16.8s total (1.8x vs Python). Slower than CPU because ~121x121 matrices are too small for GPU dispatch overhead to be fully amortized, and the iterative Newton approach requires multiple LU factorizations per searchlight.

### Apple Accelerate Does NOT Use the GPU
Accelerate's BLAS/LAPACK runs exclusively on the CPU via Apple Silicon's AMX coprocessor. GPU acceleration requires Metal Performance Shaders (MPS).

### MPS Capabilities
MPS provides GPU-accelerated GEMM (`MPSMatrixMultiplication`), Cholesky, LU (`MPSMatrixDecompositionLU`), and triangular solve (`MPSMatrixSolveTriangular`). MPS does **NOT** provide SVD — this is why we use polar decomposition for Procrustes and why ridge regression has no Metal backend.

## CUDA Path (Linux/NVIDIA) — Research Notes

Target hardware for upcoming Linux testing: AMD Threadripper (96 cores / 192 threads) + NVIDIA RTX 4090.

### RTX 4090 Specs
- 16,384 CUDA cores, 24 GB GDDR6X
- ~82.6 TFLOPS FP32, **~1.3 TFLOPS FP64** (1:64 ratio — consumer GPUs heavily deprioritize FP64)
- PCIe 4.0 x16 (~25 GB/s bidirectional)
- Compute capability 8.9 (Ada Lovelace)

FP64 is extremely slow on consumer GPUs. The FP32 path (local Procrustes in FP32, accumulation in FP64 on CPU) is mandatory.

### Our Workload on CUDA
~9,675 independent searchlight problems per hemisphere, each: extract ~121 columns from 1818 x 9675 matrix, compute 121x121 SVD + GEMM, scatter-add into 9675x9675 dense output.

### cuSOLVER SVD — The Critical Constraint

1. **`cusolverDnSgesvdjBatched`** — Batched Jacobi SVD. **Hard size limit: m <= 32 and n <= 32.** Our searchlights are ~121x121, so **NOT usable**.

2. **`cusolverDnSgesvdj`** — Non-batched Jacobi SVD. No size limit, but **effectively synchronous** per stream. Cannot trivially parallelize 9,675 SVDs using streams.

3. **`cusolverDnSgesvdaStridedBatched`** — Approximate SVD (polar decomposition-based), strided batched. No documented size limit. Computes top-k singular values/vectors. For Procrustes we need all singular vectors (T = U @ Vt), so k must equal matrix dimension — needs accuracy validation at k=121.

4. **`cusolverDnSgesvd`** — QR-based SVD. No size limit but slower than Jacobi and synchronous.

**Bottom line**: cuSOLVER does NOT provide a drop-in batched SVD for 121x121 matrices. Viable strategies:
- **`gesvdaStridedBatched`** with k=121 — test accuracy first
- **Custom one-sided Jacobi SVD kernel** — true batched parallelism, any size
- **CPU-only with Threadripper**: 96 cores may be fast enough that GPU SVD is unnecessary. At M4 Pro's 7.4x scaling on 10 cores, 96 cores could yield ~50-70x if scaling holds, putting total time under 0.5s. Memory bandwidth is the likely limiter.

### cuBLAS Batched GEMM — Fully Capable

`cublasSgemmStridedBatched` has no size limits and works well for our ~121x121 matrices. Covers:
- `A = X^T @ Y` (cross-correlation)
- `T = U @ Vt` (transformation assembly)

### Dense Scatter-Add on GPU

Our dense output pattern (accumulate 121x121 local T blocks into overlapping regions of 9675x9675 matrix) requires atomic operations. CUDA `atomicAdd` on `double` is supported since compute capability 6.0 (RTX 4090 is 8.9). A simple custom kernel can parallelize this.

### Discrete GPU: PCIe Transfer Strategy

RTX 4090 is a discrete GPU — data must be explicitly copied:
- Subject data: 1818 timepoints x 9675 vertices x 8 bytes = ~141 MB per subject
- Output: 9675 x 9675 x 8 bytes = ~749 MB
- PCIe 4.0: ~25 GB/s — transfer overhead is ~36ms for subject data, negligible vs compute

**Key insight**: Transfer subject data + searchlight indices to GPU once, do ALL computation on GPU, transfer result back once. No per-searchlight transfers. This requires solving the batched SVD problem on GPU, OR doing CPU-only with Threadripper.

### Threadripper Strategy (96 cores / 192 threads)

The existing OpenMP code should work directly with OpenBLAS on Linux. Key considerations:
- **NUMA topology**: Threadrippers are typically 2-socket or multi-die. Thread pinning (`OMP_PROC_BIND=close`) may be important.
- **Memory bandwidth**: Each searchlight workspace is ~121x121 x 8 x ~12 buffers = ~1.4 MB per thread. 192 threads x 1.4 MB = ~269 MB total workspace — fits in L3 but may pressure bandwidth.
- **OpenBLAS threading**: Set `OPENBLAS_NUM_THREADS=1` to avoid nested parallelism (OpenMP outer loop + BLAS inner threads). This is critical — same issue as `VECLIB_MAXIMUM_THREADS` on macOS.
- **Scaling prediction**: M4 Pro gets 7.4x on 10 cores (FP32). 96 cores could yield ~50-70x if memory bandwidth allows. Sub-linear scaling is expected.

### Recommended CUDA Strategy for RTX 4090

1. **Try CPU-only first**: Threadripper 96 cores with OpenMP may be fast enough. Build with `make OPENMP=1`, run with `OMP_NUM_THREADS=96`. If total time is under ~1s, GPU acceleration has diminishing returns.

2. **If GPU needed**: Use FP32 path. Pre-extract all searchlight submatrices into contiguous batched buffer on GPU. Try `cusolverDnSgesvdaStridedBatched` first (easiest, test accuracy). If insufficient, write custom Jacobi SVD kernel. Use `cublasSgemmStridedBatched` for GEMMs. Custom kernel for dense scatter-add with `atomicAdd`.

3. **Hybrid as last resort**: SVD on CPU (96 Threadripper cores), GEMM on GPU. Requires PCIe round-trips for intermediate data — only viable if SVD dominates and GPU GEMM savings outweigh transfer cost.

### GPU SVD Research References

- [NVIDIA gesvdjBatched 32x32 limit](https://forums.developer.nvidia.com/t/the-origin-of-the-m-32-and-n-32-limitations-in-gesvdjbatched/266434)
- [cuSOLVER SVD not overlapping on streams](https://forums.developer.nvidia.com/t/cusolver-svd-not-overlapping-using-streams/306782)
- [GTC 2019: Fast SVD on GPUs](https://developer.nvidia.com/gtc/2019/video/s9226)
- [cuSOLVER documentation](https://docs.nvidia.com/cuda/cusolver/index.html)
- [cuBLAS Strided Batched GEMM](https://developer.nvidia.com/blog/cublas-strided-batched-matrix-multiply/)
- [Ringoot et al. (2025)](https://arxiv.org/abs/2508.06339) — portable GPU SVD on Metal (Julia), singular values only, poor for <256x256
- [philipturner Metal GEMM kernel](https://gist.github.com/philipturner/84f613a5cc745460a914d2c6ad226131) — FP32/FP16/BF16 reference

## Python ctypes Wrapper (`hyperalignment_c.py`)

Loads `csrc/libhyperalignment.dylib` (or `.so`) via ctypes. Main function:
```python
searchlight_procrustes(X, Y, sls, dists, radius, backend='cpu64',
                       reflection=True, scaling=False, n_jobs=1)
# Returns dense numpy ndarray (nv x nv)
```

Uses `ha_searchlight_procrustes_dense` — a single C call that takes flat numpy arrays and writes into a caller-provided dense output matrix. Weights are computed internally. No sparse structure, no pointer-of-pointers, no per-searchlight memory management.

Input is converted to Fortran order (`np.asfortranarray`) so C can extract searchlight columns via contiguous memcpy. Output is zero-copy — numpy owns the buffer, C writes directly into it.

**Installed vs source API difference**: The installed `hyperalignment` package has `searchlight_procrustes(X, Y, sls, dists, radius, ...)` while the source in `src/` has a different signature `(X, Y, sls, sls_Y=None, mat0=None, ...)`. The benchmark uses the installed API.

## Known Issues in Python Code

- `ensemble.py:77,81`: References undefined variables `y` and `T` (should be `Y` and `xmat`)
- `local_template.py:61`: Calls `safe_svd(X, demean=demean)` but the parameter name is `remove_mean`
- `test_searchlight.py` depends on `neuroboros` package (not in requirements.txt)

## Testing

```bash
# Python tests
pytest tests/

# C tests (2621 deterministic tests covering all modules and all three backends)
make -C csrc test
make -C csrc DEBUG=1 test       # with AddressSanitizer
make -C csrc OPENMP=1 test      # with OpenMP

# Real-data benchmark (requires neuroboros + Forrest dataset)
python benchmark_neuroboros.py /path/to/data --backend python --repeat 3
python benchmark_neuroboros.py /path/to/data --backend c64  --n-jobs 10 --repeat 3
python benchmark_neuroboros.py /path/to/data --backend c32  --n-jobs 10 --repeat 3
python benchmark_neuroboros.py /path/to/data --backend metal --n-jobs 10 --repeat 3
```

Key invariant: SVD is unique up to sign flips of singular vector columns, so compare `abs(U_c)` vs `abs(U_py)` or compare the reconstructed matrix `U @ diag(s) @ Vt`.
