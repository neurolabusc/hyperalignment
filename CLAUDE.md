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

### Memory
- `malloc`/`free` exclusively (no C++ `new`/`delete`)
- Always check allocation: `if (!ptr) { ... return NULL; }`
- Caller owns returned pointers unless documented otherwise
- Large buffers always heap-allocated (brain data can be hundreds of MB)

### Error Handling
- Return `NULL` for pointer-returning functions on failure
- Return `int` error codes (0 = success) for void-like functions
- Print errors to stderr via `fprintf(stderr, ...)`
- No `exit()` calls in library code; only in CLI main

## Core Data Structures

```c
// Dense matrix — FP64 (row-major for LAPACK compatibility with transpose)
typedef struct {
    double *data;
    int32_t rows, cols, stride;
} TMat;

// Dense matrix — FP32 (for CPU FP32 and Metal backends)
typedef struct {
    float *data;
    int32_t rows, cols, stride;
} TMatF;

// Sparse CSC matrix (matches scipy.sparse.csc_matrix)
typedef struct {
    double *data;
    int32_t *indices;   // row indices
    int32_t *indptr;    // column pointers (length cols+1)
    int32_t rows, cols;
    int64_t nnz;
} TSparseCSC;

// Searchlight definition
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

Precision conversion helpers: `ha_mat_to_float()` (TMat→TMatF) and `ha_matf_to_double()` (TMatF→TMat) in `ha_common.h`.

## Algorithm-to-Function Mapping

| Python | C (FP64) | C (FP32) | Core Operation |
|--------|----------|----------|---------------|
| `safe_svd(X)` | `ha_svd()` | `ha_svd_f32()` | LAPACK `dgesdd`/`sgesdd`, fallback `dgesvd`/`sgesvd` |
| `svd_pca(X)` | `ha_pca()` | — | SVD then `U * s` |
| `procrustes(X, Y)` | `ha_procrustes()` | `ha_procrustes_f32()` | `A = Y'X`, SVD of A, `T = U @ Vt` |
| — | — | `ha_procrustes_metal()` | Polar decomposition via Newton iteration (GPU) |
| `ridge(X, Y, alpha)` | `ha_ridge()` | — | SVD of X, damped solve |
| `ridge_grid(...)` | `ha_ridge_grid()` | — | Vectorized over alpha/npc grid |
| `initialize_sparse_matrix(sls)` | `ha_sparse_init()` | — | Build CSC sparsity pattern |
| `compute_searchlight_weights(sls)` | `ha_searchlight_weights()` | — | Distance-based weighting |
| `searchlight_procrustes(...)` | `ha_searchlight_procrustes()` | (dispatches via `THaBackend`) | Loop: local align + sparse accumulate |
| `searchlight_ridge(...)` | `ha_searchlight_ridge()` | — | Loop: local ridge + sparse accumulate |
| `compute_template(dss)` | `ha_template()` | — | Iterative Procrustes/GPA/PCA |
| `compute_ensemble_indices(nt)` | `ha_ensemble_indices()` | — | Block permutation CV splits |

## Build System

Uses Make with platform detection. Links against Accelerate on macOS, system LAPACK/BLAS on Linux.

```
csrc/
    Makefile              Platform-detecting build system (static + shared library)
    hyperalignment.h      Umbrella header (includes all modules + ha_metal.h)
    ha_common.h           Types (TMat, TMatF, TSparseCSC, TSearchlights, THaBackend), alloc helpers
    ha_linalg.h/c         SVD FP64 (dgesdd/dgesvd), SVD FP32 (sgesdd/sgesvd), PCA, z-score
    ha_procrustes.h/c     Procrustes FP64 + FP32
    ha_ridge.h/c          Ridge regression, grid search, ensemble ridge
    ha_sparse.h/c         CSC sparse matrix init from searchlight patterns, scatter-add
    ha_searchlight.h/c    Searchlight weights, alignment loops with THaBackend dispatch
    ha_template.h/c       Template construction (Procrustes, GPA, PCA)
    ha_ensemble.h/c       Cross-validation index generation
    ha_metal.h            Metal API header (inline stubs on non-Apple)
    ha_metal.m            Metal/MPS implementation (macOS only, Objective-C)
    ha_test.c             Test harness (2621 tests)
```

### Build commands
```bash
cd csrc
make                  # Release build -> libhyperalignment.a + .dylib/.so
make test             # Build and run tests
make DEBUG=1 test     # Build with AddressSanitizer and run tests
make OPENMP=1         # Build with OpenMP searchlight parallelism
make clean
```

### Compiler flags
- `-std=c11 -O2 -Wall -Wextra -Wpedantic -fPIC` for release
- `-std=c11 -g -O0 -fsanitize=address` for debug
- `-framework Accelerate -framework Metal -framework MetalPerformanceShaders -framework Foundation` on macOS
- `-llapack -lblas -lm` on Linux
- OpenMP opt-in via `OPENMP=1` flag

### Row-major SVD trick
Row-major `X(M,N)` is column-major `X^T(N,M)` to LAPACK. We call `dgesdd("S", N, M, ...)` and swap the U/Vt buffer assignments: LAPACK's "U" output becomes our Vt, LAPACK's "Vt" output becomes our U. No explicit transpose needed.

## LAPACK/BLAS Functions Used

FP64: `dgesdd_`/`dgesvd_` (SVD), `dgemm_` (GEMM), `dgemv_` (matvec), `dnrm2_`, `dscal_`, `dcopy_`
FP32: `sgesdd_`/`sgesvd_` (SVD), `cblas_sgemm` (GEMM), `sgetrf_`/`sgetri_` (LU inverse, used in Metal Newton iteration), `cblas_sger`, `cblas_sscal`

## GPU Implementation Status

### Metal GPU Backend (Implemented)

The Metal backend (`ha_metal.m`) computes Procrustes alignment via **polar decomposition using Newton iteration**, avoiding SVD entirely:

```
X_0 = X^T @ Y                          (GPU GEMM via MPSMatrixMultiplication)
X_{k+1} = (X_k + X_k^{-T}) / 2        (CPU LAPACK inverse + transpose, ~6-10 iterations)
T = X_converged                          (orthogonal polar factor)
```

**Architecture**: GPU GEMM for the initial `X^T @ Y` multiplication, then CPU LAPACK (`sgetrf_`/`sgetri_`) for the Newton iteration inverse (small ~200x200 matrices where GPU dispatch overhead dominates). Falls back to `ha_procrustes_f32` (CPU FP32 SVD) if the matrix is singular.

**Key implementation detail**: `matrix_inverse_cpu` passes row-major data to column-major LAPACK, so it returns `A^{-1}` (not `A^{-T}`). An explicit `transpose_inplace()` is needed after the inverse to get `A^{-T}` for Newton iteration correctness.

**Current performance**: Slower than CPU (~121s vs ~73s Python) because each of the ~19K searchlights submits a separate GPU command buffer (~10µs dispatch overhead each). Optimization opportunity: batch multiple searchlights per command buffer or pipeline submissions.

### Benchmark Results (Forrest dataset, Apple M3 Max)

| Backend | Total (s) | vs Python |
|---------|-----------|-----------|
| Python (numpy/BLAS) | 73 | 1.0x |
| C CPU FP64 | 89 | 0.8x |
| C CPU FP32 | 82 | 0.9x |
| C Metal GPU FP32 | 121 | 0.6x |

All backends produce numerically identical percentile distributions at 4 decimal places. FP32 backends match FP64 because searchlight weight normalization and sparse accumulation are done in FP64 regardless.

### Metal Optimization TODO

- **Command buffer batching**: Submit multiple searchlights per GPU command buffer to amortize dispatch overhead
- **Pipeline submissions**: Overlap GPU execution with CPU sparse accumulation
- **Pre-cached sparse init**: `ha_sparse_init` sorts all (row, col) pairs (~few seconds for ~19K searchlights), could be cached

### Apple Accelerate Does NOT Use the GPU
Accelerate's BLAS/LAPACK runs exclusively on the CPU via Apple Silicon's AMX coprocessor. GPU acceleration requires Metal Performance Shaders (MPS).

### MPS Capabilities
MPS provides GPU-accelerated GEMM (`MPSMatrixMultiplication`), Cholesky, LU (`MPSMatrixDecompositionLU`), and triangular solve (`MPSMatrixSolveTriangular`). MPS does **NOT** provide SVD — this is why we use polar decomposition for Procrustes and why ridge regression has no Metal backend.

### Future: GPU SVD Strategy (for Ridge and General Use)
Since there is no GPU SVD in MPS, options for algorithms that require full SVD (ridge regression):
- **One-sided Jacobi SVD** in a custom Metal compute shader — parallelizes well for our small matrices (~200x200), each searchlight gets one threadgroup
- **Eigendecomposition approach**: compute `A^T A`, GPU eigensolver (Jacobi), then recover singular vectors — avoids full SVD but loses some numerical precision
- **Hybrid CPU/GPU**: keep SVD on CPU (Accelerate), offload GEMM and scatter-add to GPU — viable because Apple Silicon unified memory has zero copy cost

### Existing GPU SVD Research
- [Ringoot et al. (2025)](https://arxiv.org/abs/2508.06339) — first portable GPU SVD on Metal (Julia/KernelAbstractions.jl). However, computes **singular values only** (no vectors), **no batched mode**, and optimized for large matrices (poor performance <256x256). Not suitable for our workload.
- [philipturner GEMM kernel](https://gist.github.com/philipturner/84f613a5cc745460a914d2c6ad226131) — optimized Metal GEMM (FP32/FP16/BF16, no FP64). Useful reference for custom Metal shader patterns but does not address SVD.

### CUDA Path (Linux/NVIDIA) — Detailed Audit

Our workload: ~10,000 independent searchlight problems, each involving SVD + GEMM on ~200x200 matrices, followed by sparse scatter-add into a global transformation matrix.

#### cuSOLVER SVD — The Critical Constraint

cuSOLVER provides three SVD methods:

1. **`cusolverDnDgesvdjBatched`** — Batched Jacobi SVD. Launches all matrices in one kernel. **Hard size limit: m ≤ 32 and n ≤ 32.** Our searchlights are ~200x200, so this is **NOT usable** for our workload.
   - [NVIDIA gesvdjBatched sample code](https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuSOLVER/gesvdjBatched)
   - [Forum discussion on 32x32 limit](https://forums.developer.nvidia.com/t/the-origin-of-the-m-32-and-n-32-limitations-in-gesvdjbatched/266434)

2. **`cusolverDnDgesvdj`** — Non-batched Jacobi SVD. No size limit, works on larger matrices. The Jacobi method applies up to n/2 non-overlapping Givens rotations in parallel and converges quadratically (typically 3-4 sweeps). However, it is **effectively synchronous** — concurrent launches on separate CUDA streams [do not overlap](https://forums.developer.nvidia.com/t/cusolver-svd-not-overlapping-using-streams/306782). This means we cannot trivially parallelize 10,000 SVDs on the GPU using streams.
   - [GTC 2019: Fast SVD on GPUs](https://developer.nvidia.com/gtc/2019/video/s9226)

3. **`cusolverDnDgesvdaStridedBatched`** — Approximate SVD (polar decomposition-based), strided batched. No documented size limit like gesvdjBatched. Computes only the top-k singular values/vectors, not full SVD. May be suitable if we only need the leading singular components (which is true for our low-rank Procrustes problems). Requires testing for numerical accuracy vs full SVD.
   - [cuSOLVER documentation](https://docs.nvidia.com/cuda/cusolver/index.html)

4. **`cusolverDnDgesvd`** — QR-based SVD (same algorithm as LAPACK). No size limit but slower than Jacobi for small matrices and also synchronous per-stream.

**Bottom line for SVD**: Unlike what we initially assumed, cuSOLVER does NOT provide a drop-in batched SVD for 200x200 matrices. The batched API is limited to 32x32. For our workload, viable strategies are:
- **Approximate batched SVD** (`gesvdaStridedBatched`) — if top-k singular components suffice (likely yes for Procrustes where we use all of U @ Vt, but the approximation quality needs validation)
- **Custom one-sided Jacobi SVD kernel** — same approach needed for Metal, achieves true batched parallelism for any matrix size
- **Hybrid CPU/GPU** — SVD on CPU, GEMM and scatter-add on GPU (viable on unified memory systems like DGX Spark; costly on discrete GPUs due to PCIe transfers)
- **Concurrent streams** — launch non-batched `gesvdj` on multiple streams. Limited by synchronous behavior, but may still achieve some occupancy overlap on modern GPUs

#### cuBLAS Batched GEMM — Fully Capable

[`cublasDgemmBatched`](https://docs.nvidia.com/cuda/cublas/index.html) and `cublasDgemmStridedBatched` have **no size limits** and work well for small matrices. This covers all our GEMM operations:
- `A = X^T @ Y` (cross-correlation for Procrustes)
- `T = U @ Vt` (transformation assembly)
- `X_aligned = X @ T` (applying transformations)

The strided batched variant (`cublasDgemmStridedBatched`) avoids pointer-array overhead and is preferred when matrices are packed contiguously.

Reference: [cuBLAS Strided Batched Matrix Multiply](https://developer.nvidia.com/blog/cublas-strided-batched-matrix-multiply/)

#### cuSPARSE — Partial Coverage

[cuSPARSE](https://docs.nvidia.com/cuda/cusparse/index.html) supports CSC/CSR formats with SpMM and SpMV operations, 30-150x faster than CPU for sparse matrix-dense vector/matrix operations. However:
- **No built-in scatter-add with atomics** — our pattern (accumulate local transformations into overlapping regions of a global sparse matrix) requires a custom CUDA kernel with `atomicAdd` on double-precision values
- cuSPARSE is useful for the *final* sparse matrix-dense matrix multiply (`X @ T_sparse`), but not for building the sparse matrix itself

#### Discrete GPU: PCIe Transfer Overhead

For discrete GPUs (most CUDA-capable cards), data must be explicitly copied between host and device memory:
- **PCIe 3.0 x16**: ~12 GB/s bidirectional
- **PCIe 4.0 x16**: ~25 GB/s bidirectional
- **PCIe 5.0 x16**: ~64 GB/s bidirectional
- **Per-transfer latency**: ~10-30 µs overhead per `cudaMemcpy` call

For our workload, the data to transfer per hemisphere:
- Subject data: 500 timepoints × 40,000 vertices × 8 bytes = ~160 MB per subject
- Searchlight extraction + results: comparable in size
- If SVD stays on CPU in a hybrid approach, intermediate results must bounce back and forth — PCIe becomes the bottleneck

**Implication**: On discrete GPUs, the hybrid CPU/GPU approach (SVD on CPU, GEMM on GPU) is impractical due to per-searchlight transfer overhead. Everything must stay on GPU, which means solving the batched SVD problem is mandatory.

Reference: [How to Optimize Data Transfers in CUDA](https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/)

#### DGX Spark: Unified Memory Advantage

The [DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/) (Grace Blackwell GB10) uses unified memory, similar to Apple Silicon:
- **128 GB LPDDR5x** shared coherently between CPU (20 Arm cores) and GPU (6,144 CUDA cores, 192 Tensor cores)
- **~273 GB/s** shared memory bandwidth (vs ~64 GB/s for PCIe 5.0 x16)
- **No explicit `cudaMemcpy` needed** — both CPU and GPU access the same address space via hardware coherence
- CUDA 13.0, Blackwell architecture

This makes the **hybrid CPU/GPU approach viable** on DGX Spark — SVD on CPU (Arm NEON + system LAPACK), GEMM on GPU (cuBLAS), zero-copy data sharing. Same architectural advantage as Apple Silicon.

References:
- [DGX Spark Unified Memory Architecture](https://deepwiki.com/NVIDIA/dgx-spark-playbooks/9.1-unified-memory-architecture)
- [DGX Spark Hardware Overview](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)
- [DGX Spark In-Depth Review (LMSYS)](https://lmsys.org/blog/2025-10-13-nvidia-dgx-spark/)

#### CUDA vs Metal — Summary Comparison

| Capability | CUDA | Metal/MPS |
|------------|------|-----------|
| **Batched SVD** | Only ≤32x32 (`gesvdjBatched`); approximate batched available (`gesvdaStridedBatched`) | Not available — custom Jacobi shader needed |
| **Non-batched SVD** | `gesvdj` (Jacobi, any size, but synchronous per-stream) | Not available |
| **Batched GEMM** | `cublasDgemmStridedBatched` — no limits | `MPSMatrixMultiplication` — no limits |
| **Sparse ops** | cuSPARSE for SpMM/SpMV; custom kernel for scatter-add | Custom Metal compute shader |
| **Unified memory** | DGX Spark only; discrete GPUs need explicit transfers | All Apple Silicon (M1+) |
| **Peak TFLOPS (FP64)** | DGX Spark ~1 PFLOP (FP4), FP64 TBD; high-end discrete (A100): ~9.7 TFLOPS | M4 Max GPU: ~2-7 TFLOPS |

#### Recommended CUDA Strategy

1. **Discrete GPU (most common)**: Must keep everything on GPU. Use `cusolverDnDgesvdaStridedBatched` (approximate batched SVD) if accuracy is acceptable, or write a custom one-sided Jacobi SVD kernel. Use `cublasDgemmStridedBatched` for all GEMMs. Custom kernel for sparse scatter-add with `atomicAdd`.

2. **DGX Spark (unified memory)**: Hybrid approach — SVD on CPU (system LAPACK), batched GEMM on GPU (cuBLAS), sparse accumulation on CPU. Zero-copy via unified memory. Simpler to implement, avoids the batched SVD problem entirely.

3. **Both architectures**: Pre-extract all searchlight submatrices into contiguous batched buffers for coalesced GPU memory access. One kernel launch for all GEMMs, one for all SVDs (if batched).

### Searchlight Parallelism
The searchlight loop iterates over ~10,000 independent local problems (each a small SVD + GEMM on ~200x200 matrices). This maps naturally to:
- **CUDA (discrete)**: Custom batched SVD kernel or `gesvdaStridedBatched` + `cublasDgemmStridedBatched` — all searchlights in one or two kernel launches
- **CUDA (DGX Spark)**: CPU SVD (LAPACK) + GPU batched GEMM (cuBLAS) — hybrid via unified memory
- **Metal**: Custom Jacobi SVD compute shader dispatching one threadgroup per searchlight + `MPSMatrixMultiplication`

### Sparse Accumulation
The scatter-add from local transformations into the global sparse matrix requires atomic operations on GPU. Each searchlight writes to a (sl_size x sl_size) block of the sparse matrix, with overlapping searchlights causing write conflicts. On CUDA, use `atomicAdd` (double-precision supported since compute capability 6.0). On Metal, use `atomic_fetch_add_explicit`. Alternative: per-searchlight result buffers, then reduce into sparse matrix on CPU (simpler, avoids atomics).

### Memory Layout
- Searchlight data should be packed contiguously for coalesced GPU memory access
- Pre-extract all searchlight submatrices into a batched buffer before launching GPU kernels
- Result buffer per searchlight, then reduce into sparse matrix on CPU (simpler than GPU atomics)
- Apple Silicon and DGX Spark unified memory eliminates CPU-GPU transfer overhead — a significant advantage over discrete GPU architectures

## Performance Notes

### Pre-allocated workspace does NOT help (tested and reverted)

We tried pre-allocating thread-local work buffers (`TProcWorkspace` structs holding all SVD/Procrustes scratch arrays) to eliminate ~145K malloc/free pairs per hemisphere. **Result: 3-5% regression, not improvement.** Root causes:

1. **macOS magazine allocator is near-zero-cost** for repeated same-sized allocations — it recycles blocks via per-CPU free lists without system calls
2. **Oversized buffers hurt L2 cache** — workspace sized for max searchlight (~300 verts, ~10 MB/thread) exceeds M4 Pro's 4 MB L2 per core, while per-call right-sized buffers (~200 verts, ~4 MB) fit better
3. **SVD computation dominates** — LAPACK dgesdd/sgesdd is ~95%+ of runtime; allocation overhead is negligible

**Do not re-attempt this optimization.** The per-searchlight malloc/free pattern is already efficient on macOS. The ~2.5x gap between Python and single-threaded C is due to Accelerate/numpy internal optimization (cache tiling, vectorized small-matrix paths), not allocation overhead.

### Benchmark baselines (M4 Pro, 10P+4E cores, Forrest dataset, median of 3)

| Backend | Threads | Total (s) | vs Python |
|---------|---------|-----------|-----------|
| Python (numpy/Accelerate) | auto | 30.9 | 1.0x |
| C CPU FP64 | 1 | 77.4 | 0.4x |
| C CPU FP64 | 10 | 31.0 | 1.0x |
| C CPU FP32 | 1 | 69.9 | 0.4x |
| **C CPU FP32** | **10** | **28.8** | **1.1x** |
| C Metal GPU | 1 | 89.8 | 0.3x |
| C Metal GPU | 10 | 61.7 | 0.5x |

### Remaining optimization ceiling

C FP32 10-thread (28.8s) already beats Python (30.9s) by 7%. Further gains are marginal:

- **`#pragma omp critical` contention**: all scatter-adds serialize through one global lock. Fine-grained per-column locks or precomputed sparse offsets could help, but SVD is ~95% of runtime so <2% expected gain.
- **`dgesvd` vs `dgesdd`**: divide-and-conquer (`dgesdd`) has setup overhead that may not pay off for N~200. QR-iteration (`dgesvd`) might be marginally faster for small matrices.
- **CPU polar Newton (skip SVD)**: `ha_polar_newton` exists for Metal path. FP64 variant would need ~4-6 LU iterations, likely comparable to 1 SVD — no clear win.
- **Linux may benefit from workspace pre-allocation**: glibc malloc is less efficient than macOS magazine allocator for this pattern.
- **More threads**: M4 Pro has 14 cores (10P+4E). Using 14 instead of 10 might squeeze 10-15% more, but efficiency cores are slower.

The 2.5x single-threaded gap vs Python is a numpy/Accelerate integration advantage (internal workspace caching, cache-optimized call patterns) that cannot be closed from external C code.

## Python ctypes Wrapper (`hyperalignment_c.py`)

Loads `csrc/libhyperalignment.dylib` (or `.so`) via ctypes. Main function:
```python
searchlight_procrustes(X, Y, sls, dists, radius, backend='cpu64', reflection=True, scaling=False)
# Returns scipy.sparse.csc_matrix
```

**Memory management**: Input matrices are zero-copy (numpy `.ctypes.data` passed to TMat). C-allocated memory (sparse matrix, weights) is freed via libc `free()` obtained from `ctypes.CDLL(None)` — the `ha_sparse_free`/`ha_mat_free` helpers are `static inline` in `ha_common.h` and not exported from the shared library.

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
make -C csrc DEBUG=1 test   # with AddressSanitizer

# Real-data benchmark (requires neuroboros + Forrest dataset)
python benchmark_neuroboros.py /path/to/data --backend python   # Python baseline
python benchmark_neuroboros.py /path/to/data --backend c64      # C CPU FP64
python benchmark_neuroboros.py /path/to/data --backend c32      # C CPU FP32
python benchmark_neuroboros.py /path/to/data --backend metal    # Metal GPU FP32
```

Key invariant: SVD is unique up to sign flips of singular vector columns, so compare `abs(U_c)` vs `abs(U_py)` or compare the reconstructed matrix `U @ diag(s) @ Vt`.
