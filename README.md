# Hyperalignment

Hyperalignment algorithms for functional alignment of fMRI brain data across subjects.

This library implements the core algorithms from Haxby et al. (2011) and related work for aligning functional brain data into a common representational space. Given fMRI timeseries from multiple subjects, hyperalignment finds orthogonal (or regularized) transformations that maximize cross-subject similarity while preserving representational geometry.

## Algorithms

### Orthogonal Procrustes
Finds the rotation (with optional reflection and scaling) that minimizes `||XT - Y||_F`. The SVD of `Y'X` gives the optimal orthogonal transformation. This preserves representational dissimilarity matrices (RDMs).

### Ridge Regression
SVD-based ridge regression with regularization parameter `alpha`. Solves via damped singular values: `d = s / (alpha + s^2)`. Includes grid search over multiple alpha values and PC counts, with cross-validated ensemble averaging.

### Searchlight
Applies alignment locally within overlapping brain surface searchlights, then combines transformations using distance-based or uniform weighting. Each searchlight's local transformation is computed independently (Procrustes or ridge), then accumulated into a global dense transformation matrix with weights normalized to sum to 1 at each vertex.

### Template Construction
Builds a common representational space from multiple subjects via:
- **Procrustes template**: Sequential alignment to a running average, with iterative refinement (leave-one-out re-alignment)
- **GPA template**: Generalized Procrustes Analysis with simultaneous alignment
- **PCA template**: SVD of concatenated subject data with variance normalization variants

### Ensemble Cross-Validation
Generates temporally-blocked cross-validation splits for fMRI data (respecting autocorrelation via buffer zones around test blocks), trains searchlight models on each fold, and selects optimal hyperparameters by held-out prediction accuracy.

## Project Structure

```
src/hyperalignment/
    __init__.py          Public API exports
    linalg.py            SVD with LAPACK driver fallback, PCA
    procrustes.py        Orthogonal Procrustes algorithm
    ridge.py             Ridge regression with grid search and ensemble CV
    sparse.py            Sparse matrix initialization for searchlight patterns
    searchlight.py       Searchlight-based alignment and template building
    local_template.py    Template construction (Procrustes, GPA, PCA variants)
    ensemble.py          Cross-validation index generation, ensemble searchlight
tests/
    test_linalg.py       SVD and PCA correctness tests
    test_searchlight.py  Searchlight weight normalization tests
```

## Key Data Structures

| Name | Shape | Description |
|------|-------|-------------|
| X, Y | (n_samples, n_voxels) | fMRI timeseries for a subject |
| T | (n_voxels, n_voxels) | Transformation matrix: `X_aligned = X @ T` |
| dss | (n_subjects, n_samples, n_voxels) | Stacked subject data for template building |
| sls | list of int arrays | Searchlight vertex indices (each ~50-500 vertices) |
| dists | list of float arrays | Distances from searchlight centers |
| weights | list of float arrays | Per-searchlight combination weights (sum to 1 per vertex) |
| mat | sparse csc_matrix (nv, nv) | Sparse global transformation matrix |

Typical dimensions: n_samples ~ 200-500 timepoints, n_voxels ~ 10,000-40,000 cortical vertices per hemisphere.

## Dependencies

- numpy
- scipy (linalg, sparse)
- scikit-learn (PCA, randomized_svd)
- joblib (parallel searchlight computation)

## C Library

A complete C implementation lives in `csrc/` with four compute backends:

| Backend | Precision | Platform | Description |
|---------|-----------|----------|-------------|
| **CPU FP64** | double | All | Default. LAPACK/BLAS (Accelerate on macOS, MKL on Linux) |
| **CPU FP32** | float | All | Single-precision variant. Faster SVD, same output accuracy |
| **Metal GPU** | float | macOS | Apple Metal Performance Shaders via polar decomposition |
| **CUDA GPU** | float | Linux/NVIDIA | cuBLAS batched GEMM + cuSOLVER batched SVD |

Backend selection is at runtime via the `THaBackend` enum — no recompilation needed.

### Building

**macOS** — Apple Accelerate is used automatically:
```bash
cd csrc
make OPENMP=1           # CPU backends with OpenMP
make OPENMP=1 test      # build and run tests
```

**Linux (Anaconda)** — uses Intel MKL from the Anaconda environment:
```bash
# One-time: install MKL development headers
conda install mkl-devel

cd csrc
make OPENMP=1           # CPU backends with OpenMP
make OPENMP=1 test      # build and run tests
make CUDA=1 OPENMP=1    # also build CUDA backend (requires NVIDIA GPU + CUDA toolkit)
```

Other build targets:
```bash
make                    # builds libhyperalignment.a + .dylib/.so
make DEBUG=1 test       # AddressSanitizer build
make clean
```

Set `MKL_NUM_THREADS=1` (or `OPENBLAS_NUM_THREADS=1`) at runtime to suppress BLAS internal threading, which conflicts with the OpenMP outer loop.

### Running Tests

On macOS (all 4 backends including Metal): **2621 tests**. On Linux (no Metal): **2216 tests**. The difference is Metal-specific tests that are excluded on non-Apple platforms.

```bash
cd csrc
make OPENMP=1 test           # CPU tests
make CUDA=1 OPENMP=1 test    # CPU + CUDA tests (Linux)
```

Expected output:
```
Running hyperalignment C tests...
...
2216 tests: 2216 passed, 0 failed   # Linux
2621 tests: 2621 passed, 0 failed   # macOS
```

### C Source Layout

```
csrc/
    Makefile              Platform-detecting build (macOS Accelerate, Linux MKL, CUDA opt-in)
    ha_common.h           Types, LAPACK/BLAS includes (platform-conditional), THaBackend enum
    ha_linalg.h/.c        SVD FP64/FP32, PCA, z-score
    ha_procrustes.h/.c    Procrustes FP64 + FP32
    ha_ridge.h/.c         Ridge regression, grid search, ensemble ridge
    ha_sparse.h/.c        CSC sparse matrix init, scatter-add
    ha_searchlight.h/.c   Searchlight weights, alignment loops with backend dispatch
    ha_template.h/.c      Template construction (Procrustes, GPA, PCA)
    ha_ensemble.h/.c      Cross-validation index generation
    ha_metal.h            Metal API header (inline stubs on non-Apple)
    ha_metal.m            Metal/MPS implementation (macOS only, Objective-C)
    ha_cuda.h             CUDA API header (inline stubs when CUDA=1 not set)
    ha_cuda.cu            CUDA implementation (gather, batched SVD, scatter-add)
    hyperalignment.h      Umbrella header
    ha_test.c             Test harness
```

### GPU Backends

**Metal (macOS)** — polar decomposition via Newton iteration:
```
X_0 = X^T @ Y                          (GPU GEMM via MPSMatrixMultiplication)
X_{k+1} = (X_k + X_k^{-T}) / 2        (CPU LAPACK inverse + transpose, ~6-10 iterations)
T = X_converged                          (orthogonal polar factor)
```
Metal Performance Shaders does not provide SVD, so an iterative Newton approach is used. Falls back to CPU FP32 if singular.

**CUDA (Linux/NVIDIA)** — batched GEMM + batched approximate SVD:
```
A_i = local_X_i^T @ local_Y_i          (cublasSgemmStridedBatched)
A_i = U_i * S_i * V_i^T                (cusolverDnSgesvdaStridedBatched)
T_i = U_i @ V_i^T                       (cublasSgemmStridedBatched)
T_out[sl_r, sl_c] += T_i[r,c] * w[r]   (custom kernel, FP64 atomicAdd)
```
Processes 256 searchlights per batch. All matrices are zero-padded to the maximum searchlight size for uniform batch dimensions. Subject data (FP64) is converted to FP32 on-GPU; scatter-add uses FP64 for numerical accuracy.

### Python ctypes Wrapper

`hyperalignment_c.py` provides a Python interface to the C library:

```python
import hyperalignment_c as hac

W = hac.searchlight_procrustes(
    X, Y, sls, dists, radius,
    backend='cpu64',  # 'cpu64', 'cpu32', 'metal' (macOS), 'cuda' (Linux/NVIDIA)
    n_jobs=96,        # OpenMP threads (requires OPENMP=1 build)
)
# W is a dense (nv, nv) numpy float64 array
```

Input matrices are converted to Fortran (column-major) order for cache-efficient column extraction in C. Searchlight index arrays are built with vectorized numpy ops. The dense output matrix is allocated in Python and written directly by C (zero-copy).

## Benchmarks

### Setup

Benchmarked on the StudyForrest dataset (Hanke et al., 2014) via `neuroboros`: 2 subjects, 4 training runs, 4 test runs, radius-20mm searchlights (~9,675 searchlights per hemisphere, ~9,675 cortical vertices per hemisphere). The benchmark runs searchlight Procrustes alignment on both hemispheres and reports vertex-wise correlation between aligned and target timeseries.

All times are **median of 3 repeats**. Data is loaded once before timing begins. BLAS internal threading disabled (`MKL_NUM_THREADS=1` / `VECLIB_MAXIMUM_THREADS=1`) to prevent interference with the OpenMP outer loop.

### How to Run

```bash
# 1. Build the C library with OpenMP (and optionally CUDA)
cd csrc
make OPENMP=1                  # macOS or Linux CPU
make CUDA=1 OPENMP=1           # Linux + NVIDIA GPU
cd ..

# 2. Run benchmarks (3 repeats, report median)
python benchmark_neuroboros.py /path/to/data --backend python --repeat 3
MKL_NUM_THREADS=1 python benchmark_neuroboros.py /path/to/data --backend c64 --n-jobs 1   --repeat 3
MKL_NUM_THREADS=1 python benchmark_neuroboros.py /path/to/data --backend c64 --n-jobs 96  --repeat 3
MKL_NUM_THREADS=1 python benchmark_neuroboros.py /path/to/data --backend c32 --n-jobs 96  --repeat 3
python benchmark_neuroboros.py /path/to/data --backend cuda --repeat 3
```

Options:
- `--backend`: `python`, `c64`, `c32`, `metal` (macOS), `cuda` (Linux/NVIDIA)
- `--n-jobs N`: OpenMP threads for C backends (default: 1; requires `OPENMP=1` build)
- `--repeat N`: Number of repeats (default: 1); reports median when N > 1

### Results

All backends produce identical output (test-set vertex-wise correlation percentiles):
```
[-0.2561 -0.0149  0.0030  0.0160  0.0282  0.0424  0.0594  0.0840  0.1252  0.2105  0.6171]
```

#### Apple M4 Pro — 10 performance cores, 48 GB unified memory, macOS

| Backend | Threads | L hemi (s) | R hemi (s) | Total (s) | vs Python |
|---------|---------|-----------|-----------|-----------|-----------|
| Python (numpy/Accelerate) | 1 | 15.1 | 15.0 | 30.1 | 1.0x |
| C CPU FP64 | 1 | 14.4 | 14.6 | 28.9 | 1.04x |
| **C CPU FP64** | **10** | **2.2** | **2.2** | **4.3** | **7.0x** |
| C CPU FP32 | 1 | 9.3 | 9.2 | 18.5 | 1.6x |
| **C CPU FP32** | **10** | **1.25** | **1.24** | **2.5** | **12.0x** |
| C Metal GPU FP32 | 10 | 8.3 | 8.4 | 16.8 | 1.8x |

#### AMD Threadripper 7995WX + NVIDIA RTX 4090 — 96 cores, Ubuntu 24.04

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

### Analysis

**Correctness**: All backends produce numerically identical percentile distributions at 4 decimal places. FP32 backends (CPU FP32, Metal, CUDA) match FP64 because searchlight weight normalization and scatter-add accumulation are done in FP64 regardless of local Procrustes precision.

**OpenMP scaling**: The dominant factor is thread count. C CPU FP32 scales well up to core count — on Threadripper reaching **100x** vs Python at 96 threads (1.2s total). Scaling is sub-linear (96 cores → ~28x over 1 thread, not 96x) due to memory bandwidth saturation and NUMA effects across the 4-die Threadripper topology.

| Machine | Backend | 1 thread | N threads | Speedup |
|---------|---------|----------|-----------|---------|
| M4 Pro | C FP32 | 18.5s | 2.5s (10T) | 7.4x |
| Threadripper | C FP32 | 34.3s | 1.2s (96T) | 28.6x |

**Column-major input is critical**: Searchlight indices are highly scattered across ~9,675 vertices (mean gap ~80). With row-major data, extracting columns for one searchlight requires ~220K scattered reads per searchlight, thrashing the CPU cache. The C wrapper converts input to Fortran order (`np.asfortranarray`) so each column is contiguous — extraction becomes `memcpy` per column. This single change gives a **2x single-threaded speedup**.

**Dense output with OMP atomic**: The library accumulates local transformation matrices into a dense (nv×nv) output using `#pragma omp atomic` for lock-free scatter-add. Compared to the earlier sparse approach (qsort + binary-search per scatter-add + `#pragma omp critical`), this eliminates the global serialization bottleneck that dominated at high thread counts.

**CUDA GPU is slower than CPU here**: The RTX 4090 CUDA backend (**42.3s**) is much slower than 96-thread CPU FP32 (**1.2s**). The reason is that `cusolverDnSgesvdaStridedBatched` — the only viable batched SVD for 121×121 matrices on GPU — computes an approximate SVD via iterative polar decomposition (Jacobi sweeps). For this problem size, the GPU does not have enough parallelism per matrix to amortize the dispatch overhead, and the 256-matrix batch runs largely serially within the warp structure. Contrast with `cusolverDnSgesvdjBatched` (hard limit: 32×32) or the non-batched routines (effectively serial across 9,675 calls).

**Metal GPU is slower than CPU**: The ~121×121 local matrices are too small for GPU dispatch overhead to be amortized. Metal Performance Shaders does not provide SVD, requiring an iterative Newton approach (multiple LU factorizations per searchlight), which limits throughput.

**The GPU opportunity for this problem**: A custom one-sided Jacobi SVD CUDA kernel (one thread per matrix element, true batched parallelism at any size) would likely break the GPU bottleneck. Alternatively, a hybrid approach — GPU for `A = X^T @ Y` GEMM, CPU for SVD, GPU for `T = U @ Vt` GEMM — would leverage the GPU for the large matrix multiply while avoiding the SVD dispatch overhead. For the current dataset size, the 96-core CPU already achieves 1.2s total, so GPU acceleration has limited practical benefit.

### Replicating

**macOS (Apple Silicon)**:
```bash
cd csrc && make OPENMP=1 test
cd .. && python benchmark_neuroboros.py /path/to/data --backend c32 --n-jobs 10 --repeat 3
```

**Linux (Anaconda + MKL)**:
```bash
conda install mkl-devel
cd csrc && make OPENMP=1 test
MKL_NUM_THREADS=1 python benchmark_neuroboros.py /path/to/data --backend c32 --n-jobs $(nproc) --repeat 3
```

**Linux + CUDA (Anaconda + MKL + NVIDIA)**:
```bash
conda install mkl-devel
cd csrc && make CUDA=1 OPENMP=1 test
MKL_NUM_THREADS=1 python benchmark_neuroboros.py /path/to/data --backend cuda --repeat 3
```

## References

- Haxby, J.V., Guntupalli, J.S., Connolly, A.C., et al. (2011). A common, high-dimensional model of the representational space in human ventral temporal cortex. *Neuron*, 72(2), 404-416.
- Hanke, M., Baumgartner, F.J., Ibe, P., et al. (2014). A high-resolution 7-Tesla fMRI dataset from complex natural stimulation with an audio movie. *Scientific Data*, 1, 140003.
