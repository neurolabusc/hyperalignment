# Hyperalignment

Hyperalignment algorithms for functional alignment of fMRI brain data across subjects.

This library implements the core algorithms from Haxby et al. (2011) and related work for aligning functional brain data into a common representational space. Given fMRI timeseries from multiple subjects, hyperalignment finds orthogonal (or regularized) transformations that maximize cross-subject similarity while preserving representational geometry.

## Algorithms

### Orthogonal Procrustes
Finds the rotation (with optional reflection and scaling) that minimizes `||XT - Y||_F`. The SVD of `Y'X` gives the optimal orthogonal transformation. This preserves representational dissimilarity matrices (RDMs).

### Ridge Regression
SVD-based ridge regression with regularization parameter `alpha`. Solves via damped singular values: `d = s / (alpha + s^2)`. Includes grid search over multiple alpha values and PC counts, with cross-validated ensemble averaging.

### Searchlight
Applies alignment locally within overlapping brain surface searchlights, then combines transformations using distance-based or uniform weighting. Each searchlight's local transformation is computed independently (Procrustes or ridge), then accumulated into a global sparse transformation matrix with weights normalized to sum to 1 at each vertex.

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

A complete C implementation lives in `csrc/` with three compute backends:

| Backend | Precision | Platform | Description |
|---------|-----------|----------|-------------|
| **CPU FP64** | double | All | Default. Uses LAPACK/BLAS (Accelerate on macOS, OpenBLAS on Linux) |
| **CPU FP32** | float | All | Single-precision variant for comparison |
| **Metal GPU** | float | macOS | Apple Metal Performance Shaders via polar decomposition |

Backend selection is at runtime via the `THaBackend` enum — no recompilation needed.

### Building

Requires a C11 compiler and a LAPACK/BLAS provider. On macOS the Apple Accelerate framework is used automatically; on Linux, install `liblapack-dev` and `libopenblas-dev` (or equivalent).

```bash
cd csrc
make                # builds libhyperalignment.a (static) and .dylib/.so (shared)
make DEBUG=1        # debug build with AddressSanitizer
make OPENMP=1       # enable OpenMP for searchlight parallelism
make test           # build and run all tests
make clean          # remove build artifacts
```

On macOS, the Metal frameworks are linked automatically. On Linux, only the CPU backends are available; the Metal header provides inline stubs that return error codes.

### Running Tests

The test suite (`ha_test.c`) contains 2621 deterministic tests covering all modules and all three backends.

```bash
cd csrc
make test           # build and run tests (optimized)
make DEBUG=1 test   # build and run tests with AddressSanitizer
```

Expected output:

```
Running hyperalignment C tests...

test_svd_reconstruction
test_svd_wide_matrix
test_svd_mean_removal
test_pca
test_procrustes_identity
test_procrustes_known_rotation
test_procrustes_no_reflection
test_ridge_small_alpha
test_sparse_init
test_sparse_scatter_add
test_searchlight_weights_uniform
test_zscore
test_template_identical_subjects
test_ensemble_indices
test_svd_f32_reconstruction
test_procrustes_f32_identity
test_procrustes_f32_known_rotation
test_procrustes_metal_identity
test_procrustes_metal_known_rotation
test_metal_cleanup

2621 tests: 2621 passed, 0 failed
```

### C Source Layout

```
csrc/
    Makefile              Platform-detecting build system (static + shared library)
    ha_common.h           Types (TMat, TMatF, TSparseCSC, TSearchlights, THaBackend), alloc helpers
    ha_linalg.h/.c        SVD FP64 (dgesdd/dgesvd), SVD FP32 (sgesdd/sgesvd), PCA, z-score
    ha_procrustes.h/.c    Procrustes FP64 + FP32
    ha_ridge.h/.c         Ridge regression, grid search, ensemble ridge
    ha_sparse.h/.c        CSC sparse matrix init, scatter-add
    ha_searchlight.h/.c   Searchlight weights, alignment loops with backend dispatch
    ha_template.h/.c      Template construction (Procrustes, GPA, PCA)
    ha_ensemble.h/.c      Cross-validation index generation
    ha_metal.h            Metal API header (stubs on non-Apple)
    ha_metal.m            Metal/MPS implementation (macOS only, Objective-C)
    hyperalignment.h      Umbrella header
    ha_test.c             Test harness (2621 tests)
```

### Metal GPU Backend

The Metal backend computes Procrustes alignment without SVD by using **polar decomposition via Newton iteration**:

```
X_0 = X^T @ Y                          (GPU GEMM via MPSMatrixMultiplication)
X_{k+1} = (X_k + X_k^{-T}) / 2        (CPU LAPACK inverse + transpose, ~6-10 iterations)
T = X_converged                          (orthogonal polar factor)
```

This avoids the SVD entirely — Metal Performance Shaders provides GEMM, LU decomposition, and triangular solve, but not SVD. The Newton iteration converges cubically and typically needs 6-10 iterations for FP32 precision. Falls back to CPU FP32 SVD if the matrix is singular.

### Python ctypes Wrapper

`hyperalignment_c.py` provides a Python interface to the C library via ctypes. It loads the shared library (`libhyperalignment.dylib` or `.so`) and exposes a `searchlight_procrustes()` function compatible with the benchmark script.

```python
import hyperalignment_c as hac

W = hac.searchlight_procrustes(
    X, Y, sls, dists, radius,
    backend='cpu64'  # or 'cpu32', 'metal'
)
# W is a scipy.sparse.csc_matrix
```

The wrapper handles all memory management: numpy arrays are passed to C by pointer (zero-copy for the input matrices), and the sparse result is copied into a scipy CSC matrix before freeing the C allocations.

## Benchmarks

### Setup

Benchmarked on the StudyForrest dataset (Hanke et al., 2014) via `neuroboros`: 2 subjects, 4 training runs, 4 test runs, radius-20mm searchlights (~9,670 searchlights per hemisphere, ~19,341 cortical vertices per hemisphere). The benchmark runs searchlight Procrustes alignment on both hemispheres and evaluates vertex-wise correlation between aligned and target timeseries.

Hardware: Apple M4 Pro (10 performance + 4 efficiency cores), 48 GB unified memory, macOS.

All times are median of 3 repeats. Data is loaded once before timing begins.

### How to Run

```bash
# 1. Build the C library with OpenMP support
cd csrc && make OPENMP=1 && cd ..

# 2. Install Python dependencies
pip install neuroboros hyperalignment scipy numpy

# 3. Download the Forrest dataset (first run only, ~2 GB)
python -c "import neuroboros; neuroboros.Forrest()"

# 4. Run benchmarks (3 repeats, report median)
python benchmark_neuroboros.py /path/to/data --backend python --repeat 3
python benchmark_neuroboros.py /path/to/data --backend c64  --n-jobs 1  --repeat 3
python benchmark_neuroboros.py /path/to/data --backend c64  --n-jobs 10 --repeat 3
python benchmark_neuroboros.py /path/to/data --backend c32  --n-jobs 1  --repeat 3
python benchmark_neuroboros.py /path/to/data --backend c32  --n-jobs 10 --repeat 3
python benchmark_neuroboros.py /path/to/data --backend metal --n-jobs 1  --repeat 3
python benchmark_neuroboros.py /path/to/data --backend metal --n-jobs 10 --repeat 3
```

Options:
- `--backend`: `python`, `c64`, `c32`, `metal`
- `--n-jobs N`: Number of OpenMP threads for C backends (default: 1). Requires `OPENMP=1` build.
- `--repeat N`: Number of alignment repeats (default: 1). Reports median when N > 1.

On Linux, omit the `metal` backend (it will fall back to `c32` automatically).

### Results

| Backend | Threads | L hemi (s) | R hemi (s) | Total (s) | vs Python |
|---------|---------|-----------|-----------|-----------|-----------|
| Python (numpy/Accelerate) | auto | 15.5 | 15.4 | 30.9 | 1.0x |
| C CPU FP64 | 1 | 38.7 | 38.7 | 77.4 | 0.4x |
| C CPU FP64 | 10 | 15.5 | 15.5 | 31.0 | 1.0x |
| C CPU FP32 | 1 | 35.0 | 35.0 | 69.9 | 0.4x |
| **C CPU FP32** | **10** | **14.4** | **14.4** | **28.8** | **1.1x** |
| C Metal GPU FP32 | 1 | 44.7 | 45.1 | 89.8 | 0.3x |
| C Metal GPU FP32 | 10 | 30.7 | 31.0 | 61.7 | 0.5x |

All backends produce identical output (test-set vertex-wise correlation percentiles):

```
[-0.2561 -0.0149  0.0030  0.0160  0.0282  0.0424  0.0594  0.0840  0.1252  0.2105  0.6171]
```

### Analysis

**Correctness**: All backends produce numerically identical percentile distributions at 4 decimal places. The FP32 backends (CPU and Metal) match FP64 because searchlight weight normalization and sparse accumulation are done in FP64 regardless of the local Procrustes precision.

**Performance**: The Python baseline is fast because numpy calls Apple Accelerate which internally multithreads all BLAS calls across all cores. The Python `searchlight_procrustes` is a single-threaded `for` loop, but each `numpy.linalg.svd` call dispatches multithreaded BLAS underneath (controlled by `VECLIB_MAXIMUM_THREADS` on macOS).

With 10 OpenMP threads, the C CPU FP32 backend (**28.8s**) is the fastest, edging out Python (**30.9s**). C CPU FP64 with 10 threads (31.0s) matches Python. Single-threaded C is 2-2.5x slower than Python, confirming that Python's speed comes from implicit Accelerate multithreading.

**Metal GPU**: The Metal backend uses batched GPU GEMM (256 searchlights per command buffer) with CPU-GPU pipelining — while the GPU computes the next batch of `X^T @ Y` products, the CPU runs Newton iterations for the previous batch. OpenMP parallelizes the CPU Newton stage. Despite these optimizations, Metal is slower than CPU because the ~200x200 local matrices are too small for GPU dispatch overhead to be fully amortized.

**OpenMP scaling** (M4 Pro, 10 performance cores):

| Backend | 1 thread | 10 threads | Speedup |
|---------|----------|------------|---------|
| C CPU FP64 | 77.4s | 31.0s | 2.5x |
| C CPU FP32 | 69.9s | 28.8s | 2.4x |
| C Metal GPU FP32 | 89.8s | 61.7s | 1.5x |

The sub-linear scaling (2.5x on 10 cores) is expected: each searchlight's local SVD/Newton calls LAPACK, which itself uses multiple Accelerate threads internally. The `#pragma omp critical` on scatter-add adds minimal overhead since the SVD computation dominates.

### Replicating on Other Machines

**macOS (Apple Silicon)**:
```bash
cd csrc && make OPENMP=1 && make OPENMP=1 test
cd .. && python benchmark_neuroboros.py /path/to/data --backend c32 --n-jobs 10 --repeat 3
```

**macOS (Intel)**:
```bash
# Metal available but may not have MPS support for all operations
# CPU backends recommended
cd csrc && make OPENMP=1 && make OPENMP=1 test
cd .. && python benchmark_neuroboros.py /path/to/data --backend c64 --n-jobs 4 --repeat 3
```

**Linux (Ubuntu/Debian)**:
```bash
# Install LAPACK/BLAS and OpenMP
sudo apt install liblapack-dev libopenblas-dev libomp-dev

# Build with OpenMP (Metal stubs are automatic; only CPU backends work)
cd csrc && make OPENMP=1 && make OPENMP=1 test

# Run benchmark (metal backend will fall back to c32 automatically)
cd .. && python benchmark_neuroboros.py /path/to/data --backend c32 --n-jobs $(nproc) --repeat 3
```

## References

- Haxby, J.V., Guntupalli, J.S., Connolly, A.C., et al. (2011). A common, high-dimensional model of the representational space in human ventral temporal cortex. *Neuron*, 72(2), 404-416.
- Hanke, M., Baumgartner, F.J., Ibe, P., et al. (2014). A high-resolution 7-Tesla fMRI dataset from complex natural stimulation with an audio movie. *Scientific Data*, 1, 140003.
