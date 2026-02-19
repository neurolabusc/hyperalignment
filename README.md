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

A complete C implementation lives in `csrc/`. It targets Apple Accelerate on macOS and LAPACK/BLAS on Linux, with planned GPU backends (CUDA, Metal).

### Building

Requires a C11 compiler and a LAPACK/BLAS provider. On macOS the Apple Accelerate framework is used automatically; on Linux, install `liblapack-dev` and `libopenblas-dev` (or equivalent).

```bash
cd csrc
make            # optimized build (-O2)
make DEBUG=1    # debug build with AddressSanitizer
make OPENMP=1   # enable OpenMP for searchlight parallelism
make clean      # remove build artifacts
```

This produces a static library `libhyperalignment.a` and all object files.

### Running Tests

The test suite (`ha_test.c`) contains 1811 deterministic tests covering SVD, PCA, Procrustes, ridge regression, sparse matrices, searchlight weights, z-score, template construction, and ensemble cross-validation.

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

1811 tests: 1811 passed, 0 failed
```

### C Source Layout

```
csrc/
    Makefile              Platform-detecting build system
    ha_common.h           Types (TMat, TSparseCSC, TSearchlights), alloc helpers
    ha_linalg.h/.c        SVD (dgesdd + dgesvd fallback), PCA, z-score
    ha_procrustes.h/.c    Orthogonal Procrustes
    ha_ridge.h/.c         Ridge regression, grid search, ensemble ridge
    ha_sparse.h/.c        CSC sparse matrix init, scatter-add
    ha_searchlight.h/.c   Searchlight weights and alignment loops
    ha_template.h/.c      Template construction (Procrustes, GPA, PCA)
    ha_ensemble.h/.c      Cross-validation index generation
    hyperalignment.h      Umbrella header
    ha_test.c             Test harness
```

### Future GPU Backends

**Phase 2 - CUDA**
- cuSOLVER for batched SVD (all searchlights in one launch)
- cuBLAS for batched GEMM
- Custom kernels for sparse accumulation and normalization

**Phase 3 - Metal**
- Metal Performance Shaders for GEMM
- Custom compute shaders for SVD
- Apple Silicon unified memory

## References

- Haxby, J.V., Guntupalli, J.S., Connolly, A.C., et al. (2011). A common, high-dimensional model of the representational space in human ventral temporal cortex. *Neuron*, 72(2), 404-416.
