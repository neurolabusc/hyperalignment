# CLAUDE.md

## Project Overview

Hyperalignment library: aligns fMRI brain data across subjects by finding optimal transformations that maximize representational similarity. Being ported from Python (numpy/scipy) to C with planned CUDA and Metal GPU backends.

The Python code in `src/hyperalignment/` is the reference implementation. The C port lives in `csrc/`. The `dcm/` directory contains reference C code for style conventions and build patterns but is not part of the hyperalignment library.

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
csrc/                     C port (Phase 1: CPU with Accelerate/LAPACK)
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
- `double` for all floating-point computation (matching Python float64)
- `float` only for NIfTI I/O (NIfTI headers use float32)
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

## Core Data Structures for C Port

```c
// Dense matrix (row-major for LAPACK compatibility with transpose)
typedef struct {
    double *data;       // row-major storage
    int32_t rows;
    int32_t cols;
    int32_t stride;     // >= cols, allows submatrix views
} TMat;

// Sparse CSC matrix (matches scipy.sparse.csc_matrix)
typedef struct {
    double *data;       // non-zero values
    int32_t *indices;   // row indices
    int32_t *indptr;    // column pointers (length cols+1)
    int32_t rows;
    int32_t cols;
    int64_t nnz;
} TSparseCSC;

// Searchlight definition
typedef struct {
    int32_t **indices;  // vertex indices per searchlight
    double **dists;     // distances from center (NULL for uniform weighting)
    int32_t *sizes;     // number of vertices per searchlight
    int32_t count;      // number of searchlights
    double radius;      // searchlight radius
} TSearchlights;
```

## Algorithm-to-Function Mapping

Python function → C function (planned):

| Python | C | Core Operation |
|--------|---|---------------|
| `safe_svd(X)` | `ha_svd(X, U, s, Vt)` | LAPACK `dgesdd`, fallback `dgesvd` |
| `svd_pca(X)` | `ha_pca(X, out)` | SVD then `U * s` |
| `procrustes(X, Y)` | `ha_procrustes(X, Y, T)` | `A = Y'X`, SVD of A, `T = U @ Vt` |
| `ridge(X, Y, alpha)` | `ha_ridge(X, Y, alpha, betas)` | SVD of X, damped solve |
| `ridge_grid(X, y, alphas, npcs)` | `ha_ridge_grid(...)` | Vectorized over alpha/npc grid |
| `initialize_sparse_matrix(sls)` | `ha_sparse_init(sls, mat)` | Build CSC sparsity pattern |
| `compute_searchlight_weights(sls)` | `ha_searchlight_weights(sls, w)` | Distance or uniform weighting |
| `searchlight_procrustes(X, Y, sls)` | `ha_searchlight_procrustes(...)` | Loop: local SVD + sparse accumulate |
| `searchlight_ridge(X, Y, sls)` | `ha_searchlight_ridge(...)` | Loop: local ridge + sparse accumulate |
| `compute_template(dss)` | `ha_template(dss, n_subj, out)` | Iterative Procrustes/GPA/PCA |
| `compute_ensemble_indices(nt)` | `ha_ensemble_indices(...)` | Block permutation CV splits |

## Build System

Uses Make with platform detection. Links against Accelerate on macOS, system LAPACK/BLAS on Linux.

```
csrc/
    Makefile              Platform-detecting build system
    hyperalignment.h      Umbrella header
    ha_common.h           Types (TMat, TSparseCSC, TSearchlights), platform abstraction
    ha_linalg.h/c         SVD (dgesdd + dgesvd fallback), PCA, column mean removal, z-score
    ha_procrustes.h/c     Orthogonal Procrustes
    ha_ridge.h/c          Ridge regression, grid search, ensemble ridge
    ha_sparse.h/c         CSC sparse matrix init from searchlight patterns, scatter-add
    ha_searchlight.h/c    Searchlight weights, searchlight procrustes/ridge loops
    ha_template.h/c       Template construction (Procrustes, GPA, PCA)
    ha_ensemble.h/c       Cross-validation index generation
    ha_test.c             Test harness (1811 tests)
```

### Build commands
```bash
cd csrc
make                  # Release build -> libhyperalignment.a
make test             # Build and run tests
make DEBUG=1 test     # Build with AddressSanitizer and run tests
make OPENMP=1         # Build with OpenMP searchlight parallelism
make clean
```

### Compiler flags
- `-std=c11 -O2 -Wall -Wextra -Wpedantic` for release
- `-std=c11 -g -O0 -fsanitize=address` for debug
- `-framework Accelerate` on macOS (vecLib LAPACK/BLAS)
- `-llapack -lblas -lm` on Linux
- OpenMP opt-in via `OPENMP=1` flag

### Row-major SVD trick
Row-major `X(M,N)` is column-major `X^T(N,M)` to LAPACK. We call `dgesdd("S", N, M, ...)` and swap the U/Vt buffer assignments: LAPACK's "U" output becomes our Vt, LAPACK's "Vt" output becomes our U. No explicit transpose needed.

## LAPACK/BLAS Functions Used

- `dgesdd_` / `dgesvd_`: SVD (the main computational kernel)
- `dgemm_`: Matrix multiply (Procrustes, ridge, template alignment)
- `dgemv_`: Matrix-vector multiply (ridge with single target)
- `dnrm2_`: Vector norm (for normalization)
- `dscal_`: Vector scale
- `dcopy_`: Vector copy

## GPU Porting Notes

### Searchlight Parallelism
The searchlight loop iterates over ~10,000 independent local problems (each a small SVD + GEMM on ~200x200 matrices). This maps naturally to:
- **CUDA**: cuSOLVER batched SVD (`cusolverDnDgesvdjBatched`) + cuBLAS batched GEMM
- **Metal**: Compute shader dispatching one threadgroup per searchlight

### Sparse Accumulation
The scatter-add from local transformations into the global sparse matrix requires atomic operations on GPU. Each searchlight writes to a (sl_size x sl_size) block of the sparse matrix, with overlapping searchlights causing write conflicts.

### Memory Layout
- Searchlight data should be packed contiguously for coalesced GPU memory access
- Pre-extract all searchlight submatrices into a batched buffer before launching GPU kernels
- Result buffer per searchlight, then reduce into sparse matrix on CPU (simpler than GPU atomics)

## Known Issues in Python Code

- `ensemble.py:77,81`: References undefined variables `y` and `T` (should be `Y` and `xmat`)
- `local_template.py:61`: Calls `safe_svd(X, demean=demean)` but the parameter name is `remove_mean`
- `test_searchlight.py` depends on `neuroboros` package (not in requirements.txt)

## Testing

```bash
# Python tests
pytest tests/

# C tests (planned)
make -C csrc test
```

Validate C output against Python reference on identical inputs. Key invariant: SVD is unique up to sign flips of singular vector columns, so compare `abs(U_c)` vs `abs(U_py)` or compare the reconstructed matrix `U @ diag(s) @ Vt`.
