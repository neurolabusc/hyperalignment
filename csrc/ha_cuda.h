/*
 * ha_cuda.h -- CUDA GPU backend (NVIDIA only)
 *
 * Provides GPU-accelerated searchlight Procrustes via:
 *   - cuBLAS cublasSgemmStridedBatched  for batched GEMM (X^T @ Y, U @ V^T)
 *   - cuSOLVER gesvdaStridedBatched     for approximate batched SVD
 *   - Custom CUDA kernels               for gather and scatter-add
 *
 * All local Procrustes computation is in FP32. Scatter-add accumulation
 * into the output matrix uses FP64 atomicAdd (native on compute cap >= 6.0).
 *
 * On non-CUDA builds, inline stubs return kHaErrorInternal and
 * ha_cuda_available() returns false.
 */

#ifndef HA_CUDA_H
#define HA_CUDA_H

#include "ha_common.h"

#ifdef HA_CUDA_ENABLED

// extern "C" ensures C linkage (no C++ name mangling) when included from .cu
#ifdef __cplusplus
extern "C" {
#endif

/*
 * Initialize CUDA device, cuBLAS, and cuSOLVER handles.
 * Call once before any CUDA operations. Returns kHaSuccess or kHaErrorInternal.
 */
int ha_cuda_init(void);

/*
 * Release CUDA resources. Safe to call even if ha_cuda_init() was not called.
 */
void ha_cuda_cleanup(void);

/*
 * Returns true if CUDA is initialized and an NVIDIA GPU is available.
 */
bool ha_cuda_available(void);

/*
 * GPU searchlight Procrustes with flat-array input and dense output.
 * Uses batched GEMM + batched approximate SVD entirely on GPU.
 *
 * Restrictions:
 *   - isReflection must be true (reflection=false falls back to CPU FP32)
 *   - isScaling must be false (scaling not supported on GPU)
 *   - col_major must be true (Fortran-order input required)
 *   - X_data, Y_data must be double precision (converted to FP32 internally)
 *
 * Returns kHaSuccess or an error code.
 */
int ha_searchlight_procrustes_cuda(
    const double *X_data, const double *Y_data,
    int32_t nt, int32_t nv,
    const int32_t *sl_indices,
    const int32_t *sl_offsets,
    const double *sl_dists,
    int32_t count,
    double radius,
    double *T_out,
    bool isReflection, bool isScaling,
    bool col_major);

#ifdef __cplusplus
}
#endif

#else  /* !HA_CUDA_ENABLED */

// Stubs for non-CUDA builds
static inline int ha_cuda_init(void) { return kHaErrorInternal; }
static inline void ha_cuda_cleanup(void) {}
static inline bool ha_cuda_available(void) { return false; }

static inline int ha_searchlight_procrustes_cuda(
    const double *X_data, const double *Y_data,
    int32_t nt, int32_t nv,
    const int32_t *sl_indices,
    const int32_t *sl_offsets,
    const double *sl_dists,
    int32_t count,
    double radius,
    double *T_out,
    bool isReflection, bool isScaling,
    bool col_major)
{
    (void)X_data; (void)Y_data; (void)nt; (void)nv;
    (void)sl_indices; (void)sl_offsets; (void)sl_dists;
    (void)count; (void)radius; (void)T_out;
    (void)isReflection; (void)isScaling; (void)col_major;
    return kHaErrorInternal;
}

#endif /* HA_CUDA_ENABLED */

#endif /* HA_CUDA_H */
